import logging
import copy
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from icecream import ic

from rf_diffusion.chemical import ChemicalData as ChemData
import rf2aa.util
import rf2aa.data.data_loader
from rf2aa.util_module import XYZConverter
import rf2aa.data.parsers
import rf2aa.tensor_util
import rf_diffusion.aa_model as aa_model

from rf_diffusion.inference import utils as iu
from rf_diffusion.inference import old_symmetry
from hydra.core.hydra_config import HydraConfig
from rf_diffusion.frame_diffusion.data import all_atom
import rf_diffusion.frame_diffusion.data.utils as du
from rf_diffusion.frame_diffusion.rf_score.model import RFScore
from rf_diffusion import features
from rf_diffusion import noisers
from rf_diffusion.config import config_format
from paths import evaluate_path
from pathlib import Path
import os

import rf_diffusion.inference.data_loader

import sys

# When you import this it causes a circular import due to the changes made in apply masks for self conditioning
# This import is only used for SeqToStr Sampling though so can be fixed later - NRB
# import data_loader 
# from rf_diffusion.model_input_logger import pickle_function_call

logger = logging.getLogger(__name__)

class Sampler:

    def __init__(self, conf: DictConfig):
        """Initialize sampler.
        Args:
            conf: Configuration.
        """
        self.initialized = False
        self.initialize(conf)
    
    def load_model(self):
        """
        Load the model from the checkpoint. Also sets the diffuser

        Returns:
            None
        """
        # Assemble config from the checkpoint
        ic(self._conf.inference.ckpt_path)

        weights_pkl = du.read_pkl(
            evaluate_path(self._conf.inference.ckpt_path), use_torch=True,
                map_location=self.device)

        # WIP: if the conf must be read from a different checkpoint for backwards compatibility
        if hasattr( self._conf, 'score_model') and hasattr( self._conf.score_model, 'conf_pkl_path') and self._conf.score_model.conf_pkl_path:
            print(f'WARNING: READING CONF FROM NON-MODEL PICKLE: {self._conf.score_model.conf_pkl_path} THIS SHOULD ONLY BE DONE FOR DEBUGGING PURPOSES')
            weights_conf = du.read_pkl(
                self._conf.score_model.conf_pkl_path, use_torch=True,
                    map_location=self.device)['conf']
        else:
            weights_conf = weights_pkl['conf']

        weights_conf = config_format.translate_obsolete_weight_options(weights_conf)

        # Load the base training conf based on config path relative to the location of model_runners.py
        file_dir = Path(__file__).resolve().parent
        training_config_fp = os.path.join(file_dir, '../', 'config/training/base.yaml')
        base_training_conf = OmegaConf.load(training_config_fp)

        # Merge base experiment config with checkpoint config.
        OmegaConf.set_struct(self._conf, False)
        OmegaConf.set_struct(weights_conf, False)
        OmegaConf.set_struct(base_training_conf, False)
        self._conf = OmegaConf.merge(
            base_training_conf, weights_conf, self._conf)
        config_format.alert_obsolete_options(self._conf)

        self.diffuser = noisers.get(self._conf.diffuser)
        self.model = RFScore(self._conf.rf.model, self.diffuser, self.device)
        
        ema = 'unknown'
        if self._conf.inference.state_dict_to_load == 'final_state_dict':
            ema = False
        elif self._conf.inference.state_dict_to_load == 'model_state_dict':
            ema = True

        if 'final_state_dict' in weights_pkl:
            ic(ema)
            model_weights = weights_pkl[self._conf.inference.state_dict_to_load] # model_state_dict | final_state_dict
        else:
            model_weights = weights_pkl['model']

        self.model.load_state_dict(model_weights)
        self.model.to(self.device)        

    def initialize(self, conf: DictConfig):
        self._log = logging.getLogger(__name__)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Assign config to Sampler
        self._conf = conf

        # self.initialize_sampler(conf)
        self.initialized=True
        self.load_model()

        # Initialize helper objects
        self.inf_conf = self._conf.inference
        self.denoiser_conf = self._conf.denoiser
        self.ppi_conf = self._conf.ppi
        self.potential_conf = self._conf.potentials
        self.diffuser_conf = self._conf.diffuser
        self.preprocess_conf = self._conf.preprocess
        self.model_adaptor = aa_model.Model(self._conf)

        # TODO: Add symmetrization RMSD check here
        if self._conf.seq_diffuser.seqdiff is None:
            self.seq_diffuser = None

            assert(self._conf.preprocess.seq_self_cond is False), 'AR decoding does not make sense with sequence self cond'
            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        elif self._conf.seq_diffuser.seqdiff == 'continuous':
            ic('Doing Continuous Bit Diffusion')

            kwargs = {
                     'T': self._conf.diffuser.T,
                     's_b0': self._conf.seq_diffuser.s_b0,
                     's_bT': self._conf.seq_diffuser.s_bT,
                     'schedule_type': self._conf.seq_diffuser.schedule_type,
                     'loss_type': self._conf.seq_diffuser.loss_type
                     }
            self.seq_diffuser = seq_diffusion.ContinuousSeqDiffuser(**kwargs)

            self.seq_self_cond = self._conf.preprocess.seq_self_cond

        else:
            sys.exit(f'Seq Diffuser of type: {self._conf.seq_diffuser.seqdiff} is not known')

        if self.inf_conf.old_symmetry is not None:
            self.old_symmetry = old_symmetry.SymGen(
                self.inf_conf.old_symmetry,
                self.inf_conf.model_only_neighbors,
                self.inf_conf.recenter,
                self.inf_conf.radius, 
            )
        else:
            self.old_symmetry = None


        self.converter = XYZConverter()
        self.chain_idx = None

        # self.potential_manager = PotentialManager(self.potential_conf, 
        #                                           self.ppi_conf, 
        #                                           self.diffuser_conf, 
        #                                           self.inf_conf)
        
        # Get recycle schedule    
        recycle_schedule = str(self.inf_conf.recycle_schedule) if self.inf_conf.recycle_schedule is not None else None
        self.recycle_schedule = iu.recycle_schedule(self.diffuser_conf.T, recycle_schedule, self.inf_conf.num_recycles)

        self.dataset = rf_diffusion.inference.data_loader.InferenceDataset(self._conf, self.diffuser)
        
    def sample_init(self, i_des=0):
        """Initial features to start the sampling process.
        
        Modify signature and function body for different initialization
        based on the config.

        Args:
            i_des (int): Design number
        
        Returns:
            indep (Indep): the holy Indep,
            contig_map (ContigMap): the contig_map used to make this indep            
            atomizer (Atomizer): the atomizer,
            t_step_input (torch.tensor): the t_step_input
        """
        indep_uncond, self.indep_orig, self.indep_cond, metadata, self.is_diffused, self.atomizer, contig_map, t_step_input, self.conditions_dict = self.dataset[i_des % len(self.dataset)]
        indep = self.indep_cond.clone()
        self.metadata = metadata
        return indep, contig_map, self.atomizer, t_step_input

    def symmetrise_prev_pred(self, px0, seq_in, alpha):
        """
        Method for symmetrising px0 output, either for recycling or for self-conditioning
        """
        _,px0_aa = self.converter.compute_all_atom(torch.argmax(seq_in, dim=-1), px0, alpha)
        px0_sym,_ = self.old_symmetry.apply_symmetry(px0_aa.to('cpu').squeeze()[:,:14], torch.argmax(seq_in, dim=-1).squeeze().to('cpu'))
        px0_sym = px0_sym[None].to(self.device)
        return px0_sym


class NRBStyleSelfCond(Sampler):
    """
    Model Runner for self conditioning in the style attempted by NRB.

    Works for diffusion and flow matching models.
    """

    def sample_step(self, t, indep, rfo, extra, features_cache):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
            features_cache (dict): Cache of initialized and stored values for t1d/t2d features 
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''

        if self._conf.inference.get('recenter_xt'):
            indep_cond = copy.deepcopy(indep)
            indep_uncond_com = indep.xyz[:,1,:].mean(dim=0)
            indep.xyz = indep.xyz - indep_uncond_com
            indep = aa_model.make_conditional_indep(indep, indep_cond, self.is_diffused)

        extra_tXd_names = getattr(self._conf, 'extra_tXd', [])
        t_cont = t/self._conf.diffuser.T
        indep.extra_t1d, indep.extra_t2d = features.get_extra_tXd_inference(indep, extra_tXd_names, self._conf.extra_tXd_params, self._conf.inference.conditions, t_cont=t_cont, features_cache=features_cache, metadata=getattr(self, 'metadata', {}), **self.conditions_dict)
        rfi = self.model_adaptor.prepro(indep, t, self.is_diffused)

        rf2aa.tensor_util.to_device(rfi, self.device)

        ##################################
        ######## Str Self Cond ###########
        ##################################
        if all([t < self._conf.diffuser.T,
                t != self._conf.diffuser.partial_T,
                self._conf.inference.str_self_cond]):
            rfi = aa_model.self_cond(indep, rfi, rfo, use_cb=self._conf.preprocess.use_cb_to_get_pair_dist)

        if self.old_symmetry is not None:
            idx_pdb, self.chain_idx = self.old_symmetry.res_idx_procesing(res_idx=idx_pdb)

        with torch.no_grad():
            if self.recycle_schedule[t-1] > 1:
                raise Exception('not implemented')
            for _ in range(self.recycle_schedule[t-1]):
                # This is the assertion we should be able to use, but the
                # network's ComputeAllAtom requires even atoms to have N and C coords.
                # aa_model.assert_has_coords(rfi.xyz[0], indep)
                assert not rfi.xyz[0,:,:3,:].isnan().any(), f'{t}: {rfi.xyz[0,:,:3,:]}'
                # Model does not have side chain outputs
                model_out = self.model.forward_from_rfi(rfi, torch.tensor([t/self._conf.diffuser.T]).to(rfi.xyz.device), use_checkpoint=False)

        # Generate rigids
        rigids_t = du.rigid_frames_from_atom_14(rfi.xyz)

        # Default behavior
        rigid_pred = model_out['rigids_raw'][:,-1]
        trans_score = du.move_to_np(model_out['trans_score'][:,-1])
        rot_score = du.move_to_np(model_out['rot_score'][:,-1])

        # Allow control over px0 selection, keeping this if-statement outside of function for back-compatability
        if 'px0_source' in self._conf.inference.keys(): 
            px0 = iu.conf_select_px0(model_out, px0_source=self._conf.inference.px0_source)
        else:
            px0 = model_out['atom37'][0, -1] # Default behavior (fine for proteins only)

        px0 = px0.cpu()

        n_steps = 1
        if 'n_steps' in extra and extra['n_steps'] is not None:
            n_steps = extra['n_steps']
        # This isn't exactly an elegant way to take multiple steps but diffuser.reverse can be very non-linear depending on the diffuser settings
        for step in range(n_steps):
            step_t = t + n_steps - 1 - step
            rigids_t = self.diffuser.reverse(
                rigid_t=rigids_t,
                rot_score=rot_score,
                trans_score=trans_score,
                diffuse_mask=du.move_to_np(self.is_diffused.float()[None,...]),
                t=step_t/self._conf.diffuser.T,
                dt=1/self._conf.diffuser.T,
                center=self._conf.denoiser.center,
                noise_scale=self._conf.denoiser.noise_scale,
                rigid_pred=rigid_pred,
            )
        return px0, get_x_t_1(rigids_t, indep.xyz, self.is_diffused), get_seq_one_hot(indep.seq), model_out['rfo'], {'traj':{}}

def get_x_t_1(rigids_t, xyz, is_diffused):
    x_t_1 = all_atom.atom37_from_rigid(rigids_t)
    x_t_1 = x_t_1[0,:,:ChemData().NTOTAL]  # Conversion from 37 style to 36 style
    # Replace the xyzs of the motif
    x_t_1[~is_diffused.bool(), :ChemData().NHEAVY] = xyz[~is_diffused.bool(), :ChemData().NHEAVY]
    x_t_1 = x_t_1.cpu()
    return x_t_1

def get_seq_one_hot(seq):
    seq_init = torch.nn.functional.one_hot(
            seq, num_classes=ChemData().NAATOKENS).float()
    return seq_init.cpu()
    # seq_t = torch.clone(seq_init)
    # seq_t_1 = seq_t
    # seq_t_1 = seq_t_1.cpu()
    # return seq_t_1

class FlowMatching(Sampler):
    """
    Model Runner for flow matching.
    """

    def run_model(self, t, indep, rfo, is_diffused, features_cache):
        extra_tXd_names = getattr(self._conf, 'extra_tXd', [])
        t_cont = t/self._conf.diffuser.T
        indep.extra_t1d, indep.extra_t2d = features.get_extra_tXd_inference(indep, extra_tXd_names, self._conf.extra_tXd_params, self._conf.inference.conditions, t_cont=t_cont, features_cache=features_cache, metadata=getattr(self, 'metadata', {}), **self.conditions_dict)
        rfi = self.model_adaptor.prepro(indep, t, is_diffused)
        rf2aa.tensor_util.to_device(rfi, self.device)

        ##################################
        ######## Str Self Cond ###########
        ##################################
        if all([t < self._conf.diffuser.T,
                t != self._conf.diffuser.partial_T,
                self._conf.inference.str_self_cond]):
            rfi = aa_model.self_cond(indep, rfi, rfo, use_cb=self._conf.preprocess.use_cb_to_get_pair_dist)

        if self.old_symmetry is not None:
            idx_pdb, self.chain_idx = self.old_symmetry.res_idx_procesing(res_idx=idx_pdb)

        with torch.no_grad():
            # assert not rfi.xyz[0,:,:3,:].isnan().any(), f'{t}: {rfi.xyz[0,:,:3,:]}'
            model_out = self.model.forward_from_rfi(rfi, torch.tensor([t/self._conf.diffuser.T]).to(rfi.xyz.device), use_checkpoint=False)
        return model_out

    def get_grads_rigid(self, rigids_t, rigids_pred, t, model_out):
        trans_grad, rots_grad = self.diffuser.get_grads(
            rigid_t=rigids_t,
            rot_score=du.move_to_np(model_out['rot_score'][:,-1]),
            trans_score=du.move_to_np(model_out['trans_score'][:,-1]),
            diffuse_mask=np.ones(rigids_pred.shape, dtype=bool),
            t=t/self._conf.diffuser.T,
            dt=1/self._conf.diffuser.T,
            center=self._conf.denoiser.center,
            noise_scale=self._conf.denoiser.noise_scale,
            rigid_pred=rigids_pred,
        )
        return trans_grad, rots_grad

    def get_grads(self, t, indep_in, indep_t, rfo, is_diffused, features_cache):

        model_out = self.run_model(t, indep_in, rfo, is_diffused, features_cache)
        rigids_pred = model_out['rigids_raw'][:,-1]
        rigids_t = du.rigid_frames_from_atom_14(indep_t.xyz.to(self.device))
        trans_grad, rots_grad = self.get_grads_rigid(rigids_t, rigids_pred, t, model_out)

        px0 = model_out['atom37'][0, -1]
        px0 = px0.cpu()

        return trans_grad, rots_grad, px0, model_out
    
    def get_rigids(self):
        rigids = du.rigid_frames_from_atom_14(self.xyz)
        return rigids

    def sample_step(self, t, indep, rfo, extra, features_cache):
        '''
        Generate the next pose that the model should be supplied at timestep t-1.
        Args:
            t (int): The timestep that has just been predicted
            seq_t (torch.tensor): (L,22) The sequence at the beginning of this timestep
            x_t (torch.tensor): (L,14,3) The residue positions at the beginning of this timestep
            seq_init (torch.tensor): (L,22) The initialized sequence used in updating the sequence.
            features_cache (dict): data cache for features
        Returns:
            px0: (L,14,3) The model's prediction of x0.
            x_t_1: (L,14,3) The updated positions of the next step.
            seq_t_1: (L) The updated sequence of the next step.
            tors_t_1: (L, ?) The updated torsion angles of the next  step.
            plddt: (L, 1) Predicted lDDT of x0.
        '''
        ic('sample using FM model')
        trans_grad, rots_grad, px0, model_out = self.get_grads(t, indep, indep, rfo, self.is_diffused)
        trans_dt, rots_dt = self.diffuser.get_dt(t/self._conf.diffuser.T, 1/self._conf.diffuser.T)
        rigids_t = du.rigid_frames_from_atom_14(indep.xyz.to(self.device))[None,...]
        rigids_t = self.diffuser.apply_grads(rigids_t, trans_grad, rots_grad, trans_dt, rots_dt)
        x_t_1 = get_x_t_1(rigids_t, indep.xyz, self.is_diffused)
    
        return px0, x_t_1, get_seq_one_hot(indep.seq), model_out['rfo'], {'traj':{}}


class DifferentialAtomizedDecoder(FlowMatching):

    def __init__(self, conf):
        super().__init__(conf)
        atomized_diffuser_conf = copy.deepcopy(self._conf.diffuser)
        OmegaConf.set_struct(self._conf.diffuser, False)
        OmegaConf.set_struct(self._conf.atomized_diffuser_overrides, False)
        atomized_diffuser_conf = OmegaConf.merge(
            self._conf.diffuser, self._conf.atomized_diffuser_overrides)
        self.atomized_diffuser = noisers.get(atomized_diffuser_conf)

    def sample_step(self, t, indep, rfo, extra, features_cache):
        
        # res_atom_by_i = atomize.get_res_atom_name_by_atomized_idx(atomizer)
        atomized_res_idx_from_res = self.atomizer.get_atom_idx_by_res()
        atomized_indices = []
        for v in atomized_res_idx_from_res.values():
            atomized_indices.extend(v)
        
        if self._conf.inference.differential_atomized_decoder_include_sm:
            atomized_indices = indep.is_sm
        
        trans_grad, rots_grad, px0, model_out = self.get_grads(t, indep, indep, rfo, self.is_diffused, features_cache)
        trans_dt, rots_dt = self.diffuser.get_dt(t/self._conf.diffuser.T, 1/self._conf.diffuser.T)

        ic(
            trans_dt,
            rots_dt,
        )
        trans_dt = torch.full((indep.length(), 3), trans_dt, device=self.device)
        rots_dt = torch.full((indep.length(), 3), rots_dt, device=self.device)

        atomized_trans_dt, atomized_rots_dt = self.atomized_diffuser.get_dt(t/self._conf.diffuser.T, 1/self._conf.diffuser.T)
        trans_dt[atomized_indices] = atomized_trans_dt
        rots_dt[atomized_indices] = atomized_rots_dt

        rigids_t = du.rigid_frames_from_atom_14(indep.xyz.to(self.device))[None,...]
        rigids_t = self.diffuser.apply_grads(rigids_t, trans_grad, rots_grad, trans_dt, rots_dt)
        x_t_1 = get_x_t_1(rigids_t, indep.xyz, self.is_diffused)
    
        return px0, x_t_1, get_seq_one_hot(indep.seq), model_out['rfo'], {'traj':{}}

    

def sampler_selector(conf: DictConfig):
    if conf.inference.model_runner == 'default':
        sampler = Sampler(conf)
    elif conf.inference.model_runner == 'NRBStyleSelfCond':
        sampler = NRBStyleSelfCond(conf)
    elif conf.inference.model_runner == 'FlowMatching':
        sampler = FlowMatching(conf)
    elif conf.inference.model_runner == 'FlowMatching_make_conditional':
        sampler = FlowMatching_make_conditional(conf)
    elif conf.inference.model_runner == 'NRBStyleSelfCond_debug':
        sampler = NRBStyleSelfCond_debug(conf)
    elif conf.inference.model_runner == 'ClassifierFreeGuidance':
        sampler = ClassifierFreeGuidance(conf)
    elif conf.inference.model_runner == 'DifferentialAtomizedDecoder':
        sampler = DifferentialAtomizedDecoder(conf)
    elif conf.inference.model_runner in globals():
        sampler = globals()[conf.inference.model_runner](conf)
    else:
        raise ValueError(f'Unrecognized sampler {conf.inference.model_runner}')
    return sampler


def assemble_config_from_chk(conf, ckpt) -> None:
    """
    Function for loading model config from checkpoint directly.

    Takes:
        - config file

    Actions:
        - Replaces all -model and -diffuser items
        - Throws a warning if there are items in -model and -diffuser that aren't in the checkpoint
    
    This throws an error if there is a flag in the checkpoint 'config_dict' that isn't in the inference config.
    This should ensure that whenever a feature is added in the training setup, it is accounted for in the inference script.

    JW
    """
    
    # get overrides to re-apply after building the config from the checkpoint
    overrides = []
    if HydraConfig.initialized():
        overrides = HydraConfig.get().overrides.task
        ic(overrides)
    if 'config_dict' in ckpt.keys():
        print("Assembling -model, -diffuser and -preprocess configs from checkpoint")

        # First, check all flags in the checkpoint config dict are in the config file
        for cat in ['model','diffuser','seq_diffuser','preprocess']:
            #assert all([i in self._conf[cat].keys() for i in self.ckpt['config_dict'][cat].keys()]), f"There are keys in the checkpoint config_dict {cat} params not in the config file"
            for key in conf[cat]:
                if key == 'chi_type' and ckpt['config_dict'][cat][key] == 'circular':
                    ic('---------------------------------------------SKIPPPING CIRCULAR CHI TYPE')
                    continue
                try:
                    print(f"USING MODEL CONFIG: self._conf[{cat}][{key}] = {ckpt['config_dict'][cat][key]}")
                    conf[cat][key] = ckpt['config_dict'][cat][key]
                except KeyError:
                    print(f'WARNING: config {cat}.{key} is not saved in the checkpoint. Check that conf.{cat}.{key} = {conf[cat][key]} is correct')
        # add back in overrides again
        for override in overrides:
            if override.split(".")[0] in ['model','diffuser','seq_diffuser','preprocess']:
                print(f'WARNING: You are changing {override.split("=")[0]} from the value this model was trained with. Are you sure you know what you are doing?') 
                mytype = type(conf[override.split(".")[0]][override.split(".")[1].split("=")[0]])
                conf[override.split(".")[0]][override.split(".")[1].split("=")[0]] = mytype(override.split("=")[1])
    else:
        print('WARNING: Model, Diffuser and Preprocess parameters are not saved in this checkpoint. Check carefully that the values specified in the config are correct for this checkpoint')     

    print('self._conf:')
    ic(conf)

class FlowMatching_make_conditional(FlowMatching):
    
    def sample_step(self, t, indep, *args, **kwargs):
        indep = aa_model.make_conditional_indep(indep, self.indep_cond, self.is_diffused)
        return super().sample_step(t, indep, *args, **kwargs)

class FlowMatching_make_conditional_diffuse_all(FlowMatching_make_conditional):

    def sample_init(self, i_des=0):
        indep_uncond, self.indep_orig, self.indep_cond, metadata, self.is_diffused, atomizer, contig_map, t_step_input, self.conditions_dict = self.dataset[i_des % len(self.dataset)]
        self.metadata = metadata
        return indep_uncond, contig_map, atomizer, t_step_input

class FlowMatching_make_conditional_diffuse_all_xt_unfrozen(FlowMatching):

    def sample_init(self, i_des=0):
        indep_uncond, self.indep_orig, self.indep_cond, metadata, self.is_diffused, atomizer, contig_map, t_step_input, self.conditions_dict = self.dataset[i_des % len(self.dataset)]
        self.metadata = metadata
        return indep_uncond, contig_map, atomizer, t_step_input
    
    def sample_step(self, t, indep, rfo, extra, features_cache):
        indep_cond = aa_model.make_conditional_indep(indep, self.indep_cond, self.is_diffused)
        trans_grad, rots_grad, px0, model_out = self.get_grads(t, indep_cond, indep, rfo, self.is_diffused)
        trans_dt, rots_dt = self.diffuser.get_dt(t/self._conf.diffuser.T, 1/self._conf.diffuser.T)
        rigids_t = du.rigid_frames_from_atom_14(indep.xyz)[None,...]
        rigids_t = self.diffuser.apply_grads(rigids_t, trans_grad, rots_grad, trans_dt, rots_dt)
    
        uncond_is_diffused = torch.ones_like(self.is_diffused).bool()
        x_t_1 = get_x_t_1(rigids_t, indep.xyz, uncond_is_diffused)
        return px0, x_t_1, get_seq_one_hot(indep.seq), model_out['rfo'], {'traj':{}}


class ClassifierFreeGuidance(FlowMatching):
    # WIP
    def sample_init(self, i_des=0):
        indep_uncond, self.indep_orig, self.indep_cond, metadata, self.is_diffused, atomizer, contig_map, t_step_input, self.conditions_dict = self.dataset[i_des % len(self.dataset)]
        self.metadata = metadata
        return indep_uncond, contig_map, atomizer, t_step_input
    
    def get_grads(self, t, indep_in, indep_t, rfo, is_diffused, features_cache):

        model_out = self.run_model(t, indep_in, rfo, is_diffused, features_cache)
        rigids_pred = model_out['rigids_raw'][:,-1]
        rigids_t = du.rigid_frames_from_atom_14(indep_t.xyz.to(self.device))
        trans_grad, rots_grad = self.get_grads_rigid(rigids_t, rigids_pred, t, model_out)

        px0 = model_out['atom37'][0, -1]
        px0 = px0.cpu()

        return trans_grad, rots_grad, px0, model_out
    
    def sample_step(self, t, indep, rfo, extra, features_cache):
        if self._conf.inference.get('classifier_free_guidance_recenter_xt'):
            if self._conf.inference.str_self_cond:
                print('warning, self._conf.inference.str_self_cond is true, may need to change')
            indep_uncond_com = indep.xyz[:,1,:].mean(dim=0)
            indep.xyz = indep.xyz - indep_uncond_com

        uncond_is_diffused = torch.ones_like(self.is_diffused).bool()
        indep_cond = aa_model.make_conditional_indep(indep, self.indep_cond, self.is_diffused)
        with torch.random.fork_rng():
            trans_grad_cond, rots_grad_cond, px0_cond, model_out_cond = self.get_grads(t, indep_cond, indep, extra['rfo_cond'], self.is_diffused, features_cache)

        extra_out = {'rfo_cond': model_out_cond['rfo']}
        trans_grad, rots_grad, px0_uncond, model_out_uncond = self.get_grads(t, indep, indep, extra['rfo_uncond'], uncond_is_diffused, features_cache)
        extra_out['rfo_uncond'] = model_out_uncond['rfo']
        w = self._conf.inference.classifier_free_guidance_scale
        if self._conf.inference.get('classifier_free_guidance_ignore_rots'):
            rots_grad = rots_grad_cond
        else:
            rots_grad = (1-w) * rots_grad + w * rots_grad_cond
        if self._conf.inference.get('classifier_free_guidance_ignore_trans'):
            trans_grad = trans_grad_cond
        else:
            trans_grad = (1-w) * trans_grad + w * trans_grad_cond
        trans_dt, rots_dt = self.diffuser.get_dt(t/self._conf.diffuser.T, 1/self._conf.diffuser.T)
        rigids_t = du.rigid_frames_from_atom_14(indep.xyz)
        rigids_t = self.diffuser.apply_grads(rigids_t, trans_grad, rots_grad, trans_dt, rots_dt)

        # TODO: write both px0 trajectories
        px0 = px0_cond
        if w == 0:
            px0 = px0_uncond

        extra_out['traj'] = {
            'px0_cond': px0_cond[:,:ChemData().NHEAVY],
            'px0_uncond': px0_uncond[:,:ChemData().NHEAVY],
            'Xt_cond': indep_cond.xyz[:,:ChemData().NHEAVY],
            'Xt_uncond': indep.xyz[:,:ChemData().NHEAVY],
        }
        x_t_1 = get_x_t_1(rigids_t, indep.xyz, uncond_is_diffused)
        return px0, x_t_1, get_seq_one_hot(indep.seq), extra_out['rfo_cond'], extra_out
