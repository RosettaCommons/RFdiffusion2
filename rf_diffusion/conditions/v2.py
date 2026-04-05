import torch

from rf_diffusion import aa_model

from rf_diffusion import sasa
from rf_diffusion import nucleic_compatibility_utils as nucl_utils

import torch.nn.functional as F
import numpy as np

def get_relative_sasa(indep, conf=None, **kwargs):
    if 1 - torch.rand(1) > conf.get('prob', 0.5): # 1 - for test consistency
        return {'t1d':torch.zeros((indep.length(), conf.n_bins + 1))}
    rasa = sasa.get_relative_sasa(indep)
    is_feature_applicable = indep.is_sm
    one_hot = one_hot_buckets(rasa, conf.low, conf.high, conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}

def radius_of_gyration_xyz(xyz):
    L, _ = xyz.shape
    com = torch.mean(xyz, dim=0)
    dist = torch.cdist(xyz[None,...], com[None,...])[0]
    return torch.sqrt( torch.sum(torch.square(dist)) / L)

def get_radius_of_gyration(indep, conf=None, **kwargs):
    if 1 - torch.rand(1) > conf.get('prob', 0.5): # 1 - for test consistency
        return {'t1d':torch.zeros((indep.length(), conf.n_bins + 1))}
    rog = torch.zeros((indep.length(),))
    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = ~indep.is_sm * ~indep.is_gp * ~is_nucl
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)
    rog_prot = torch.full((indep_prot.length(),), 0.0)
    
    # Iterate over all protein chains and calculate radii of gyration
    for is_chain in indep_prot.chain_masks():
        rog_chain = radius_of_gyration_xyz(indep_prot.xyz[is_chain, 1])
        rog_prot[is_chain] = rog_chain
    rog[is_prot] = rog_prot
    is_feature_applicable = is_prot
    one_hot = one_hot_buckets(rog, conf.low, conf.high, conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}


def one_hot_buckets(a, low, high, n, eps=1e-6):
    '''
    First category absorbs anything below low
    Last category absorbs anything above high
    '''
    a = a.float()
    buckets = torch.linspace(low, high+eps, n+1)
    bucket_idx = torch.searchsorted(buckets, a) - 1
    bucket_idx = torch.clamp(bucket_idx, 0, n-1)
    return F.one_hot(bucket_idx, n)

def init_radius_of_gyration(indep, feature_conf, feature_inference_conf, **kwargs):
    """
    Initialize the radius of gyration fature


    During interface use the following additional parameters
    "spread" to give a normal distribution std in addition to rog, which now becomes the mean

    Args:
        indep (Indep): The independent variable.
        feature_conf (omegaconf): The feature config.
        feature_inference_conf (omegaconf): The feature inference config.

    Returns:
        None
    """
    cache = {}

    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = ~indep.is_sm * ~indep.is_gp * ~is_nucl
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)     
    # Create random values if required
    rog_vals = [(max([0, np.random.normal(feature_inference_conf.rog, 
                                            feature_inference_conf.spread)]) 
                if 'spread' in feature_inference_conf 
                else feature_inference_conf.rog) # Use default value if not randomized
                for _ in indep_prot.chain_masks()]
    cache['rog_vals_cache'] = rog_vals

    return cache

def get_radius_of_gyration_inference(indep, feature_conf, feature_inference_conf, cache, **kwargs):
    """
    Calculates the radius of gyration fature

    Args:
        indep (Indep): The holy indep.
        feature_conf (omegaconf): The feature config.
        feature_inference_conf (omegaconf): The feature inference config.
        cache (dict): data cache

    Returns:
        rog feature
    """    
    # TODO: Currently assumes single diffusing chain
    if not feature_inference_conf.active:
        return {'t1d':torch.zeros((indep.length(), feature_conf.n_bins + 1))}
    rog = torch.zeros((indep.length(),))
    is_nucl = nucl_utils.get_resi_type_mask(indep.seq, 'na')
    is_prot = ~indep.is_sm * ~indep.is_gp * ~is_nucl
    indep_prot, _ = aa_model.slice_indep(indep, is_prot)    
    rog_prot = torch.full((indep_prot.length(),), 0.0)
    rog_vals = cache['rog_vals_cache']

    for is_chain, rog_chain in zip(indep_prot.chain_masks(), rog_vals):
        rog_prot[is_chain] = rog_chain

    rog[is_prot] = rog_prot
    is_feature_applicable = is_prot
    one_hot = one_hot_buckets(rog, feature_conf.low, feature_conf.high, feature_conf.n_bins)
    one_hot[~is_feature_applicable] = 0
    out = torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)
    ic(out[0:2, :], out[-3:-1, :])
    return out

def parse_atomwise_rasa_config(rasa_config, indep, metadata):
    """
    Parse the atomwise RASA configuration string and create a per-atom RASA tensor.
    
    Args:
        rasa_config (str or float): Either a single float for global RASA, or 
                                   a string like "0.0,O7:0.8,C8:1.0,C9:1.0"
        indep (Indep): The indep object containing is_sm mask
        metadata (dict): Metadata containing ligand_atom_names
        
    Returns:
        torch.Tensor: Per-atom RASA values for the entire indep
    """
    rasa = torch.full((indep.length(),), 0.0)
    
    # If it's just a number, apply globally to small molecules
    if isinstance(rasa_config, (float, int)):
        rasa[indep.is_sm] = float(rasa_config)
        return rasa
    
    # Parse the string format: "global_value,atom1:value1,atom2:value2,..."
    config_str = str(rasa_config)
    parts = [p.strip() for p in config_str.split(',')]
    global_value = float(parts[0])
    rasa[indep.is_sm] = global_value
    
    if not metadata or 'ligand_atom_names' not in metadata:
        print("[RASA WARNING] No metadata or ligand_atom_names found, using global RASA")
        return rasa
    
    # Build atom name to specific RASA mapping
    atom_rasa_map = {}
    for part in parts[1:]:
        if ':' not in part:
            continue
        atom_name, value_str = part.split(':', 1)
        atom_rasa_map[atom_name.strip()] = float(value_str.strip())
    if not atom_rasa_map:
        return rasa
    
    # Apply atom-specific values
    ligand_atom_names = metadata['ligand_atom_names']
    sm_indices = torch.where(indep.is_sm)[0]
    n_sm_atoms = len(sm_indices)
    
    # Validate indices
    if n_sm_atoms > len(ligand_atom_names):
        print(f"[RASA ERROR] More SM atoms ({n_sm_atoms}) than ligand names ({len(ligand_atom_names)})")
        return rasa
    
    # Ligand atom names are stored at the end of the array
    ligand_names_start = len(ligand_atom_names) - n_sm_atoms
    matched_atoms = []
    for i, sm_idx in enumerate(sm_indices):
        ligand_name_idx = ligand_names_start + i
        if ligand_name_idx < len(ligand_atom_names):
            atom_name_in_metadata = ligand_atom_names[ligand_name_idx].strip()
            if atom_name_in_metadata in atom_rasa_map:
                rasa[sm_idx] = atom_rasa_map[atom_name_in_metadata]
                matched_atoms.append(f"{atom_name_in_metadata}={atom_rasa_map[atom_name_in_metadata]}")
    
    if matched_atoms:
        print(f"[RASA] Set atom-specific RASA for {len(matched_atoms)} atoms: {', '.join(matched_atoms)}")
    
    return rasa

def get_relative_sasa_inference(indep, feature_conf, feature_inference_conf, cache, **kwargs):
    """
    Calculates the relative SASA feature with support for atom-wise specification

    Args:
        indep (Indep): The holy indep.
        feature_conf (omegaconf): The feature config.
        feature_inference_conf (omegaconf): The feature inference config.
        cache (dict): data cache
        **kwargs: Additional keyword arguments including metadata

    Returns:
        dict: Dictionary with 't1d' key containing the SASA feature tensor
    """  
    if not feature_inference_conf.active:
        return {'t1d':torch.zeros((indep.length(), feature_conf.n_bins + 1))}
    
    metadata = kwargs.get('metadata', {})
    rasa = parse_atomwise_rasa_config(feature_inference_conf.rasa, indep, metadata)
    one_hot = one_hot_buckets(rasa, feature_conf.low, feature_conf.high, feature_conf.n_bins)
    is_feature_applicable = indep.is_sm
    one_hot[~is_feature_applicable] = 0
    return {'t1d':torch.cat((~is_feature_applicable[:, None], one_hot), dim=1)}
