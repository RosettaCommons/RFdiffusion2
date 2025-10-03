from functools import partial
import glob
import os
import re
import logging

import itertools
from dataclasses import dataclass
from itertools import *
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from tqdm.notebook import trange, tqdm
import numpy as np

from rf_diffusion import aa_model
import rf_diffusion.parsers as parsers
import rf_diffusion
PKG_DIR = rf_diffusion.__path__[0]
REPO_DIR = os.path.dirname(PKG_DIR)

from rf_diffusion.dev.pymol import cmd
from rf_diffusion.dev.pymol import mass_paper_rainbow_sel

import rf_diffusion.estimate_likelihood as el
from rf_diffusion.inference import utils
from itertools import takewhile

import time

logger = logging.getLogger(__name__)


class ExecutionTimer:

    def __init__(self, message='Execution time', unit='s'):
        self.message = message
        self.unit = unit
        self.start_time = None
        self.end_time = None
        self.result = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.result = self.end_time - self.start_time
        self.print_time()

    def print_time(self):
        elapsed_time = self.result
        if self.unit == 'ms':
            elapsed_time *= 1000  # convert to milliseconds
        elif self.unit == 'us':
            elapsed_time *= 1e6  # convert to microseconds
        elif self.unit == 'ns':
            elapsed_time *= 1e9  # convert to nanoseconds
        else:
            assert self.unit == 's', f"Invalid unit: {self.unit}"

        print(f"{self.message}: {elapsed_time:.2f}{self.unit}")


logger.debug(f'initializing analyze, cmd: {cmd}')
DESIGN = 'design_path'

def common_prefix(strlist):
    strlist=[e if isinstance(e, str) else 'notfound' for e in strlist]
    return ''.join(c[0] for c in takewhile(lambda x:
                all(x[0] == y for y in x), zip(*strlist)))


def get_pdb_path(row):
    possible_paths = [
            os.path.join(row['rundir'], 'ligmpnn/packed', f"{row['name']}_{row.get('mpnn_index', -1)}.pdb"),
            os.path.join(row['rundir'], 'ligmpnn', f"{row['name']}_{row.get('mpnn_index')}.pdb"),
            os.path.join(row['rundir'], f"{row['name']}.pdb"),
            os.path.join(row['rundir'], 'mpnn/packed', f"{row['name']}_{row.get('mpnn_index', -1)}.pdb"),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    raise Exception(f'pdb not found in {possible_paths} ')

def get_epoch(row, model_key='inference.ckpt_path'):
    ckpt = row[model_key]
    ckpt = ckpt.split('_')[-1]
    ckpt = ckpt[:-3]
    return float(ckpt)

def get_source(rundir):
    return os.path.basename(rundir.removesuffix('/').removesuffix('out'))

def read_metrics_simple(df_path, **kwargs):
    print(f'read_metrics_simple: reading metrics from {df_path}')
    with ExecutionTimer(f'read_metrics_simple: read_csv(*args, {kwargs=})'):
        df = pd.read_csv(df_path, **kwargs)

    df['rundir'] = os.path.split(df_path)[0]
    add_derived_values(df)
    return df

def get_cond(name):
    if 'cond' not in name:
        return -1
    return int(name.split('_cond')[1].split('_')[1].split('-')[0])

def add_derived_values(df):

    with ExecutionTimer('read_metrics_simple: computing derived values'):
        df['method'] = 'placeholder_method'
        df['run'] = df['name'].apply(lambda x: x.split('_')[0])
        # Parse the cond.
        try:
            df['cond'] = df['name'].apply(lambda x: x.split('_cond')[1].split('_')[0])
        except Exception as e:
            print('failed to get cond', e)
        try:
            df['benchmark'] = [n[n.index('_')+1:n.index('_cond')] for n in df.name]
        except Exception as e:
            print('failed to get benchmark', e)

        try:
            df['run'] = [n[:n.index('_')] for n in df.name]
        except Exception as e:
            print('failed to get run', e)
        # For backwards compatibility
        model_key = 'inference.ckpt_path'
        possible_model_keys = ['inference.ckpt_path', 'score_model.weights_path', 'inference.ckpt_override_path']
        count_by_model_key = {k: len(df.value_counts(k)) if k in df.columns else 0 for k in possible_model_keys}
        model_key = max(count_by_model_key, key=count_by_model_key.get)
        if not any(k in df.columns for k in possible_model_keys):
            model_key = 'fake_model_key'
            df[model_key] = 'SOME_MODEL_1.pt'
        df[model_key] = df[model_key].map(lambda x: x if isinstance(x, str) else "MODEL_NOT_FOUND")
        df['model_key_name'] = model_key
        # df = df[df[model_key] != "MODEL_NOT_FOUND"]
        if model_key in df.columns:
            models = df[model_key].unique()
            common = common_prefix(models)
            df['model'] = df[model_key].apply(lambda x: x[len(common):])
            df['seed'] = df.name.apply(get_cond)
        if 'diffuser.type' not in df:
            df['diffuser.type'] = 'diffuser_unknown'
        df['diffuser.type'] = df['diffuser.type'].fillna('diffusion')
        df['mpnn_index'] = df['mpnn_index'].astype(int)
        df['des_color'] = 'rainbow'

def read_metrics(df_path, **kwargs):
    df = read_metrics_simple(df_path, **kwargs)
    with ExecutionTimer('read_metrics: computing derived values'):
        model_key = df['model_key_name'].iloc[0]
        df['epoch'] = df.apply(partial(get_epoch, model_key=model_key), axis=1)
        df['pdb_path'] = df.apply(get_pdb_path, axis=1)

    #get_epoch  = lambda x: re.match('.*_(\w+).*', x).groups()[0]

    #df['model'] = df['inference.ckpt_path'].apply(get_epoch)
    # for tm_cluster in ['tm_cluster_0.40', 'tm_cluster_0.60', 'tm_cluster_0.80']:
    #     df['i_'+tm_cluster] = df.apply(lambda x: x[tm_cluster].split('_clus')[-1], axis=1)
    #df['contig_rmsd'] = df.apply(lambda x: get_contig_c_alpha_rmsd(x).item(), axis=1)
    return df

def combine(*df_paths, names=None, simple=True, **kwargs):
    f = read_metrics
    if simple:
        f = read_metrics_simple
    to_cat = []
    for i,p in enumerate(df_paths):
        with ExecutionTimer(f'Loaded {p}'):
            df = f(p, **kwargs)
        #_, base = os.path.split(p)
        root, _ = os.path.splitext(os.path.abspath(p))
        root = root.split('/')[-3]
        df['metrics_path'] = p
        if names:
            df['source'] = names[i]
        else:
            df['source'] = df['rundir'].map(get_source)

        df['design_id'] = df['source'] + '_' + df['name']
        to_cat.append(df)
    return pd.concat(to_cat)

num2aa=[
    'ALA','ARG','ASN','ASP','CYS',
    'GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO',
    'SER','THR','TRP','TYR','VAL', 'MAS'
    ]

aa2num= {x:i for i,x in enumerate(num2aa)}
aa2num['MEN'] = 20

# full sc atom representation (Nx14)
aa2long=[
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None), # arg
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None), # asn
    (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None), # asp
    (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None), # cys
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None), # gln
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None), # glu
    (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None), # his
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None), # ile
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None), # lys
    (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None), # met
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None), # phe
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None), # pro
    (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None), # ser
    (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # thr
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE2"," CE3"," NE1"," CZ2"," CZ3"," CH2"), # trp
    (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None), # tyr
    (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None), # val
    (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), #21 mask
]

def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True):
    # indices of residues observed in the structure
    res = [(l[22:26],l[17:20]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
    seq = [aa2num[r[1]] if r[1] in aa2num.keys() else 20 for r in res]
    pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
    for l in lines:
        if l[:4] != "ATOM":
            continue
        chain, resNo, atom, aa = l[21:22], int(l[22:26]), ' '+l[12:16].strip().ljust(3), l[17:20]
        idx = pdb_idx.index((chain,resNo))
        for i_atm, tgtatm in enumerate(aa2long[aa2num[aa]]):
            if tgtatm is not None and tgtatm.strip() == atom.strip(): # ignore whitespace
                xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[...,0]))
    xyz[np.isnan(xyz[...,0])] = 0.0

    # remove duplicated (chain, resi)
    new_idx = []
    i_unique = []
    for i,idx in enumerate(pdb_idx):
        if idx not in new_idx:
            new_idx.append(idx)
            i_unique.append(i)

    pdb_idx = new_idx
    xyz = xyz[i_unique]
    mask = mask[i_unique]
    seq = np.array(seq)[i_unique]

    out = {'xyz':xyz, # cartesian coordinates, [Lx14]
            'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
            'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
            'seq':np.array(seq), # amino acid sequence, [L]
            'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
           }

    # heteroatoms (ligands, etc)
    if parse_hetatom:
        xyz_het, info_het = [], []
        for l in lines:
            if l[:6]=='HETATM' and not (ignore_het_h and l[77]=='H'):
                info_het.append(dict(
                    idx=int(l[7:11]),
                    atom_id=l[12:16],
                    atom_type=l[77],
                    name=l[16:20]
                ))
                xyz_het.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])

        out['xyz_het'] = np.array(xyz_het)
        out['info_het'] = info_het

    return out


def get_input_pdb(row):
    trb = get_trb(row)
    # if args.template_dir is not None and os.path.exists(trbname):
    refpdb_fn = trb['config']['inference']['input_pdb'] # diffusion outputs
    return refpdb_fn

def pdb_to_xyz_idx(pdb, chain_i):
    parsed = parsers.parse_pdb(pdb)
    idxmap = dict(zip(parsed['pdb_idx'],range(len(parsed['pdb_idx']))))
    idx = [idxmap[e] for e in chain_i]
    return idx

def get_idx_motif(row, mpnn=True):
    rundir = row['rundir']
    trb = get_trb(row)
    # if args.template_dir is not None and os.path.exists(trbname):
    refpdb_fn = trb['config']['inference']['input_pdb'] # diffusion outputs
    pdb_ref = parse_pdb(refpdb_fn)
    if mpnn:
        pdb_des = parse_pdb(os.path.join(rundir, 'mpnn', row['name']+'_0.pdb'))
    else:
        pdb_des = parse_pdb(os.path.join(rundir, row['name']+'.pdb'))
    #pdb_des = parse_pdb(os.path.join(rundir, 'mpnn', row['name']+'.pdb'))
    # pdb_ref = parse_pdb(template_dir+trb['settings']['pdb'].split('/')[-1])

    # calculate 0-indexed motif residue positions (ignore the ones from the trb)
    # if os.path.exists(trbname):
    idxmap = dict(zip(pdb_ref['pdb_idx'],range(len(pdb_ref['pdb_idx']))))
    trb['con_ref_idx0'] = np.array([idxmap[i] for i in trb['con_ref_pdb_idx']])
    idxmap = dict(zip(pdb_des['pdb_idx'],range(len(pdb_des['pdb_idx']))))
    trb['con_hal_idx0'] = np.array([idxmap[i] for i in trb['con_hal_pdb_idx']])

    # calculate rmsds
    # row['rmsd_af2_des'] = calc_rmsd(xyz_pred.reshape(L*3,3), xyz_des.reshape(L*3,3))

    # load contig position
    # if os.path.exists(trbname): 
    idx_motif = [i for i,idx in zip(trb['con_hal_idx0'],trb['con_ref_pdb_idx']) 
                 if idx[0]!='R']


    idx_motif_ref = [i for i,idx in zip(trb['con_ref_idx0'],trb['con_ref_pdb_idx']) 
                     if idx[0]!='R']
    # row['contig_rmsd_af2_des'] = calc_rmsd(xyz_pred[idx_motif].reshape(L_motif*3,3), 
    #                                        xyz_des[idx_motif].reshape(L_motif*3,3))
    # row['contig_rmsd_af2'] = calc_rmsd(xyz_pred[idx_motif].reshape(L_motif*3,3), xyz_ref_motif.reshape(L_motif*3,3))
    return idx_motif, idx_motif_ref

def get_trb(row):
    path = os.path.join(row['rundir'], f'{row["name"]}.trb')
    return np.load(path,allow_pickle=True)

def get_mpnn_pdb(row):
    name = row['name']
    mpnn_i = row['mpnn_index']
    possible_paths = []
    possible_paths.append(os.path.join(row['rundir'], 'mpnn/packed', f'{name}_{mpnn_i}_1.pdb'))
    possible_paths.append(os.path.join(row['rundir'], 'mpnn/packed', f'{name}_{mpnn_i}.pdb'))
    possible_paths.append(os.path.join(row['rundir'], 'ligmpnn/packed', f'{name}_{mpnn_i}_1.pdb'))
    possible_paths.append(os.path.join(row['rundir'], 'ligmpnn/packed', f'{name}_{mpnn_i}.pdb'))
    for pdb in possible_paths:
        if os.path.exists(pdb):
            return pdb
    raise Exception(f'could not find mpnn_packed pdb at any of {possible_paths}')

def get_unidealized_pdb(row, return_design_if_backbone_only=False):
    if is_rfd(row):
        assert return_design_if_backbone_only, 'row is from a backbone-only model, so get_unidealized_pdb must be called with return_design_if_backbone_only=True'
        return os.path.join(row['rundir'], f'{row["name"]}.pdb')

    return os.path.join(row['rundir'], 'unidealized', f'{row["name"]}.pdb')

def get_chai1_df(row):

    mpnn_pdb = get_mpnn_pdb(row)
    head, tail = os.path.split(mpnn_pdb)
    tail = tail.split('.')[0]

    pattern = os.path.join(head, 'chai1/out', f'pred.{tail}_*.pdb')
    pdb_paths = glob.glob(pattern)
    model_idxs = []
    # print(f'{pattern=}')
    # print(f'{len(pdb_paths)} pdb_paths found')
    # print(f'{pdb_paths=}')
    for path in pdb_paths:
        model_idx = re.match('.*_model_idx_(\d+).pdb', path).groups()[0]
        model_idxs.append(int(model_idx))

    df_stack = []
    for model_idx in model_idxs:
        pdb_path = os.path.join(head, 'chai1/out', f'pred.{tail}_model_idx_{model_idx}.pdb')
        scores_path = os.path.join(head, 'chai1/out', f'scores.{tail}_model_idx_{model_idx}.json')
        assert os.path.exists(pdb_path), f'{pdb_path} does not exist'
        assert os.path.exists(scores_path), f'{scores_path} does not exist'
        df_tmp = pd.read_json(scores_path)
        df_tmp['model_idx'] = model_idx
        df_tmp['pdb_path'] = pdb_path
        df_stack.append(df_tmp)

    return pd.concat(df_stack)

def has_chai1(row):
    mpnn_pdb = get_mpnn_pdb(row)
    head, tail = os.path.split(mpnn_pdb)
    tail = tail.split('.')[0]
    pdb_path = os.path.join(head, 'chai1/out', f'pred.{tail}_model_idx_0.pdb')
    logger.debug(f'has_chai1: {pdb_path=}')
    return os.path.exists(pdb_path)

def get_best_chai1(row, key='aggregate_score'):

    if not has_chai1(row):
        return None, False
    
    df = get_chai1_df(row)
    best = df.nlargest(1, key)
    return best, True

def get_chai1_path(row, model_idx):
    mpnn_pdb = get_mpnn_pdb(row)
    head, tail = os.path.split(mpnn_pdb)
    tail = tail.split('.')[0]
    return os.path.join(head, 'chai1/out', f'pred.{tail}_model_idx_{model_idx}.pdb')

def get_best_chai1_path(row, key='aggregate_score'):
    best, has_chai1 = get_best_chai1(row, key)
    if not has_chai1:
        return None, False
    return best['pdb_path'].iloc[0], True

def get_af2(row):

    mpnn_pdb = get_mpnn_pdb(row)
    head, tail = os.path.split(mpnn_pdb)
    return os.path.join(head, 'af2', tail)

    mpnn_flavor = 'mpnn'
    if not pd.isna(row['inference.ligand']):
        mpnn_flavor = 'ligmpnn'
    mpnn_dir = os.path.join(row['rundir'], mpnn_flavor)
    mpnn_packed_dir = os.path.join(mpnn_dir, 'packed')
    if os.path.exists(mpnn_packed_dir):
        mpnn_dir = mpnn_packed_dir
    path = os.path.join(mpnn_dir, f'af2/{row["name"]}_{row["mpnn_index"]}.pdb')
    return path

def load_af2(row, name=None):
    rundir = row['rundir']
    d = rundir
    # if row['mpnn']:
    path = os.path.join(d, f'af2/{row["name"]}.pdb')
    if row.get('mpnn'):
        d = os.path.join(d, 'mpnn')
        path = os.path.join(d, f'af2/{row["name"]}_{row["mpnn_index"]}.pdb')
    if row.get('ligmpnn'):
        d = os.path.join(d, 'ligmpnn')
        packed_dir = os.path.join(d, 'packed')
        ic(os.path.exists(packed_dir))
        if os.path.exists(packed_dir):
            d= packed_dir
        path = os.path.join(d, f'af2/{row["name"]}_{row["mpnn_index"]}.pdb')
    name = (name or row['model']) + '_af2'
    cmd.load(path, name)
    return name

def get_ligmpnn_path(row):
    rundir = row['rundir']
    return os.path.join(rundir, 'ligmpnn', f"{row['name']}_{row['mpnn_index']}.pdb")


def to_resi(chain_idx):
    return f'resi {"+".join(str(i) for _, i in chain_idx)}'

def to_chain(chain_idx):
    chains = set(ch for ch,_ in chain_idx)
    assert len(chains)==1
    return list(chains)[0]

def get_traj_path(row,  traj='X0'):
    rundir = row['rundir']
    if traj == 'X0':
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_pX0_traj.pdb')
    elif traj == 'Xt':
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_Xt-1_traj.pdb')
    else:
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_{traj}_traj.pdb')
    return traj_path

def load_traj(row, name=None, traj='X0'):
    rundir = row['rundir']
    if traj == 'X0':
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_pX0_traj.pdb')
    elif traj == 'Xt':
        traj_path = os.path.join(rundir, f'traj/{row["name"]}_Xt-1_traj.pdb')
    else:
        trb = get_trb(row)
        if DESIGN in trb:
            traj_path = DESIGN
        else:
            #traj_path = os.path.join(row['rundir'], 'mpnn', row['name'] + '_'+str(row['mpnn_index'])+'.pdb')
            traj_path = os.path.join(row['rundir'], 'rethreaded', row['name'] + '_'+str(row['mpnn_index'])+'.pdb')
            if not os.path.exists(traj_path):
                traj_path = os.path.join(row['rundir'], row['name'] +'.pdb')
    name = name or row['model']
    cmd.load(traj_path, name)
    return name

from icecream import ic
def show_traj(row, strat, traj_type='X0'):
    strat_name = strat.replace(' ', '_')
    trb = get_trb(row)

    traj = load_traj(row, strat_name+'_'+traj_type, traj=traj_type)
    cmd.do('util.chainbow')
    color = 'white'
    if np.any(trb['con_hal_pdb_idx']):
        pymol_color(traj, trb['con_hal_pdb_idx'], color)
    only_backbones()


def show_traj_path(path):
    cmd.load(path, path.split('/')[-1])
    cmd.do('util.chainbow')
    only_backbones()
    
def pymol_color(name, chain_idx, color='red'):
    sel = f'{name} and resi {"+".join(str(i) for _, i in chain_idx)}'
    cmd.color(color, sel)

def get_motif_idx(row):
    #input_pdb = get_input_pdb(row)
    trb = get_trb(row)
    chain_idx = trb["con_hal_pdb_idx"]
    return torch.tensor([i for _, i in chain_idx])

def get_name(row, strat=None):
    strat = strat or row['name']
    strat_name = strat.replace(' ', '_')
    return strat_name

def to_selector(motif_resi):
    #if len(chains) == 1:
    #    return f'{name} and chain {to_chain(self.motif_resi)} and ({to_resi(self.motif_resi)})'
    chain_sels = []
    for ch, g in itertools.groupby(motif_resi, lambda x: x[0]):
        chain_sels.append(f'(chain {ch} and {to_resi(g)})')
    return ' or '.join(chain_sels)

@dataclass
class Structure:
    name: str
    motif_resi: any
    #diffusion_mask: torch.BoolTensor

    def motif_sele(self):
        return f'({self.name} and {to_selector(self.motif_resi)})'
        return to_selector(self.motif_resi)
        chains = set(ch for ch,_ in self.motif_resi)
        if len(chains) == 1:
            return f'{self.name} and chain {to_chain(self.motif_resi)} and ({to_resi(self.motif_resi)})'
        chain_sels = []
        for ch, g in itertools.groupby(self.motif_resi):
            chain_sels.append(f'(chain {ch} and resi {to_resi(g)})')
        return f"({self.name} and ({' or '.join(chain_sels)}))"

def load_motif_same_chain(row, strat=None):
    strat_name = get_name(row, strat)
    motif_suffix = '_motif'
    native = strat_name+motif_suffix
    input_pdb = get_input_pdb(row)
    trb = get_trb(row)
    ref_idx = trb["con_ref_pdb_idx"]
    if cmd.is_network:
        #cmd.do(f'load {input_pdb}, {native}; remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
        cmd.load(f'{input_pdb}, {native}')
        cmd.remove(f'({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
    else:
        cmd.load(input_pdb, native)
        cmd.do(f'remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
    for i, (chain, resi_i) in enumerate(ref_idx):
        cmd.alter(f'{native} and chain {chain} and resi {resi_i}', f'resi={i}')
        cmd.sort()
    return Structure(native, [(chi[0], i) for i, chi in enumerate(trb['con_ref_pdb_idx'])])
 
def load_motif(row, strat=None, show=True):
    strat_name = get_name(row, strat)
    motif_suffix = '_motif'
    native = strat_name+motif_suffix
    input_pdb = get_input_pdb(row)
    trb = get_trb(row)
    # ic(trb)
    # ligand_name = trb['inference.ligand']
    ligand_name = row['inference.ligand']
    ref_idx = trb["con_ref_pdb_idx"]


    #cmd.do(f'load {input_pdb}, {native}; remove ({native} and not ({to_selector(ref_idx)} or resn {row["inference.ligand"]}))')
    cmd.load(f'{input_pdb}', f'{native}')
    cmd.remove(f'({native} and not ({to_selector(ref_idx)} or resn {ligand_name}))')
    #cmd.remove(f'({native} and not ({to_selector(ref_idx)}))')
    # cmd.do(f'load {input_pdb}, {native}; remove ({native} and not {to_selector(ref_idx)})')
    # cmd.load(input_pdb, native)
    #ic(f'remove ({native} and not ({to_selector(ref_idx)}))')
    #ipdb.set_trace()
    #cmd.do(f'remove ({native} and not ({to_selector(ref_idx)}))')
    #ipdb.set_trace()
    cmd.show_as('licorice', native)
    #ipdb.set_trace()
    #print(f'load {input_pdb}, {native}; remove ({native} and not ({to_selector(ref_idx)})')
    #for i, (chain, resi_i) in enumerate(ref_idx):
    #    cmd.alter(f'{native} and chain {chain} and resi {resi_i}', f'resi={i}')
    #for i, (chain, resi_i) in enumerate(ref_idx):
    #    cmd.alter(f'{native}', f'chain={chain}')
    #for i, (chain, resi_i) in enumerate(ref_idx):
    #    cmd.alter(f'{native} and resi {resi_i}', f'resi={i}')
    #cmd.sort()

    # for i, (chain, resi_i) in enumerate(ref_idx):
    #     cmd.alter(f'{native}', f'chain="A"')
    for i, (chain, resi_i) in enumerate(ref_idx):
        cmd.alter(f'{native} and resi {resi_i}', f'resi={i}')
    cmd.sort()
    return Structure(native, [(ch, i) for i, (ch, _) in enumerate(trb['con_ref_pdb_idx'])])
    # return Structure(native, [('A', i) for i, chi in enumerate(trb['con_ref_pdb_idx'])])
    #return Structure(native, [(chi[0], i) for i, chi in enumerate(trb['con_ref_pdb_idx'])])
    #return Structure(native, trb['con_ref_pdb_idx'])

def show_motif(row, strat, traj_types='X0', show_af2=True, show_true=False):
    structures = {}
    strat_name = strat.replace(' ', '_')
    trb = get_trb(row)
    native = load_motif(row, strat_name)
    structures['native'] = native
    
#     input_pdb = get_input_pdb(row)
#     motif_suffix = '_motif'
#     native = strat_name+motif_suffix
#     ic(input_pdb)
#     # cmd.load(input_pdb, native)
#     ref_idx = trb["con_ref_pdb_idx"]
#     # cmd.do(f'load {input_pdb}, {native}; remove not (chain {to_chain(ref_idx)} and {to_resi(ref_idx)})')
#     # cmd.do(f'load {input_pdb}, {native}; remove not ({native} and chain {to_chain(ref_idx)} and {to_resi(ref_idx)})')
#     # cmd.do(f'load {input_pdb}, {native}')
#     # cmd.do(f'load {input_pdb}, {native}; remove ({native} and chain {to_chain(ref_idx)} and not ({to_resi(ref_idx)}))')
#     cmd.do(f'load {input_pdb}, {native}; remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
#     cmd.load(input_pdb, native)
#     cmd.do(f'remove ({native} and not (chain {to_chain(ref_idx)} and ({to_resi(ref_idx)})))')
    # return
    
    if not isinstance(traj_types, list):
        traj_types = [traj_types]
    trajs = []
    traj_motifs = []
    structures['trajs'] = []
    for traj_type in traj_types:
        traj = load_traj(row, strat_name+'_'+traj_type, traj_type)
        trajs.append(traj)
        traj_motif = f'{traj} and {to_resi(trb["con_hal_pdb_idx"])}'
        traj_motifs.append(traj_motif)
        #ic(traj_motif, native)
        # cmd.align(traj_motif, native, 'mobile_state=1')
        cmd.align(traj_motif, native.name)
        structures['trajs'].append(Structure(traj, trb['con_hal_pdb_idx']))
    # traj = strat_name
    # traj_path = os.path.join(rundir, 'mpnn', row['name'])+'.pdb'
    # # print(traj_path)
    # cmd.load(os.path.join(rundir, 'mpnn', row['name'])+'.pdb', strat_name)
    # cmd.show_as('
    #native_motif = f'{traj} and {to_resi(trb["con_ref_pdb_idx"])}'
    # cmd.do(f'load {input_pdb} {native}; remove not ({native_motif})')
    # des_motif_resi = to_resi(trb["con_hal_pdb_idx"])
    # cmd.align(af2, traj)
    # cmd.align(native , traj)
    #cmd.align(traj, native)
    
    if show_af2:
        af2 = load_af2(row, strat_name)
        #cmd.align(af2, native)
        #cmd.align(af2, traj)
        af2 = Structure(af2, trb['con_hal_pdb_idx'])
        structures['af2'] = af2
        #ic(af2.motif_sele(), native.motif_sele())
        cmd.align(af2.motif_sele(), native.motif_sele())
        cmd.set('stick_transparency', 0.7, af2.name)
    #cmd.center(traj)
    cmd.center(native.motif_sele())
    cmd.do('util.chainbow')
    color = 'white'
    # pymol_color(native, trb['con_ref_pdb_idx'], color)
    for traj in trajs:
        pymol_color(traj, trb['con_hal_pdb_idx'], color)
    if show_af2:
        pymol_color(af2.name, trb['con_hal_pdb_idx'], color)
    whole_native = None
    if show_true:
        #input_pdb = get_input_pdb(row)
        whole_native = strat_name+'_true'
        ref_idx = trb["con_ref_pdb_idx"]
        #cmd.do(f'load {input_pdb}, {whole_native}')
        cmd.load(input_pdb, whole_native)
        pymol_color(whole_native, ref_idx, color)
        
        
    #cmd.color('pink', '*'+motif_suffix)
    #cmd.color('pink', native.name)
    cmd.color('pink', f'{native.name} and elem C')
    # cmd.hide('cartoon', native)
    # cmd.show('cartoon', f'{native} and {to_resi(trb["con_ref_pdb_idx"])}')
    only_backbones()
    return structures
    #return ructure(t,[]) for t,m in zip(trajs, traj_motifs)], 
    #return trajs, traj_motifs, whole_native, native, native

def only_backbones(o=False):
    cmd.hide('all')
    cmd.show('licorice', f'(name ca or name c or name n{" or name o" if o else ""})')


def get_spread(bench_des, key='contig_rmsd_af2'):
    bench_des.reset_index()
    bench_des = bench_des.sort_values(key)
    worst = bench_des.iloc[-1]
    best = bench_des.iloc[0]
    median = bench_des.iloc[len(bench_des)//2]
    rows = [best, median, worst]
    return rows

def show_spread(bench_des, key='contig_rmsd_af2'):
    clear(cmd)
    rows = get_spread(bench_des, key=key)
    # rows = rows[:2]
    for row in rows:
        name = f"ep{row['model']}_{row[key]:.2f}"
        show_motif(row, name)
        # break
        

def get_examples(df, benchmark, model, mpnn=False, key='rmsd_af2_des'):
    bench_des = df[(df['benchmark'] == benchmark) & (df['model'] == model) & (df['mpnn'] == mpnn)]
    bench_des.reset_index()
    return get_spread(bench_des, key)

def get_traj_xyz(row, traj_type='X0', n=None):
    #p = parse_pdb(os.path.join(row['rundir'], 'mpnn', row['name']+'.pdb'))

    traj_path = get_traj_path(row, traj_type)
    return read_traj_xyz(traj_path)

def read_traj_xyz(traj_path, seq=False):
    with open(traj_path) as f:
        s = f.read()
        models = s.strip().split('ENDMDL')
        parsed = []
        seqs = []
        for i, m in enumerate(models):
            if not m:
                continue
            # o = parsers.parse_pdb_lines(m.split('\n'), False, False, lambda x: 0)
            o = parsers.parse_pdb_lines(m.split('\n'), False, seq, lambda x: 0)
            xyz = o[0]
            if seq:
                seqs.append(o[-1])
            parsed.append(xyz)
        # parsed = torch.concat(parsed)
    parsed = torch.tensor(np.array(parsed))
    if seqs:
        return parsed, seqs
    return parsed
        # parsed = [parse_pdb_lines(m.split('\n')) for m in models]

def get_contig_c_alpha_rmsd(row, all_idxs=False):
    print('.', end='')
    traj = get_traj_xyz(row, 'Xt')
    motif_idx, native_motif_idx = get_idx_motif(row, mpnn=False)
    i = torch.tensor([0,1])
    if all_idxs:
        #i = torch.arange(traj.shape[1])
        ic(traj.shape)
        i = torch.arange(traj.shape[0])
    return el.c_alpha_rmsd_traj(traj[i][:,motif_idx])

def flatten_dict(dd, separator ='.', prefix =''):
    return { prefix + separator + k if prefix else k : v
             for kk, vv in dd.items()
             for k, v in flatten_dict(vv, separator, kk).items()
             } if isinstance(dd, dict) else { prefix : dd }

def make_row_from_traj(traj_prefix, use_trb=True):
    synth_row = {}
    
    # Minor fix: handles pdbs as traj_prefix for ease of use
    if traj_prefix.endswith('.pdb'):
        traj_prefix = traj_prefix[:-4]
    pdb_path = traj_prefix 
    pdb_path += '.pdb'
    
    synth_row['pdb_path'] = pdb_path
    
    synth_row['rundir'], synth_row['name'] = os.path.split(traj_prefix)
    synth_row['mpnn_index'] = 0
    if 'packed' in synth_row['rundir']:
        synth_row['rundir'] = os.path.dirname(synth_row['rundir'])
        synth_row['name'] = re.sub(r'_\d+$', '', synth_row['name'])
    if 'mpnn/' in traj_prefix:
        synth_row['mpnn_index'] = int(synth_row['name'].split('_')[-1])
        synth_row['rundir'] = os.path.dirname(synth_row['rundir'])
        synth_row['name'] = '_'.join(synth_row['name'].split('_')[:-1])
        
    synth_row['mpnn'] = True
    if use_trb:
        trb = get_trb(synth_row)
        rundir = synth_row['rundir']
        config = trb.get('config', {})
        config = flatten_dict(config)
        synth_row.update(config)
        synth_row['rundir'] = rundir
    return synth_row

def show_row(row, traj_name, traj_type='X0'):
    show_traj(row, traj_name, traj_type)
    #input_pdb = get_input_pdb(row)
    
    af2 = load_af2(row, traj_name)
    cmd.align(af2, traj_name)
    only_backbones()
    cmd.do('util.chainbow')
    cmd.set('stick_transparency', 0.7, af2)
    # break

def calc_rmsd(xyz1, xyz2, eps=1e-6):
    #ic(xyz1.shape, xyz2.shape)

    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U

    L = xyz2_.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_-xyz1)*(xyz2_-xyz1), axis=(0,1)) / L + eps)

    return rmsd


import os
from itertools import permutations

def get_benchmark(spec, interc=10-100, length=150):
	'''
	Spec like:
		{'GAA_0_0': {
			'pdb': '/home/heisen/0_projects/4_enzymedesign/input/GAA/5nn5_aligned_sub_rot1.pdb',
		  	'ligand_resn': 'UNL',
		  	'pdb_contig': [('A', 518), ('A', 616), ('A', 404)]
			},
		}

	'''
	benchmarks_json = '''
	{
	'''

	benchmark_dict = OrderedDict()
	for name, d in spec.items():
		for motif in permutations(row.motif_selector):
			motif_suffix = ''.join(f'{ch}{i}' for ch,i in motif)
			benchmark_name = f'{input_pdb}_{motif_suffix}'
			# bench_by_pdb[row.pdb].append(
			contig = interc + ',' + (','+interc+',').join(f'{ch}{r}-{r}' for ch, r in motif) +',' + interc
			benchmark_dict[benchmark_name] = f"inference.input_pdb={d['pdb']} contigmap.contigs=[\\\\'{contig}\\\\']"
	benchmarks_json = json.dumps(benchmark_dict, indent=4)
	return benchmarks_json
	# print(benchmarks_json)

def make_script(args, run, debug=False, n_sample=1, num_per_condition=1):
    T_arg = ''
    seq_per_target = 8
    lengths = '150-150|200-200'
    if debug:
        T_arg = 'diffuser.T=5 '
        seq_per_target = 1
        lengths = '150-150'
        num_per_condition = 1
    
    arg_strs = []
    for a in args:
        arg_strs.append(f"""        "--config-name=base inference.deterministic=False inference.align_motif=True inference.annotate_termini=True inference.model_runner=NRBStyleSelfCond inference.ckpt_path=/home/ahern/projects/rf_diffusion/models/theo_pdb/BFF_4.pt contigmap.length={lengths} {T_arg}{a}" \\""")

    nl = '\n'
    script = f"""#!/bin/bash 

source activate /home/dimaio/.conda/envs/SE3nv 

./pipeline.py \\
        --num_per_condition {num_per_condition} --num_per_job 1 --out {run}/run \\
        --args \\
{nl.join(arg_strs)}
        --num_seq_per_target {seq_per_target} --af2_gres=gpu:a6000:1 -p cpu
    """
    return script

import functools

def add_filters(df, **kwargs):
    filter_names, filter_unions = add_filters_multi(df, **kwargs)
    return filter_names + filter_unions

def add_filters_multi(df, threshold_columns = ['contig_rmsd_af2', 'rmsd_af2_des'], threshold_signs = ['-', '-'], thresholds = [(1, 3)]):
    filter_names = []
    filter_unions = []
    for i, threshold in enumerate(thresholds):
        filter_names = []
        for metric, sign, value in zip(threshold_columns, threshold_signs, threshold):
            if sign == '+':
                geq_or_leq = '>'
                passes_filter = df[metric] > value
            else:
                geq_or_leq = '<'
                passes_filter = df[metric] < value
            filter_name = f'{metric}_{geq_or_leq}_{value}'
            filter_names.append(filter_name)
            df[filter_name] = passes_filter
        filter_union_name = f'filter_set_{i}'
        filter_unions.append(filter_union_name)
        # df[functools.reduce(lambda a,b: df[a] & df[b], filter_names)]
        #df[filter_union_name] = functools.reduce(lambda a,b: df[a] & df[b], filter_names)
        df[filter_union_name] = functools.reduce(lambda a,b: a & b, [df[f] for f in filter_names])
    return filter_names, filter_unions

def melt_filters(df, filter_names):
    data = df.melt(id_vars='name', value_vars=filter_names, var_name='filter_name', value_name='pass')
    merged=data.merge(right=df, on='name', how='outer')
    return merged

def plot_melted(df, filter_names):
    hue_order = sorted(df['contigmap.length'].unique())
    x_order = sorted(df['benchmark'].unique())
    g = sns.catplot(data=df, y='pass', hue='contigmap.length', x='benchmark', kind='bar', orient='v', col='filter_name', hue_order=hue_order, height=8.27, aspect=11.7/8.27, legend_out=True, order=x_order, ci=None)
    # iterate through axes
    for ax in g.axes.ravel():

        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()*100):.1f}%' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)
        ax.tick_params(axis='x', rotation=90)
    plt.xticks(rotation=90)
    return g
    # filter_union = functools.reduce(lambda a,b: a | b, filters)

def plot_melted_by_column(df, filter_names, column):
    for column_v in df[column].unique():
        g = plot_melted(df[df[column]==column_v], filter_names)
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(f'{column}:{column_v}')

def add_benchmark_from_input_pdb(df):
    def get_benchmark(row):
            return os.path.basename(os.path.splitext(row['inference.input_pdb'])[0])
    df['benchmark'] = df.apply(get_benchmark, axis=1)

def get_lengths(row):
    m = analyze.get_trb(row)['sampled_mask'][0]
    l = [e for e in m.split(',') if e[0] != 'A']
    l = [int(e.split('-')[0]) for e in l]
    return l

def get_sidechain_rmsd(row):
    af2_motif = get_af2_motif(row)
    native_motif = get_native_motif(row)
    is_atom = get_motif_atom_mask(row)
    is_atom = is_atom[:,:14]
    return analyze.calc_rmsd(af2_motif[is_atom].numpy(), native_motif[is_atom].numpy())

def get_motif_from_pdb(pdb, chain_i):
    feats = utils.process_target(pdb, parse_hetatom=True, center=False)
    xyz_motif_idx = pdb_to_xyz_idx(pdb, chain_i)
    motif = feats['xyz_27'][xyz_motif_idx]
    return motif

def get_min_dist_to_het(pdb, chain_i):

    dgram, het_names = get_dist_to_het(pdb, chain_i)
    minidx  = torch.argmin(dgram, keepdim=True)
    _, M, H = dgram.shape
    hetidx = minidx // M
    return torch.min(dgram), het_names[hetidx]
    print(f'{torch.min(dgram)} to {het_names[hetidx]}')

def get_xyz_nonhet(pdb, chain_i):
    feats = utils.process_target(pdb, parse_hetatom=True, center=False)
    xyz_motif_idx = pdb_to_xyz_idx(pdb, chain_i)
    motif = feats['xyz_27'][xyz_motif_idx]
    motif_atms = motif[feats['mask_27'][xyz_motif_idx]]
    return motif_atms

def get_dist_to_het(pdb, chain_i, ligands=None):
    '''
    Returns [M, L]
    '''
    feats = utils.process_target(pdb, parse_hetatom=True, center=False)
    xyz_motif_idx = pdb_to_xyz_idx(pdb, chain_i)
    motif = feats['xyz_27'][xyz_motif_idx]
    motif_atms = motif[feats['mask_27'][xyz_motif_idx]]
    het_names = np.array([i['name'].strip() for i in feats['info_het']])
    het_xyz = feats['xyz_het'][het_names != 'HOH']
    het_names =  het_names[het_names != 'HOH']
    if ligands:
        is_ligand = [h in ligands for h in het_names]
        het_xyz = het_xyz[is_ligand]
        het_names =  het_names[is_ligand]
    if het_xyz.size == 0:
        return 999.0, ''
    try:
        dgram = torch.cdist(motif_atms[None], torch.tensor(het_xyz[None], dtype=torch.float32))
    except Exception as e:
        print(motif_atms.shape, het_xyz.shape, len(het_xyz))
        raise e
    # print(dgram.shape)
    # print(set(het_names))
    return dgram, het_names

def get_xyz_het(pdb, ligands):
    feats = utils.process_target(pdb, parse_hetatom=True, center=False)
    het_names = np.array([i['name'].strip() for i in feats['info_het']])
    het_xyz = feats['xyz_het'][het_names != 'HOH']
    het_names =  het_names[het_names != 'HOH']
    if ligands:
        is_ligand = [h in ligands for h in het_names]
        het_xyz = het_xyz[is_ligand]
        het_names =  het_names[is_ligand]
    return het_xyz, het_names

def get_traj_motif(row):
    motif_idx, native_motif_idx = analyze.get_idx_motif(row, mpnn=False)
    traj = analyze.get_traj_xyz(row, 'X0')
    return traj[0, motif_idx]

def get_design_pdb(row):
    return row['pdb_path']

def get_diffusion_pdb(row):
    return os.path.join(row['rundir'], f'{row["name"]}.pdb')

def get_design(row):
    rundir = row['rundir'] 
    des =  utils.process_target(os.path.join(rundir, row['name']+'.pdb'))
    return des['xyz_27']

def get_af2_xyz(row):
    af2_path = get_af2(row)
    des =  utils.process_target(af2_path)
    return des['xyz_27']


# def get_af2_motif(row):
#     motif_idx, native_motif_idx = analyze.get_idx_motif(row, mpnn=False)
#     af2 = analyze.get_af2(row)
#     af2 = utils.process_target(af2)
#     return af2['xyz_27'][motif_idx][:,:14]

def get_native(row):
    # motif_idx, native_motif_idx = analyze.get_idx_motif(row, mpnn=False)
    input_pdb = get_input_pdb(row)
    native = utils.process_target(input_pdb, center=False, parse_hetatom=True)
    #print(native.keys())
    het_names = [i['name'].strip() for i in native['info_het']]
    het_names = np.array(het_names)
    # ic(native['info_het'])
    # import ipbd
    # ipdb.set_trace()
    # assert len(het_names) <= 1, f'more than 1 het: {het_names}'
    is_design_ligand = het_names == row.get('inference.ligand', None)
    # ic(is_design_ligand.sum(), len(is_design_ligand))
    # ic(native['xyz_het'].shape, native['xyz_het'][is_design_ligand].shape)
    return native['xyz_27'][:,:14], native['xyz_het'][is_design_ligand]
    # return native['xyz_27'][native_motif_idx][:,:14]

def get_registered_ligand(row, af2=False):
    motif_idx, native_motif_idx = get_idx_motif(row, mpnn=False)
    # traj_motif = get_traj_motif(row)
    if af2:
        des = get_af2_xyz(row)
    else:
        des = get_design(row)
    native, het = get_native(row)
    motif_des = des[motif_idx]
    motif_native = native[native_motif_idx]
    T = register_full_atom(motif_des[:,1:2,:], motif_native[:,1:2,:])
    des = T(des)
    return des, native, het

def get_dist_to_ligand(row, af2=False, c_alpha=False):
    if row.get('inference.ligand', False):
        design_pdb = get_design_pdb(row)
        design_info = utils.process_target(design_pdb, center=False, parse_hetatom=True)
        des = design_info['xyz_27'][:, :14]
        het = design_info['xyz_het']
    else:
        des, _, het = get_registered_ligand(row, af2=af2)

    motif_idx, native_motif_idx = get_idx_motif(row, mpnn=False)
    L, _, _ = des.shape
    if c_alpha:
        bb_des = des[:,1]
    else:
        bb_des = des[:,:3].reshape(L*3, 3)
    dgram = torch.cdist(bb_des[None,...], torch.tensor(het[None, ...], dtype=torch.float32), p=2)
    # ic(dgram)
    return dgram[0]

null_structure = Structure('null', [])

def show_motif_simple(row, strat, traj_types='X0', show_af2=True, show_true=False):
    structures = {}
    strat_name = strat.replace(' ', '_')
    trb = get_trb(row)
    has_motif = bool(trb['con_hal_pdb_idx'])
    # native=null_structure
    native=None
    if has_motif:
        native = load_motif(row, strat_name)
    structures['native'] = native
    
    if not isinstance(traj_types, list):
        traj_types = [traj_types]
    trajs = []
    traj_motifs = []
    structures['trajs'] = []
    for traj_type in traj_types:
        traj = load_traj(row, strat_name+'_'+traj_type, traj_type)
        trajs.append(traj)
        traj_motif = f'{traj} and {to_resi(trb["con_hal_pdb_idx"])}'
        traj_motifs.append(traj_motif)
        if has_motif:
            cmd.align(traj_motif, native.name)
        structures['trajs'].append(Structure(traj, trb['con_hal_pdb_idx']))
    
    if show_af2:
        af2 = load_af2(row, strat_name)
        af2 = Structure(af2, trb['con_hal_pdb_idx'])
        structures['af2'] = af2
        if has_motif:
            cmd.align(af2.motif_sele(), native.motif_sele())
        cmd.set('stick_transparency', 0.7, af2.name)
    color = 'white'
    for traj in trajs:
        if has_motif:
            pymol_color(traj, trb['con_hal_pdb_idx'], color)
    if show_af2 and has_motif:
        pymol_color(af2.name, trb['con_hal_pdb_idx'], color)
    whole_native = None
    if show_true:
        whole_native = strat_name+'_true'
        ref_idx = trb["con_ref_pdb_idx"]
        cmd.load(input_pdb, whole_native)
        pymol_color(whole_native, ref_idx, color)
    
    #if not has_motif:
    #    cmd.hide('everything', native.name)
    if has_motif:
        cmd.color('pink', f'{native.name} and elem C')
    return structures

def show_paper_pocket(row, des=True, ligand=False):
    # b=row['benchmark']
    b = row['name']
    structures = show_motif_simple(row, b, traj_types=['des'])
    # structures = analyze.show_motif(row, b, traj_types=['X0'])
    af2 = structures['af2']
    des = structures['trajs'][0]
    native = structures['native']
    af2_scaffold = Structure(af2.name +'_scaffold', af2.motif_resi)
    cmd.copy(af2_scaffold.name, af2.name)

    cmd.do(f'util.cbag {af2.name}')
    cmd.do(f'util.cbag {des.name}')
    cmd.hide('everything', f'{af2_scaffold.name} or {af2.name} or {native.name} or {des.name}')
    # Scaffold
    cmd.show('cartoon', f'{af2_scaffold.name}')
    cmd.set('stick_transparency', 0, af2.name)
    cmd.color('gray', f'{af2_scaffold.name}')
    cmd.color('teal', f'{af2.motif_sele()} and elem C and not (name ca or name c or name n)')
    
    # Design
    if des:
        #cmd.align(des.name, af2_scaffold.name)
        cmd.align(f'{des.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
        cmd.show('cartoon', f'{des.name}')
        mass_paper_rainbow_sel(des.name)

    # AF2 sidechains
    cmd.show('licorice', f'({af2.motif_sele()}) and not (name o)')
    cmd.color('paper_pink', f'({af2.motif_sele()}) and (elem C or name n)')

    # Desired sidechains
    cmd.show('licorice', f'{native.name} and not (name o)')
    cmd.color('paper_teal', f'{native.name} and (elem C or name n)')

    cmd.set('cartoon_transparency', 0)
    cmd.center(af2.name)
    cmd.hide('everything', af2.name)
    cmd.hide('everything', af2_scaffold.name)
    cmd.hide('cartoon', des.name)
    cmd.show('licorice', f'{des.name} and (name c or name ca or name n)')
    cmd.set('stick_transparency', 0, des.name)
    # cmd.show('licorice'
    # cmd.show('licorice', f'{des.motif_sele()}')
    
    # #
    # cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
    # cmd.show('licorice', f'{af2.name} and (name c or name ca or name n)')
    # cmd.show('licorice', f'{af2.motif_sele()}')
    # cmd.color('good_gray', f'{af2.name} and (name c or name ca or name n)')
    
    # Ligand
    lig = f'lig_{b}'
    if ligand:
        cmd.load(get_input_pdb(row), lig)
        cmd.color('orange', f'{lig} and elem C')
    # cmd.orient(lig)
    return [af2.name, af2_scaffold.name, des.name, native.name, lig]

def show_paper_pocket_af2(row, b=None, des=True, ligand=False, traj_types=None, show_af2=True):
    # b=row['benchmark']
    b = b or f"{row['name']}_{row['mpnn_index']}"
    traj_types = traj_types or ['des']
    structures = show_motif_simple(row, b, traj_types=traj_types, show_af2=show_af2)
    has_motif = structures['native'] is not None
    # structures = analyze.show_motif(row, b, traj_types=['X0'])
    des = structures['trajs'][0]
    native = structures['native']
    af2 = None
    af2_scaffold=None
    if show_af2:
        af2 = structures['af2']
        af2_scaffold = Structure(af2.name +'_scaffold', af2.motif_resi)
        if has_motif:
            cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
        else:
            cmd.align(f'{des.name} and name ca', f'{af2.name} and name ca')
            # import ipdb
            # ipdb.set_trace()
        # else:
        #     cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')

        cmd.copy(af2_scaffold.name, af2.name)

        cmd.do(f'util.cbag {af2.name}')
    cmd.do(f'util.cbag {des.name}')
    if has_motif:
        cmd.hide('everything', f'{native.name}')
    cmd.hide('everything', f'{des.name}')
    if show_af2:
        cmd.hide('everything', f'{af2_scaffold.name} or {af2.name}')
        # Scaffold
        cmd.show('cartoon', f'{af2_scaffold.name}')
        cmd.set('stick_transparency', 0, af2.name)
        cmd.color('good_gray', f'{af2_scaffold.name}')
    
    # Design
    if des:
        #cmd.align(des.name, af2_scaffold.name)
        if show_af2:
            cmd.align(f'{des.name}', f'{af2.name}')
        if has_motif:
            cmd.align(f'{des.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
        cmd.show('cartoon', f'{des.name}')
        mass_paper_rainbow_sel(des.name)

    if show_af2 and has_motif:
        # AF2 sidechains
        cmd.show('licorice', f'({af2.motif_sele()}) and not (name o)')
        cmd.color('good_gray', f'({af2.motif_sele()}) and (elem C or name n)')

    # Desired sidechains
    if has_motif:
        cmd.show('licorice', f'{native.name} and not (name o)')
        cmd.color('paper_teal', f'{native.name} and (elem C or name n)')
        cmd.hide('everything', f'{native.name} and not {native.motif_sele()}')
        # trb = get_trb(row)
        cmd.show('licorice', f'{native.name} and resn {row["inference.ligand"]}')
        cmd.color('orange', f'{native.name} and resn {row["inference.ligand"]} and elem C')

    cmd.set('cartoon_transparency', 0)
    if show_af2:
        cmd.center(af2.name)
    else:
        cmd.center(des.name)
    if has_motif:
        cmd.hide('sticks', f'{native.name} and elem H')
    #cmd.hide('everything', af2.name)
    #cmd.hide('everything', af2_scaffold.name)
    #cmd.hide('cartoon', des.name)
    #cmd.show('licorice', f'{des.name} and (name c or name ca or name n)')
    #cmd.set('stick_transparency', 0, des.name)
    # cmd.show('licorice'
    # cmd.show('licorice', f'{des.motif_sele()}')
    
    # #
    #cmd.align(f'{af2.motif_sele()} and name ca', f'{native.motif_sele()} and name ca')
    # cmd.show('licorice', f'{af2.name} and (name c or name ca or name n)')
    # cmd.show('licorice', f'{af2.motif_sele()}')
    # cmd.color('good_gray', f'{af2.name} and (name c or name ca or name n)')
    
    # Ligand
    lig = f'lig_{b}'
    if ligand:
        cmd.load(get_input_pdb(row), lig)
        cmd.color('orange', f'{lig} and elem C')
    # cmd.orient(lig)
    identifiers =  [af2, af2_scaffold, des, native]
    if ligand:
        identifiers.append(Structure(lig, None))
    return identifiers


def calc_success(df, threshold_columns=['contig_rmsd_af2', 'rmsd_af2_des', 'af2_pae_mean'], threshold_signs=['-', '-', '-'], named_thresholds=[('excellent', (1,2,5)), ('good', (1.5,3,7.5)), ('okay', (2,3,10))], recompute=True):
    filter_names = [name for name, threshold in named_thresholds]
    thresholds = [threshold for name, threshold in named_thresholds]
    if recompute:
        filters,  filter_unions = add_filters_multi(df, threshold_columns=threshold_columns, thresholds=thresholds, threshold_signs=threshold_signs)
        df.drop(columns=filter_names, inplace=True, errors='ignore')
        df.rename(columns=dict(zip(filter_unions, filter_names)), inplace=True)
    else: 
        filter_unions = []
        for i in range(len(named_thresholds)):
            filter_union_name = f'filter_set_{i}'
            filter_unions.append(filter_union_name)

    melts = []
    for filter_union in filter_names:
        best_filter_passers = df.groupby(["name"]).apply(lambda grp: grp.sort_values([filter_union, 'contig_rmsd_af2_full_atom'], ascending=[False, True]).head(1))
        best_filter_passers.index =best_filter_passers.index.droplevel()
        melted = melt_filters(best_filter_passers, [filter_union])
        melted['filter_set'] = filter_union
        melts.append(melted)

    melted = pd.concat(melts)
    return melted

def plot_success(df, threshold_columns=['contig_rmsd_af2', 'rmsd_af2_des', 'af2_pae_mean'], threshold_signs=['-', '-', '-'], named_thresholds=[('excellent', (1,2,5)), ('good', (1.5,3,7.5)), ('okay', (2,3,10))], recompute=True):
    #filters,  filter_unions = analyze.add_filters_multi(df, threshold_columns=['contig_rmsd_af2','contig_rmsd_af2_full_atom', 'rmsd_af2_des', 'af2_pae_mean'], thresholds=[(1,1.5,2,5), (1,999,2,5), (1.5,3,3,7.5), (1.5,2,3,10)], threshold_signs=['-','-', '-', '-'])
    # filter_names = ['excellent', 'backbone excellent', 'good', 'okay']
    filter_names = [name for name, threshold in named_thresholds]
    thresholds = [threshold for name, threshold in named_thresholds]
    if recompute:
        filters,  filter_unions = add_filters_multi(df, threshold_columns=threshold_columns, thresholds=thresholds, threshold_signs=threshold_signs)
        df.drop(columns=filter_names, inplace=True, errors='ignore')
        df.rename(columns=dict(zip(filter_unions, filter_names)), inplace=True)
    else: 
        filter_unions = []
        for i in range(len(named_thresholds)):
            filter_union_name = f'filter_set_{i}'
            filter_unions.append(filter_union_name)

    melts = []
    for filter_union in filter_names:
        best_filter_passers = df.groupby(["name"]).apply(lambda grp: grp.sort_values([filter_union, 'contig_rmsd_af2_full_atom'], ascending=[False, True]).head(1))
        best_filter_passers.index =best_filter_passers.index.droplevel()
    # best_filter_passers = df.loc[df.groupby(["name"])["contig_rmsd_af2"].idxmin()]
    # filters = analyze.add_filters(best_filter_passers, thresholds=[(1,2)])
        melted = melt_filters(best_filter_passers, [filter_union])
        melted['filter_set'] = filter_union
        melts.append(melted)

    melted = pd.concat(melts)
    # melted['filter_name'] = melted['filter_name'].replace(filter_unions, filter_names)
    # analyze.plot_melted(melted, filter_unions)

    # hue_order = sorted(df['contigmap.length'].unique())
    import matplotlib.pyplot as plt
    x_order = sorted(df['benchmark'].unique())
    g = sns.catplot(data=melted, y='pass', hue='filter_name', x='benchmark', kind='bar', orient='v', height=8.27, aspect=11.7/8.27, legend_out=True, order=x_order, ci=None)
    # iterate through axes
    for ax in g.axes.ravel():

        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()*100):.1f}%' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)
        ax.tick_params(axis='x', rotation=90)
    plt.xticks(rotation=90)
    return g, melted

from tqdm.notebook import tqdm
def apply(df, name, f):
    df = df.copy()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        df.loc[i, name] = f(row)
    return df

def apply_arr(df, name, f):
    df = df.copy()
    kw = {name: None}
    df = df.assign(**kw)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        df.at[i, name] = f(row)
    return df


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def add_ligand_dist(df, c_alpha=False):
    groupers = ['name']
    if 'source' in df.columns:
        groupers = ['source'] + groupers
    designs = df.drop_duplicates(groupers, ignore_index=True)
    name = ('c_alpha' if c_alpha else 'bb') + '_ligand_dist'
    designs = apply_arr(designs, name, lambda x: get_dist_to_ligand(x, c_alpha=c_alpha).min(-1)[0].numpy())
    # ic(designs)
    return df.merge(designs[groupers + [name]], on=groupers, how='inner')



def show_df(data, cols=['af2_pae_mean', 'rmsd_af2_des'], traj_types=None, n=999):
    i=1
    all_structures = []
    for _, row in itertools.islice(data.iterrows(), n):
        rmsd_too_high = row['rmsd_af2_des'] > 2
        pae_too_high =  row['af2_pae_mean'] > 5
        key_val = [f'i_{i}']
        for k in cols:
            v = row[k]
            if not isinstance(v, str):
                v = f'{v:.1f}'
            key_val.append(f'{k}_{v}')
            # print(key_val)
        design_name = '__'.join(key_val)
        # print(design_name)
        structures = show_paper_pocket_af2(row, design_name, traj_types=traj_types)
        all_structures.append(structures)
        for s in structures:
            if s:
                cmd.set('grid_slot', i, s.name)
        i += 1
        af2, af2_scaffold, des, motif = structures
        cmd.super(f'{des.name} and name ca', f'{af2.name} and name ca')
        if rmsd_too_high and pae_too_high:
            cmd.color('purple', af2_scaffold.name)
        elif rmsd_too_high: 
            cmd.color('red', af2_scaffold.name)
        elif pae_too_high: 
            cmd.color('blue', af2_scaffold.name)
    cmd.set('grid_mode', 1)
    return all_structures

def set_remote_cmd(remote_ip):
    cmd = get_cmd(f'http://{remote_ip}:9123')
    make_network_cmd(cmd)
    return cmd

def clear():
    # cmd.do('reinitialize everything')
    cmd.delete('all')
    # cmd.do(f'cd {REPO_DIR}/pymol_config')
    # cmd.do('@./pymolrc')

def register_full_atom(pred, true, log=False, gamma=0.95):
    '''
    Calculate coordinate RMSD
    Input:
        - pred: predicted coordinates (L, n_atom, 3)
        - true: true coordinates (L, n_atom, 3)
    Output: RMSD after superposition
    '''
    #ic(pred.shape, true.shape)
    for name, xyz in (('pred', pred), ('true', true)):
        m = f'wrong shape for {name}: {xyz.shape}'
        assert len(xyz.shape) == 3, m
        assert xyz.shape[2] == 3, m
    assert pred.shape == true.shape, f'{pred.shape} != {true.shape}'
    pred = pred[None, None]
    true = true[None]

    def rmsd(V, W, eps=1e-6):
        L = V.shape[1]
        return torch.sqrt(torch.sum((V - W) * (V - W), dim=(1, 2)) / L + eps)

    def centroid(X):
        return X.mean(dim=-2, keepdim=True)

    pred = pred[:, :, :, :3, :].contiguous()
    true = true[:, :, :3, :].contiguous()
    I, B, L, n_atom = pred.shape[:4]

    # center to centroid
    pred_centroid = centroid(pred.view(I, B, n_atom * L,
                                       3)).view(I, B, 1, 1, 3)
    true_centroid = centroid(true.view(B, n_atom * L, 3)).view(B, 1, 1, 3)
    pred = pred - pred_centroid
    true = true - true_centroid

    # reshape true crds to match the shape to pred crds
    true = true.unsqueeze(0).expand(I, -1, -1, -1, -1)
    pred = pred.view(I * B, L * n_atom, 3)
    true = true.view(I * B, L * n_atom, 3)

    # Computation of the covariance matrix
    C = torch.matmul(pred.permute(0, 2, 1), true)

    # Compute optimal rotation matrix using SVD
    V, S, W = torch.svd(C)

    # get sign to ensure right-handedness
    d = torch.ones([I * B, 3, 3], device=pred.device)
    d[:, :, -1] = torch.sign(torch.det(V) * torch.det(W)).unsqueeze(1)

    # Rotation matrix U
    U = torch.matmul(d * V, W.permute(0, 2, 1))  # (IB, 3, 3)

    # Rotate pred
    rP = torch.matmul(pred, U)  # (IB, L*3, 3)
    pred, true = rP[0, ...] + true_centroid, true[0, ...] + true_centroid

    ## On FA coords.
    def T(crds):
        L, n_atom, _ = crds.shape
        #         ic(L, n_atom)
        crds = crds[None, None]
        I, B = 1, 1

        crds = crds - pred_centroid

        # reshape true crds to match the shape to pred crds
        crds = crds.view(I * B, L * n_atom, 3)

        # Rotate pred
        rcrds = torch.matmul(crds, U)  # (IB, L*3, 3)
        crds = rcrds[0, ...] + true_centroid
        crds = crds.reshape(L, n_atom, 3)
        #         return crds[0,0]
        return crds

    return T

def make_network_cmd(cmd):
    # old_load = cmd.load
    def new_load(*args, **kwargs):
        path = args[0]
        with open(path) as f:
            contents = f.read()
        # args[0] = contents
        args = (contents,) + args[1:]
        #print('writing contents')
        cmd.read_pdbstr(*args, **kwargs)
    cmd.is_network = True
    cmd.load = new_load

def show_percents(g):
    # iterate through axes
    for ax in g.axes.ravel():
        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height()*100):.1f}%' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)
        _ =ax.tick_params(axis='x', rotation=90)
        
def add_metrics_sc(df):
    df['self_consistent'] = df['rmsd_af2_des'] < 2.0
    df['self_consistent_and_motif'] = df['self_consistent'] & (np.isnan(df['contig_rmsd_af2_des']) | (df['contig_rmsd_af2_des'] < 1.0))

def get_best(df):
    data = df.groupby(["design_id"]).apply(lambda grp: grp.sort_values(['self_consistent_and_motif', 'contig_rmsd_af2_des', 'rmsd_af2_des'], ascending=[False, True, True]).head(1)).reset_index(drop=True)
    return data

def get_best_design(df, column, ascending=True):
    data = df.groupby(["design_id"]).apply(lambda grp: grp.sort_values([column], ascending=[ascending]).head(1)).reset_index(drop=True)
    return data

def get_best_design_multi(df, columns, ascendings):
    
    data = df.groupby(["design_id"]).apply(lambda grp: grp.sort_values(columns, ascending=ascendings).head(1)).reset_index(drop=True)
    return data

def get_best_n_designs_in_group(df, groups, n, columns, ascendings):
    assert max(df['design_id'].value_counts()) == 1
    data = df.groupby(groups).apply(lambda grp: grp.sort_values(columns, ascending=ascendings).head(n)).reset_index(drop=True)
    return data

def get_keys(df, substring):
    return [k for k in df.keys() if substring in k]


def get_motif_indep(pdb, motif_i):
    indep = aa_model.make_indep(pdb)
    is_motif = aa_model.make_mask(motif_i, indep.length())
    indep_motif, _ = aa_model.slice_indep(indep, is_motif)
    return indep_motif


def get_rog(row):
    trb = dev.analyze.get_trb(row)
    indep = trb['indep']
    rog = conditions.v2.radius_of_gyration_xyz(torch.tensor(indep['xyz'])[~indep['is_sm'], 1])
    return {'design_rog': rog}

    
def get_method(r):
    method = []
    if r['is_aa']:
        method.append('aa')
        if r['inference.ckpt_path'] == '/home/ahern/projects/rf_diffusion/models/aa_v1/BFF_10_remapped.pt':
            method.append('vanilla_ep10')
            if r['potentials.guide_scale'] > 0:
                method.append('potential')

        elif r['inference.conditions.relative_sasa_v2.active']:
            method.append('rasa-conditioned')
    else:
        method.append('rfd')
        if r.get('contigmap.shuffle', False):
            method.append('shuffled')
        else:
            method.append('native')
    return '_'.join(method)


# def add_info


def get_condition(row, key='name'):
    return re.match('.*cond\d+', row[key])[0]

known_rfd_model_paths = (
    '/home/ahern/projects/rf_diffusion/models/theo_pdb/BFF_4.pt',
    '/databases/lab/diffusion/models/hotspot_models/base_complex_finetuned_BFF_9.pt',
)
def is_rfd(row):
    """
    Returns True if the row comes from a vanilla RFDiffusion model.

    Uses heuristics.
    """
    return row.get('inference.ckpt_override_path', '') in known_rfd_model_paths

def parse_rfd_aa_df(df, drop_missing_af2=True):
    try:
        df['seed'] = df.name.apply(lambda x: int(x.split('_cond')[1].split('_')[1].split('-')[0]))
    except Exception as e:
        print('failed to get seed', e)
#     print(
# data['seed'].value_counts())

    # df['n_motif'] = df['contigmap.contigs'].map({
    #     "['10,A1051-1051,31,A1083-1083,26,A1110-1110,69,A1180-1180,10']": 4,
    #     "['10,A1051-1051,31,A1083-1083,96,A1180-1180,10']": 3})

    # df['n_motif'].value_counts()

    df['is_aa'] = df['source'].apply(lambda v: v.startswith('aa_'))
    df['condition'] = df.apply(get_condition, axis=1)


    def get_target_contig_rmsd_af2(r):
        assert r['is_aa'] in [True, False]
        if r['is_aa']:
            return r['contig_rmsd_af2_des']
        else:
            return r['contig_rmsd_af2']

    try:
        df['target_contig_rmsd_af2'] = df.apply(get_target_contig_rmsd_af2, axis=1)
    except Exception as e:
        print('failed to get target_contig_rmsd_af2', e)

    df['design_id'] = df['source'] + '_' + df['name']

    if drop_missing_af2:

        df['missing_af2'] = df['rmsd_af2_des'].isna()

        missing_af2 = df['rmsd_af2_des'].isna()
        print(f'{missing_af2.sum()=}')
        designs = df.groupby('design_id').agg({'missing_af2': 'sum'})
        designs['has_all_af2'] = designs['missing_af2'] == 0
        # print(f'{designs["has_all_af2"].value_counts()=}')
        print(f'dropping {(~designs["has_all_af2"]).sum()}/{len(designs["has_all_af2"])} designs missing any of their AF2 outputs')
        
        # n_drop = has_all_af2['missing_af2'] == 
        design_ids_with_all_af2 = designs[designs['has_all_af2']]
        # df[df['design_id'].isin(design_ids_with_all_af2.index)].shape
        # design_ids_with_all_af2['missing
        df = df[df['design_id'].isin(design_ids_with_all_af2.index)]
    
    df = df.copy()
    df['method'] = df.apply(get_method, axis=1)
    # import assertpy
    # assertpy.assert_that(df['name'].nunique() * 8).is_equal_to(df.shape[0])
    return df

def drop_designs_missing_any_af2(df, expected_af2=8):
    df['has_af2'] = ~df['rmsd_af2_des'].isna()
    designs = df.groupby('design_id').agg({'has_af2': 'sum'})
    designs_with_expected_af2 = designs[designs['has_af2'] == expected_af2]
    return df[df['design_id'].isin(designs_with_expected_af2.index)]

def get_min_max(df, col):
    # Find the row with the minimum value in 'col'
    min_row = df[df[col] == df[col].min()]

    # Find the row with the maximum value in 'col'
    max_row = df[df[col] == df[col].max()]

    # Concatenate the two rows into a new dataframe
    result_df = pd.concat([min_row, max_row])

    # Reset the index of the new dataframe if desired
    result_df.reset_index(drop=True, inplace=True)
    return result_df

def AND(i):
    return '('+ ' and '.join(i) + ')'

def OR(i):
    return '(' + ' or '.join(i) +')'

def NOT(e):
    return f'not ({e})'


def center_entities(all_entities, pymol_selection='hetatm'):
    for entities in all_entities:
        first_e = list(entities.values())[0]
        com = cmd.centerofmass(f"{first_e.name} and {pymol_selection}")
        dx = [-e for e in com]
        for entity in entities.values():
            cmd.translate(dx, entity.name, -1, 0)

def center_object(obj):
    com = cmd.centerofmass(obj)
    dx = [-e for e in com]
    cmd.translate(dx, obj, -1, 0)

def get_trb_path(row):
    return os.path.join(row['rundir'], f'{row["name"]}.trb')

def translate(pymol_name, dxyz):
    dxyz = [e for e in dxyz]
    cmd.translate(dxyz, pymol_name, -1, 0)

def fast_apply(df, f, input_columns):
    o = []

    input_values = {c: df[c] for c in input_columns}
    for c in input_columns:
        assert len(input_values[c]) == df.shape[0], f'{c=}, {len(input_values[c])=} != {df.shape[0]=}'

    for i in range(df.shape[0]):
        o.append(f({c: input_values[c].iloc[i] for c in input_columns}))
    
    return o

