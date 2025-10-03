#!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/../exec/rf_diffusion_aa_shebang.sh" "$0" "$@"'

from rf_diffusion.dev import pymol
import fire
import glob
from icecream import ic

import logging
import os
from rf_diffusion.dev import show_tip_pa

cmd = show_tip_pa.cmd

import numpy as np
import itertools

from rf_diffusion.dev import show_tip_pa
from rf_diffusion.dev import analyze
from rf_diffusion.parsers import parse_pdb_lines_target
from rf_diffusion.dev.pymol import mass_paper_rainbow_sel

logger = logging.getLogger(__name__)

def model_generator(traj_path, seq=False):
    with open(traj_path) as f:
        s = f.read()
        models = s.strip().split('ENDMDL')
        for i, m in enumerate(models):
            if not m:
                continue
            yield m
    #         # o = parsers.parse_pdb_lines(m.split('\n'), False, False, lambda x: 0)
    #         o = inference.utils.parse_pdb_lines(m.split('\n'), True)
    #         xyz = o[0]
    #         if seq:
    #             seqs.append(o[-1])
    #         parsed.append(xyz)
    #     # parsed = torch.concat(parsed)
    # parsed = torch.tensor(np.array(parsed))
    # if seqs:
    #     return parsed, seqs
    # return parsed

def parse_traj(traj_path, n=None):
    
    # d = defaultdict(list)
    d = []
    for pdb_lines in itertools.islice(model_generator(traj_path), n):
        o = parse_pdb_lines_target(pdb_lines.split('\n'), True)
        d.append(o)
        # print(list(o.keys()))
        # print(list((k, type(v)) for k,v in o.items()))
        # for k, v in o.items():
        #     if isinstance(v, np.ndarray):
        #         d[k].append(v)
    
    # for k, v in d.items():
    #     d[k] = np.stack(v)
    
    return d
        
# traj_path = os.path.join(row['rundir'], f'traj/{row["name"]}_pX0_traj.pdb')
# parsed = parse_traj(traj_path)
# parsed['xyz'].shape
def get_last_px0(row):
    px0_traj_path = analyze.get_traj_path(row, 'X0')
    if not os.path.exists(px0_traj_path):
        px0_traj_path = analyze.get_traj_path(row, 'x0')

    parsed = parse_traj(px0_traj_path, n=1)[0]
    n_prot, n_heavy, _ = parsed['xyz'].shape
    n_het, _ = parsed['xyz_het'].shape

    xyz = np.full((n_prot + n_het, n_heavy, 3), float('nan'))
    xyz[:n_prot] = parsed['xyz']
    xyz[n_prot:, 1] = parsed['xyz_het']
    is_het = np.zeros((n_prot + n_het)).astype(bool)
    is_het[n_prot:] = True
    return xyz, is_het


# def motif_backbone_dists(px0, inferred_atom_names_by_i, gp_atom_names_by_i):
def motif_backbone_dists(px0_xyz, inferred_i, gp_i):
    # backbone_crds = []
    # motif_ca
    prot_bb = px0_xyz[inferred_i, :3]
    gp_bb = px0_xyz[gp_i, :3]
    # print(f'{inferred_i=}, {gp_i=}')
    # print(f'{prot_bb=}')
    # print(f'{gp_bb=}')
    d = np.linalg.norm(prot_bb - gp_bb, axis=-1) # L, 3
    mean_d = np.mean(d, axis=0)
    return {f'dist_backbone_gp_{k}':v for k,v in enumerate(mean_d)}

def motif_backbone_dists_row(row):
    trb = analyze.get_trb(row)
    is_sm = trb['indep']['is_sm']
    sm_i = is_sm.nonzero()[0]
    gp_i = list(trb['motif'].keys())
    gp_i = [i for i in gp_i if i not in sm_i]
    inferred_i = trb['con_hal_idx0']
    px0_xyz = trb['px0_xyz_stack'][0]
    assert len(inferred_i) == len(gp_i), f'{gp_i=}, {inferred_i=}'
    # Need to get ordering to do this properly
    # return motif_backbone_dists(px0_xyz, inferred_i, gp_i)
    dists = []
    for c in itertools.permutations(inferred_i, len(inferred_i)):
        dists.append(motif_backbone_dists(px0_xyz, c, gp_i))
    
    dist = min(dists, key=lambda x: sum(map(np.abs, x.values())))
    return dist
        

from tqdm.notebook import tqdm
def apply_dict(df, f, safe=True):
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            metrics_dict = f(row)
        except Exception as e:
            print(safe)
            if safe:
                print(f'Caught exception at row {i}: {row["name"]}: {str(e)}')
                continue
            else:
                raise e
        for k, v in metrics_dict.items():
            df.loc[i, k] = v.item()
    return metrics_dict.keys()

def get_design_df(df):
    groupers = ['name']
    if 'source' in df.columns:
        groupers = ['source'] + groupers
    designs = df.drop_duplicates(groupers, ignore_index=True).reset_index(drop=True)
    return designs

def apply_dict_design(df, f, **kwargs):
    designs = get_design_df(df)
    print(f'{df.shape=}, {designs.shape=}')
    keys = apply_dict(designs, f, **kwargs)
    groupers = ['name']
    if 'source' in df.columns:
        groupers = ['source'] + groupers
    return df.merge(designs[groupers + list(keys)], on=groupers, how='inner')

# smol_df = df.sample(1).reset_index(drop=True)

# data = df[df['self_consistent']]
# data = data[data['model'] == '8_1681815206.1173074/models/BFF_8.pt']
# data = data[data['benchmark'] == 'tip_2_lysozyme_rigid']
# data = data[data['contig_rmsd_af2_atomized'] == data['contig_rmsd_af2_atomized'].min()]
# df = analyze.read_metrics('/home/ahern/benchmarks/aa_diffusion/tip_atoms/220420_tip_cmp/out/compiled_metrics.csv')
# df = apply_dict_design(df, motif_backbone_dists_row, safe=False)


def is_self_consistent(row):
    return (row['rmsd_af2_des'] < 2) and (row['af2_pae_mean'] < 5)
# df['self_consistent'] = df.apply(is_self_consistent, axis=1)

import pandas as pd

import re

def glob_re(pattern, strings):
    return filter(re.compile(pattern).match, strings)

# '/mnt/home/ahern/projects/dev_rf_diffusion/debug/no_so3_0_*.pdb'a
def get_sdata(path, pattern=None, progress=False):
    traj_paths = glob.glob(path)
    if pattern:
        traj_paths = glob_re(pattern, traj_paths)
    traj_paths = [p[:-4] for p in traj_paths]
    traj_paths = sorted(traj_paths)
    srows = []
    for traj_path in tqdm(traj_paths, disable=not progress):
        srows.append(analyze.make_row_from_traj(traj_path))
    data = pd.DataFrame.from_dict(srows)
    data['des_color'] = 'rainbow'
    try:
        data['seed'] = data.name.apply(lambda x: int(x.split('_cond')[1].split('_')[1].split('-')[0]))
    except Exception as e:
        print(e)
    return data


def transform_file_path(file_path):
    # Extract date and epoch from the file path using regular expressions
    pattern = r'train_session(\d{4}-\d{2}-\d{2})_\d+\.\d+/models/BFF_(\d+)\.pt'
    match = re.search(pattern, file_path)
    
    if not match:
        raise Exception('not match')
    # Extract the date and epoch from the matched groups
    date = match.group(1)
    epoch = match.group(2)

    # Rearrange the extracted date and epoch to the desired format
    formatted_date = '-'.join(date.split('-')[1:])  # Extract month and day from the date
    formatted_epoch = f'epoch_{epoch:>02}'

    # Return the transformed string
    return f'{formatted_date}_{formatted_epoch}'
    
def get_epoch(row):
    ckpt = row['inference.ckpt_path']
    ckpt = ckpt.split('_')[-1]
    ckpt = ckpt[:-3]
    return float(ckpt)

def load_df(metrics_path):
    df = analyze.read_metrics(metrics_path)
    df['seed'] = df.name.apply(lambda x: int(x.split('_cond')[1].split('_')[1].split('-')[0]))
    try:
        df['model_number'] = df['model'].apply(lambda x: int(x.split('.')[0]))
    except Exception as e:
        print(e)
    # df['epoch'] = df.name.apply(lambda x: int(x.split('cond')[1].split('_')[1].split('-')[0]))
    df['des_color'] = pd.NA
    try:
        df['model'] = df['inference.ckpt_path'].apply(transform_file_path)
    except Exception:
        pass
    try:
        df['epoch'] = df.apply(get_epoch, axis=1)
    except Exception as e:
        print(f'caught exception {e}')
    
    root, _ = os.path.splitext(metrics_path)
    root = os.path.abspath(root)
    root = root.split('/')[-3]
    df['source'] = root
    df['design_id'] = df['source'] + '_' + df['name']

    return df

def show_df(data, structs={'X0'}, af2=False, chai1_best=False, des=False, pair_seeds=False, return_entities=False, **kwargs):
    cmd.set('grid_mode', 1)
    all_pymol = []
    all_entities = []
    for i, (_, row) in enumerate(data.iterrows(), start=1):
        # print(row[['benchmark', 'name', 'dist_backbone_gp_sum', 'contig_rmsd_af2_des', 'contig_rmsd_af2_atomized']])
        # print(f'{(row['rmsd_af2_des'] < 2)=}, {row['contig_rmsd_af2_des'] < 2=} and {row['af2_ligand_dist'] > 2=}')
        # print(row[['benchmark', 'name', 'rmsd_af2_des', 'contig_rmsd_af2_des_atomized','contig_rmsd_af2_des', 'dist_backbone_gp',  'contig_rmsd_af2_atomized', 'inference.guidepost_xyz_as_design_bb']]) 
        des_color = None
        hetatm_color = None
        if not pd.isna(row['des_color']):
            des_color = row['des_color']
        if 'hetatm_color' in row and not pd.isna(row['hetatm_color']):
            hetatm_color = row['hetatm_color']
        entities = show_tip_pa.show(row, structs=structs, af2=af2, chai1_best=chai1_best, des=des, des_color=des_color, hetatm_color=hetatm_color, **kwargs)
        
        all_entities.append(entities)

        # Uncomment to show only residue motifs
        # for label, entity in entities.items():
        #     ic(entity)
        #     cmd.hide(AND([entity.name, NOT(entity['residue_motif'])]))

        print(f'{row["name"]=}')
        for e in entities.values():
            print(f'{e.name=}')
            v = e.name
            grid_slot = i
            if pair_seeds:
                grid_slot = row['seed'] + 1
            cmd.set('grid_slot', grid_slot, v)
        all_pymol.append(v)
    cmd.color('atomic', 'hetatm and not elem C')
    cmd.set('valence', 1)
    if return_entities:
        return all_entities
    return all_pymol
        
            
def write_png(path):
    cmd.png(path, 0, 0, 100, 0)
    
def add_pymol_name(data, keys):
    def f(row):
        pymol_prefix = []
        for k in keys:
            if k == 'model':
                v = row['inference.ckpt_path']
                v = f"{v.split('/')[-4].split('202')[0]}_{v.split('/')[-1]}"
            else:
                v = row[k]
                             
            if k == 'inference.ckpt_path':
                v = v.split('/')[-1]
            if k == 'inference.input_pdb':
                v = v.split('/')[-1]
            if k == 'inference.ckpt_override_path':
                v = v.split('/')[-1]
            if k == 'pdb_path':
                v = v.split('/')[-1]
            k_str = k.replace('.', '_')
            v = str(v)
            v = v.replace('.', '_')
            v = v.replace(',', '_')
            v = v.replace(' ', '_')
            if k == 'rundir':
                vs = v.split('/')
                if 'out' in vs:
                    v = vs[vs.index('out')-1]
                else:
                    v = vs[-2]

            pymol_prefix.append(f"{k_str}-{v}")
        pymol_prefix = '_'.join(pymol_prefix)
        return pymol_prefix
    data['pymol'] = data.apply(f, axis=1)

def get_sweeps(data):
    uniques = {}
    for k in data.keys():
        try:
            uniques[k] = data[k].unique()
        except Exception:
            continue
        

    sweeps = {k:v for k,v in uniques.items() if len(v) > 1}

    for k in ['name', 'seed', 'inference.output_prefix', 'pdb_path', 'transforms.configs.AddConditionalInputs.p_is_guidepost_example']:
        _ = sweeps.pop(k, None)

    return sweeps

backbone = '(name ca or name c or name n or name o)'

def main(path,
         name=None,
         clear=False,
         structs=['X0'],
         pymol_keys=None,
         pymol_url='http://localhost:9123',
         max_seed = None,
         only_seed=None,
         pair_seeds=False,
         des=False,
         af2=False,
         chai1_best=False,
         mpnn_packed=False,
         ga_lig=False,
         rosetta_lig=False,
         input=False,
         sidechains=False,
         hydrogenated=False,
         key=None,
         show_origin=False,
         hide_oxygen=False,
         debug=False,
         filt=None,
         extra=None,
         ppi=False,
         cartoon=False,
         ):
    ic(pymol_url)
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    pymol.init(pymol_url, init_colors=True)
    # cmd = analyze.get_cmd(pymol_url)
    # analyze.cmd = cmd
    # show_tip_pa.cmd = cmd
    # show_tip_row.cmd = cmd
    # ic('before show pro3')
    # cmd.fragment('pro')
    # ic('after show pro')
    ic.configureOutput(includeContext=True)
    datas = []
    for p in path.split(','):
        datas.append(get_sdata(p))
    data = pd.concat(datas, ignore_index=True)
    assert len(data) > 0
    data['des_color'] = 'rainbow'
    print(f'1 {data.shape=}')
    if name:
        data['pymol'] = name
    # if pymol_keys:
    #     ic(pymol_keys, structs)
    #     pymol_keys = pymol_keys.split(',')
    #     add_pymol_name(data, pymol_keys)
    # ic(data)
    ic(data.shape)
    if only_seed is not None:
        data = data[data['seed'] == only_seed]
    elif max_seed is not None:
        data = data[data['seed'] <= max_seed]
    if filt:
        k, v = filt.split('=')
        ic(k,v)
        ic(data.value_counts(k))
        data = data[data[k].astype(str) == v]

    sweeps = get_sweeps(data)
    logger.debug(f'{sweeps=}')
    if len(sweeps):
        if key:
            keys = key
            if isinstance(key, str):
                keys = key.split(',')
        else:
            keys = [k for k in sweeps.keys() if k not in ['contigmap.contig_atoms']]
        ic(keys)
        add_pymol_name(data, keys)
    if clear:
        show_tip_pa.clear()
    if extra:
        data['extra'] = extra
    
    if 'pymol' in data.columns:
        data.sort_values('pymol', inplace=True)
    ic(extra)
    all_entities = show_df(
            data,
            structs=structs,
            des=des,
            pair_seeds=pair_seeds,
            af2=af2,
            chai1_best=chai1_best,
            mpnn_packed=mpnn_packed,
            ga_lig=ga_lig,
            rosetta_lig=rosetta_lig,
            input=input,
            hydrogenated=hydrogenated,
            return_entities=True)

    if sidechains:
        ic(all_entities)
        show_sidechains(all_entities, ['mpnn_packed', 'ga_lig'])
    
    if show_origin:
        pa =pseudoatom(cmd, label='the_origin')
        cmd.center(pa)
        cmd.color('red', pa)
        cmd.set('grid_slot', -2, pa)

    # Assume chain B is the target
    if ppi:
        format_ppi(all_entities)

    # Show as cartoons:
    if cartoon:
        format_cartoon(all_entities)


    # cmd.do('mass_paper_rainbow')
def format_cartoon(all_entities):
    for entities in all_entities:
        for name, e in entities.items():
            print(f'{name=} {list(e.selectors.keys())}')
            cmd.show_as('cartoon', e['protein'])

def format_ppi(all_entities):
    score_names = ['af2', 'chai1']
    for entities in all_entities:
        for name, e in entities.items():
            if name in score_names:
                continue
            e.selectors['target'] = "chain B and not hetatm"
            cmd.color('paper_teal', e['target'])
            mass_paper_rainbow_sel(f"({e.NOT('target')} and not hetatm)")

def pseudoatom(
        cmd,
        pos: list = [0,0,0],
        label='origin',
        ):
    cmd.pseudoatom(label,'', 'PS1','PSD', '1', 'P',
        'PSDO', 'PS', -1.0, 1, 0.0, 0.0, '',
        '', pos)
    # cmd.do(f'label {label}, "{label}"')
    return label


# # TODO: make this monadic
# cmd = analyze.get_cmd('http://10.64.100.67:9123')
# analyze.cmd = cmd
# show_tip_pa.cmd = cmd
# show_tip_row.cmd = cmd

def show_sidechains(all_entities, which_entities=['mpnn_packed']):
    for entity in all_entities:
        for name in which_entities:
            if name not in entity:
                continue
            packed = entity[name]
            diffused = AND([packed.name, NOT(OR([packed['lig'], packed['sidechains_diffused'], packed['sidechains_motif']]))])
            cmd.show('licorice', diffused)
            cmd.color('lightteal', AND([diffused, NOT(backbone)]))

if __name__ == '__main__':
    fire.Fire(main)
