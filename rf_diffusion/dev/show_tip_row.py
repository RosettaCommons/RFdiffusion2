
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import functools

from rf_diffusion.dev import analyze
from rf_diffusion.dev.pymol import mass_paper_rainbow_sel
cmd = analyze.cmd

class PymolPalette:

    def __init__(self, cmd, cmap_name, start_val, stop_val, reverse=False):
        self.cmd = cmd
        self.start_val = start_val
        self.stop_val = stop_val
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        if reverse:
            self.cmap = self.cmap.reversed()
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.defined = {}

    def __call__(self, val):
        rgb = self.scalarMap.to_rgba(val)[:3]
        rgb = list(map(float, rgb))
        return rgb
        # if val not in self.defined:
        #     self.defined[val] = self.define(cmd, val)
        # return self.defined[val]

    @functools.lru_cache(maxsize=None)
    def name(self, v):
        # for i in range(start_val
        rgb = self(v)
        name = f'{self.cmap_name}_{v}'
        self.cmd.set_color(name, rgb)
        return name

    def all(self):
        return [self.name(v) for v in range(self.start_val, self.stop_val)]

def AND(i):
    return '('+ ' and '.join(i) + ')'

def OR(i):
    return '(' + ' or '.join(i) +')'

def NOT(e):
    return f'not ({e})'

def get_atom_names_by_res_idx(srow):
    trb = analyze.get_trb(srow)
    is_atom_motif = trb['atomize_indices2atomname']
    idx = trb['indep']['idx']
    atom_names_by_res_idx = {}
    for i0, atom_names in is_atom_motif.items():
        idx_pdb = idx[i0]
        atom_names_by_res_idx[idx_pdb] = atom_names
    
    return atom_names_by_res_idx

def get_motif_selectors(atom_names_by_res_idx):
    motif_resi_selectors = []
    motif_atom_selectors = []
    for idx_pdb, atom_names in atom_names_by_res_idx.items():
        motif_resi_selectors.append(f'resi {idx_pdb}')
        atom_sel = ' or '.join(f'name {a}' for a in atom_names)
        motif_atom_selectors.append(f'({atom_sel})')
    
    return motif_resi_selectors, motif_atom_selectors

def get_selectors(atom_names_by_res_idx):
    motif_resi_selectors, motif_atom_selectors = get_motif_selectors(atom_names_by_res_idx)
    sidechains_motif = OR(map(AND, zip(motif_resi_selectors, motif_atom_selectors)))
    sidechains_diffused = OR(map(AND, zip(motif_resi_selectors, map(NOT, motif_atom_selectors))))
    lig = 'hetatm'
    protein = f'(not {lig})'
    # lig = AND([carbon, lig])
    sidechains_motif = AND([protein, sidechains_motif])
    sidechains_diffused = AND([protein, sidechains_diffused])
    shown = OR([
            '(name C or name N or name CA)',
            OR(motif_resi_selectors),
            lig,
        ])
    selectors = {
        'shown': shown,
        'protein': protein,
        'sidechains_diffused':sidechains_diffused,
        'sidechains_motif': sidechains_motif,
        'lig': lig,
    }
    return selectors

def get_motif_selectors_2(atom_names_by_res_idx):
    motif_resi_selectors = []
    motif_atom_selectors = []
    for idx_pdb, atom_names in atom_names_by_res_idx.items():
        motif_resi_selectors.append(f'resi {idx_pdb}')
        atom_sel = ' or '.join([f'name {a}' for a in atom_names] + ['name placeholder'])
        motif_atom_selectors.append(f'({atom_sel})')
    
    return motif_resi_selectors, motif_atom_selectors

def get_selectors_2(atom_names_by_res_idx_0):

    token_by_selector_name = {
        'residue_gp_motif': ['ALL'],
        'residue_motif': 'RES',
    }

    residue_selectors = {}
    for selector_name, token in token_by_selector_name.items():
        residue_idxs = [i for i, atom_names in atom_names_by_res_idx_0.items() if atom_names == token]
        selectors = ['resi 9999']
        for resi in residue_idxs:
            selectors.append(f'resi {resi}')
        residue_selectors[selector_name] = OR(selectors)

    residue_tokens = list(token_by_selector_name.values())
    atom_names_by_res_idx = {i:atom_names for i, atom_names in atom_names_by_res_idx_0.items() if atom_names not in residue_tokens}
    motif_resi_selectors, motif_atom_selectors = get_motif_selectors_2(atom_names_by_res_idx)
    sidechains_motif = OR(map(AND, zip(motif_resi_selectors, motif_atom_selectors)))
    sidechains_diffused = OR(map(AND, zip(motif_resi_selectors, map(NOT, motif_atom_selectors))))
    if sidechains_motif == '()':
        sidechains_motif = '(resi 9999)'
    if sidechains_diffused == '()':
        sidechains_diffused = '(resi 9999)'
    if len(motif_resi_selectors) == 0:
        motif_resi_selectors = ['(resi 9999)']
    lig = 'hetatm'
    protein = f'(not {lig})'
    sidechains_motif = AND([protein, sidechains_motif])
    sidechains_diffused = AND([protein, sidechains_diffused])
    shown = OR([
            '(name C or name N or name CA)',
            OR(motif_resi_selectors),
            # resi_motifs_selector,
            residue_selectors['residue_motif'],
            lig,
        ])
    selectors = {
        'shown': shown,
        'protein': protein,
        'sidechains_diffused':sidechains_diffused,
        'sidechains_motif': sidechains_motif,
        'lig': lig,
        **residue_selectors,
    }
    selectors.update(get_individual_residue_selectors(motif_resi_selectors, motif_atom_selectors))
    return selectors

def get_individual_residue_selectors(motif_resi_selectors, motif_atom_selectors):
    individual_residue_selectors = {}
    for i, (selector) in enumerate(motif_atom_selectors):
        individual_residue_selectors[f'motif_resi_{i}'] = selector
    return individual_residue_selectors

def get_atom_selector(obj, ch, idx, atom_names):
    atom_sel = ' or '.join(f'name {a}' for a in atom_names)
    return f'({obj} and chain {ch} and resi {idx} and ({atom_sel}))'


def colored_selectors(selectors):
    return {k: v for k,v in selectors.items() if 'motif_resi_' not in k}

def color_selectors(selectors, carbon=True, verbose=False, des_color=None, hetatm_color=None, palette_name='Pastel1', palette_n_colors=9):
    palette = PymolPalette(cmd, palette_name, 0, palette_n_colors)
    # if not carbon:
    #     selectors = [AND([s, carbon]) for s in selectors]
    selectors = colored_selectors(selectors)
    for j,sel in enumerate(selectors.values()):
        color = palette.name(j)
        if verbose:
            print(f'{color} --> {sel}')
        if des_color and j==0:
            if des_color == 'rainbow':
                mass_paper_rainbow_sel(sel)
            else:
                cmd.color(des_color, sel)
        else:
            # print(f'{sel=}')
            cmd.color(color, sel)

        if hetatm_color:
            cmd.color(hetatm_color, f'({sel}) and hetatm and elem C')
    return palette

def show_design(srow, name=None):
    des_pdb = analyze.get_design_pdb(srow)
    des = f'design_{name}'
    cmd.load(des_pdb, des)
    pymol_objects = [des]
    atom_names_by_res_idx = get_atom_names_by_res_idx(srow)
    selectors = get_selectors(atom_names_by_res_idx)
    # shown = selectors.pop('shown')
    for o in pymol_objects:
        cmd.hide('everything', o)
        cmd.show('licorice', AND([o, selectors['shown']]))
    color_selectors(selectors)
    return pymol_objects, selectors

def show_protein(pdb, atom_names_by_res_idx, name=None):
    des = f'protein_{name}'
    cmd.load(pdb, des)
    pymol_objects = [des]
    selectors = get_selectors(atom_names_by_res_idx)
    # shown = selectors.pop('shown')
    for o in pymol_objects:
        cmd.hide('everything', o)
        cmd.show('licorice', AND([o, selectors['shown']]))
    color_selectors(selectors)
    return pymol_objects, selectors

