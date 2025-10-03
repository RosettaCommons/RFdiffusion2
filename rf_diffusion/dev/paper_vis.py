from datetime import datetime
import os
import colorsys

import numpy as np
import matplotlib as mpl

from rf_diffusion.dev import show_tip_row
from rf_diffusion.dev import show_bench
from rf_diffusion.dev import analyze
from rf_diffusion.dev import pymol_atom_colors
from rf_diffusion.dev.pymol import mass_paper_rainbow_sel

cmd = analyze.cmd

ligand_hsv_v = 0.35
paper_teal_hsv_v = 0.72
paper_teal_hex = "4fb8ae"

def visualize_design_entities(all_entities, *args, **kwargs):
    for entities in all_entities:
        print(f'visualize_design_entities: {list(entities.values())[0].name}: {datetime.now().strftime("%M:%S")}')
        for label, e in entities.items():
            if label in ['af2', 'chai1_index', 'chai1_best']:    
                continue
            visualize_design(e, *args, **kwargs)

def visualize_design(e, bb_cartoon=True, pretty=True):
    if pretty:
        set_pymol_settings()
    print(f'fix_spheres: {datetime.now().strftime("%M:%S")}')
    fix_spheres()
    # color_design(e)
    print(f'show_backbone_cartoon: {datetime.now().strftime("%M:%S")}')
    if bb_cartoon:
        show_backbone_cartoon(e)
    # cmd.do(f'mass_paper_rainbow {e["protein"]}')
    # make_transparent_rainbow(e, diffused_transparency=0)
    print(f'color_backbone: {datetime.now().strftime("%M:%S")}')
    color_backbone(e)

    print(f'color_lig_sidechain: {datetime.now().strftime("%M:%S")}')
    color_lig_sidechain(e, ligand_hsv_v=ligand_hsv_v)
    print(f'color_lig_atom_spheres: {datetime.now().strftime("%M:%S")}')
    color_lig_atom_spheres(e)
    print(f'color_motif_atom_spheres: {datetime.now().strftime("%M:%S")}')
    color_motif_atom_spheres(e)
    cmd.set('sphere_scale', 0.4, 'metals')


def set_pymol_settings():
    cmd.do('set sphere_scale, 0.25')
    cmd.do('set valence, 1')
    cmd.set('valence', 0)
    # Light/rendering effects
    for l in (
    """set specular, 0
    set ray_shadow, off
    set antialias, 2
    set ray_trace_mode, 1
    set ray_trace_disco_factor, 1
    set ray_trace_gain, 0.1
    set ambient, 0.4
    set direct, 0.45
    set cartoon_sampling, 10
    set ray_trace_color, black
    set reflect, 1
    set reflect_power, 0
    set ribbon_width, 8
    set line_width, 2.5
    set cartoon_flat_sheets, off
    set valence, off
    set cartoon_gap_cutoff, 0""").split('\n'):
        cmd.do(l)
    cmd.set('valence', 0)
    cmd.do('set sphere_scale, 0.25')
    cmd.do('bg white')

def make_transparent_rainbow(
    e,
    diffused_transparency = 0.45):
    cmd.set('cartoon_transparency', diffused_transparency)
    cmd.set('stick_transparency', diffused_transparency, e['sidechains_diffused'])
    mass_paper_rainbow_sel(f'{e["protein"]}')

def show_backbone_cartoon(e):
    cmd.show_as('cartoon', f"({e['protein']}) and not (({e['sidechains_motif']}) or ({e['sidechains_diffused']}))")
    # motif_resi_selector = ' OR '.join(e.selectors[s] for s in ['residue_motif', 'residue_gp_motif', 'sidechains_motif', 'sidechains_diffused'])
    # cmd.show_as('cartoon', f"({e['protein']}) and not ({motif_resi_selector})")
    cmd.show('cartoon', f"({e['protein']})") # and not ({e['residue_motif']})")
    cmd.show('licorice', f"(({e['sidechains_motif']}) or ({e['sidechains_diffused']}))")
    cmd.hide('licorice', f"(({e['sidechains_motif']}) or ({e['sidechains_diffused']})) and (name C or name N or name O)")
    cmd.hide('spheres', f"(({e['sidechains_motif']}) or ({e['sidechains_diffused']})) and (name CA)")
    cmd.set('cartoon_side_chain_helper', 1)

def fix_spheres():
    cmd.remove('hydrogens')
    cmd.set('sphere_transparency', 0.)
    cmd.show('spheres', 'elem Mg')

def color_lig_sidechain(
        e,
        ligand_hsv_v = 0.4,
        cp_2 = {
            # "sidechain":"9F53AC",
            "sidechain":"4fb8ae",
            "ligand": pymol_atom_colors.rgb2hex(*(colorsys.hsv_to_rgb(*[0, 0, ligand_hsv_v])))
        }):

    palette = get_pymol_palette(cp_2)
    for i,k in [
            # (2, 'sidechains_diffused'),
            (0, 'sidechains_motif'),
            (1, 'lig'),
    ]:
        if k != 'lig':
            # cmd.color(palette.name(i), e[k])
            cmd.set('stick_color', palette.name(i), e[k])
            cmd.set('sphere_color', palette.name(i), e[k])
        else:
            cmd.set('stick_color', palette.name(i), e[k])
            cmd.set('sphere_color', palette.name(i), e[k])
            # cmd.do('util.cbac hetatm')
            # cmd.color(palette.name(i), f'({e[k]}) and elem C')
    
    cmd.do('show spheres, hetatm')
    cmd.do('util.cbac hetatm')
    cmd.color(palette.name(1), 'hetatm and elem C')
    cmd.set('stick_color', palette.name(1), 'hetatm')

def get_atom_symbol_hex(
        atom_saturation = 0.5,
        atom_value = 0.8,
        ):
    rgb_by_symbol = pymol_atom_colors.get_rgb_by_symbol()
    hsv_by_symbol = {k:colorsys.rgb_to_hsv(*rgb) for k,rgb in rgb_by_symbol.items()}

    hsv_by_symbol = {k:[h, atom_saturation, atom_value] for k, (h,_,_) in hsv_by_symbol.items()}
    rgb_by_symbol = {k:colorsys.hsv_to_rgb(*hsv) for k,hsv in hsv_by_symbol.items()}

    hex_by_symbol = {k:pymol_atom_colors.rgb2hex(*rgb) for k, rgb in rgb_by_symbol.items()}
    return hex_by_symbol

def color_lig_atom_spheres(
        e,
        atom_saturation = 0.5,
        atom_value = ligand_hsv_v+0.2,
        ):
    color_atom_spheres(
        e['lig'],
        atom_saturation=atom_saturation,
        atom_value=atom_value)


def color_motif_atom_spheres(
        e,
        atom_saturation = 0.5,
        atom_value = paper_teal_hsv_v+0.2
        ):
    color_atom_spheres(
        e['sidechains_motif'],
        atom_saturation=atom_saturation,
        atom_value=atom_value)

def color_atom_spheres(
        selector,
        atom_saturation = 0.5,
        atom_value = ligand_hsv_v+0.2,
        ):
    hex_by_symbol = get_atom_symbol_hex(
        atom_saturation = atom_saturation,
        atom_value = atom_value
    )
    cmd.show('spheres', selector)
    atom_pal = get_pymol_palette(hex_by_symbol)
    for i, atom_type in enumerate(hex_by_symbol.keys()):
        # cmd.color(
        #     atom_pal.name(i), f'hetatm and elem {atom_type}')
        cmd.set(
            # 'sphere_color', atom_pal.name(i), f'hetatm and elem {atom_type}')
            'sphere_color', atom_pal.name(i), f"({selector}) and elem {atom_type}")


def make_more_white(hsv_color, amount=0.5):
    """
    Make an HSV color more white by increasing its Value (brightness) and
    reducing its Saturation.

    :param hsv_color: List or tuple of HSV values [H, S, V]
    :param amount: The factor by which to increase brightness and reduce saturation (between 0 and 1)
    :return: Modified HSV color as a list [H, S, V]
    """
    if len(hsv_color) != 3:
        raise ValueError("Input color must be a list or tuple of length 3.")
    
    H, S, V = hsv_color
    
    if not (0 <= H <= 1) or not (0 <= S <= 1) or not (0 <= V <= 1):
        raise ValueError("HSV values must be between 0 and 1.")
    
    if not (0 <= amount <= 1):
        raise ValueError("Amount must be between 0 and 1.")
    
    # Increase Value towards 1 (white) and decrease Saturation towards 0
    new_V = min(1.0, V + amount * (1.0 - V))  # Increase brightness towards 1
    new_S = max(0.0, S * (1 - amount))  # Decrease saturation towards 0

    return [H, new_S, new_V]

def get_pymol_palette(palette_json):
    colormap_name = 'colormap_name'
    colormap_name = 'colormap_' + '_'.join(palette_json.values())
    rfflow_colormap = mpl.colors.ListedColormap([f'#{v}' for v in palette_json.values()], name=colormap_name)
    mpl.colormaps.register(rfflow_colormap, force=True)
    palette = show_tip_row.PymolPalette(cmd, colormap_name, start_val=0, stop_val=len(palette_json))
    return palette

def color_spectrum(colors, name):
    cmd.do(f'spectrum count, {" ".join(colors)}, {name}')


def color_spectrum_pymol_palette(name, palette, palette_indices=None):
    color_names = palette.all()
    if palette_indices:
        color_names = [palette.name(i) for i in palette_indices]
    color_spectrum(color_names, name)

default_palette_json = {
    "Charcoal":"264653",
    "Burnt sienna":"e76f51",
    "Persian green":"2a9d8f",
    "Olivine":"8ab17d",
    "Saffron":"e9c46a",
    "Sandy brown":"f4a261",
}

def color_spectrum_palette(name, palette_indices=None, palette_json=default_palette_json):
    palette = get_pymol_palette(palette_json)
    color_spectrum_pymol_palette(name, palette, palette_indices)

def color_backbone(
        e,
        paper_rainbow_rgb = dict(enumerate(np.array([
            [255,224,172],
            [255,198,178],
            [255,172,183],
            [213,154,181],
            [149,150,198],
            [102,134,197],
            [75,95,170],
        ], dtype=float) / 255)),
        whitening=0.5,
):
    paper_rainbow_hsv = {k:colorsys.rgb_to_hsv(*rgb) for k,rgb in paper_rainbow_rgb.items()}
    paper_rainbow_hsv = {k:make_more_white(hsv, whitening) for k,hsv in paper_rainbow_hsv.items()}
    paper_rainbow_rgb = {k:colorsys.hsv_to_rgb(*hsv) for k,hsv in paper_rainbow_hsv.items()}
    paper_rainbow_hex = {k:pymol_atom_colors.rgb2hex(*rgb) for k,rgb in paper_rainbow_rgb.items()}
    color_spectrum_palette(e["protein"], palette_json=paper_rainbow_hex)


def paper_rainbow(
        name,
        paper_rainbow_rgb = dict(enumerate(np.array([
            [255,224,172],
            [255,198,178],
            [255,172,183],
            [213,154,181],
            [149,150,198],
            [102,134,197],
            [75,95,170],
        ], dtype=float) / 255)),
        whitening=0.5,
):
    paper_rainbow_hsv = {k:colorsys.rgb_to_hsv(*rgb) for k,rgb in paper_rainbow_rgb.items()}
    paper_rainbow_hsv = {k:make_more_white(hsv, whitening) for k,hsv in paper_rainbow_hsv.items()}
    paper_rainbow_rgb = {k:colorsys.hsv_to_rgb(*hsv) for k,hsv in paper_rainbow_hsv.items()}
    paper_rainbow_hex = {k:pymol_atom_colors.rgb2hex(*rgb) for k,rgb in paper_rainbow_rgb.items()}
    color_spectrum_palette(name, palette_json=paper_rainbow_hex)

def make_suffixed_dir(path):
    i = 0
    while i==0 or os.path.exists(outpath):
        i += 1
        outpath = f'{path}_{i}'
    os.makedirs(outpath)
    return outpath


def save_pngs(
        output_dir_basename, # /path/to/dir --> /path/to/dir_1
        names_per_png, # [[name1, name2], [name3]]
        views_per_png = None, # [view1, view2]
        cm = 20,
        png_names = None,
        zoom_to_fit = True,
):
    
    n_pngs = len(names_per_png)
    if views_per_png is None:
        views_per_png = [cmd.get_view() for i in range(n_pngs)]
    if png_names is None:
        png_names = []
        for names in names_per_png:
            combined_name = '___'.join(names)
            png_names.append(combined_name)
    output_dir = make_suffixed_dir(output_dir_basename)

    pymol_iterator = get_pymol_iterator(names_per_png, views_per_png)

    for i, (_, png_name) in enumerate(zip(pymol_iterator, png_names)):
        png_path = os.path.join(output_dir, f'{png_name}.png')
        print(f'saving image {i}/{n_pngs} to {png_path}...')
        cmd.png(png_path, f'{cm}cm', 0, 300, 1)
        print('Done')
        
        yield png_path
    
    pse_path = os.path.join(output_dir, 'session.pse')
    cmd.enable('all')
    cmd.save(pse_path)
    raise StopIteration
    

def get_pymol_iterator(
        names_per_png, # [[name1, name2], [name3]]
        views_per_png = None, # [view1, view2]
):
    n_pngs = len(names_per_png)
    if views_per_png is None:
        views_per_png = [cmd.get_view() for i in range(n_pngs)]
    cmd.viewport(800, 800)
    for names, view in zip(names_per_png, views_per_png):
        cmd.disable('all')
        for entity in names:
            cmd.enable(entity)
        cmd.set_view(view)
        yield names

from PIL import Image

def show_pngs(image_paths):
    for image_path in image_paths:
        show_image(image_path)

def show_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    print(f'{width=} {height=}: {image_path}')
    image.show()

def get_views(
        pymol_iterator,
):
    views = []
    for _ in pymol_iterator:
        inp = input('please reorient the camera so everything is in frame and press enter (q to quit)')
        if inp == 'q':
            raise Exception('user quit')
        views.append(cmd.get_view())
    return views

from collections import defaultdict
def get_views_and_save_pngs(
        output_dir_basename, # /path/to/dir --> /path/to/dir_1
        names_per_png, # [[A], [B,C], [D]]
        view_groups = None, # [0, 0, 1]
        cm = 20,
        png_names = None,
        views_by_group = None,
        zoom_to_fit = True,
):

    if view_groups is None:
        view_groups = list(range(len(names_per_png)))
    if views_by_group is None:
        base_view = cmd.get_view()
        views_by_group = {i:base_view for i in set(view_groups)}

    views = []
    def get_grouped_views():
        nonlocal views_by_group
        grouped_names = defaultdict(list)
        for i, j in enumerate(view_groups):
            grouped_names[j].extend(names_per_png[i])

        print(f'{list(grouped_names.values())=}')
        views_by_group = get_views(get_pymol_iterator(grouped_names.values(), views_by_group.values()))
        views_by_group = {i:v for i,v in zip(grouped_names.keys(), views_by_group)}

        # Zoom to fit
        for i, j in enumerate(view_groups):
            views.append(views_by_group[j])
        
        if zoom_to_fit:
            for i, objects in enumerate(get_pymol_iterator(names_per_png, views)):
                objects = ' or '.join(objects)
                cmd.do('zoom visible, complete=1, buffer=1')
                views[i] = cmd.get_view()

        return views_by_group
    
    def capture_images(cm=cm):
        print(f'capture_images({cm=})')
        # views = []
        # for i, j in enumerate(view_groups):
        #     views.append(views_by_group[j])
        
        image_paths = []
        for image_path in save_pngs(
            output_dir_basename,
            names_per_png,
            views,
            png_names=png_names,
            cm=cm,
        ):
            image_paths.append(image_path)
            show_image(image_path)
        return image_paths
    
    return get_grouped_views, capture_images, get_pymol_iterator(names_per_png, views)

def capture_session(
        output_dir_basename,
        *args,
        **kwargs,
):
    get_grouped_views, capture_images, pymol_iterator = get_views_and_save_pngs(output_dir_basename, *args, **kwargs)
    get_grouped_views()
    if input('Capture images now? (y/n)') == 'y':
        capture_images()

    return get_grouped_views, capture_images, pymol_iterator

def view_rfd_trajectory(show, struct):
    all_entities = show_bench.show_df(
        show,
        structs={struct},
        des=False,
        af2=False,
        mpnn_packed=False,
        return_entities=True,
    )

    return all_entities

def iter_grid_slot():
    i = 1
    while True:
        yield i
        i += 1

def focus_on_motif(design, structure_predictions):

    # cmd.do(f'mass_paper_rainbow {design["protein"]}')
    cmd.hide('spheres', design.name)
    cmd.hide('everything', f'{design.name} and hetatm')

    sidechains = f"{design['sidechains_motif']} or {design['sidechains_diffused']}"
    cmd.unset('stick_color', sidechains)
    cmd.unset('sphere_color', sidechains)
    color_backbone(design)
    cmd.set('cartoon_transparency', 0.7, design.name)
    for pred in structure_predictions:
        cmd.hide('spheres', pred.name)
        cmd.hide('everything', f'{pred.name} and hetatm')

        sidechains = f"{pred['sidechains_motif']} or {pred['sidechains_diffused']}"
        cmd.unset('stick_color', sidechains)
        cmd.unset('sphere_color', sidechains)
        cmd.set('cartoon_transparency', 0.7, pred.name)

import re
def hacky_get_guidepost_resi(sidechains_diffused):
    resi = re.findall(r'resi\s(\d+)', sidechains_diffused)
    return f"(resi {'+'.join(resi)})"

def align_chai_to_design(chai, design, motif='global'):
    # chai = entities['chai1_index_chai']
    # design = entities['unidealized_chai']
    if motif == 'motif_allatom':
        selector = hacky_get_guidepost_resi(chai['sidechains_diffused'])
        # print(f"cmd.super('{chai.name} and {selector}', '{design.name} and {selector}')")
        cmd.align(f'{chai.name} and {selector}', f'{design.name} and {selector}')
    elif motif == 'global':
        cmd.super(f'{chai.name} and name N+CA+C', f'{design.name} and name N+CA+C')
    elif motif == 'motif_bb':
        selector = hacky_get_guidepost_resi(chai['sidechains_diffused'])
        cmd.align(f'{chai.name} and {selector} and name N+CA+C', f'{design.name} and {selector}  and name N+CA+C')


# def all
# for ee in all_entities:
#     align_chai_to_design(ee['chai1_index_coarse_chai'], ee['unidealized_coarse_chai'], motif=False)
#     align_chai_to_design(ee['chai1_index_fine_chai'], ee['unidealized_fine_chai'], motif=True)