
import xmlrpc.client as xmlrpclib

class XMLRPCWrapperProxy(object):
    def __init__(self, wrapped=None):
        self.name = 'cmd'
        self.wrapped = wrapped

    def __getattr__(self, name):
        attr = getattr(self.wrapped, name)
        # ic(type(self), attr)
        wrapped = type(self)(attr)
        wrapped.name = name
        return wrapped

    def __call__(self, *args, **kw):
        all_args = tuple(map(repr, args))
        all_args += tuple(f'{k}={repr(v)}' for k,v in kw.items())
        try:
            return self.wrapped(*args, **kw)
        except Exception as e:
            all_args = tuple(map(str, args))
            all_args += tuple(f'{k}={v}' for k,v in kw.items())
            raise Exception(f"cmd.{self.name}('{','.join(all_args)})'") from e

def get_cmd(pymol_url='http://localhost:9123'):
    cmd = xmlrpclib.ServerProxy(pymol_url, allow_none=True)
    if not  ('ipd' in pymol_url or 'localhost' in pymol_url):
        make_network_cmd(cmd)
    return cmd

cmd = None
def init(pymol_url='http://localhost:9123', init_colors=False):
    global cmd
    cmd_inner = get_cmd(pymol_url)
    if cmd is None:
        cmd = XMLRPCWrapperProxy(cmd_inner)
    else:
        print(f'cmd.wrapped = {cmd_inner=}')
        cmd.wrapped = cmd_inner
    
    if init_colors:
        set_colors()
        cmd.set('sphere_scale', 0.3)
    
def mass_paper_rainbow_sel(name):
    """
    Colors the given selection with a pastel gradient according to residue index.
    """
    cmd.spectrum("count", "paper_navaho paper_melon paper_pink paper_purple paper_lightblue paper_blue paper_darkblue", name, None, None, 0, 1)

def set_colors():
    """
    Creates extra colors in pymol 
    """
    print('Setting colors in pymol')
    cmd.set_color("nitrogen",        [2,118,253])
    cmd.set_color("pymol_gray",      [51,51,51])
    cmd.set_color("pymol_black",     [34,34,34])
    cmd.set_color("good_yellow",     [250,199,44])
    cmd.set_color("good_teal",       [41,176,193])
    cmd.set_color("good_green",      [170,195,47])
    cmd.set_color("good_pink",       [236,114,164])
    cmd.set_color("good_blue",       [68,153,231])
    cmd.set_color("good_gray",       [220,220,220])
    cmd.set_color("good_red",        [228,74,62])
    cmd.set_color("good_light_green",[101,179,124])

    #cmd.set_color("paper_yellow",    [1,0.878,0.675])
    #cmd.set_color("paper_pink",      [1.0,0.675,0.718])
    #cmd.set_color("paper_blue",      [0.408,0.525,0.773])
    cmd.set_color("paper_teal",      [ 0.310, 0.725, 0.686 ])
    cmd.set_color("paper_navaho",     [255,224,172]) #FFE0AC1
    cmd.set_color("paper_melon",      [255,198,178]) #FFC6B2
    cmd.set_color("paper_pink",       [255,172,183]) #FFACB7
    cmd.set_color("paper_purple",     [213,154,181]) #D59AB5
    cmd.set_color("paper_lightblue",  [149,150,198]) #9596C6
    cmd.set_color("paper_blue",       [102,134,197]) #6686C5
    cmd.set_color("paper_darkblue",   [75,95,170]) #4B5FAA

def make_network_cmd(cmd):
    # old_load = cmd.load
    def new_load(*args, **kwargs):
        path = args[0]
        if path.endswith('.pse'):
            return cmd.do(f"load {path}")
        with open(path) as f:
            contents = f.read()
        # args[0] = contents
        args = (contents,) + args[1:]
        #print('writing contents')
        cmd.read_pdbstr(*args, **kwargs)
    cmd.is_network = True
    cmd.load = new_load

init()
