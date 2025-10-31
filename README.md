> 🚧 **Under Construction:**  
> This is an initial release of the functionality — further documentation and cleanup is in progress.  Currently only inference is supported. But everything currently documented in this README should be runnable. If you are experiencing a bug with inference, feel free to file an issue and attach the output .pdb, .trb, **and** a picture of the design visualized according to the pymol visualization section of this README.

# RFdiffusion2

Open source code for RFdiffusion2 as described in the following pre-print.

```bib
@article{ahern2025atom,
  title={Atom level enzyme active site scaffolding using RFdiffusion2},
  author={Ahern, Woody and Yim, Jason and Tischer, Doug and Salike, Saman and Woodbury, Seth M and Kim, Donghyo and Kalvet, Indrek and Kipnis, Yakov and Coventry, Brian and Altae-Tran, Han Raut and others},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

More detailed information about how to run, install, and use RFdiffusion2 can be found [here](https://rosettacommons.github.io/RFdiffusion2/). 

<!-- ## Table of Contents
- [Setup](readme_link.html#setup)
- [Inference Example](readme_link.html#inference)
- [Viewing Designs](readme_link.html#viewing-designs)
  - [PyMOL and designs on the same machine](readme_link.html#same_machine_pymol)
  - [PyMOL running locally, designs on remote GPU](readme_link.html#remote_machine_pymol)
- [Additional Info](readme_link.html#additional_info)
  - [Running the AME benchmark](readme_link.html#ame_benchmark)
  - [Pipeline metrics](readme_link.html#pipeline_metrics)
    - [RFdiffusion2 outputs](readme_link.html#rfdiffusion2_outputs)
    - [LigandMPNN outputs](readme_link.html#ligandmpnn_outputs)
-->

## Set-up
<a id="set-up"></a>

If these set-up instructions do not work for your system see the [Installation Guide](https://rosettacommons.github.io/RFdiffusion2/installation.html) for troubleshooting issues with the
provided image and alternative instructions for how to install RFdiffusion2 from source. 

1. **Clone the repo.**

2. **Add the repo to your PYTHONPATH**

   If you installed the repo into `/my/path/to/RFdiffusion2`, then run:
   ```bash
   export PYTHONPATH="/my/path/to/RFdiffusion2"
   ```
   You will need to export this variable every time you use RFdiffusion2.

3. **Download the model weights and containers:**
   ```bash
   cd /my/path/to/RFdiffusion2
   python setup.py
   ```
   These files are quite large so the download process can take over half an hour. If the download process gets terminated before finishing run `python setup.py overwrite` so that any partially downloaded files can be overwritten. 

4. **Install Apptainer**

   *Note: You can also run RFdiffusion2 with [Singularity](https://sylabs.io/singularity/).*

   RFdiffusion2 (RFD2) uses [Apptainer](https://apptainer.org) to simplify the environment setup.
   If you do not already have Apptainer on your system, please follow the [Apptainer installation instructions](https://apptainer.org/docs/admin/main/installation.html) for your operating system.

   If you manage your packages on linux with `apt` you can simply run:
   ```bash
   sudo add-apt-repository -y ppa:apptainer/ppa
   sudo apt update
   sudo apt install -y apptainer
   ```

   Once apptainer is installed, the image is provided as:
   ```
   rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif
   ```

   You can then run commands as such:
   ```bash
   apptainer exec --nv exec/bakerlab_rf_diffusion_aa.sif <path-to-python_file> <args>
   ```

## Inference Example
<a id="inference"></a>

For other usage examples, see the [Usage page](rosettacommons.github.io/RFdiffusion2/usage/usage.html) in the external documentation. 

The below commands run several demos that show off some of the inference capabilities of RFdiffusion2 including: 
- enzyme design from an atomic motif and small molecule
- enzyme design from an atomic motif of unknown sequence positions and a small molecule
- small-molecule binder design with RASA conditioning 
The possible demo options can be found in `rf_diffusion/benchmark/open_source_demo.json` which also shows the different inference configurations
used in each demo. Note that this will be extremely slow if not run on a GPU.

<!--The default argument `in_proc=True` in `open_source_demo.yaml` makes the script run locally. With `in_proc=False` the pipeline will automatically distribute the tasks using SLURM, but this is not yet supported.-->  
The demo generates 150 residue proteins with many of the residues atomized, so it can take upwards of 30 minutes to run all cases.  Each case may take up to 10 minutes on an RTX2060.

**Run single demo case:**
```bash
apptainer exec --nv rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif rf_diffusion/benchmark/pipeline.py --config-name=open_source_demo sweep.benchmarks=active_site_unindexed_atomic_partial_ligand
```
You can replace `active_site_unindexed_atomic_partial_ligand` with any of the other demos included in `rf_diffusion/benchmark/open_source_demo.json`.

**Run all demo cases:**
```bash
apptainer exec --nv rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif rf_diffusion/benchmark/pipeline.py --config-name=open_source_demo
```

The outputs will be written to:
```
pipeline_outputs/${now:%Y-%m-%d}_${now:%H-%M-%S}_open_source_demo
```
in the directory that you are running the image file from. 

This runs only the design stage of the pipeline.  In order to continue through sequence-fitting with [LigandMPNN](https://github.com/dauparas/LigandMPNN) and folding with [Chai1](https://github.com/chaidiscovery/chai-lab), pass the command line argument: `stop_step='end'`.  Note that Chai1 cannot run on all GPU architectures.

Pipeline runs can be resumed by passing `outdir=/path/to/your/output/directory`.


## Viewing Designs
<a id="viewing-designs"></a>

Visualizing the design outputs can be confusing when looking at the raw .pdb files, especially for unindexed motif scaffolding, in which the input motif is incorporated into the protein at indices of the network's choice.  
To simplify this, we provide scripts for visualizing the outputs of the network that interact with a local PyMOL instance over XMLRPC.

Download [PyMOL](https://www.pymol.org/) and run it as an XML-RPC server with:
```bash
pymol -R
```

### PyMOL and designs on the same machine
<a id="same_machine_pymol"></a>

Run:
```bash
apptainer exec rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif rf_diffusion/dev/show_bench.py --clear=True --key=name '/absolute/path/to/pipeline_outputs/output_directory/*.pdb'
```

This command will display the predicted X1 at each step in the flow-matching trajectory for each design in the output directory.  
You should see something like:

![unindexed_atomic](assets/unindexed_atomic_0_ray.png)

**Legend:**
- Atomized protein motif atoms : green.
- Atomized protein motif atoms : blue.
- Backbone protein motif atoms : yellow.
- The generated backbone will be colored a pastel gradient.
- Any small molecules will have their carbon atoms colored purple.

### PyMOL running locally, designs on remote GPU
<a id="remote_machine_pymol"></a>

It is common for users to be sshed into a gpu cluster for running designs.  
It is still possible to view designs on a remote computer from your local PyMOL, as long as your remote computer has a route to your local computer (via VPN or ssh proxy).

You will need to know an IP address that will point back to your computer, typically you can use 127.0.0.1.

You will also need to add the -R option when you are signing into your cluster and provide the path back to your PyMOL server:
```
ssh username@hostname -R 9123:localhost:9123
```
You do **not** need to replace `localhost` with an IP address. The 9123 is the port that should have been printed in the PyMOL terminal
when you first set up the XML-RPC server.

If you need to sign into multiple servers before you can run Apptainer, you will need to sign into your cluster like this:
```
ssh -J username@first_hostname username@second_hostname -R 9123:localhost:9123
```
The -J option stands for 'jump host' and allows you to connect to a remote host through an intermediate server in one command.

Simply append `--pymol_url=http://127.0.0.1:9123` to the command, i.e. from your remote machine (cluster) run:
```bash
apptainer exec rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif rf_diffusion/dev/show_bench.py --clear=True --key=name '/absolute/path/to/pipeline_outputs/output_directory/*.pdb' --pymol_url=http://127.0.0.1:9123
```
If 127.0.0.1 does not point to your local machine, replace it in all the above steps in this section with an IP address that does point to your machine.  

For more information about using PyMOL remotely, see this blog post on [Controlling PyMOL from afar](https://www.blopig.com/blog/2024/11/controlling-pymol-from-afar/).

## Additional Information
<a id="additional_info"></a>

### Running the AME benchmark
<a id="ame_benchmark"></a>

We crawled M-CSA for 41 enzymes where all reactants and products are present to create this benchmark.  
Only positon-agnostic tip atoms are provided to the network. 100 designs for each case are created. Run it with:
```bash
apptainer exec --nv rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif rf_diffusion/benchmark/pipeline.py --config-name=enzyme_bench_n41_fixedligand in_proc=True
```
Running this entire benchmark will perform [41 active sites * 100 designs per active site * 8 sequences per design] chai folding runs, which will take a prohibitively long time on a single machine, but for reproducibility it is included.

### Pipeline metrics
<a id="pipeline_metrics"></a>

We also include the code that was used to benchmark the network.  
The outline of the benchmarking process is:
1. Use RFdiffusion2 to generate backbones
2. LigandMPNN for sequence fitting
3. Chai-1 for refolding

The final output csv shows many metrics about the designed structure.  
Notably, we consider motif recapitulation metrics to measure how well the network scaffolded the given motif.  
Motif recapitulation metrics follow the following formula:

```
contig_rmsd_a_b_s
```

Where:

- **a, b**: the proteins being compared:
  - `des`: The MPNN packed protein
  - `pred`: The Chai prediction
  - `ref`: The input PDB  
    (With the caveat that 'ref' is always omitted from the name.)

- **s**: the comparison type:
  - ` ` : backbone (N, Ca, C)
  - `c_alpha`: Ca
  - `full_atom`: All heavy atoms
  - `motif_atom`: Only motif heavy atoms

#### RFdiffusion2 outputs
<a id="rfdiffusion2_outputs"></a>

The network outputs the protein with both the indexed backbone region and the unindexed atomized region.  
After that several idealization steps are conducted; the backbone is idealized, the protein is deatomized and the unindexed residues are assigned their corresponding indexed residue (using a greedy algorithm that searches for the closest C-alpha in the indexed backbone).

There are two further idealization steps that are optional to users:
- If `inference.guidepost_xyz_as_design_bb == True`, then the unindexed coordinates overwrite the matched backbone. Otherwise only the sidechain (including C-beta) coordinates of the guidepost are used.
- If `inference.idealize_sidechain_outputs == True` then all atomized sidechains are idealized. This amounts to finding the set of torsions angles that minimizes the RMSD between the unidealized residue and the residue reconstructed from those torsion angles. Note: these torsions are the full RF All-Atom torsion set which includes not only torsions but also bends and twists e.g. C-Beta bend which can adopt values which would be of higher-strain than that seen in nature.

The protein at this point has sequence and structure for the motif regions but only backbone (N,Ca,C,O,C-Beta) coordinates for diffused residues (as well as any non-protein components e.g. small molecules).

#### LigandMPNN outputs
<a id="ligandmpnn_outputs"></a>

Sequence is fit using LigandMPNN in a ligand-aware, motif-rotamer-aware mode. LigandMPNN also performs packing. LigandMPNN attempts to keep the motif rotamers unchanged, however the pack uses a more conservative set of torsions than RF All-Atom (i.e. fewer DoF) to pack the rotamers and thus there is often some deviation between the RF All-Atom-idealized and ligandmpnn-idealized motif rotamers. The idealization gap between the diffusion-output rotamer set and the RF All-Atom-idealized rotamer set can be found with metrics key: `metrics.IdealizedResidueRMSD.rmsd_constellation`. The corresponding gap between the rf2aa-idealized (or not idealized if `inference.idealize_sidechain_outputs == False`) rotamer set and the ligandmpnn-idealized rotamer set can be found with metrics key: `motif_ideality_diff`.

