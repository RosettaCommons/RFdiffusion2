# Installation Guide

## Apptainer Image (Recommended)
There is an Apptainer image provided in the RFdiffusion2 repository, it is located at `RFdiffusion2/rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif`. This file can be run with either Apptainer or Singularity, if you have any issues using it please [create an issue](https://github.com/RosettaCommons/RFdiffusion2/issues). An example of how to use this image is given in the [README](readme_link.html#inference).

If you need to generate your own image, the `.spec` file used to generate the given `.sif` file can be found at `RFdiffusion2/rf_diffusion/exec/rf_diffusion_aa.spec`.

### Troubleshooting
<a id="image_troubleshooting"></a>

<details>
<summary>lz4 compression issues</summary>

Full error message you might see: 
```
FATAL: container creation failed: mount hook function failure: mount /proc/self/fd/3->/var/apptainer/mnt/session/rootfs error: while mounting image /proc/self/fd/3: squashfuse_ll exited with status 255: Squashfs image uses lz4 compression, this version supports only zlib.
```
Or you may see
```
FATAL: kernel reported a bad superblock for squashfs image partition,possible causes are that your kernel doesn't support the compression algorithm or the image is corrupted.
```

To fix this issue you can rebuild the sif on your HPC cluster: 
```
apptainer build --sandbox rfd2_sandbox /path/to/bakerlab_rf_diffusion_aa.sif
apptainer build rfd2_zlib.sif rfd2_sandbox
```
Thank you to those who posted in [Issue 10](https://github.com/RosettaCommons/RFdiffusion2/issues/10) for reporting this problem and documenting a
solution.  
</details>


## Creating Your Own Environment
You do not need to install RFdiffusion2 itself, but you do need to install several dependencies to be able to use the Python scripts that will run the inference calculations. 
This is what the Apptainer image above supplies, an environment where the dependencies required by RFdiffusion2 have already been installed. 
If this container works on your computing system, we highly recommend using it. 

However, if you need to set up your own environment, the instructions below should help you determine the dependency versions you need to get RFdiffusion2 running on your system. 

### Using Provided Environment Files
We have created a few environment files to automatically generate a conda environment that will allow RFdiffusion2 to run. 
> Note: Due to variations in GPU types and drivers, we are not able to guarantee that any of the provided environment files successfully install all the required dependencies. See the section below if none of the provided environment files are appropriate for your computing system. 

You can find the prepared environment files in the `envs` directory
- `cuda121_env.yml` - This is appropriate for systems able to run CUDA 12.1 and PyTorch 2.4.0
    - This uses requirements_cuda121.txt to install dependencies via `pip`
- `cuda124_env.yml` - This is appropriate for systems able to run CUDA 12.4 and PyTorch 2.4.0
    - This uses requirements_cuda124.txt to install dependencies via `pip`

If you have trouble with these files but they *should* work based on your system specifications here are a few things to try: 
1. Separate the creation of the environment and the installation of dependencies via pip: 
    1. Remove the last two lines from the above `.yml` files
    2. 
        ```
        conda env create -f cuda121_env.yml
        conda activate rfd2_env
        pip install -r requirements_cuda121.txt
        ```
    This will force the dependencies you want installed by CUDA to be installed before pip is used. 
2. Check to make sure the python that is being referenced is the one from your conda environment once it is activated. On clusters different modules you have imported might overrule the python in your conda environment. You can either manually give the path to your Python or change your system settings or environment variables to prefer the environment's python installation.
3. You can try to install any dependencies that pip hangs on using CUDA instead of pip.
If you have created an environment file that runs RFdiffusion for a different CUDA version or other dependency versions, create a PR to add it to the `envs` directory. 

### Creating the Environment Manually
Some of the dependencies listed below will vary based on your system, especially the version of CUDA available on your cluster. 
You will likely need to change some of the versions of the tools below to successfully install RFdiffusion2. 
The instructions below are for CUDA 12.4 and PyTorch 2.4.
For some useful troubleshooting tips, see the [Troubleshooting](#install_troubleshooting) section below. 

1. Create a conda environment using [miniforge](https://github.com/conda-forge/miniforge) and activate it
1. Point to the correct [NVIDIA-CUDA channel](https://anaconda.org/nvidia/cuda/labels),  and install [PyTorch](https://pytorch.org/), Python 3.11, and [pip](https://pip.pypa.io/en/latest/) based on what is available on your system:
    ```
    conda install --yes \
     -c nvidia/label/cuda-12.4.0 \
     -c https://conda.rosettacommons.org \
     -c pytorch \
     -c dglteam/label/th24_cu124 \
     python==3.11 \
     pip \
     numpy"<2" \
     matplotlib \
     jupyterlab \
     conda-forge::openbabel==3.1.1 \
     cuda \
     pytorch==2.4 \
     pytorch-cuda==12.4 \
     pyrosetta
    ```
    > **REMEMBER:** You will need to change your CUDA version based on what is available on your system. This will need to be changed in the 
    > NVIDIA channel, the dglteam channel, the pytorch version, and the pytorch-cuda version. 

1. Use pip to install several Python libraries: 
    ```
    pip install \
    hydra-core==1.3.1 \
    ml-collections==0.1.1 \
    addict==2.4.0 \
    assertpy==1.1.0 \
    biopython==1.83 \
    colorlog \
    compact-json \
    cython==3.0.0 \
    cytoolz==0.12.3 \
    debugpy==1.8.5 \
    deepdiff==6.3.0 \
    dm-tree==0.1.8 \
    e3nn==0.5.1 \
    einops==0.7.0 \
    executing==2.0.0 \
    fastparquet==2024.5.0 \
    fire==0.6.0 \
    GPUtil==1.4.0 \
    icecream==2.1.3 \
    ipdb==0.13.11 \
    ipykernel==6.29.5 \
    ipython==8.27.0 \
    ipywidgets \
    mdtraj==1.10.0 \
    numba \
    omegaconf==2.3.0 \
    opt_einsum==3.3.0 \
    pandas==1.5.0 \
    plotly==5.16.1 \
    pre-commit==3.7.1 \
    py3Dmol==2.2.1 \
    pyarrow==17.0.0 \
    pydantic \
    pyrsistent==0.19.3 \
    pytest-benchmark \
    pytest-cov==4.1.0 \
    pytest-dotenv==0.5.2 \
    pytest==8.2.0 \
    rdkit==2024.3.5 \
    RestrictedPython \
    ruff==0.6.2 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    submitit \
    sympy==1.13.2 \
    tmtools \
    tqdm==4.65.0 \
    typer==0.12.5 \
    wandb==0.13.10
    ```
1. Install [Biotite](https://www.biotite-python.org/latest/index.html) and several libraries related to PyTorch, and [pylibcugraphops](https://pypi.org/project/pylibcugraphops-cu12/):
    ```
    pip install biotite
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
    pip install -U -i https://pypi.anaconda.org/rapidsai-wheels-nightly/simple "pylibcugraphops-cu12>=24.6.0a24" 
    ```
    > **REMEMBER:** You will need to change the link for installing the PyTorch-related libraries (the second line in the code block above) to have it match your PyTorch and CUDA versions. 
1. Install a version of [TorchData](https://pypi.org/project/torchdata/#what-is-torchdata) that still has DataPipes:
    ```
    pip install torchdata==0.9.0
    ```
1. Install a version of the [Deep Graph Library](https://www.dgl.ai/pages/start.html) based on the version of PyTorch and CUDA you are using: 
    ```
    conda install -c dglteam/label/th24_cu124 dgl
    ```
    > **REMEMBER:** You will need to change the conda channel to the correct version of PyTorch (`th24` in the line above) and CUDA (`cu124` in the line above). Use the [Deep Graph Library's Installation guide](https://www.dgl.ai/pages/start.html) to determine the correct conda or pip command. 
1. Set your `PYTHONPATH` environment variable:
    ```
    export PYTHONPATH=$PYTHONPATH:/path/to/RFdiffusion2
    ```
    
    You can add this to your environment via
    ```
    conda env config vars set PYTHONPATH=$PYTHONPATH:/path/to/RFdiffusion2
    ```
    so that you do not need to set it every time.

.. _troubleshooting:

### Troubleshooting
<a id="install_troubleshooting"></a>
Ran into an installation issue not covered here? [Create a new issue!](https://github.com/RosettaCommons/RFdiffusion2/issues)


<details>
<summary>How to determine the highest available CUDA version on your system</summary>

The `nvidia-smi` command will print out information about the available GPUs you can access on your cluster. 
The first line in the result will look something like:
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.230.02             Driver Version: 535.230.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
```
Here, this means that this system can only support up to CUDA 12.2. However, if you look at the possible [PyTorch versions](https://pytorch.org/get-started/previous-versions/)
and [Deep Graph Library versions](https://www.dgl.ai/pages/start.html) on their installation pages, you'll notice that they don't
have versions for 12.2, so in this situation you would need to change the installation instructions to work with CUDA 12.1.
</details>

<details>
<summary>Cannot find DGL C++ graphbolt library at...</summary>

Seeing this error likely means that the version of the Deep Graph Library (DGL) that you have installed does not match
the corresponding version of PyTorch your system is finding. Double check that you installed the correct versions of 
these tools and ensure that your system does not have a different version of PyTorch it is finding. 

It might also be useful to `ls` in the given directory to see what version of the DGL libraries you have installed. 
For example, if your error says it is looking for `graphbolt/libgraphbolt_pytorch_2.4.0.so` it means your system is
using Pytorch version 2.4.0. Meanwhile if you `ls` in the directory you might see that you only have `libgraphbolt_pytorch_2.1.2.so`
meaning that the version of DGL you downloaded was only mean to work with PyTorch versions up to 2.1.2.
</details>

<details>
<summary>No module named 'torchdata.datapipes'</summary>

Newer versions of TorchData have stopped supporting their DataPipes tools. You will need to downgrade the version of TorchData
you have installed to one at or below version 0.9.0. You can learn more about this change on [TorchData's PyPI page](https://pypi.org/project/torchdata/). 
</details>


