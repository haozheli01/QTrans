==============
Installation
==============

Requirements
-------------
**QTrans** requires Python 3.9 or later. Python 3.9 is recommended for optimal compatibility and stability.

The following Python packages are required:

* Core dependencies:

  * pymatgen     == 2023.5.10
  * phonopy      == 2.19.1
  * mpi4py       == 3.1.4
  * opt_einsum   == 3.4.0
  * yaml         == 0.2.5
  * pytorch      == 1.11.0
  * numpy        == 1.24.3
  * scipy        == 1.10.1
  * tqdm         == 4.65.0
  * easydict     == 1.9
  * setuptools   == 77

* Additional dependencies for HamGNN integration:

  * torch-geometric == 2.0.4
  * torch_sparse    == 0.6.13
  * torch-scatter   == 2.0.9

* Compiler requirements:

  * A C compiler with OpenMP support is required for compilation.

Installation
-------------
There are two ways to set up the Python environment and install **QTrans**:

1. Conda-based Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This method uses the provided environment.yaml file to create a Conda environment with the required dependencies.

**Step 1.Create the Conda environment** 

Run the following command::

  conda env create -f environment.yml -n QTrans

**Step 2.Activate the environment**::

  conda activate QTrans

**Step 3. Download and install** **QTrans**:

Clone the **QTrans** repository and install the package in editable mode::

  git clone ***.git
  cd QTrans
  python -m pip install -e .

2. Pre-built Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For convenience, we also provide a pre-built Conda environment, 
allowing users to set up **QTrans** without manually resolving its dependencies.

**Step 1.Download the pre-built environment** 
Download the pre-built **QTrans** environment archive from the following link:  `<download_link>`.

**Step 2. Extract the environment archive** 
Move ``QTrans.tar.gz`` to your Conda environments directory and extract it::

  cd ~/softwares/miniconda/miniconda3/envs/ # Modify this path according to your environment
  mkdir QTrans && cd QTrans
  mv QTrans.tar.gz ./
  tar xvf QTrans.tar.gz

**Step 3. Verify the environment** 

Check whether the **QTrans** environment appears in the Conda environment list::

  conda env list

**Step 4. Download and install** **QTrans**

Clone the **QTrans** repository and install the package in editable mode::

  git clone ***.git
  cd QTrans
  python -m pip install -e .