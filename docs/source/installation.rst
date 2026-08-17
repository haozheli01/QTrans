==============
Installation
==============

Requirements
-------------
**QTrans** requires the following prerequisites for installation:

* `Python`_ 3.9 or newer (We recommend Python 3.9 for stability)
* Python libraries:

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
  * torch-geometric == 2.0.4
  * torch_sparse    == 0.6.13
  * torch-scatter   == 2.0.9
  * The pre-built C extension ``numpy_extension-1.0.1-cp39-cp39-linux_x86_64.whl``

Installation
-------------
There are two main methods for setting up the Python environment and installing **QTrans**:

1. Conda-based Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using provided ``environment.yaml``, one can build the environment for **QTrans** via Conda.

1.Run the following command to create the environment::

  [user]$ conda env create -f environment.yaml -n QTrans

2.After creating the environment, install **QTrans**::

  [user]$ git clone ***.git
  [user]$ cd QTrans
  [user]$ python setup.py install

3.Install the pre-built C extension::

  [user]$ pip install numpy_extension-1.0.1-cp39-cp39-linux_x86_64.whl

2. Using the Pre-built Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way to install **QTrans** is by using the pre-built Conda environment we provide:

1.Download the pre-built **QTrans** Conda environment from `<download_link>`.

2.Move it into your Conda environment folder and unpack the file ``QTrans.tar.gz``::

  [user]$ cd ~/softwares/miniconda/miniconda3/envs/ # Modify this path according to your environment
  [user]$ mkdir QTrans && cd QTrans
  [user]$ mv QTrans.tar.gz ./
  [user]$ tar xvf QTrans.tar.gz

3.Check the conda environment list, there would be an environment named **QTrans**::

  [user]$ conda env list
    
4.After unpacking the environment, install **QTrans**::

  [user]$ git clone ***.git
  [user]$ cd QTrans
  [user]$ python setup.py install

5.Install the pre-built C extension::

  [user]$ pip install numpy_extension-1.0.1-cp39-cp39-linux_x86_64.whl


.. _Python: https://www.python.org/