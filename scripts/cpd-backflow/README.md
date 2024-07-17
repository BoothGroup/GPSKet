# Scripts for CPD backflow
This folder contains the code to generate the data presented in our manuscript [*"Simple Fermionic backflow states via a systematically improvable tensor decomposition"*](https://arxiv.org/abs/2407.11779) by Massimo Bortone, Yannic Rath and George H. Booth.

## Organization
The code is structured as follows:
- [vmc.py](cpd_backflow/vmc.py): main entry point to the application and runs a VMC optimization as specified by a configuration file in the `configs/` folder
- [models.py](cpd_backflow/models.py): setup code for the models used in the paper
- [systems.py](cpd_backflow/systems.py): setup code for the systems studied in the paper
- [utils.py](cpd_backflow/utils.py): small collection of utility functions
- [rdms.py](cpd_backflow/rdms.py): script to evaluate the RDMs of a molecular system from an optimized model, which are for example used to compute correlation functions
- [corrfunc.py](cpd_backflow/corrfunc.py): functions used to compute the spin-spin correlations of the 6x6 hydrogen sheet

The configuration files are:
- [fermi_hubbard.py](cpd_backflow/configs/fermi_hubbard.py): 2D Fermi-Hubbard system
- [h2o.py](cpd_backflow/configs/h2o.py): ab-initio Hamiltonian in second quantization for the water molecule in the 6-31g basis
- [hsheet.py](cpd_backflow/configs/hsheet.py): ab-initio Hamiltonian in second quantization for molecular hydrogen sheet

## Installation

### Requirements
To run the code make sure the following packages are installed:
- [NetKet](https://github.com/netket/netket): quantum many-body machine learning framework written in [JAX](https://github.com/google/jax)
- [GPSKet](https://github.com/BoothGroup/GPSKet): NetKet plugin for the family of GPS models
- [pyscf](https://github.com/pyscf/pyscf): Python module for quantum chemistry
- [ml_collections](https://github.com/google/ml_collections): configuration file data structures

### Conda

We provide a `conda` environment file that installs the necessary dependencies to reproduce the calculations in the paper.
To create the environment and install the packages run the following command from the `scripts/cpd-backflow` folder:
```
conda env create -f environment.yml
```

### Manual
Install `GPSKet`
```
pip install --upgrade pip
pip install git+https://github.com/BoothGroup/GPSKet.git
```
This will also install `NetKet`, but without `MPI` support.
To install `NetKet` with `MPI` support run
```
pip install --upgrade "netket[mpi]"
```
If this fails because you do not have a working `MPI` installation, see the documentation [here](https://github.com/netket/netket?tab=readme-ov-file#installation-and-usage) for help.

Install the remaining dependencies
```
pip install pyscf==2.5.0 ml-collections
```

## Example
To start a VMC optimization on the Fermi-Hubbard model with the CPD backflow run the following command from the `scripts/cpd-backflow` folder:
```
python -m cpd_backflow.vmc --workdir=/path/to/run --config=./cpd_backflow/configs/fermi_hubbard.py --config.variational_state.n_samples=4096
```
This will load the configuration file `cpd_backflow/configs/fermi_hubbard.py` and override the default setting of the `n_samples` parameter of the variational state.
Any other parameter in the config can be changed like this.
The application will create the working directory `/path/to/run`, if it doesn't exist yet, and store the config as a YAML file.
Optimization metrics are logged as a CSV file `metrics.csv` and checkpoints are stored in the subfolder `checkpoints` of the working directory.

If the optimization run is interrupted, and you want to resume from the latest available checkpoint simply run the command:
```
python vmc.py --workdir=/path/to/run
```