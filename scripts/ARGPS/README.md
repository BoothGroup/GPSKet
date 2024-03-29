# Scripts for AutoRegressive GPS (ARGPS)
This folder contains the code to generate the data presented in our upcoming manuscript [*"Impact of conditional modelling for a universal autoregressive quantum state"*](https://arxiv.org/abs/2306.05917) by Massimo Bortone, Yannic Rath and George H. Booth.

## Organization
The code is structured as follows:
- [vmc.py](argps/vmc.py): main entry point to the application and runs a VMC optimization as specified by a configuration file in the `configs/` folder
- [models.py](argps/models.py): setup code for the models used in the paper
- [systems.py](argps/systems.py): setup code for the systems studied in the paper
- [sampler.py](argps/samplers.py): setup code for the samplers used in the VMC algorithm
- [utils.py](argps/utils.py): small collection of utility functions

The configuration files are:
- [heisenberg2d.py](argps/configs/heisenberg2d.py): configurations for VMC on a 2D Heisenberg system; accepts a valid model name as option (`GPS`, `ARGPS`, `MaskedGPS`, `ARFilterGPS`, `MaskedFilterGPS`; default: `ARGPS`)
- [hubbard1d.py](argps/configs/hubbard1d.py): configurations for VMC on a 1D Hubbard system in second quantization with `ARFilterGPS`
- [hydrogen.py](argps/configs/hydrogen.py): configurations for VMC on an ab-initio Hamiltonian in second quantization for molecular hydrogen; accepts geometry (`chain` or `sheet`), basis (`canonical` or `local`) and dtype (`real` or `complex`) as options (default: `chain,canonical,real`).

## Installation

### Requirements
To run the code make sure the following packages are installed:
- [NetKet](https://github.com/netket/netket): quantum many-body machine learning framework written in [JAX](https://github.com/google/jax)
- [GPSKet](https://github.com/BoothGroup/GPSKet): NetKet plugin for the family of GPS models
- [pyscf](https://github.com/pyscf/pyscf): Python module for quantum chemistry
- [ml_collections](https://github.com/google/ml_collections): configuration file data structures

### Conda

We provide a `conda` environment file that installs the necessary dependencies to reproduce the calculations in the paper.
To create the environment and install the packages run the following command from the `scripts/ARGPS` folder:
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
pip install pyscf ml-collections
```

## Example
To start a VMC optimization on the Heisenberg lattice with the `ARGPS` model for 100 steps, for example, run the following command from the `scripts/ARGPS` folder:
```
python -m argps.vmc --workdir=/path/to/run --config=./argps/configs/heisenberg2d.py:ARGPS --config.max_steps=100
```
This will load the configuration file `argps/configs/heisenberg2d.py`, set the model to `ARGPS` and override the default setting of the `max_steps` parameter.
Any other parameter in the config can be overridden like this.
The application will create the working directory `/path/to/run`, if it doesn't exist yet, and store the config as a YAML file.
Optimization metrics are logged as a CSV file `metrics.csv` and checkpoints are stored in the subfolder `checkpoints` of the working directory.

If the optimization run is interrupted, and you want to resume from the latest available checkpoint simply run the command:
```
python vmc.py --workdir=/path/to/run
```