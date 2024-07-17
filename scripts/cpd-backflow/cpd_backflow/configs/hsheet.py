import math
from ml_collections.config_dict import placeholder
from cpd_backflow.configs.common import get_config as get_base_config


def closest_divisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a, n//a

def sheet(config):
    Lx, Ly = closest_divisors(config.system.n_atoms)
    molecule = []
    dist = config.system.distance
    for x in range(Lx):
        for y in range(Ly):
            molecule.append(('H', (x*dist, y*dist, 0.)))
    config.system.molecule = molecule
    return config

def get_config():
    config = get_base_config()

    config.total_steps = 5000

    # System
    config.system_name = "Hsheet"
    config.system.n_atoms = 36
    config.system.distance = 1.0
    config.system.basis = "local-boys"
    config.system.basis_set = "sto-6g"
    config.system.molecule = placeholder(list)
    config.system.symmetry = True
    config.system.unit = "angstrom"
    with config.ignore_type():
        config.system.set_molecule = sheet

    # Model
    config.model.M = 1
    config.model.dtype = "real"
    config.model.init_fun = "hf"
    config.model.sigma = 0.01
    config.model.restricted = False
    config.model.fixed_magnetization = True
    config.model.exchange_cutoff = placeholder(int)

    # Variational state
    config.variational_state.n_samples = 4096
    config.variational_state.n_discard_per_chain = 128
    config.variational_state.chunk_size = placeholder(int)
    config.variational_state.seed = placeholder(int)

    # Sampler
    config.sampler.hop_probability = 1.0
    config.sampler.n_chains_per_rank = 4
    config.sampler.n_sweeps = 36

    # Optimizer
    config.optimizer_name = "kernelSR"
    config.optimizer.learning_rate = 0.01
    config.optimizer.mode = "real"
    config.optimizer.diag_shift = 0.1

    # Descent finishing
    config.descent_finishing.total_steps = 1000
    config.descent_finishing.learning_rate = 0.001

    # Evaluation
    config.evaluate.total_steps = 50
    config.evaluate.n_samples = 16384
    config.evaluate.chunk_size = 1024

    return config
