import math
from ml_collections.config_dict import placeholder
from configs.common import get_config as get_base_config


def closest_divisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a, n//a

def chain(config):
    config.system.molecule = [('H', (x*config.system.distance, 0., 0.)) for x in range(config.system.n_atoms)]
    return config

def sheet(config):
    Lx, Ly = closest_divisors(config.system.n_atoms)
    molecule = []
    dist = config.system.distance
    for x in range(Lx):
        for y in range(Ly):
            molecule.append(('H', (x*dist, y*dist, 0.)))
    config.system.molecule = molecule
    return config

def get_config(geometry):
    if geometry not in ['chain', 'sheet']:
        raise ValueError(f"{geometry} is not a valid config option: choose between `chain` and `sheet`.")

    config = get_base_config()

    # System
    config.system_name = 'H'+geometry
    config.system.n_atoms = 16
    config.system.distance = 1.8
    config.system.basis_set = 'sto-6g'
    config.system.basis = 'canonical'
    config.system.symmetry = True
    config.system.unit = 'Bohr'
    config.system.molecule = placeholder(list)
    with config.ignore_type():
        if geometry == 'chain':
            config.system.set_molecule = chain
        elif geometry == 'sheet':
            config.system.set_molecule = sheet

    # Model
    config.model_name = 'ARGPS'
    config.model.M = '64' # To allow int as well as tuples, set support dimension as string first and parse it later
    config.model.dtype = 'real'
    config.model.sigma = 0.1
    config.model.symmetries = 'none'
    config.model.apply_exp = True

    # Variational state
    config.variational_state.n_samples = 4096
    config.variational_state.chunk_size = placeholder(int)
    config.variational_state.seed = placeholder(int)

    # Sampler
    config.sampler_name = 'ARDirectSampler'

    # Optimizer
    config.optimizer.learning_rate = 0.01
    config.optimizer.mode = 'real'
    config.optimizer.diag_shift = 0.01
    config.optimizer.decay = 0.9
    config.optimizer.eps = 1e-8

    return config