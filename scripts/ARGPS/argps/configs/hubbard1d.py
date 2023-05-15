from ml_collections.config_dict import placeholder
from argps.configs.common import get_config as get_base_config


def get_config():
    config = get_base_config()

    # System
    config.system_name = 'Hubbard1d'
    config.system.Lx = 32
    config.system.t = 1.0
    config.system.U = 0.0
    config.system.pbc = True

    # Model
    config.model_name = 'ARFilterGPS'
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