from ml_collections.config_dict import placeholder
from argps.configs.common import get_config as get_base_config
from argps.models import _MODELS


def get_config(model):
    if model not in _MODELS.keys():
        models = ', '.join([f"`{model}`" for model in _MODELS.keys()])
        raise ValueError(f"{model} is not a valid config option: choose between {models}.")

    config = get_base_config()

    # System
    config.system_name = 'Heisenberg2d'
    config.system.Lx = 6
    config.system.Ly = 6
    config.system.J1 = 1.0
    config.system.pbc = True
    config.system.sign_rule = True
    config.system.total_sz = 0

    # Model
    config.model_name = model
    config.model.M = '1' # To allow int as well as tuples, set support dimension as string first and parse it later
    config.model.dtype = 'real'
    config.model.sigma = 0.1
    config.model.symmetries = 'none'
    config.model.apply_exp = True

    # Variational state
    config.variational_state.n_samples = 4096
    if 'AR' not in model:
        config.variational_state.n_discard_per_chain = config.variational_state.get_ref('n_samples') // 10
    config.variational_state.chunk_size = placeholder(int)
    config.variational_state.seed = placeholder(int)

    # Sampler
    if 'AR' in model:
        config.sampler_name = 'ARDirectSampler'
    else:
        config.sampler_name = 'MetropolisExchange'
        config.sampler.n_chains = config.variational_state.get_ref('n_samples')
        config.sampler.n_sweeps = config.system.get_ref('Lx')
        config.sampler.d_max = config.system.get_ref('Lx')//2

    # Optimizer
    config.optimizer.learning_rate = 0.01
    config.optimizer.mode = 'real'
    config.optimizer.diag_shift = 0.01
    config.optimizer.decay = 0.9
    config.optimizer.eps = 1e-8

    return config