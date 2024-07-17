from ml_collections.config_dict import placeholder
from cpd_backflow.configs.common import get_config as get_base_config


def get_config():
    config = get_base_config()

    config.total_steps = 2000

    # System
    config.system_name = "FermiHubbard"
    config.system.Lx = 4
    config.system.Ly = 4
    config.system.t = 1.0
    config.system.U = 8.0
    config.system.pbc = "PBC-PBC"
    config.system.n_elec = (7, 7)

    # Model
    config.model.M = 1
    config.model.dtype = "real"
    config.model.init_fun = "hf"
    config.model.sigma = 0.01
    config.model.restricted = False
    config.model.fixed_magnetization = False

    # Variational state
    config.variational_state.n_samples = 4096
    config.variational_state.n_discard_per_chain = 128
    config.variational_state.chunk_size = placeholder(int)
    config.variational_state.seed = placeholder(int)

    # Sampler
    config.sampler.hop_probability = 1.0
    config.sampler.n_chains_per_rank = 4
    config.sampler.n_sweeps = 16

    # Optimizer
    config.optimizer.learning_rate = 0.01
    config.optimizer.mode = "real"
    config.optimizer.diag_shift = 0.1
    config.optimizer.decay = 0.9
    config.optimizer.eps = 1e-8

    # Descent finishing
    config.descent_finishing.total_steps = 400
    config.descent_finishing.learning_rate = 0.001

    # Evaluation
    config.evaluate.total_steps = 50
    config.evaluate.n_samples = 65536
    config.evaluate.chunk_size = placeholder(int)

    return config
