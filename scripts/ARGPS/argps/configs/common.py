import numpy as np
from netket.utils.mpi import MPI_py_comm as MPI_comm, node_number as MPI_rank
from ml_collections import ConfigDict


def get_config() -> ConfigDict:
    """Returns base config values other than system, model, sampler, variational state and optimizer parameters."""

    config = ConfigDict()

    # Maximum number of optimization steps
    config.max_steps = 100
    # How often to report progress
    config.progress_every = 10
    # How often to write checkpoints
    config.checkpoint_every = 10

    # Will be set later
    config.system_name = None
    config.system = ConfigDict()
    config.model_name = None
    config.model = ConfigDict()
    config.sampler_name = None
    config.sampler = ConfigDict()
    config.variational_state_name = "MCState"
    config.variational_state = ConfigDict()
    config.optimizer_name = "SRRMSProp"
    config.optimizer = ConfigDict()

    return config


def resolve(config: ConfigDict) -> ConfigDict:
    # Set random seed
    if MPI_rank == 0:
        seed = np.random.randint(np.iinfo(np.uint32).max)
    else:
        seed = None
    seed = MPI_comm.bcast(seed, root=0)
    if (
        config.variational_state_name != "FullSumState"
        and config.variational_state.get("seed", None) is None
    ):
        config.variational_state.seed = seed

    # Resolve molecular configuration
    if "set_molecule" in config.system and callable(config.system.set_molecule):
        config = config.system.set_molecule(config)
        with config.ignore_type():
            # Replace the function with its name so we know how the molecule was set
            # This makes the ConfigDict object serialisable.
            if callable(config.system.set_molecule):
                config.system.set_molecule = config.system.set_molecule.__name__

    # Support dimension can be int or tuple
    if isinstance(config.model.M, str):
        M = config.model.M.split(",")
        if len(M) > 1:
            M = tuple(map(int, M))
        else:
            M = int(M[0])
        with config.ignore_type():
            config.model.M = M

    config = config.copy_and_resolve_references()
    return config.lock()
