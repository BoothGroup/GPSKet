import os
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
from absl import app
from absl import flags
from absl import logging
from netket.utils.mpi import node_number as _MPI_rank
from netket.utils.mpi import n_nodes as _MPI_n_nodes
from ml_collections import config_flags, ConfigDict
from netket.vqs import MCState
from GPSKet.sampler import MetropolisHopping
from cpd_backflow.configs.common import resolve
from cpd_backflow.systems import get_system
from cpd_backflow.models import get_model
from cpd_backflow.utils import save_config, read_config, restore_best_params
from GPSKet.operator.hamiltonian.ab_initio import local_en_on_the_fly


_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "File path to a configuration file", lock_config=True
)
_WORKDIR = flags.DEFINE_string(
    "workdir",
    None,
    "Directory in which results of the optimization run are stored",
    required=True,
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Parse flags
    workdir = _WORKDIR.value
    config = _CONFIG.value
    filename = os.path.join(workdir, "config.yaml")
    if os.path.isfile(filename) and config is None:
        config = ConfigDict(read_config(workdir))
    config = resolve(config)

    # Print and save config
    if _MPI_rank == 0:
        logging.info(f"\n{config}")
        save_config(workdir, config)

    # System
    ha = get_system(config, workdir=workdir)
    hi = ha.hilbert
    n_elec = hi._n_elec
    norb = ha.hilbert.size

    # Model
    ma = get_model(config, hi, hamiltonian=ha, workdir=workdir)

    # Sampler
    sa = MetropolisHopping(hi, **config.sampler)

    # Variational state
    vs = MCState(sa, ma, **config.variational_state)

    # Load best parameters
    best_params = restore_best_params(workdir)['Parameters']
    best_params = jax.tree_util.tree_map(lambda x: jnp.array(x, x.dtype), best_params)
    vs.parameters = best_params

    # Compute RDMs
    try:
        use_fast_update = vs.model.apply_fast_update
    except:
        use_fast_update = False
    if _MPI_rank == 0:
        logging.info(f"Computing RDMs: fast update {'ON' if use_fast_update else 'OFF'}")
    n_samples = config.evaluate.n_samples
    if n_samples % _MPI_n_nodes != 0:
        raise ValueError("The number samples must be divisible by the number of MPI ranks")
    n_samples_per_rank = config.evaluate.n_samples // _MPI_n_nodes
    chunk_size = config.evaluate.chunk_size
    if n_samples_per_rank % chunk_size != 0:
        raise ValueError("The number of samples per rank must be divisible by the chunk size")
    vs.n_samples = chunk_size
    n_chunks = n_samples_per_rank // chunk_size
    rdm1 = jnp.zeros((norb, norb), dtype=jnp.float32)
    rdm2 = jnp.zeros((norb, norb, norb, norb), dtype=jnp.float32)
    for i in range(1, n_chunks+1):
        # Sample state
        vs.reset()
        samples = vs.sample()
        samples = samples.reshape((-1, norb))

        # Evaluate RDMs
        local_en, local_rdm1, local_rdm2 = local_en_on_the_fly(
            n_elec,
            vs._apply_fun,
            vs.variables,
            samples,
            (jnp.array(ha.t_mat), jnp.array(ha.eri_mat)),
            use_fast_update=use_fast_update,
            # chunk_size=config.evaluate.chunk_size,
            return_local_RDMs=True
        )
        stats = nk.stats.statistics(local_en)
        acceptance = vs.sampler_state.acceptance
        if _MPI_rank == 0:
            logging.info(f"[Chunk: {i}/{n_chunks}] E: {stats}, acceptance: {acceptance*100:.2f}%")
        rdm1 += nk.utils.mpi.mpi_sum_jax(jnp.sum(local_rdm1, axis=0))[0]
        rdm2 += nk.utils.mpi.mpi_sum_jax(jnp.sum(local_rdm2, axis=0))[0]
    rdm1 = np.array(rdm1) / n_samples
    rdm2 = np.array(rdm2) / n_samples

    # Re-order into (p^+ r^+ s q) form
    rdm2 *= 2
    for k in range(norb):
        rdm2[:, k, k, :] -= rdm1.T

    # Save result
    if _MPI_rank == 0:
        np.save(os.path.join(workdir, "rdm_1.npy"), rdm1)
        np.save(os.path.join(workdir, "rdm_2.npy"), rdm2)
        logging.info(f"Saved RDMs at {workdir}")


if __name__ == '__main__':
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)