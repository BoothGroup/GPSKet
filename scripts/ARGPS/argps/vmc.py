import os
import time
import jax
import numpy as np
import netket as nk
import GPSKet as qk
from netket.utils.mpi import (
    MPI_py_comm as MPI_comm,
    node_number as MPI_rank
)
from ml_collections import config_flags, ConfigDict
from absl import app
from absl import flags
from absl import logging
from argps.configs.common import resolve
from argps.systems import get_system
from argps.models import get_model
from argps.samplers import get_sampler
from argps.utils import save_config, read_config, CSVLogger, VMCState, Timer
from flax.training.checkpoints import save_checkpoint, restore_checkpoint


_CONFIG = config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to a configuration file',
    lock_config=True)
WORKDIR = flags.DEFINE_string('workdir', None, 'Directory in which to store results of the optimization run')

flags.mark_flag_as_required('workdir')


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Set random seed
    if MPI_rank == 0:
        seed = np.random.randint(np.iinfo(np.uint32).max)
    else:
        seed = None
    seed = MPI_comm.bcast(seed, root=0)
    
    # Parse config
    workdir = WORKDIR.value
    config = _CONFIG.value
    filename = os.path.join(workdir, "config.yaml")
    if os.path.isfile(filename) and config is None:
        config = ConfigDict(read_config(workdir))
    if config.variational_state.get('seed', None) is None:
        config.variational_state.seed = seed
    config = resolve(config)

    # Print and ave config
    if MPI_rank == 0:
        logging.info(f'\n{config}')
        save_config(workdir, config)

    # System
    ha = get_system(config)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, 'graph') else None

    # Model
    ma = get_model(config, hi, g)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Variational state
    vs = nk.vqs.MCState(sa, ma, **config.variational_state)

    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
    pars_struct = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        vs.parameters
    )
    sr = qk.optimizer.SRRMSProp(
        pars_struct,
        qk.optimizer.qgt.QGTJacobianDenseRMSProp,
        solver=jax.scipy.sparse.linalg.cg,
        diag_shift=config.optimizer.diag_shift,
        decay=config.optimizer.decay,
        eps=config.optimizer.eps,
        mode=config.optimizer.mode
    )

    # Driver
    vmc = nk.driver.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Restore checkpoint
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    state = VMCState(vmc.state.parameters, (vmc._optimizer_state, vmc.preconditioner._ema), 0)
    state = restore_checkpoint(checkpoints_dir, state)
    vmc.state.parameters = state.parameters
    vmc._optimizer_state = state.opt_state[0]
    vmc.preconditioner._ema = state.opt_state[1]
    initial_step = state.step+1
    step = initial_step
    if MPI_rank == 0:
        logging.info('Will start/continue training at initial_step=%d', initial_step)

    # Logger
    if MPI_rank == 0:
        fieldnames = list(nk.stats.Stats().to_dict().keys())
        logger = CSVLogger(os.path.join(workdir, "metrics.csv"), fieldnames)

    # Run optimization loop
    if MPI_rank == 0:
        logging.info('Starting training loop; initial compile can take a while...')
        timer = Timer(config.max_steps)
        t0 = time.time()
    while step <= config.max_steps:
        # Training step
        vmc.advance()

        # Report compilation time
        if MPI_rank == 0 and step == initial_step:
            logging.info('First step took %.1f seconds.', time.time() - t0)

        # Update timer
        if MPI_rank == 0:
            timer.update(step)

        # Log data
        if MPI_rank == 0:
            logger(step, vmc.energy.to_dict())

        # Report training metrics
        if MPI_rank == 0 and config.progress_every and step % config.progress_every == 0:
            grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
            grad_norm = np.linalg.norm(grad)
            done = step / config.max_steps
            logging.info(f'Step: {step}/{config.max_steps} {100*done:.1f}%, '
                         f'E: {vmc.energy}, '
                         f'||∇E||: {grad_norm:.4f}, '
                         f'{timer}')

        # Store checkpoint
        if MPI_rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == config.max_steps):
            state = VMCState(vmc.state.parameters, (vmc._optimizer_state, vmc.preconditioner._ema), step)
            checkpoint_path = save_checkpoint(checkpoints_dir, state, step, keep_every_n_steps=config.checkpoint_every)
            logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

        step += 1

    return


if __name__ == '__main__':
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)