import os
import time
import flax
import jax
import numpy as np
import netket as nk
import GPSKet as qk
from absl import app
from absl import flags
from absl import logging
from netket.utils.mpi import node_number as _MPI_rank
from ml_collections import config_flags, ConfigDict
from argps.configs.common import resolve
from argps.systems import get_system
from argps.models import get_model
from argps.samplers import get_sampler
from argps.utils import save_config, read_config, CSVLogger, Timer
from flax import serialization
from flax.training.checkpoints import save_checkpoint, restore_checkpoint


flax.config.update('flax_use_orbax_checkpointing', False)

_CONFIG = config_flags.DEFINE_config_file(
    "config", None, "File path to a configuration file", lock_config=True
)
_WORKDIR = flags.DEFINE_string(
    "workdir",
    None,
    "Directory in which to store results of the optimization run",
    required=True,
)


def serialize_VMC(driver: nk.driver.VMC):
    state_dict = {
        "variables": serialization.to_state_dict(driver.state.variables),
        "optimizer": serialization.to_state_dict(driver._optimizer_state),
        "preconditioner": serialization.to_state_dict(driver.preconditioner._ema),
        "step": driver._step_count,
    }
    return state_dict


def deserialize_VMC(driver: nk.driver.VMC, state_dict: dict):
    import copy

    new_driver = copy.copy(driver)
    new_driver.state.variables = serialization.from_state_dict(
        driver.state.variables, state_dict["variables"]
    )
    new_driver._optimizer_state = serialization.from_state_dict(
        driver._optimizer_state, state_dict["optimizer"]
    )
    new_driver._step_count = serialization.from_state_dict(
        driver._step_count, state_dict["step"]
    )
    new_driver.preconditioner._ema = serialization.from_state_dict(
        driver.preconditioner._ema, state_dict["preconditioner"]
    )
    return new_driver


serialization.register_serialization_state(
    nk.driver.VMC, serialize_VMC, deserialize_VMC
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Parse config
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
    ha = get_system(config)
    hi = ha.hilbert
    g = ha.graph if hasattr(ha, "graph") else None

    # Model
    ma = get_model(config, hi, g)

    # Sampler
    sa = get_sampler(config, hi, g)

    # Variational state
    vs = nk.vqs.MCState(sa, ma, **config.variational_state)

    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=config.optimizer.learning_rate)
    pars_struct = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vs.parameters
    )
    sr = qk.optimizer.SRRMSProp(
        pars_struct,
        qk.optimizer.qgt.QGTJacobianDenseRMSProp,
        solver=jax.scipy.sparse.linalg.cg,
        diag_shift=config.optimizer.diag_shift,
        decay=config.optimizer.decay,
        eps=config.optimizer.eps,
        mode=config.optimizer.mode,
    )

    # Driver
    vmc = nk.driver.VMC(ha, op, variational_state=vs, preconditioner=sr)

    # Restore checkpoint
    checkpoints_dir = os.path.join(workdir, "checkpoints")
    vmc = restore_checkpoint(checkpoints_dir, vmc)
    initial_step = vmc.step_count + 1
    step = initial_step
    if _MPI_rank == 0:
        logging.info(f"Will start/continue training at initial_step={initial_step}")

    # Logger
    if _MPI_rank == 0:
        fieldnames = list(nk.stats.Stats().to_dict().keys()) + ["Runtime"]
        logger = CSVLogger(os.path.join(workdir, "metrics.csv"), fieldnames)

    # Run optimization loop
    if _MPI_rank == 0:
        logging.info(f"Model has {vs.n_parameters} parameters")
        logging.info("Starting training loop; initial compile can take a while...")
        timer = Timer(config.max_steps)
        t0 = time.time()
    while step <= config.max_steps:
        # Training step
        vmc.advance()

        # Report compilation time
        if _MPI_rank == 0 and step == initial_step:
            logging.info(f"First step took {time.time() - t0:.1f} seconds.")

        # Update timer
        if _MPI_rank == 0:
            timer.update(step)

        # Log data
        if _MPI_rank == 0:
            logger(step, {**vmc.energy.to_dict(), "Runtime": timer.runtime})

        # Report training metrics
        if (
            _MPI_rank == 0
            and config.progress_every
            and step % config.progress_every == 0
        ):
            grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
            grad_norm = np.linalg.norm(grad)
            done = step / config.max_steps
            logging.info(
                f"Step: {step}/{config.max_steps} {100*done:.1f}%, "
                f"E: {vmc.energy}, "
                f"||∇E||: {grad_norm:.4f}, "
                f"{timer}"
            )

        # Store checkpoint
        if _MPI_rank == 0 and (
            (config.checkpoint_every and step % config.checkpoint_every == 0)
            or step == config.max_steps
        ):
            checkpoint_path = save_checkpoint(
                checkpoints_dir, vmc, step, keep_every_n_steps=config.checkpoint_every
            )
            logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

        step += 1

    return


if __name__ == "__main__":
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)
