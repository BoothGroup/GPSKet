import os
import time
import flax
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk
import GPSKet as qk
from absl import app
from absl import flags
from absl import logging
from netket.utils.mpi import node_number as _MPI_rank
from ml_collections import config_flags, ConfigDict
from netket.vqs import MCState
from netket.optimizer import Sgd
from netket.driver import VMC
from netket.experimental.driver import VMC_SRt
from GPSKet.sampler import MetropolisHopping
from cpd_backflow.configs.common import resolve
from cpd_backflow.systems import get_system
from cpd_backflow.models import get_model
from cpd_backflow.utils import save_config, read_config, CSVLogger, Timer, save_best_params, restore_best_params
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


def serialize_VMC(driver: VMC):
    state_dict = {
        "variables": serialization.to_state_dict(driver.state.variables),
        "optimizer": serialization.to_state_dict(driver._optimizer_state),
        "step": driver._step_count,
    }
    if hasattr(driver, "preconditioner") and type(driver.preconditioner).__name__ == "SRRMSProp":
        state_dict["preconditioner"] = serialization.to_state_dict(driver.preconditioner._ema)
    return state_dict


def deserialize_VMC(driver: VMC, state_dict: dict):
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
    if hasattr(driver, "preconditioner") and type(driver.preconditioner).__name__ == "SRRMSProp":
        new_driver.preconditioner._ema = serialization.from_state_dict(
            driver.preconditioner._ema, state_dict["preconditioner"]
        )
    return new_driver

serialization.register_serialization_state(
    VMC, serialize_VMC, deserialize_VMC
)

serialization.register_serialization_state(
    VMC_SRt, serialize_VMC, deserialize_VMC
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
    ha = get_system(config, workdir)
    hi = ha.hilbert

    # Model
    ma = get_model(config, hi, ha, workdir)

    # Sampler
    sa = MetropolisHopping(hi, **config.sampler)

    # Variational state
    vs = MCState(sa, ma, **config.variational_state)

    # Optimizer and driver
    op = Sgd(learning_rate=config.optimizer.learning_rate)
    if config.optimizer_name == 'kernelSR':
        vmc = VMC_SRt(ha, op, variational_state=vs, jacobian_mode=config.optimizer.mode, diag_shift=config.optimizer.diag_shift)
    else:
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
        vmc = VMC(ha, op, variational_state=vs, preconditioner=sr)

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

    # Run training loop
    if initial_step < config.total_steps:
        if _MPI_rank == 0:
            logging.info(f"Model has {vs.n_parameters} parameters")
            logging.info('Starting training loop; initial compile can take a while...')
            timer = Timer(config.total_steps)
            t0 = time.time()
            best_params = restore_best_params(workdir)
            best_energy = best_params["Energy"] if best_params else np.inf
            best_variance = best_params["Variance"] if best_params else np.inf
        for step in range(initial_step, config.total_steps + 1):
            # Training step
            vmc.advance()
            acceptance = vmc.state.sampler_state.acceptance

            # Report compilation time
            if _MPI_rank == 0 and step == initial_step:
                logging.info(f"First step took {time.time() - t0:.1f} seconds.")

            # Update timer
            if _MPI_rank == 0:
                timer.update(step)

            # Log data
            if _MPI_rank == 0:
                logger(step, {**vmc.energy.to_dict(), "Runtime": timer.runtime})

            # Save best energy params
            if _MPI_rank == 0 and vmc.energy.mean.real < best_energy and vmc.energy.variance < best_variance:
                best_energy = vmc.energy.mean.real
                best_variance = vmc.energy.variance
                save_best_params(workdir, {"Energy": best_energy, "Variance": best_variance, "Parameters": vmc.state.parameters})
                logging.info(f"Stored best parameters at step {step} with energy {vmc.energy}")

            # Report training metrics
            if _MPI_rank == 0 and config.progress_every and step % config.progress_every == 0:
                if hasattr(vmc, "_loss_grad"):
                    grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
                    grad_norm = np.linalg.norm(grad)
                else:
                    grad_norm = np.nan
                done = step / config.total_steps
                logging.info(f"Step: {step}/{config.total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                            f"E: {vmc.energy}, "
                            f"||∇E||: {grad_norm:.4f}, "
                            f"acceptance: {acceptance*100:.2f}%, "
                            f"{timer}")

            # Store checkpoint
            if _MPI_rank == 0 and ((config.checkpoint_every and step % config.checkpoint_every == 0) or step == config.total_steps):
                checkpoint_path = save_checkpoint(checkpoints_dir, vmc, step, keep_every_n_steps=config.checkpoint_every)
                logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    # Descent finishing
    if config.get('descent_finishing', None) and initial_step < config.total_steps + config.descent_finishing.total_steps:
        # Switch to first SGD optimizer
        op = nk.optimizer.Sgd(learning_rate=config.descent_finishing.learning_rate)
        vmc.optimizer = op

        # Run training loop
        if _MPI_rank == 0:
            logging.info('Starting descent finishing loop...')
            timer = Timer(config.descent_finishing.total_steps)
            t0 = time.time()
            best_params = restore_best_params(workdir)
            best_energy = best_params["Energy"] if best_params else np.inf
            best_variance = best_params["Variance"] if best_params else np.inf
        total_steps = config.total_steps + config.descent_finishing.total_steps
        for step in range(step+1, total_steps + 1):
            # Training step
            vmc.advance()
            acceptance = vmc.state.sampler_state.acceptance

            # Report compilation time
            if _MPI_rank == 0 and step == 1:
                logging.info(f"First step took {time.time() - t0:.1f} seconds.")

            # Update timer
            if _MPI_rank == 0:
                timer.update(step)

            # Log data
            if _MPI_rank == 0:
                logger(step, {**vmc.energy.to_dict(), "Runtime": timer.runtime})

            # Save best energy params
            if _MPI_rank == 0 and vmc.energy.mean.real < best_energy and vmc.energy.variance < best_variance:
                best_energy = vmc.energy.mean.real
                best_variance = vmc.energy.variance
                save_best_params(workdir, {"Energy": best_energy, "Variance": best_variance, "Parameters": vmc.state.parameters})
                logging.info(f"Stored best parameters at step {step} with energy {vmc.energy}")

            # Report training metrics
            if _MPI_rank == 0 and config.progress_every and step % config.progress_every == 0:
                if hasattr(vmc, "_loss_grad"):
                    grad, _ = nk.jax.tree_ravel(vmc._loss_grad)
                    grad_norm = np.linalg.norm(grad)
                else:
                    grad_norm = np.nan
                done = step / total_steps
                logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                            f"E: {vmc.energy}, "
                            f"||∇E||: {grad_norm:.4f}, "
                            f"acceptance: {acceptance*100:.2f}%, "
                            f"{timer}")

            # Store checkpoint
            if _MPI_rank == 0:
                checkpoint_path = save_checkpoint(checkpoints_dir, vmc, step, keep_every_n_steps=1)
                logging.info(f"Stored checkpoint at step {step} to {checkpoint_path}")

    # Evaluate model
    if config.get('evaluate', None):
        # Restore model
        best = restore_best_params(workdir)
        if best is not None:
            best_params = best["Parameters"]
            best_params = jax.tree_map(lambda x: jnp.array(x, x.dtype), best_params)
            vs.parameters = best_params

        # Update variational state settings
        vs.n_samples = config.evaluate.n_samples
        vs.chunk_size = config.evaluate.chunk_size

        # Logger
        if _MPI_rank == 0:
            fieldnames = list(nk.stats.Stats().to_dict().keys())+["n_samples", "Runtime"]
            logger = CSVLogger(os.path.join(workdir, "evals.csv"), fieldnames)

        # Run evaluation loop
        if _MPI_rank == 0:
            logging.info('Starting evaluation loop...')
            timer = Timer(config.evaluate.total_steps)
            t0 = time.time()
        total_steps = config.evaluate.total_steps
        for step in range(1, total_steps + 1):
            # Evaluation step
            vs.reset()
            energy = vs.expect(ha)

            # Report compilation time
            if _MPI_rank == 0 and step == 1:
                logging.info(f"First step took {time.time() - t0:.1f} seconds.")

            # Update timer
            if _MPI_rank == 0:
                timer.update(step)

            # Log data
            if _MPI_rank == 0:
                logger(step, {**energy.to_dict(), "n_samples": config.evaluate.n_samples, "Runtime": timer.runtime})

            # Report evaluation metrics
            if _MPI_rank == 0:
                done = step / total_steps
                logging.info(f"Step: {step}/{total_steps} {100*done:.1f}%, "  # pylint: disable=logging-format-interpolation
                            f"E: {energy}, "
                            f"{timer}")

    return


if __name__ == "__main__":
    # Provide access to --jax_log_compiles, --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(main)
