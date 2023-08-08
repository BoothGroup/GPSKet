import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, Optional
from netket.utils.types import Array, SeedT
from netket.operator import AbstractOperator
from netket.utils import mpi
from netket.driver.vmc_common import info
from .abstract_state_fitting import AbstractStateFittingDriver


class ARStateFitting(AbstractStateFittingDriver):
    """
    Fit an autoregressive Ansatz to data from another state by minimizing the distance between two normalized quantum states
    """

    def __init__(
        self,
        dataset: Tuple[Array, Array],
        hamiltonian: AbstractOperator,
        optimizer,
        *args,
        variational_state=None,
        mini_batch_size: int = 32,
        seed: Optional[SeedT] = None,
        **kwargs,
    ):
        super().__init__(
            dataset,
            hamiltonian,
            optimizer,
            *args,
            variational_state=variational_state,
            mini_batch_size=mini_batch_size,
            seed=seed,
            **kwargs,
        )

        if not hasattr(self._variational_state.model, "conditionals"):
            raise ValueError(f"{self._variational_state.model} is not autoregressive.")

        self._probas = np.abs(self._dataset[1]) ** 2
        self._ids = np.arange(self._size_dataset)

    def _forward_and_backward(self):
        # Sample mini-batch
        self._key, _ = jax.random.split(self._key)
        mini_batch_ids = jax.random.choice(
            self._key,
            self._ids,
            (self._mini_batch_size,),
            p=self._probas,
            replace=False,
        )
        mini_batch = (
            self._dataset[0][mini_batch_ids],
            self._dataset[1][mini_batch_ids],
        )

        # Compute loss and gradient
        self.loss, self._loss_grad = _loss_and_grad(
            self.state.parameters,
            self.state.model_state,
            self.state._apply_fun,
            mini_batch,
        )

        return self._loss_grad

    def __repr__(self):
        return (
            "ARStateFitting("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian    ", self._ham),
                ("Optimizer      ", self._optimizer),
                ("State          ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)


@partial(jax.jit, static_argnums=2)
def _loss(params, model_state, logpsi, mini_batch):
    # TODO: this might need some chunking/vmapping
    x, y = mini_batch
    model_amplitudes = logpsi({"params": params, **model_state}, x)
    loss = jnp.mean(jnp.abs(jnp.exp(model_amplitudes) - y) ** 2)
    return loss


@partial(jax.jit, static_argnums=2)
def _loss_and_grad(params, model_state, logpsi, mini_batch):
    loss, grad = jax.value_and_grad(_loss, argnums=0)(
        params, model_state, logpsi, mini_batch
    )
    loss, _ = mpi.mpi_mean_jax(loss)
    grad = jax.tree_util.tree_map(lambda p: mpi.mpi_sum_jax(p)[0], grad)
    grad = jax.tree_map(jnp.conj, grad)
    return loss, grad
