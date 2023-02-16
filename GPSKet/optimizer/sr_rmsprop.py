import jax
import jax.numpy as jnp
import netket.jax as nkjax
from dataclasses import dataclass
from typing import Callable, Optional, Any
from netket.utils.types import PyTree, Scalar, ScalarOrSchedule
from netket.vqs import VariationalState
from netket.optimizer.preconditioner import AbstractLinearPreconditioner
from netket.optimizer.qgt import QGTJacobianDense
from .solvers import pinv


@dataclass
class SRRMSProp(AbstractLinearPreconditioner):

    def __init__(
        self,
        params: PyTree,
        qgt: QGTJacobianDense,
        solver: Callable = pinv,
        *,
        diag_shift: ScalarOrSchedule = 0.01,
        decay: Scalar = 0.9,
        eps: Scalar = 1e-8,
        initial_scale: Scalar = 0.0,
        **kwargs,
    ):
        self.qgt_constructor = qgt
        self.qgt_kwargs = kwargs
        self.diag_shift = diag_shift
        self.decay = decay
        self.eps = eps
        super().__init__(solver)

        params, _ = nkjax.tree_ravel(params)
        self._nu = jnp.full_like(params, initial_scale)
        del params

    def lhs_constructor(self, vstate: VariationalState, step: Optional[Scalar] = None):
        return self.qgt_constructor(
            vstate,
            diag_shift=None,
            diag_scale=None,
            **self.qgt_kwargs
        )

    def __call__(self, vstate: VariationalState, gradient: PyTree, step: Optional[Scalar] = None) -> PyTree:
        # Ravel gradient
        gradient, unravel_fun = nkjax.tree_ravel(gradient)

        # Compute S matrix
        self._lhs = self.lhs_constructor(vstate, step)
        S = self._lhs.to_dense()

        # Update moving average
        self._nu = self.decay*self._nu + (1-self.decay)*gradient**2

        # Compute bias correction
        t = step+1
        nu_hat = self._nu / (1-self.decay**t)

        # Compute diagonal shift and apply it to S matrix
        diag = jnp.diag(jnp.sqrt(nu_hat) + self.eps)
        S = S + self.diag_shift*diag

        # Solve system
        x0 = self.solver(S, gradient)
        self.x0 = unravel_fun(x0)

        return self.x0