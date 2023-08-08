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
class SRDense(AbstractLinearPreconditioner):
    def __init__(self, qgt: QGTJacobianDense, solver: Callable = pinv):
        self.qgt_constructor = qgt
        super().__init__(solver)

    def lhs_constructor(self, vstate: VariationalState, step: Optional[Scalar] = None):
        return self.qgt_constructor(vstate)

    def __call__(
        self, vstate: VariationalState, gradient: PyTree, step: Optional[Scalar] = None
    ) -> PyTree:
        # Ravel gradient
        gradient, unravel_fun = nkjax.tree_ravel(gradient)

        # Compute S matrix
        self._lhs = self.lhs_constructor(vstate, step)
        S = self._lhs.to_dense()

        # Solve system
        x0 = self.solver(S, gradient)
        self.x0 = unravel_fun(x0)

        return self.x0
