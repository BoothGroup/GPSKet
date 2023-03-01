import jax.numpy as jnp
from jax.tree_util import tree_map
from dataclasses import dataclass
from typing import Callable, Optional
from netket.utils.types import PyTree, Scalar
from netket.vqs import VariationalState
from netket.optimizer.preconditioner import AbstractLinearPreconditioner
from .solvers import pinv
from .qgt import QGTJacobianDenseRMSProp


@dataclass
class SRRMSProp(AbstractLinearPreconditioner):

    def __init__(
        self,
        params_structure: PyTree,
        qgt: Callable = QGTJacobianDenseRMSProp,
        solver: Callable = pinv,
        *,
        diag_shift: Scalar = 0.01,
        decay: Scalar = 0.9,
        eps: Scalar = 1e-8,
        initial_scale: Scalar = 0.0,
        **kwargs,
    ):
        self.qgt_constructor = qgt
        self.qgt_kwargs = kwargs
        assert (diag_shift >= 0.0) and (diag_shift <= 1.0)
        self.diag_shift = diag_shift
        self.decay = decay
        self.eps = eps
        super().__init__(solver)

        self._ema = tree_map(
            lambda p: jnp.full(p.shape, initial_scale, p.dtype),
            params_structure
        )
        del params_structure

    def lhs_constructor(self, vstate: VariationalState, ema: PyTree, step: Optional[Scalar] = None):
        return self.qgt_constructor(
            vstate,
            ema,
            diag_shift=self.diag_shift,
            eps=self.eps,
            **self.qgt_kwargs
        )

    def __call__(self, vstate: VariationalState, gradient: PyTree, step: Optional[Scalar] = None) -> PyTree:
        # Update exponential moving average
        self._ema = tree_map(
            lambda nu, g: self.decay*nu + (1-self.decay)*g**2,
            self._ema,
            gradient
        )

        # Compute bias correction
        t = step+1
        ema_hat = tree_map(
            lambda nu: nu / (1-self.decay**t),
            self._ema
        )

        # Compute S matrix
        self._lhs = self.lhs_constructor(vstate, ema_hat, step)

        # Solve system
        self.x0, self.info = self._lhs.solve(self.solver, gradient)

        return self.x0