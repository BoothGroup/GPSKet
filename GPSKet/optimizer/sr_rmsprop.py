import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from dataclasses import dataclass
from typing import Callable, Optional
from netket.jax.utils import is_scalar
from netket.utils.types import PyTree, Scalar, ScalarOrSchedule
from netket.vqs import VariationalState
from netket.optimizer.preconditioner import AbstractLinearPreconditioner
from .qgt import QGTJacobianDenseRMSProp


@dataclass
class SRRMSProp(AbstractLinearPreconditioner):
    def __init__(
        self,
        params_structure: PyTree,
        qgt: Callable = QGTJacobianDenseRMSProp,
        solver: Callable = jax.scipy.sparse.linalg.cg,
        *,
        diag_shift: ScalarOrSchedule = 0.01,
        decay: Scalar = 0.9,
        eps: Scalar = 1e-8,
        initial_scale: Scalar = 0.0,
        **kwargs,
    ):
        self.qgt_constructor = qgt
        self.qgt_kwargs = kwargs
        if is_scalar(diag_shift):
            assert (diag_shift >= 0.0) and (diag_shift <= 1.0)
        self.diag_shift = diag_shift
        self.decay = decay
        self.eps = eps
        super().__init__(solver)

        self._ema = tree_map(
            lambda p: jnp.full(p.shape, initial_scale, p.dtype), params_structure
        )
        del params_structure

    def lhs_constructor(
        self, vstate: VariationalState, ema: PyTree, step: Optional[Scalar] = None
    ):
        diag_shift = self.diag_shift
        if callable(self.diag_shift):
            if step is None:
                raise TypeError(
                    "If you use a scheduled `diag_shift`, you must call "
                    "the precoditioner with an extra argument `step`."
                )
            diag_shift = diag_shift(step)
            assert (diag_shift >= 0.0) and (diag_shift <= 1.0)
        return self.qgt_constructor(
            vstate, ema, diag_shift=diag_shift, eps=self.eps, **self.qgt_kwargs
        )

    def __call__(
        self, vstate: VariationalState, gradient: PyTree, step: Optional[Scalar] = None
    ) -> PyTree:
        # Update exponential moving average
        def update_ema(nu, g):
            if jnp.iscomplexobj(g):
                # This assumes that the parameters are split into complex and real parts later on (done in the QGT implementation)
                squared_g = g.real**2 + 1.0j * g.imag**2
            else:
                squared_g = g**2
            return self.decay * nu + (1 - self.decay) * squared_g

        self._ema = tree_map(update_ema, self._ema, gradient)

        # Compute bias correction
        t = step + 1
        ema_hat = tree_map(lambda nu: nu / (1 - self.decay**t), self._ema)

        # Compute S matrix
        self._lhs = self.lhs_constructor(vstate, ema_hat, step)

        # Solve system
        self.x0, self.info = self._lhs.solve(self.solver, gradient)

        return self.x0
