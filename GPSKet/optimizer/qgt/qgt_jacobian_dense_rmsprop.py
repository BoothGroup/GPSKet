import jax
import jax.numpy as jnp
import netket.jax as nkjax
from flax import struct
from typing import Optional, Union
from netket.utils import mpi
from netket.nn import split_array_mpi
from netket.utils.types import PyTree
from netket.optimizer import LinearOperator
from netket.optimizer.linear_operator import Uninitialized
from netket.optimizer.qgt.common import check_valid_vector_type
from netket.optimizer.qgt.qgt_jacobian_common import choose_jacobian_mode
from netket.optimizer.qgt.qgt_jacobian_dense_logic import vec_to_real, mat_vec


def QGTJacobianDenseRMSProp(
    vstate,
    ema,
    mode: str = None,
    holomorphic: bool = None,
    diag_shift=None,
    eps=None,
    chunk_size=None,
    **kwargs,
) -> "QGTJacobianDenseRMSPropT":
    if mode is not None and holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")

    assert diag_shift >= 0.0 and diag_shift <= 1.0

    # TODO: Find a better way to handle this case
    from netket.vqs import ExactState

    if isinstance(vstate, ExactState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    else:
        samples = vstate.samples
        pdf = None

    if mode is None:
        mode = choose_jacobian_mode(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            samples,
            mode=mode,
            holomorphic=holomorphic,
        )

    if chunk_size is None and hasattr(vstate, "chunk_size"):
        chunk_size = vstate.chunk_size

    jacobians = nkjax.jacobian(
        vstate._apply_fun,
        vstate.parameters,
        samples.reshape(-1, samples.shape[-1]),
        vstate.model_state,
        mode=mode,
        pdf=pdf,
        chunk_size=chunk_size,
        dense=True,
        center=True,
    )

    pars_struct = jax.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), vstate.parameters
    )
    return QGTJacobianDenseRMSPropT(
        O=jacobians,
        diag_shift=diag_shift,
        eps=eps,
        ema=ema,
        mode=mode,
        _params_structure=pars_struct,
        **kwargs,
    )

@struct.dataclass
class QGTJacobianDenseRMSPropT(LinearOperator):
    O: jnp.ndarray = Uninitialized
    diag_shift: float = Uninitialized
    eps: float = Uninitialized
    ema: PyTree = Uninitialized
    mode: str = struct.field(pytree_node=False, default=Uninitialized)
    _in_solve: bool = struct.field(pytree_node=False, default=False)
    _params_structure: PyTree = struct.field(pytree_node=False, default=Uninitialized)

    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        return _matmul(self, vec)

    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
        return _solve(self, solve_fun, y, x0=x0)

    def to_dense(self) -> jnp.ndarray:
        return _to_dense(self)

    def __repr__(self):
        return (
            f"QGTJacobianDenseRMSProp(diag_shift={self.diag_shift}, mode={self.mode})"
        )
    
########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################


@jax.jit
def _matmul(
    self: QGTJacobianDenseRMSPropT, vec: Union[PyTree, jnp.ndarray]
) -> Union[PyTree, jnp.ndarray]:

    unravel = None
    if not hasattr(vec, "ndim") and not self._in_solve:
        check_valid_vector_type(self._params_structure, vec)
        vec, unravel = nkjax.tree_ravel(vec)

    # Real-imaginary split RHS in R→R and R→C modes
    reassemble = None
    if self.mode != "holomorphic" and not self._in_solve:
        vec, reassemble = vec_to_real(vec)

    result = mat_vec(vec, self.O, self.diag_shift)

    if reassemble is not None:
        result = reassemble(result)

    if unravel is not None:
        result = unravel(result)

    return result


@jax.jit
def _solve(
    self: QGTJacobianDenseRMSPropT, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None
) -> PyTree:
    if not hasattr(y, "ndim"):
        check_valid_vector_type(self._params_structure, y)

    # Ravel input PyTrees, record unravelling function too
    y, unravel = nkjax.tree_ravel(y)

    if self.mode != "holomorphic":
        y, reassemble = vec_to_real(y)

    if x0 is not None:
        x0, _ = nkjax.tree_ravel(x0)
        if self.mode != "holomorphic":
            x0, _ = vec_to_real(x0)

    insolve_self = self.replace(_in_solve=True)

    out, info = solve_fun(insolve_self, y, x0=x0)

    if self.mode != "holomorphic":
        out = reassemble(out)

    return unravel(out), info


@jax.jit
def _to_dense(self: QGTJacobianDenseRMSPropT) -> jnp.ndarray:
    # Concatenate samples with real/imaginary dimension
    O = self.O
    O = O.reshape(-1, O.shape[-1])

    # Compute S matrix
    S = mpi.mpi_sum_jax(O.conj().T @ O)[0]

    # Compute diagonal shift and apply it to S matrix
    ema, _ = nkjax.tree_ravel(self.ema)
    diag = jnp.diag(jnp.sqrt(ema) + self.eps)
    S = (1-self.diag_shift)*S + self.diag_shift * diag
    return S