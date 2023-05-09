import jax
import jax.numpy as jnp
import netket.jax as nkjax
from flax import struct
from typing import Optional, Union
from netket.utils import mpi
from netket.nn import split_array_mpi
from netket.utils.types import PyTree, Scalar
from netket.optimizer import LinearOperator
from netket.optimizer.linear_operator import Uninitialized
from netket.optimizer.qgt.common import check_valid_vector_type

from GPSKet.vqs import MCStateUniqueSamples

from functools import partial

def QGTJacobianDenseRMSProp(
    vstate=None,
    ema=None,
    mode: str = None,
    holomorphic: bool = None,
    diag_shift=None,
    eps=None,
    chunk_size=None,
    **kwargs,
) -> "QGTJacobianDenseRMSPropT":
    if mode is not None and holomorphic is not None:
        raise ValueError("Cannot specify both `mode` and `holomorphic`.")

    if vstate is None:
        return partial(QGTJacobianDenseRMSProp, mode=mode, holomorphic=holomorphic,
                       diag_shift=diag_shift, eps=eps, chunk_size=chunk_size, **kwargs)


    assert diag_shift >= 0.0 and diag_shift <= 1.0

    # TODO: Find a better way to handle this case
    from netket.vqs import ExactState

    if isinstance(vstate, ExactState):
        samples = split_array_mpi(vstate._all_states)
        pdf = split_array_mpi(vstate.probability_distribution())
    elif isinstance(vstate, MCStateUniqueSamples):
        samples, pdf = vstate.samples_with_counts
    else:
        samples = vstate.samples
        pdf = None

    if mode is None:
        mode = nkjax.jacobian_default_mode(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            samples,
            mode=mode,
            holomorphic=holomorphic,
        )

    if mode == "holomorphic":
        raise ValueError("Mode cannot be holomorphic for the QGT with RMSProp diagonal shift")

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

    @jax.jit
    def __matmul__(self, vec: Union[PyTree, jnp.ndarray]) -> Union[PyTree, jnp.ndarray]:
        if not hasattr(vec, "ndim") and not self._in_solve:
            check_valid_vector_type(self._params_structure, vec)

        vec, reassemble = convert_tree_to_dense_format(
            vec, self.mode, disable=self._in_solve
        )

        ema, _ = convert_tree_to_dense_format(self.ema, self.mode)
        result = mat_vec(vec, self.O, self.diag_shift, ema, self.eps)

        return reassemble(result)

    @jax.jit
    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree] = None) -> PyTree:
        if not hasattr(y, "ndim"):
            check_valid_vector_type(self._params_structure, y)

        y, reassemble = convert_tree_to_dense_format(y, self.mode)

        if x0 is not None:
            x0, _ = convert_tree_to_dense_format(x0, self.mode)

        insolve_self = self.replace(_in_solve=True)
        out, info = solve_fun(insolve_self, y, x0=x0)

        return reassemble(out), info

    @jax.jit
    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        # Concatenate samples with real/imaginary dimension
        O = self.O
        O = O.reshape(-1, O.shape[-1])

        # Compute S matrix
        S = mpi.mpi_sum_jax(O.conj().T @ O)[0]

        # Compute diagonal shift and apply it to S matrix
        ema, _ = convert_tree_to_dense_format(self.ema, self.mode)
        diag = jnp.diag(jnp.sqrt(ema) + self.eps)
        return (1-self.diag_shift)*S + self.diag_shift * diag

    def __repr__(self):
        return (
            f"QGTJacobianDenseRMSProp(diag_shift={self.diag_shift}, mode={self.mode})"
        )

########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################


def mat_vec(v: PyTree, O: PyTree, diag_shift: Scalar, ema: PyTree, eps: Scalar) -> PyTree:
    w = O @ v
    res = jnp.tensordot(w.conj(), O, axes=w.ndim).conj()
    res = mpi.mpi_sum_jax(res)[0]
    return (1-diag_shift) * res + diag_shift * (jnp.sqrt(ema) + eps) * v

def convert_tree_to_dense_format(vec, mode, *, disable=False):
    """
    Converts an arbitrary PyTree/vector which might be real/complex
    to the dense-(maybe-real)-vector used for QGTJacobian.

    The format is dictated by the sequence of operations chosen by
    `nk.jax.jacobian(..., dense=True)`. As `nk.jax.jacobian` first
    converts the pytree of parameters to real and then concatenates
    real and imaginary terms with a tree_ravel, we must do the same
    in here.
    """
    unravel = lambda x: x
    reassemble = lambda x: x
    if not disable:
        if mode != "holomorphic":
            vec, reassemble = nkjax.tree_to_real(vec)
        if not hasattr(vec, "ndim"):
            vec, unravel = nkjax.tree_ravel(vec)

    return vec, lambda x: reassemble(unravel(x))