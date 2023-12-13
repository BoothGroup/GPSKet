import warnings
import jax
import jax.numpy as jnp
import netket.jax as nkjax
from flax import struct
from typing import Optional, Union, Callable
from functools import partial
from jax.tree_util import Partial, tree_map
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.stats import subtract_mean
from netket.jax import tree_conj
from netket.optimizer import LinearOperator
from netket.optimizer.linear_operator import Uninitialized
from netket.optimizer.qgt.common import check_valid_vector_type
from netket.errors import (
    IllegalHolomorphicDeclarationForRealParametersError,
    NonHolomorphicQGTOnTheFlyDenseRepresentationError,
    HolomorphicUndeclaredWarning,
)


def mat_vec(jvp_fn, v, diag_shift, ema, eps):
    # Save linearisation work
    # TODO move to mat_vec_factory after jax v0.2.19
    vjp_fn = jax.linear_transpose(jvp_fn, v)

    w = jvp_fn(v)
    w = w * (1.0 / (w.size * mpi.n_nodes))
    w = subtract_mean(w)  # w/ MPI
    # Oᴴw = (wᴴO)ᴴ = (w* O)* since 1D arrays are not transposed
    # vjp_fn packages output into a length-1 tuple
    (res,) = tree_conj(vjp_fn(w.conjugate()))
    res = tree_map(lambda x: mpi.mpi_sum_jax(x)[0], res)

    # (1-diag_shift) * res + diag_shift * (sqrt(ema)+eps) * v
    return tree_map(
        lambda r_, e_, v_: (1 - diag_shift) * r_
        + diag_shift * (jnp.sqrt(e_) + eps) * v_,
        res,
        ema,
        v,
    )


@partial(jax.jit, static_argnums=0)
def mat_vec_factory(forward_fn, params, model_state, samples):
    # "forward function" that maps params to outputs
    def fun(W):
        return forward_fn({"params": W, **model_state}, samples)

    _, jvp_fn = jax.linearize(fun, params)
    return Partial(mat_vec, jvp_fn)


def QGTOnTheFlyRMSProp(
    vstate,
    ema,
    diag_shift=None,
    eps=None,
    chunk_size=None,
    holomorphic: Optional[bool]=None,
    **kwargs,
) -> "QGTOnTheFlyRMSPropT":
    assert diag_shift >= 0.0 and diag_shift <= 1.0

    # TODO: Find a better way to handle this case
    from netket.vqs import FullSumState

    if isinstance(vstate, FullSumState):
        raise TypeError(
            "FullSumState is not supported. Use QGTJacobianDenseRMSProp instead."
        )

    from GPSKet.vqs import MCStateUniqueSamples

    if isinstance(vstate, MCStateUniqueSamples):
        raise TypeError("Unique samples state with on-the-fly QGT is not supported.")

    if jnp.ndim(vstate.samples) == 2:
        samples = vstate.samples
    else:
        samples = vstate.samples.reshape((-1, vstate.samples.shape[-1]))

    if chunk_size is not None:
        raise ValueError("Chunking is not support yet.")
    n_samples = samples.shape[0]

    if chunk_size is None or chunk_size >= n_samples:
        mv_factory = mat_vec_factory
        chunking = False

    # check if holomorphic or not
    if holomorphic:
        if nkjax.tree_leaf_isreal(vstate.parameters):
            raise IllegalHolomorphicDeclarationForRealParametersError()
        else:
            mode = "holomorphic"
    else:
        if not nkjax.tree_leaf_iscomplex(vstate.parameters):
            mode = "real"
        else:
            if holomorphic is None:
                warnings.warn(HolomorphicUndeclaredWarning(), UserWarning)
            mode = "complex"

    mat_vec = mv_factory(
        forward_fn=vstate._apply_fun,
        params=vstate.parameters,
        model_state=vstate.model_state,
        samples=samples,
    )

    return QGTOnTheFlyRMSPropT(
        diag_shift=diag_shift,
        eps=eps,
        ema=ema,
        _mat_vec=mat_vec,
        _params=vstate.parameters,
        _chunking=chunking,
        _mode=mode
    )


@struct.dataclass
class QGTOnTheFlyRMSPropT(LinearOperator):
    diag_shift: float = Uninitialized
    eps: float = Uninitialized
    ema: PyTree = Uninitialized
    _mat_vec: Callable[[PyTree, float], PyTree] = Uninitialized
    _params: PyTree = Uninitialized
    _chunking: bool = struct.field(pytree_node=False, default=False)
    _mode: str = struct.field(pytree_node=False, default=None)

    def __matmul__(self, y):
        return onthefly_mat_treevec(self, y)

    def _solve(self, solve_fun, y: PyTree, *, x0: Optional[PyTree], **kwargs) -> PyTree:
        return _solve(self, solve_fun, y, x0=x0)

    def to_dense(self) -> jnp.ndarray:
        """
        Convert the lazy matrix representation to a dense matrix representation.

        Returns:
            A dense matrix representation of this S matrix.
        """
        if self._mode == "complex":
            raise NonHolomorphicQGTOnTheFlyDenseRepresentationError()
        return _to_dense(self)

    def __repr__(self):
        return f"QGTOnTheFlyRMSProp(diag_shift={self.diag_shift})"


########################################################################################
#####                                  QGT Logic                                   #####
########################################################################################


@jax.jit
def onthefly_mat_treevec(
    S: QGTOnTheFlyRMSPropT, vec: Union[PyTree, jnp.ndarray]
) -> Union[PyTree, jnp.ndarray]:
    """
    Perform the lazy mat-vec product, where vec is either a tree with the same structure as
    params or a ravelled vector
    """

    # if has a ndim it's an array and not a pytree
    if hasattr(vec, "ndim"):
        if not vec.ndim == 1:
            raise ValueError("Unsupported mat-vec for chunks of vectors")
        # If the input is a vector
        if not nkjax.tree_size(S._params) == vec.size:
            raise ValueError(
                """Size mismatch between number of parameters ({nkjax.tree_size(S.params)})
                                and vector size {vec.size}.
                             """
            )

        _, unravel = nkjax.tree_ravel(S._params)
        vec = unravel(vec)
        ravel_result = True
    else:
        ravel_result = False

    check_valid_vector_type(S._params, vec)

    vec = nkjax.tree_cast(vec, S._params)

    res = S._mat_vec(vec, S.diag_shift, S.ema, S.eps)

    if ravel_result:
        res, _ = nkjax.tree_ravel(res)

    return res


@jax.jit
def _solve(
    self: QGTOnTheFlyRMSPropT, solve_fun, y: PyTree, *, x0: Optional[PyTree], **kwargs
) -> PyTree:
    check_valid_vector_type(self._params, y)

    y = nkjax.tree_cast(y, self._params)

    # we could cache this...
    if x0 is None:
        x0 = jax.tree_map(jnp.zeros_like, y)

    out, info = solve_fun(self, y, x0=x0)
    return out, info


@jax.jit
def _to_dense(self: QGTOnTheFlyRMSPropT) -> jnp.ndarray:
    """
    Convert the lazy matrix representation to a dense matrix representation

    Returns:
        A dense matrix representation of this S matrix.
    """
    Npars = nkjax.tree_size(self._params)
    I = jax.numpy.eye(Npars)

    if self._chunking:
        # the linear_call in mat_vec_chunked does currently not have a jax batching rule,
        # so it cannot be vmapped but we can use scan
        # which is better for reducing the memory consumption anyway
        _, out = jax.lax.scan(lambda _, x: (None, self @ x), None, I)
    else:
        out = jax.vmap(lambda x: self @ x, in_axes=0)(I)

    if jnp.iscomplexobj(out):
        out = out.T

    return out
