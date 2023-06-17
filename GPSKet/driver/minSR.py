import numpy as np

import jax
import jax.numpy as jnp

from functools import partial

import netket as nk
from netket import VMC
from netket.stats import Stats
from netket.utils import mpi
from netket.stats._autocorr import integrated_time
from netket.stats.mc_stats import _split_R_hat

from GPSKet.vqs import MCStateUniqueSamples

class minSRVMC(VMC):
    """
    VMC driver utilizing the minSR updates as proposed in https://arxiv.org/abs/2302.01941
    """
    def __init__(self, *args, mode: str = None, holomorphic: bool = None,
                 solver=lambda A, b: jnp.linalg.lstsq(A, b, rcond=1.e-12)[0], diag_shift: float = 0., **kwargs):
        super().__init__(*args, **kwargs)
        assert(not (mode is not None and holomorphic is not None))
        assert (diag_shift >= 0.) and (diag_shift <= 1.)
        if mode is None:
            self.mode = nk.jax.jacobian_default_mode(self.state._apply_fun, self.state.parameters,
                                             self.state.model_state, self.state.samples,
                                             holomorphic=holomorphic)
        else:
            self.mode = mode
        self.solver = solver
        self.diag_shift = diag_shift

    # Super simple implementation of the minSR driver
    def _forward_and_backward(self):
        self.state.reset()

        if hasattr(self.state, "samples_with_counts"):
            samples, counts = self.state.samples_with_counts
        else:
            samples = self.state.samples
            counts = jnp.ones(samples.shape[:-1])/(mpi.mpi_sum_jax(np.prod(samples.shape[:-1]))[0])

        samples = samples.reshape((-1, samples.shape[-1]))
        counts = counts.reshape((-1,))

        # Transpose as local_estimators function flips the axes
        loc_ens = self.state.local_estimators(self._ham).T.reshape(-1)

        O = nk.jax.jacobian(self.state._apply_fun, self.state.parameters, samples,
                            self.state.model_state, mode = self.mode, pdf = counts, dense=True, center=True)

        self._loss_stats, self._loss_grad, dense_update = compute_update(loc_ens, O, counts, self.solver, self.diag_shift)

        # Convert back to pytree
        unravel = lambda x : x
        reassemble = lambda x: x
        x = self.state.parameters
        if self.mode != "holomorphic":
            x, reassemble = nk.jax.tree_to_real(self.state.parameters)
        _, unravel = nk.jax.tree_ravel(x)

        self._dp = reassemble(unravel(dense_update))

        # Cast to real if necessary
        self._dp = jax.tree_map(lambda x, target: (x if jnp.iscomplexobj(x) else x.real), self._dp, self.state.parameters)

        return self._dp


@partial(jax.jit, static_argnames=("solver"))
def compute_update(loc_ens, O, counts, solver, diag_shift):
    loss_stats = _statistics(loc_ens, counts)

    loc_ens_centered = (loc_ens - loss_stats.mean) * jnp.sqrt(counts)

    loc_ens_centered = (mpi.mpi_allgather_jax(loc_ens_centered)[0]).reshape(-1)

    O = (mpi.mpi_allgather_jax(O)[0]).reshape((-1, *O.shape[1:]))

    # Complex real split, is this correct? TODO: double check
    if len(O.shape) == 3:
        O = O[:,0,:] + 1.j * O[:,1,:]

    loss_grad = jnp.dot(O.T, loc_ens_centered).real

    OO = (1-diag_shift) * O.dot(O.conj().T) + diag_shift * jnp.eye(O.shape[0])

    OO_epsilon = solver(OO, loc_ens_centered)

    dense_update = O.conj().T.dot(OO_epsilon)

    return loss_stats, loss_grad, dense_update

@jax.jit
def _statistics(data, counts):
    data = jnp.atleast_1d(data)
    if data.ndim == 1:
        data = data.reshape((1, -1))

    if data.ndim > 2:
        raise NotImplementedError("Statistics are implemented only for ndim<=2")

    batch_size = mpi.mpi_sum_jax(data.shape[0])[0]

    mean = mpi.mpi_sum_jax(jnp.sum(data * counts))[0]
    var = mpi.mpi_sum_jax(jnp.sum(abs(data - mean)**2 * counts))[0]
    error_of_mean = jnp.sqrt(var / batch_size)
    
    taus = jax.vmap(integrated_time)(data)
    tau_avg, _ = mpi.mpi_mean_jax(jnp.mean(taus))
    
    R_hat = _split_R_hat(data, var)
    
    res = Stats(mean, error_of_mean, var, tau_avg, R_hat)
    
    return res
