import numpy as np

import jax
import jax.numpy as jnp

from functools import partial

import netket as nk
from netket import VMC
from netket.stats import Stats
from netket.utils import mpi

from GPSKet.vqs import MCStateUniqueSamples

from netket.optimizer.qgt.qgt_jacobian_common import choose_jacobian_mode

class minSRVMC(VMC):
    """
    VMC driver utilizing the minSR updates as proposed in https://arxiv.org/abs/2302.01941
    """
    def __init__(self, *args, mode: str = None, holomorphic: bool = None, minSR_solver=lambda x : jnp.linalg.pinv(x), **kwargs):
        super().__init__(*args, **kwargs)
        assert(not (mode is not None and holomorphic is not None))
        if mode is None:
            self.mode = choose_jacobian_mode(self.state._apply_fun, self.state.parameters,
                                             self.state.state, self.state.samples, mode=mode,
                                             holomorphic=holomorphic)
        else:
            self.mode = mode
        self.solver = minSR_solver

    # Super simple implementation of the minSR driver
    def _forward_and_backward(self):
        self.state.reset()

        if isinstance(self.state, MCStateUniqueSamples):
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


        self._loss_stats, dense_update = compute_update(loc_ens, O, counts, self.solver)

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
def compute_update(loc_ens, O, counts, solver):
    mean = mpi.mpi_sum_jax(jnp.sum(loc_ens * counts))[0]
    var = mpi.mpi_sum_jax(jnp.sum(abs(loc_ens - mean)**2 * counts))[0]

    loss_stats = Stats(mean, np.nan, var, np.nan, np.nan)

    loc_ens_centered = (loc_ens - loss_stats.mean) * jnp.sqrt(counts)

    loc_ens_centered = (mpi.mpi_allgather_jax(loc_ens_centered)[0]).reshape(-1)

    O = (mpi.mpi_allgather_jax(O)[0]).reshape((-1, *O.shape[1:]))

    # Complex real split, is this correct? TODO: double check
    if len(O.shape) == 3:
        O = O[:,0,:] + 1.j * O[:,1,:]

    OO = O.dot(O.conj().T)

    OO_epsilon = solver(OO).dot(loc_ens_centered)

    dense_update = O.conj().T.dot(OO_epsilon)

    return loss_stats, dense_update



