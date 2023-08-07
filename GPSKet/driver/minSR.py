import numpy as np

import jax
import jax.numpy as jnp

from functools import partial

import netket as nk
from netket import VMC
from netket.stats import Stats
from netket.utils import mpi
from netket.vqs.mc.mc_state.state import check_chunk_size
from netket.stats._autocorr import integrated_time
from netket.stats.mc_stats import _split_R_hat


class minSRVMC(VMC):
    """
    VMC driver utilizing the minSR updates as proposed in https://arxiv.org/abs/2302.01941
    """
    def __init__(self, *args, mode: str = None, holomorphic: bool = None,
                 solver=lambda A, b: jnp.linalg.lstsq(A, b, rcond=1.e-12)[0], **kwargs):
        super().__init__(*args, **kwargs)
        assert(not (mode is not None and holomorphic is not None))
        if mode is None:
            self.mode = nk.jax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.samples,
                holomorphic=holomorphic
            )
        else:
            self.mode = mode
        self.solver = solver
        if self.state.chunk_size is None:
            self.state.chunk_size = self.state.n_samples
        check_chunk_size(self.state.n_samples, self.state.chunk_size)

    # Super simple implementation of the minSR driver
    def _forward_and_backward(self):
        self.state.reset()

        # Gather samples and counts
        if hasattr(self.state, "samples_with_counts"):
            samples, counts = self.state.samples_with_counts
        else:
            samples = self.state.samples
            counts = jnp.ones(samples.shape[:-1])/(mpi.mpi_sum_jax(np.prod(samples.shape[:-1]))[0])
        samples = samples.reshape((-1, samples.shape[-1]))
        counts = counts.reshape((-1,))

        # Compute local energies and center them
        loc_ens = self.state.local_estimators(self._ham).T.reshape(-1)
        self._loss_stats = _statistics(loc_ens, counts)
        loc_ens_centered = (loc_ens-self._loss_stats.mean)*jnp.sqrt(counts)

        # Prepare chunks
        n_samples_per_rank = samples.shape[0]
        n_chunks = n_samples_per_rank//self.state.chunk_size
        idx = jnp.arange(n_samples_per_rank).reshape((n_chunks, self.state.chunk_size))
        samples = jnp.reshape(samples, (n_chunks, self.state.chunk_size, samples.shape[-1]))
        counts = jnp.reshape(counts, (n_chunks, self.state.chunk_size))
        loc_ens_centered = jnp.reshape(loc_ens_centered, (n_chunks, self.state.chunk_size))

        # Initialize neural tangent kernel, gradient of loss and dense update
        dtype = jnp.complex128 if self.mode == "holomorphic" else jnp.float64
        if self.mode == "holomorphic" or self.mode == "real":
            O_avg = jnp.zeros(self.state.n_parameters, dtype=dtype)
            OO = jnp.zeros((self.state.n_samples, self.state.n_samples), dtype=dtype)
            loss_grad = jnp.zeros(self.state.n_parameters, dtype=dtype)
            dense_update = jnp.zeros(self.state.n_parameters, dtype=jnp.complex128)
        else:
            O_avg = jnp.zeros(2*self.state.n_parameters, dtype=dtype)
            OO = jnp.zeros((2*self.state.n_samples, 2*self.state.n_samples), dtype=dtype)
            loss_grad = jnp.zeros(2*self.state.n_parameters, dtype=dtype)
            dense_update = jnp.zeros(2*self.state.n_parameters, dtype=jnp.complex128)

        # Compute gradient of loss and average of log-derivatives
        for i in range(n_chunks):
            counts_i = counts[i]
            O_i = nk.jax.jacobian(self.state._apply_fun, self.state.parameters, samples[i],
                                  self.state.model_state, mode=self.mode, pdf=counts_i, dense=True, center=False)
            O_i = (mpi.mpi_allgather_jax(O_i)[0]).reshape((-1, *O_i.shape[1:]))
            if len(O_i.shape) == 3:
                O_i = O_i[:,0,:] + 1.j * O_i[:,1,:]
            counts_i = (mpi.mpi_allgather_jax(counts_i)[0]).reshape((-1,))
            loc_ens_centered_i = (mpi.mpi_allgather_jax(loc_ens_centered[i])[0]).reshape((-1,))
            loss_grad += jnp.dot(O_i.T, loc_ens_centered_i).real
            O_avg += np.sum(O_i*counts_i[:, np.newaxis], axis=0)
        
        # Compute neural tangent kernel
        mpi_rank = jnp.repeat(jnp.arange(mpi.n_nodes), self.state.chunk_size)
        for i in range(n_chunks):
            idx_i = (mpi.mpi_allgather_jax(idx[i])[0]).reshape((-1,))+mpi_rank
            O_i = nk.jax.jacobian(self.state._apply_fun, self.state.parameters, samples[i],
                                  self.state.model_state, mode=self.mode, pdf=counts[i], dense=True, center=False)
            O_i = (mpi.mpi_allgather_jax(O_i)[0]).reshape((-1, *O_i.shape[1:]))
            if len(O_i.shape) == 3:
                O_i = O_i[:,0,:] + 1.j * O_i[:,1,:]
            O_i = O_i-O_avg
            for j in range(n_chunks):
                idx_j = (mpi.mpi_allgather_jax(idx[j])[0]).reshape((-1,))+mpi_rank
                O_j = nk.jax.jacobian(self.state._apply_fun, self.state.parameters, samples[j],
                                      self.state.model_state, mode=self.mode, pdf=counts[j], dense=True, center=False)
                O_j = (mpi.mpi_allgather_jax(O_j)[0]).reshape((-1, *O_j.shape[1:]))
                if len(O_j.shape) == 3:
                    O_j = O_j[:,0,:] + 1.j * O_j[:,1,:]
                O_j = O_j-O_avg
                ii, jj = np.meshgrid(idx_i, idx_j, indexing="ij")
                OO = OO.at[ii, jj].set(O_i.dot(O_j.conj().T))

        # Solve linear system and compute parameters update
        # FIXME: shapes don't match for complex models and complex mode
        loc_ens_centered = (mpi.mpi_allgather_jax(loc_ens_centered)[0]).reshape((-1,))
        OO_epsilon = self.solver(OO, loc_ens_centered)
        for i in range(n_chunks):
            idx_i = (mpi.mpi_allgather_jax(idx[i])[0]).reshape((-1,))+mpi_rank
            O_i = nk.jax.jacobian(self.state._apply_fun, self.state.parameters, samples[i],
                                  self.state.model_state, mode=self.mode, pdf=counts[i], dense=True, center=False)
            O_i = (mpi.mpi_allgather_jax(O_i)[0]).reshape((-1, *O_i.shape[1:]))
            if len(O_i.shape) == 3:
                O_i = O_i[:,0,:] + 1.j * O_i[:,1,:]
            O_i = O_i-O_avg
            dense_update += O_i.conj().T.dot(OO_epsilon[idx_i])               

        # Convert back to pytree
        unravel = lambda x : x
        reassemble = lambda x: x
        x = self.state.parameters
        if self.mode != "holomorphic":
            x, reassemble = nk.jax.tree_to_real(self.state.parameters)
        _, unravel = nk.jax.tree_ravel(x)

        self._dp = reassemble(unravel(dense_update))
        self._loss_grad = reassemble(unravel(loss_grad))

        # Cast to real if necessary
        self._dp = jax.tree_map(lambda x, target: (x if jnp.iscomplexobj(x) else x.real), self._dp, self.state.parameters)

        return self._dp

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
