import numpy as np

import jax
import jax.numpy as jnp

from functools import partial

import netket as nk
import mpi4jax
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

    def __init__(
        self,
        *args,
        mode: str = None,
        holomorphic: bool = None,
        solver=lambda A, b: jnp.linalg.lstsq(A, b, rcond=1.0e-12)[0],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert not (mode is not None and holomorphic is not None)
        if mode is None:
            self.mode = nk.jax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.samples,
                holomorphic=holomorphic,
            )
        else:
            self.mode = mode
        self.solver = solver
        if self.state.chunk_size is None:
            self.state.chunk_size = self.state.n_samples_per_rank
        check_chunk_size(self.state.n_samples_per_rank, self.state.chunk_size)

    # Super simple implementation of the minSR driver
    def _forward_and_backward(self):
        self.state.reset()

        # Gather samples and counts
        if hasattr(self.state, "samples_with_counts"):
            samples, counts = self.state.samples_with_counts
        else:
            samples = self.state.samples
            counts = jnp.ones(samples.shape[:-1]) / (
                mpi.mpi_sum_jax(np.prod(samples.shape[:-1]))[0]
            )
        samples = samples.reshape((-1, samples.shape[-1]))
        counts = counts.reshape((-1,))

        # Compute local energies and center them
        loc_ens = self.state.local_estimators(self._ham).T.reshape(-1)
        self._loss_stats = _statistics(loc_ens, counts)
        loc_ens_centered = (loc_ens - self._loss_stats.mean) * jnp.sqrt(counts)

        # Prepare chunks
        n_samples_per_rank = samples.shape[0]
        n_chunks = n_samples_per_rank // self.state.chunk_size

        idx = jnp.arange(n_samples_per_rank).reshape((n_chunks, self.state.chunk_size))
        samples = jnp.reshape(
            samples, (n_chunks, self.state.chunk_size, samples.shape[-1])
        )
        counts = jnp.reshape(counts, (n_chunks, self.state.chunk_size))
        loc_ens_centered = jnp.reshape(
            loc_ens_centered, (n_chunks, self.state.chunk_size)
        )

        # Initialize neural tangent kernel, gradient of loss and dense update
        if mpi.rank == 0:
            OO = jnp.zeros(
                (self.state.n_samples, self.state.n_samples), dtype=jnp.complex128
            )

        dense_update = None

        O_avg = None
        loss_grad = None

        # Compute gradient of loss and average of log-derivatives
        for i in range(n_chunks):
            counts_i = counts[i]
            O_i = nk.jax.jacobian(
                self.state._apply_fun,
                self.state.parameters,
                samples[i],
                self.state.model_state,
                mode=self.mode,
                pdf=counts_i,
                dense=True,
                center=False,
            )
            if len(O_i.shape) == 3:
                O_i = O_i[:, 0, :] + 1.0j * O_i[:, 1, :]

            if loss_grad is None:
                loss_grad = mpi.mpi_sum_jax(jnp.dot(O_i.T, loc_ens_centered[i]))[0]
            else:
                loss_grad += mpi.mpi_sum_jax(jnp.dot(O_i.T, loc_ens_centered[i]))[0]

            if O_avg is None:
                O_avg = mpi.mpi_sum_jax(np.sum(O_i * counts_i[:, np.newaxis], axis=0))[
                    0
                ]
            else:
                O_avg += mpi.mpi_sum_jax(np.sum(O_i * counts_i[:, np.newaxis], axis=0))[
                    0
                ]

        # Compute neural tangent kernel
        for i in range(n_chunks):
            idx_i = (
                idx[i]
                + mpi.node_number * self.state.chunk_size
                + i * self.state.chunk_size * mpi.n_nodes
            )
            idx_i = (mpi4jax.gather(idx_i, root=0, comm=mpi.MPI_jax_comm)[0]).reshape(-1)
            O_i = nk.jax.jacobian(
                self.state._apply_fun,
                self.state.parameters,
                samples[i],
                self.state.model_state,
                mode=self.mode,
                pdf=counts[i],
                dense=True,
                center=False,
            )
            O_i = (mpi4jax.gather(O_i, root=0, comm=mpi.MPI_jax_comm)[0]).reshape((-1, *O_i.shape[1:]))
            if mpi.rank == 0:
                if len(O_i.shape) == 3:
                    O_i = O_i[:, 0, :] + 1.0j * O_i[:, 1, :]
                O_i = O_i - O_avg
            for j in range(n_chunks):
                idx_j = (
                    idx[j]
                    + mpi.node_number * self.state.chunk_size
                    + i * self.state.chunk_size * mpi.n_nodes
                )
                idx_j = (mpi4jax.gather(idx_j, root=0, comm=mpi.MPI_jax_comm)[0]).reshape(-1)
                O_j = nk.jax.jacobian(
                    self.state._apply_fun,
                    self.state.parameters,
                    samples[j],
                    self.state.model_state,
                    mode=self.mode,
                    pdf=counts[j],
                    dense=True,
                    center=False,
                )
                O_j = (mpi4jax.gather(O_j, root=0, comm=mpi.MPI_jax_comm)[0]).reshape((-1, *O_j.shape[1:]))
                if mpi.rank == 0:
                    if len(O_j.shape) == 3:
                        O_j = O_j[:, 0, :] + 1.0j * O_j[:, 1, :]
                    O_j = O_j - O_avg
                    ii, jj = np.meshgrid(idx_i, idx_j, indexing="ij")
                    OO = OO.at[ii, jj].set(O_i.dot(O_j.conj().T))

        # Solve linear system and compute parameters update
        loc_ens_centered_restacked = mpi4jax.gather(loc_ens_centered, root=0, comm=mpi.MPI_jax_comm)[0]
        if mpi.rank == 0:
            loc_ens_centered_restacked = jnp.swapaxes(
                loc_ens_centered_restacked, 0, 1
            ).reshape((-1))
            OO_epsilon = self.solver(OO, loc_ens_centered_restacked)
        else:
            OO_epsilon = jnp.zeros((self.state.n_samples,), dtype=jnp.complex128)
        mpi.MPI_jax_comm.Bcast(OO_epsilon, root=0)
        for i in range(n_chunks):
            idx_i = (
                idx[i]
                + mpi.node_number * self.state.chunk_size
                + i * self.state.chunk_size * mpi.n_nodes
            )
            O_i = nk.jax.jacobian(
                self.state._apply_fun,
                self.state.parameters,
                samples[i],
                self.state.model_state,
                mode=self.mode,
                pdf=counts[i],
                dense=True,
                center=False,
            )
            if len(O_i.shape) == 3:
                O_i = O_i[:, 0, :] + 1.0j * O_i[:, 1, :]
            O_i = O_i - O_avg

            if dense_update is None:
                dense_update = mpi.mpi_sum_jax(O_i.conj().T.dot(OO_epsilon[idx_i]))[0]
            else:
                dense_update += mpi.mpi_sum_jax(O_i.conj().T.dot(OO_epsilon[idx_i]))[0]

        # Convert back to pytree
        unravel = lambda x: x
        reassemble = lambda x: x
        x = self.state.parameters
        if self.mode != "holomorphic":
            x, reassemble = nk.jax.tree_to_real(self.state.parameters)
        _, unravel = nk.jax.tree_ravel(x)

        self._dp = reassemble(unravel(dense_update))

        self._loss_grad = reassemble(unravel(loss_grad))

        # Cast to real if necessary
        self._dp = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

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
    var = mpi.mpi_sum_jax(jnp.sum(abs(data - mean) ** 2 * counts))[0]
    error_of_mean = jnp.sqrt(var / batch_size)

    taus = jax.vmap(integrated_time)(data)
    tau_avg, _ = mpi.mpi_mean_jax(jnp.mean(taus))

    R_hat = _split_R_hat(data, var)

    res = Stats(mean, error_of_mean, var, tau_avg, R_hat)

    return res
