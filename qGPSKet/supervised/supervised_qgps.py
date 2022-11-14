import numpy as np
from numba import njit
import scipy as sp

import scipy.optimize as spo

from netket.stats import (
    mean as _mean
)

from netket.utils.mpi import (
    MPI_py_comm as _MPI_comm,
    n_nodes as _n_nodes,
    node_number as _rank,
    mpi_sum as _mpi_sum
)

from mpi4py import MPI

from threadpoolctl import threadpool_limits

class QGPSLearning():
    def __init__(self, epsilon, init_alpha=1.0, complex_expand=False, K=None, include_bias=False):
        self.K = K
        if self.K is not None:
            self.precomputed_features = True
        else:
            self.precomputed_features = False
        self.complex_expand = complex_expand
        self._epsilon = None
        self.epsilon = np.array(epsilon)
        """ We set the range ids whenever we set epsilon, this is used for indexing
        (we don't want to create a new arange each time we need it), can probably
        be done a little bit more elegantly
        """
        self.weights = None
        self.site_prod = None
        self.confs = None
        self._ref_sites = None
        if self.precomputed_features:
            if self.complex_expand and self.epsilon.dtype==complex:
                self.alpha_mat = np.ones(self.K.shape[0]*2)*init_alpha
                self.alpha_bias = np.ones(2)*init_alpha
            else:
                self.alpha_mat = np.ones(self.K.shape[0])*init_alpha
                self.alpha_bias = np.ones(1)*init_alpha
        else:
            if self.complex_expand and self.epsilon.dtype==complex:
                self.alpha_mat = np.ones((self.epsilon.shape[-1], self.epsilon.shape[0]*2*self.epsilon.shape[1]))*init_alpha
                self.alpha_bias = np.ones(2)*init_alpha
            else:
                self.alpha_mat = np.ones((self.epsilon.shape[-1], self.epsilon.shape[0]*self.epsilon.shape[1]))*init_alpha
                self.alpha_bias = np.ones(1)*init_alpha
        self.include_bias = include_bias
        self.alpha_cutoff = 1.e10
        self.kern_cutoff = 1.e-10
        self.alpha_convergence_tol = 1.e-15
        self.max_threads = 1
        self.bias = np.zeros(1, dtype=epsilon.dtype)
        self.beta = 1.0

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if self._epsilon is not None:
            assert(epsilon.shape == self.epsilon.shape)
        self._epsilon = np.array(epsilon)
        self.support_dim_range_ids = np.arange(epsilon.shape[1])
        if self.complex_expand and self.epsilon.dtype==complex:
            self.feature_ids = np.arange(2 * epsilon.shape[0] * epsilon.shape[1])
        else:
            self.feature_ids = np.arange(epsilon.shape[0] * epsilon.shape[1])
        self.reset()

    @property
    def ref_sites(self):
        return self._ref_sites

    @ref_sites.setter
    def ref_sites(self, ref_sites_or_site):
        if isinstance(ref_sites_or_site, (int, np.integer)):
            self._ref_sites = ref_sites_or_site * np.ones(self.epsilon.shape[1], dtype=int)
        else:
            self._ref_sites = ref_sites_or_site
        self._ref_sites_incr_dim = np.tile(self._ref_sites, (self.epsilon.shape[0], 1)).T.flatten()
        if self.complex_expand and self.epsilon.dtype==complex:
            self._ref_sites_incr_dim = np.tile(self._ref_sites_incr_dim, 2)

    @property
    def alpha_mat_ref_sites(self):
        if self.precomputed_features:
            full_alpha = self.alpha_mat
        else:
            full_alpha = self.alpha_mat[self._ref_sites_incr_dim, self.feature_ids]
        if self.include_bias:
            full_alpha = np.concatenate((full_alpha, self.alpha_bias))
        return full_alpha

    @alpha_mat_ref_sites.setter
    def alpha_mat_ref_sites(self, alpha):
        if self.include_bias:
            if self.complex_expand and self.epsilon.dtype==complex:
                alpha_bias = alpha[-2:]
                alpha_epsilon = alpha[:-2]
            else:
                alpha_bias = alpha[-1:]
                alpha_epsilon = alpha[:-1]
            self.alpha_bias = alpha_bias
        else:
            alpha_epsilon = alpha
        if self.precomputed_features:
            self.alpha_mat = alpha_epsilon
        else:
            self.alpha_mat[self._ref_sites_incr_dim, self.feature_ids] = alpha_epsilon

    @staticmethod
    @njit()
    def kernel_mat_inner(site_prod, confs, K, ref_sites):
        K = K.reshape(site_prod.shape[0], site_prod.shape[1], -1, site_prod.shape[2])
        K.fill(0.0)
        for i in range(site_prod.shape[0]):
            for j in range(ref_sites.shape[0]):
                for k in range(site_prod.shape[2]):
                    K[i, j, confs[i, ref_sites[j], k], k] += site_prod[i, j, k]
        return K.reshape(site_prod.shape[0], -1, site_prod.shape[2])

    @staticmethod
    @njit()
    def compute_site_prod_fast(epsilon, ref_sites, confs, site_product):
        site_product.fill(1.0)
        for i in range(confs.shape[0]):
            for w in range(epsilon.shape[1]):
                for j in range(confs.shape[1]):
                    if j != ref_sites[w]:
                        for k in range(confs.shape[2]):
                            site_product[i, w, k] *= epsilon[confs[i, j, k], w, j]
        return site_product

    @staticmethod
    @njit()
    def update_site_prod_fast(epsilon, ref_sites, ref_sites_old, confs, site_product):
        eps = 1.e2 * np.finfo(np.double).eps
        for w in range(epsilon.shape[1]):
            ref_site = ref_sites[w]
            ref_site_old = ref_sites_old[w]
            if ref_site != ref_site_old:
                for i in range(confs.shape[0]):
                    for k in range(confs.shape[2]):
                        if np.abs(epsilon[confs[i, ref_site, k], w, ref_site]) > eps:
                            site_product[i, w, k] /= epsilon[confs[i, ref_site, k], w, ref_site]
                            site_product[i, w, k] *= epsilon[confs[i, ref_site_old, k], w, ref_site_old]
                        else:
                            site_product[i, w, k] = 1.
                            for j in range(confs.shape[1]):
                                if j != ref_site:
                                    site_product[i, w, k] *= epsilon[confs[i, j, k], w, j]

        return site_product

    def compute_site_prod(self):
        self.site_prod = np.zeros((self.confs.shape[0], self.epsilon.shape[1], self.confs.shape[-1]), dtype=self.epsilon.dtype)
        self.site_prod = self.compute_site_prod_fast(self.epsilon, self.ref_sites, self.confs, self.site_prod)
        self.site_prod_ref_sites = self.ref_sites

    def update_site_prod(self):
        if not np.array_equal(self.site_prod_ref_sites, self.ref_sites):
            self.site_prod = self.update_site_prod_fast(self.epsilon, self.ref_sites, self.site_prod_ref_sites, self.confs,
                                                        self.site_prod)
        self.site_prod_ref_sites = self.ref_sites

    def set_kernel_mat(self, confs, update_K=False):
        assert(self.ref_sites is not None)
        if not self.precomputed_features:
            recompute_site_prod = False

            if len(confs.shape) == 2:
                confs = np.expand_dims(confs, axis=-1)

            if self.confs is not None:
                if not np.array_equal(self.confs, confs) or self.site_prod is None:
                    recompute_site_prod = True
                elif not np.array_equal(self.ref_sites, self.site_prod_ref_sites):
                    self.update_site_prod()
                    update_K = True
            else:
                recompute_site_prod = True

            if recompute_site_prod:
                self.confs = confs
                self.compute_site_prod()
                self.K_per_sym = None

            if self.K_per_sym is None:
                self.K_per_sym = np.zeros((confs.shape[0], self.epsilon.shape[1] * self.epsilon.shape[0], self.site_prod.shape[-1]), dtype=self.epsilon.dtype)
                update_K = True

            if update_K:
                self.K_per_sym = self.kernel_mat_inner(self.site_prod, self.confs, self.K_per_sym, self.ref_sites)

        self.K = np.sum(self.K_per_sym, axis=-1)

        if self.include_bias:
            self.K = np.concatenate((self.K, np.ones((self.K.shape[0],1))), axis=1)

        return self.K

    def reset(self):
        self.site_prod = None
        if not self.precomputed_features:
            self.K = None

    def setup_fit_alpha_dep(self):
        self.active_elements = self.alpha_mat_ref_sites < self.alpha_cutoff
        self.valid_kern = abs(np.diag(self.KtK)) > self.kern_cutoff

        self.active_elements = np.logical_and(self.active_elements, self.valid_kern)

        if self.complex_expand and self.epsilon.dtype==complex:
            self.KtK_alpha = self.beta * self.KtK + np.diag(self.alpha_mat_ref_sites/2)
        else:
            self.KtK_alpha = self.beta * self.KtK + np.diag(self.alpha_mat_ref_sites)

        self.cholesky = False
        self.Sinv = np.zeros((np.sum(self.active_elements), np.sum(self.active_elements)), dtype=self.KtK_alpha.dtype)
        weights = np.zeros(np.sum(self.active_elements), dtype=self.y.dtype)

        if self.active_elements.any():
            if _rank == 0:
                with threadpool_limits(limits=self.max_threads, user_api="blas"):
                    try:
                        L = sp.linalg.cholesky(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)], lower=True)
                        np.copyto(self.Sinv, sp.linalg.solve_triangular(L, np.eye(self.active_elements.sum()), check_finite=False, lower=True))
                        np.copyto(weights, sp.linalg.cho_solve((L, True), self.y[self.active_elements]))
                        self.cholesky = True
                    except:
                        np.copyto(self.Sinv, sp.linalg.pinvh(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)]))
                        np.copyto(weights, self.Sinv.dot(self.y[self.active_elements]))

            _MPI_comm.Bcast(self.Sinv, root=0)
            _MPI_comm.Bcast(weights, root=0)
            self.cholesky = _MPI_comm.bcast(self.cholesky, root=0)

            # This bit is just to emphasize that self.Sinv is not the inverse of sigma but its Cholesky decomposition if self.cholesky==True
            if self.cholesky:
                self.Sinv_L = self.Sinv
                self.Sinv = None

        if self.weights is None:
            if not self.complex_expand and self.epsilon.dtype==complex:
                self.weights = np.zeros(self.alpha_mat_ref_sites.shape[0], dtype=complex)
            else:
                self.weights = np.zeros(self.alpha_mat_ref_sites.shape[0], dtype=float)

        else:
            self.weights.fill(0.0)

        if self.active_elements.any() > 0:
            self.weights[self.active_elements] = self.beta * weights

    def log_marg_lik_alpha_der(self):
        derivative_alpha = np.zeros(self.alpha_mat_ref_sites.shape[0])

        if self.cholesky:
            derivative_alpha[self.active_elements] -= np.sum(abs(self.Sinv_L) ** 2, 0)
        else:
            derivative_alpha[self.active_elements] -= np.diag(self.Sinv).real

        if self.complex_expand and self.epsilon.dtype==complex:
            derivative_alpha[self.active_elements] *= 0.5

        derivative_alpha += 1/(self.alpha_mat_ref_sites)
        derivative_alpha -= (self.weights.conj() * self.weights).real

        if self.complex_expand or self.epsilon.dtype==float:
            derivative_alpha *= 0.5

        return derivative_alpha.real

    def opt_alpha_beta(self, opt_alpha=True, opt_beta=False, max_iterations=None, rvm=False):
        if opt_alpha or opt_beta:
            alpha_old = self.alpha_mat_ref_sites.copy()
            beta_old = self.beta
            converged = False
            j = 0
            if max_iterations is not None:
                if j >= max_iterations:
                    converged = True
            while not converged:
                if np.any(self.active_elements):
                    if self.cholesky:
                        diag_Sinv = np.sum(abs(self.Sinv_L) ** 2, 0)
                    else:
                        diag_Sinv = np.diag(self.Sinv).real
                        if _rank == 0:
                            print("Warning, update not applied (Cholesky decomposition failed).")

                    if self.complex_expand and self.epsilon.dtype==complex:
                        diag_Sinv = diag_Sinv * 0.5

                    gamma = (1 - (self.alpha_mat_ref_sites[self.active_elements])*diag_Sinv)

                    alpha = self.alpha_mat_ref_sites

                    if opt_alpha:
                        if rvm:
                            alpha[self.active_elements] = (gamma/((self.weights.conj()*self.weights)[self.active_elements])).real
                        else:
                            alpha.fill(((np.sum(gamma)/(self.weights.conj().dot(self.weights))).real))

                        if self.cholesky:
                            if np.any(alpha < 0.):
                                print("Warning! clipping alpha < 0")
                            self.alpha_mat_ref_sites = np.clip(alpha, 0., self.alpha_cutoff)
                    if opt_beta:
                        if self.cholesky:
                            if self.weightings is not None:
                                N = _mpi_sum(np.sum(self.weightings))
                                squared_error = _mpi_sum(np.sum(self.weightings * abs(self.fit_data - self.prediction_after_setup())**2))
                            else:
                                N = _mpi_sum(len(self.fit_data))
                                squared_error = _mpi_sum(np.sum(abs(self.fit_data - self.prediction_after_setup())**2))

                            if self.complex_expand and self.epsilon.dtype==complex:
                                self.beta = (N - 0.5 * np.sum(gamma))/(squared_error)
                            else:
                                self.beta = (N - np.sum(gamma))/(squared_error)

                    j += 1
                    convergence_loss = 0.
                    if opt_alpha:
                        convergence_loss += np.sum((abs(self.alpha_mat_ref_sites - alpha_old)**2)/(abs(alpha_old)**2))
                    if opt_beta:
                        convergence_loss += (abs((self.beta - beta_old))**2)/(abs(beta_old)**2)
                    if convergence_loss < self.alpha_convergence_tol:
                        converged = True
                    if opt_alpha:
                        np.copyto(alpha_old, self.alpha_mat_ref_sites)
                    if opt_beta:
                        beta_old = self.beta
                    if max_iterations is not None:
                        if j >= max_iterations:
                            converged = True
                    if not converged:
                        self.setup_fit_alpha_dep()
                else:
                    converged = True

    def set_up_prediction(self, confset):
        if self.ref_sites is None:
            self.ref_sites = 0

        self.set_kernel_mat(confset)

    def squared_error(self, confset, target_amplitudes, weightings = None):
        errors = abs(self.predict(confset) - target_amplitudes)**2
        if weightings is not None:
            errors *= weightings
        return _MPI_comm.allreduce(np.sum(errors))

    def squared_error_log_space(self, confset, target_amplitudes, weightings = None):
        errors = abs(np.log(self.predict(confset)) - np.log(target_amplitudes))**2
        if weightings is not None:
            errors *= weightings
        return _MPI_comm.allreduce(np.sum(errors))

    def update_epsilon_with_weights(self, prior_mean=0.):
        old_weights = (self.epsilon[:, self.support_dim_range_ids, self.ref_sites]).T.flatten()-prior_mean

        if self.include_bias:
            old_weights = np.concatenate((old_weights, self.bias))

        if self.complex_expand and self.epsilon.dtype==complex:
            old_weights = np.concatenate((old_weights.real, old_weights.imag))

        weights = np.where(self.valid_kern, self.weights, old_weights)

        no_epsilon_weights = self.epsilon.shape[0]*self.epsilon.shape[1]

        if self.include_bias:
            self.bias = np.array(self.weights[no_epsilon_weights], dtype=self.bias.dtype).reshape(1)
            if self.complex_expand and self.epsilon.dtype==complex:
                self.bias += 1.j*self.weights[-1]

        self.epsilon[:, self.support_dim_range_ids, self.ref_sites] = weights[:no_epsilon_weights].reshape(self.epsilon.shape[1], self.epsilon.shape[0]).T + prior_mean

        if self.complex_expand and self.epsilon.dtype==complex:
            if self.include_bias:
                self.epsilon[:, self.support_dim_range_ids, self.ref_sites] += 1.j * weights[(no_epsilon_weights+1):-1].reshape(self.epsilon.shape[1], self.epsilon.shape[0]).T
            else:
                self.epsilon[:, self.support_dim_range_ids, self.ref_sites] += 1.j * weights[no_epsilon_weights:].reshape(self.epsilon.shape[1], self.epsilon.shape[0]).T


class QGPSLearningExp(QGPSLearning):
    def __init__(self, epsilon, init_alpha = 1.0, init_noise_tilde = 1.e-1, complex_expand=False, K=None, include_bias=False, iterative_noise_est=False):
        super().__init__(epsilon, init_alpha=init_alpha, complex_expand=complex_expand, K=K, include_bias=include_bias)

        self.noise_tilde = init_noise_tilde
        self.iterative_noise_est = iterative_noise_est
        self.noise_reweighting = None

    def predict(self, confset):
        assert(confset.size > 0)
        self.set_up_prediction(confset)
        if self.include_bias:
            return np.exp(self.K[:,:-1].dot((self.epsilon[:, self.support_dim_range_ids, self.ref_sites].T).flatten())+self.bias)
        else:
            return np.exp(self.K.dot((self.epsilon[:, self.support_dim_range_ids, self.ref_sites].T).flatten()))

    def setup_fit_noise_dep(self, weightings=None):
        if self.noise_tilde == 0.:
            self.S_diag = np.ones(len(self.exp_amps))
        else:
            if self.iterative_noise_est:
                pred = self.predict(self.confs)
                self.S_diag = 1/(np.log1p(np.sqrt((4 * self.noise_tilde/(abs(pred)**2)) + 1)) + np.log(0.5))
            else:
                if self.noise_reweighting is None:
                    self.S_diag = 1/(np.log1p(self.noise_tilde/(abs(self.exp_amps)**2)))
                else:
                    self.S_diag = 1/(np.log1p(self.noise_tilde/((abs(self.exp_amps)**2)/self.noise_reweighting)))
        if weightings is not None:
            self.S_diag *= weightings
        self.weightings = weightings

        self.KtK = _mpi_sum(np.dot(self.K.conj().T, np.einsum("i,ij->ij", self.S_diag, self.K)))

        self.y = _mpi_sum(self.K.conj().T.dot(self.S_diag * self.fit_data))

        if self.complex_expand and self.epsilon.dtype==complex:
            self.KtK = np.block([[self.KtK.real, -self.KtK.imag],[self.KtK.imag, self.KtK.real]])
            self.y = np.concatenate((self.y.real, self.y.imag))

        self.setup_fit_alpha_dep()

    def setup_fit(self, confset, target_amplitudes, ref_sites, weightings=None, prior_mean=0.):
        self.ref_sites = ref_sites
        self.exp_amps = target_amplitudes.astype(self.epsilon.dtype)
        if self.epsilon.dtype == float:
            self.fit_data = np.log(abs(self.exp_amps))
        else:
            self.fit_data = np.log(self.exp_amps)
        self.set_kernel_mat(confset)
        if self.include_bias:
            self.fit_data -= prior_mean * np.sum(self.K[:,:-1], axis=1)
        else:
            self.fit_data -= prior_mean * np.sum(self.K, axis=1)
        self.setup_fit_noise_dep(weightings=weightings)

    def log_marg_lik(self):
        if self.weightings is not None:
            if self.epsilon.dtype==complex:
                log_lik = -(np.sum(self.weightings * np.log(np.pi/(self.S_diag/self.weightings))))
            else:
                log_lik = -(np.sum(self.weightings * np.log(2*np.pi/(self.S_diag/self.weightings))))
        else:
            if self.epsilon.dtype==complex:
                log_lik = -(np.sum(np.log(np.pi/self.S_diag)))
            else:
                log_lik = -(np.sum(np.log(2*np.pi/self.S_diag)))


        log_lik -= np.dot(self.fit_data.conj(), self.S_diag * self.fit_data)
        log_lik = _MPI_comm.allreduce(log_lik)

        if self.cholesky:
            if self.complex_expand and self.epsilon.dtype==complex:
                log_lik += 0.5 * np.sum(np.log(0.5 * abs(np.diag(self.Sinv_L))**2))
            else:
                log_lik += 2 * np.sum(np.log(abs(np.diag(self.Sinv_L))))
        else:
            if self.complex_expand and self.epsilon.dtype==complex:
                log_lik += 0.5 * np.linalg.slogdet(0.5 * self.Sinv)[1]
            else:
                log_lik += np.linalg.slogdet(self.Sinv)[1]

        active_alpha = np.logical_and(self.active_elements, self.alpha_mat_ref_sites > 0.)

        if self.complex_expand and self.epsilon.dtype==complex:
            log_lik += 0.5 * np.sum(np.log(self.alpha_mat_ref_sites[active_alpha]))
        else:
            log_lik += np.sum(np.log(self.alpha_mat_ref_sites[active_alpha]))

        weights = self.weights[self.active_elements]
        log_lik += np.dot(weights.conj(), np.dot(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)], weights))

        if self.epsilon.dtype==float:
            log_lik *= 0.5

        return log_lik.real

    def log_marg_lik_noise_der(self):
        if self.iterative_noise_est:
            pred = self.predict(self.confs)
            del_S = (1/((np.sqrt((4 * self.noise_tilde/((abs(pred)**2)/self.noise_reweighting)) + 1) + 1))) * 2/(((abs(pred)**2)/self.noise_reweighting) * np.sqrt((4 * self.noise_tilde/((abs(pred)**2)/self.noise_reweighting)) + 1))
        else:
            del_S = 1/(((abs(self.exp_amps)**2)/self.noise_reweighting) * (1 + self.noise_tilde/((abs(self.exp_amps)**2)/self.noise_reweighting)))

        Delta_S = - (self.S_diag**2 * del_S)

        if self.weightings is not None:
            Delta_S /= self.weightings

        K = self.K
        KtK_der = self.K.conj().T.dot(np.einsum("i,ij->ij", Delta_S, self.K))

        if self.complex_expand and self.epsilon.dtype==complex:
            KtK_der = np.block([[KtK_der.real, -KtK_der.imag],[KtK_der.imag, KtK_der.real]])
            K = np.hstack((K, 1.j * K))

        K = K[:,self.active_elements]
        KtK_der = KtK_der[np.ix_(self.active_elements, self.active_elements)]

        if self.cholesky:
            derivative_noise = np.trace(KtK_der.dot(self.Sinv_L.conj().T.dot(self.Sinv_L)))
        else:
            derivative_noise = np.trace(KtK_der.dot(self.Sinv))

        if self.complex_expand and self.epsilon.dtype==complex:
            derivative_noise *= 0.5

        derivative_noise -= np.sum(self.S_diag * del_S)

        weights = self.weights[self.active_elements]

        derivative_noise -= self.fit_data.conj().dot(Delta_S*self.fit_data)
        derivative_noise -= weights.conj().dot(KtK_der.dot(weights))

        derivative_noise += 2*self.fit_data.conj().dot(Delta_S*K.dot(weights))

        derivative_noise = _MPI_comm.allreduce(derivative_noise)

        if self.epsilon.dtype==float:
            derivative_noise *= 0.5

        return derivative_noise.real

    def fit_step(self, confset, target_amplitudes, ref_sites, noise_bounds=[(None, None)],
                 opt_alpha=True, opt_noise=True, max_alpha_iterations=None, max_noise_iterations=None, rvm=False,
                 weightings=None, prior_mean=0.):
        self.setup_fit(confset, target_amplitudes, ref_sites, weightings=weightings, prior_mean=prior_mean)
        if opt_noise:
            alpha_init = self.alpha_mat_ref_sites.copy()
            def ML(x):
                self.noise_tilde = np.exp(x[0])
                if opt_alpha:
                    np.copyto(self.alpha_mat_ref_sites, alpha_init)
                self.setup_fit_noise_dep(weightings=weightings)
                if opt_alpha:
                    self.opt_alpha_beta(max_iterations=max_alpha_iterations, rvm=rvm)
                return -self.log_marg_lik()

            def derivative(x):
                self.noise_tilde = np.exp(x[0])
                if opt_alpha:
                    np.copyto(self.alpha_mat_ref_sites, alpha_init)
                self.setup_fit_noise_dep(weightings=weightings)
                if opt_alpha:
                    self.opt_alpha_beta(max_iterations=max_alpha_iterations, rvm=rvm)
                der_noise = self.log_marg_lik_noise_der()
                return - der_noise * np.exp(x)

            def update_alpha(x):
                self.noise_tilde = np.exp(x[0])
                if opt_alpha:
                    self.opt_alpha_beta(max_iterations=max_alpha_iterations, rvm=rvm)
                np.copyto(alpha_init, self.alpha_mat_ref_sites)

            if max_noise_iterations is not None:
                opt = sp.optimize.minimize(ML, np.log(self.noise_tilde), options={"maxiter" : max_noise_iterations}, jac=derivative, bounds=noise_bounds, callback=update_alpha)
            else:
                opt = sp.optimize.minimize(ML, np.log(self.noise_tilde), jac=derivative, bounds=noise_bounds, callback=update_alpha)

            self.noise_tilde = np.exp(opt.x)[0]
            if opt_alpha:
                np.copyto(self.alpha_mat_ref_sites, alpha_init)
            self.setup_fit_noise_dep(weightings=weightings)

        if opt_alpha:
            self.opt_alpha_beta(max_iterations=max_alpha_iterations, rvm=rvm)
        if not self.precomputed_features:
            self.update_epsilon_with_weights(prior_mean=prior_mean)
        return

    '''
    parts of the following code are based on the code from https://github.com/AmazaspShumik/sklearn-bayes
    which is published under the following MIT license:
    Copyright (c) 2020 Amazasp Shaumyan
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    '''

    def compute_sparsity_quantities(self):
        bxy = self.y
        bxx = np.diag(self.KtK)

        if self.cholesky:
            xxr = np.dot(self.KtK[:, self.active_elements], self.Sinv_L.conj().T)
            rxy = np.dot(self.Sinv_L, self.y[self.active_elements])
            S = bxx - np.sum(abs(xxr) ** 2, axis=1)
            Q = bxy - np.dot(xxr, rxy)
        else:
            XXa = self.KtK[:, self.active_elements]
            XS = np.dot(XXa, self.Sinv)
            S = bxx - np.sum(XS * XXa, 1)
            Q = bxy - np.dot(XS, self.y[self.active_elements])

        S = S.real
        Q = Q.real
        # Use following:
        # (EQ 1) q = A*Q/(A - S) ; s = A*S/(A-S), so if A = np.PINF q = Q, s = S
        qi = np.copy(Q)
        si = np.copy(S)
        #  If A is not np.PINF, then it should be 'active' feature => use (EQ 1)
        Qa, Sa = Q[self.active_elements], S[self.active_elements]

        if self.complex_expand and self.epsilon.dtype==complex:
            qi[self.active_elements] = self.alpha_mat_ref_sites[self.active_elements] * Qa / (self.alpha_mat_ref_sites[self.active_elements] - 2*Sa)
            si[self.active_elements] = self.alpha_mat_ref_sites[self.active_elements] * Sa / (self.alpha_mat_ref_sites[self.active_elements] - 2*Sa)
        else:
            qi[self.active_elements] = self.alpha_mat_ref_sites[self.active_elements] * Qa / (self.alpha_mat_ref_sites[self.active_elements] - Sa)
            si[self.active_elements] = self.alpha_mat_ref_sites[self.active_elements] * Sa / (self.alpha_mat_ref_sites[self.active_elements] - Sa)
        return [si, qi, S, Q]

    def update_precisions(self, s, q, S, Q):
        deltaL = np.zeros(Q.shape[0])

        theta = abs(q) ** 2 - s
        add = (theta > 0) * (self.active_elements == False)
        recompute = (theta > 0) * (self.active_elements == True)
        delete = (theta <= 0) * (self.active_elements == True)

        # compute sparsity & quality parameters corresponding to features in
        # three groups identified above
        Qadd, Sadd = Q[add], S[add]

        if self.complex_expand and self.epsilon.dtype==complex:
            Qrec, Srec, Arec = Q[recompute], S[recompute], self.alpha_mat_ref_sites[recompute]/2
            Qdel, Sdel, Adel = Q[delete], S[delete], self.alpha_mat_ref_sites[delete]/2
        else:
            Qrec, Srec, Arec = Q[recompute], S[recompute], self.alpha_mat_ref_sites[recompute]
            Qdel, Sdel, Adel = Q[delete], S[delete], self.alpha_mat_ref_sites[delete]

        # compute new alpha's (precision parameters) for features that are
        # currently in model and will be recomputed

        Anew = s[recompute] ** 2 / (theta[recompute])

        delta_alpha = (1. / Anew) - (1. / Arec)

        # compute change in log marginal likelihood
        deltaL[add] = (abs(Qadd) ** 2 - Sadd) / Sadd + np.log(Sadd / abs(Qadd) ** 2)

        deltaL[recompute] = abs(Qrec) ** 2 / (Srec + 1. / delta_alpha) - np.log(1 + Srec * delta_alpha)

        deltaL[delete] = abs(Qdel) ** 2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
        deltaL = np.nan_to_num(deltaL, nan=np.NINF, posinf=np.NINF, neginf=np.NINF)

        # find feature which caused largest change in likelihood

        feature_index = np.argmax(deltaL)

        alpha = self.alpha_mat_ref_sites
        if theta[feature_index] > 0:
            if self.complex_expand and self.epsilon.dtype==complex:
                alpha[feature_index] = 2 * (s[feature_index] ** 2 / theta[feature_index])
            else:
                alpha[feature_index] = s[feature_index] ** 2 / theta[feature_index]
        else:
            # at least one active features
            if self.active_elements[feature_index] == True and np.sum(self.active_elements) >= 2:
                alpha[feature_index] = np.PINF
        self.alpha_mat_ref_sites = alpha

        return


    def fit_step_growing_RVM(self, confset, target_amplitudes, ref_site, alpha_iterations, weightings=None, prior_mean=0.):
        self.setup_fit(confset, target_amplitudes, ref_site, weightings=weightings, prior_mean=prior_mean)

        if np.max(self.active_elements) == 0:
            alpha = self.alpha_mat_ref_sites
            if np.min(abs(np.diag(self.KtK))) < np.finfo(np.float32).eps:
                alpha[0] = np.finfo(np.float32).eps
            else:
                projections = (abs(self.y) **2 / np.diag(self.KtK))
                ind = np.argmax(projections)

                alpha_est = (((np.diag(self.KtK))**2 / (abs(self.y)**2 - np.diag(self.KtK))).real)[ind]

                if alpha_est > 0.:
                    alpha[ind] = alpha_est
                    if self.complex_expand and self.epsilon.dtype==complex:
                        alpha[ind] *= 2
                else:
                    alpha[ind] = 1.
                    if self.complex_expand and self.epsilon.dtype==complex:
                        alpha[ind] *= 2
                    print(alpha_est)
            self.alpha_mat_ref_sites = alpha
            self.setup_fit_alpha_dep()

        for i in range(alpha_iterations):
            s, q, S, Q = self.compute_sparsity_quantities()
            self.update_precisions(s, q, S, Q)
            self.setup_fit_alpha_dep()

        if not self.precomputed_features:
            self.update_epsilon_with_weights()

        return

class QGPSLogSpaceFit(QGPSLearningExp):
    def __init__(self, epsilon, init_alpha = 1.0, init_noise_tilde = 1.e-1, complex_expand=False, K=None, include_bias=False):
        super().__init__(epsilon, init_alpha=init_alpha, init_noise_tilde = init_noise_tilde, complex_expand=complex_expand, K=K, include_bias=include_bias, iterative_noise_est=False)
        self.beta = 1/self.noise_tilde

    def predict(self, confset):
        assert(confset.size > 0)
        self.set_up_prediction(confset)
        if self.include_bias:
            return self.K[:,:-1].dot((self.epsilon[:, self.support_dim_range_ids, self.ref_sites].T).flatten())+self.bias
        else:
            return self.K.dot((self.epsilon[:, self.support_dim_range_ids, self.ref_sites].T).flatten())

    def prediction_after_setup(self):
        return self.K.dot(self.weights)

    def setup_fit(self, confset, log_amplitudes, ref_sites, weightings=None, prior_mean=0.):
        self.ref_sites = ref_sites
        self.fit_data = log_amplitudes
        self.set_kernel_mat(confset)
        if self.include_bias:
            self.fit_data -= prior_mean * np.sum(self.K[:,:-1], axis=1)
        else:
            self.fit_data -= prior_mean * np.sum(self.K, axis=1)
        self.weightings = weightings
        if self.weightings is not None:
            self.KtK = _mpi_sum(np.dot(self.K.conj().T, np.einsum("i,ij->ij", self.weightings, self.K)))
            self.y = _mpi_sum(self.K.conj().T.dot(self.weightings * self.fit_data))
        else:
            self.KtK = _mpi_sum(np.dot(self.K.conj().T, self.K))
            self.y = _mpi_sum(self.K.conj().T.dot(self.fit_data))

        if self.complex_expand and self.epsilon.dtype==complex:
            self.KtK = np.block([[self.KtK.real, -self.KtK.imag],[self.KtK.imag, self.KtK.real]])
            self.y = np.concatenate((self.y.real, self.y.imag))

        self.setup_fit_alpha_dep()

    def log_marg_lik(self):
        if self.weightings is not None:
            if self.epsilon.dtype==complex:
                log_lik = -(np.sum(self.weightings) * np.log(np.pi * self.noise_tilde))
            else:
                log_lik = -(np.sum(self.weightings) * np.log(2*np.pi * self.noise_tilde))
        else:
            if self.epsilon.dtype==complex:
                log_lik = -(len(self.fit_data) * np.log(np.pi * self.noise_tilde))
            else:
                log_lik = -(len(self.fit_data) * np.log(2 * np.pi * self.noise_tilde))


        log_lik -= np.dot(self.fit_data.conj(), self.fit_data)/self.noise_tilde
        log_lik = _MPI_comm.allreduce(log_lik)

        if self.cholesky:
            if self.complex_expand and self.epsilon.dtype==complex:
                log_lik += 0.5 * np.sum(np.log(0.5 * abs(np.diag(self.Sinv_L))**2))
            else:
                log_lik += 2 * np.sum(np.log(abs(np.diag(self.Sinv_L))))
        else:
            if self.complex_expand and self.epsilon.dtype==complex:
                log_lik += 0.5 * np.linalg.slogdet(0.5 * self.Sinv)[1]
            else:
                log_lik += np.linalg.slogdet(self.Sinv)[1]

        active_alpha = np.logical_and(self.active_elements, self.alpha_mat_ref_sites > 0.)

        if self.complex_expand and self.epsilon.dtype==complex:
            log_lik += 0.5 * np.sum(np.log(self.alpha_mat_ref_sites[active_alpha]))
        else:
            log_lik += np.sum(np.log(self.alpha_mat_ref_sites[active_alpha]))

        weights = self.weights[self.active_elements]
        log_lik += np.dot(weights.conj(), np.dot(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)], weights))

        if self.epsilon.dtype==float:
            log_lik *= 0.5

        return log_lik.real

    # This is not tested at all, TODO: test!
    def log_marg_lik_noise_der(self):
        if self.cholesky:
            diag_Sinv = np.sum(abs(self.Sinv_L) ** 2, 0)
        else:
            diag_Sinv = np.diag(self.Sinv).real

        if self.complex_expand and self.epsilon.dtype==complex:
            diag_Sinv = diag_Sinv * 0.5

        beta_derivative = -np.sum((1 - (self.alpha_mat_ref_sites[self.active_elements])*diag_Sinv))*self.noise_tilde

        if self.complex_expand and self.epsilon.dtype==complex:
            beta_derivative *= 0.5

        if self.weightings is not None:
            N = _mpi_sum(np.sum(self.weightings))
            squared_error = _mpi_sum(np.sum(self.weightings * abs(self.fit_data - self.predict(self.confs))**2))
        else:
            N = _mpi_sum(len(self.fit_data))
            squared_error = _mpi_sum(np.sum(abs(self.fit_data - self.predict(self.confs))**2))

        beta_derivative += N * self.noise_tilde
        beta_derivative -= squared_error

        if self.epsilon.dtype==float:
            beta_derivative *= 0.5

        return -(self.noise_tilde**2) * beta_derivative


    def fit_step(self, confset, target_amplitudes, ref_sites, opt_alpha=True, opt_noise=True, max_alpha_beta_iterations=None, rvm=False,
                 weightings=None, prior_mean=0.):
        self.beta = 1/self.noise_tilde
        self.setup_fit(confset, target_amplitudes, ref_sites, weightings=weightings, prior_mean=prior_mean)
        self.opt_alpha_beta(opt_alpha=opt_alpha, opt_beta=opt_noise, max_iterations=max_alpha_beta_iterations, rvm=rvm)
        self.noise_tilde = 1/self.beta

        if not self.precomputed_features:
            self.update_epsilon_with_weights(prior_mean=prior_mean)
        return

# TODO: double check and fix the generalized linear model implementation for complex parameters
class QGPSGenLinMod(QGPSLearningExp):
    def __init__(self, epsilon, init_alpha = 1.0, init_noise_tilde = 1.e-1, K=None, include_bias=False):
        if epsilon.dtype==complex:
            complex_expand = True
        else:
            complex_expand = False
        super().__init__(epsilon, init_alpha=init_alpha, init_noise_tilde = init_noise_tilde, complex_expand=complex_expand, K=K, include_bias=include_bias)

    def prediction_after_setup(self):
        return np.exp(self.K.dot(self.weights))

    def setup_fit(self, confset, target_amplitudes, ref_sites, minimize_fun=None, linesearch_fun=None, weightings=None, log_fit_init=False, it_rew_lst_sq_steps=0):
        self.ref_sites = ref_sites
        self.weights = None
        self.exp_amps = target_amplitudes.astype(self.epsilon.dtype)
        self.fit_data = self.exp_amps
        self.set_kernel_mat(confset)
        if weightings is None:
            self.weightings = np.ones(len(self.exp_amps))
        else:
            self.weightings = weightings
        self.N_data = _mpi_sum(np.sum(self.weightings))
        if self.noise_tilde != 0.:
            self.beta = 1/self.noise_tilde
        else:
            self.beta = 1.
        self.setup_fit_alpha_dep(minimize_fun=minimize_fun, linesearch_fun=linesearch_fun, log_fit_init=log_fit_init, it_rew_lst_sq_steps=it_rew_lst_sq_steps)

    def setup_fit_alpha_dep(self, minimize_fun=None, linesearch_fun=None, log_fit_init=False, it_rew_lst_sq_steps=0):
        self.active_elements = self.alpha_mat_ref_sites < self.alpha_cutoff

        if self.precomputed_features:
            assert self.weights is not None
            weights = self.weights
        else:
            if self.weights is None:
                weights = (self.epsilon[:, self.support_dim_range_ids, self.ref_sites]).T.flatten()
                if self.include_bias:
                    weights = np.concatenate((weights, self.bias))
                if self.complex_expand and self.epsilon.dtype==complex:
                    weights = np.concatenate((weights.real, weights.imag))
            else:
                weights = self.weights

        if self.complex_expand and self.epsilon.dtype==complex:
            K = np.hstack((self.K, 1.j * self.K))
        else:
            K = self.K

        self.valid_kern = _mpi_sum(np.sum(abs(K), axis=0)) > self.kern_cutoff
        self.active_elements = np.logical_and(self.active_elements, self.valid_kern)

        if log_fit_init:
            if self.noise_tilde == 0.:
                S_diag = np.ones(len(self.exp_amps))
            else:
                S_diag = 1/(np.log1p(self.noise_tilde/(abs(self.exp_amps)**2)))

            S_diag *= self.weightings

            KtK = _mpi_sum(np.dot(self.K.conj().T, np.einsum("i,ij->ij", S_diag, self.K)))

            y = _mpi_sum(self.K.conj().T.dot(S_diag * np.log(self.exp_amps.astype(self.epsilon.dtype))))

            if self.complex_expand and self.epsilon.dtype==complex:
                KtK = np.block([[KtK.real, -KtK.imag],[KtK.imag, KtK.real]])
                y = np.concatenate((y.real, y.imag))
                KtK_alpha = KtK + np.diag(self.alpha_mat_ref_sites/2)
            else:
                KtK_alpha = KtK + np.diag(self.alpha_mat_ref_sites)

            if _rank == 0:
                with threadpool_limits(limits=self.max_threads, user_api="blas"):
                    weights = np.linalg.lstsq(KtK_alpha, y, rcond=None)[0]
            _MPI_comm.Bcast(weights, root=0)

        # TODO: double check for non-vanishing alpha parameters
        # TODO: implement for complex parameters
        # Implements the iteratively reweighted least squares approach

        if it_rew_lst_sq_steps > 0:
            assert(self.epsilon.dtype != complex)

        for i in range(it_rew_lst_sq_steps):
            pred = np.exp(K.dot(weights))
            e = (self.exp_amps - pred)/pred
            z = np.log(pred) + e
            S_diag = abs(pred)**2 / self.noise_tilde
            KtK = _mpi_sum(np.dot(self.K.conj().T, np.einsum("i,ij->ij", S_diag, self.K)))
            y = _mpi_sum(self.K.conj().T.dot(S_diag * z))

            if self.complex_expand and self.epsilon.dtype==complex:
                KtK = np.block([[KtK.real, -KtK.imag],[KtK.imag, KtK.real]])
                y = np.concatenate((y.real, y.imag))
                KtK_alpha = KtK + np.diag(self.alpha_mat_ref_sites/2)
            else:
                KtK_alpha = KtK + np.diag(self.alpha_mat_ref_sites)

            if self.active_elements.any():
                if _rank == 0:
                    with threadpool_limits(limits=self.max_threads, user_api="blas"):
                        try:
                            L = sp.linalg.cholesky(KtK_alpha, lower=True)
                            np.copyto(Sinv, sp.linalg.solve_triangular(L, np.eye(self.active_elements.sum()), check_finite=False, lower=True))
                            weights[self.active_elements] = sp.linalg.cho_solve((L, True), y)
                            cholesky = True
                        except:
                            weights[self.active_elements] = np.linalg.lstsq(KtK_alpha, y)[0]

            _MPI_comm.Bcast(weights, root=0)

        K = K[:, self.active_elements]
        weights = weights[self.active_elements]

        def get_loss(weights):
            pred = np.exp(K.dot(weights))
            self.neg_log_likelihood = self.beta * _mpi_sum(np.sum(self.weightings * abs(self.exp_amps - pred)**2))
            if self.complex_expand and self.epsilon.dtype==complex:
                loss = self.neg_log_likelihood + np.sum(self.alpha_mat_ref_sites[self.active_elements]/2 * abs(weights)**2)
            else:
                loss = self.neg_log_likelihood + np.sum(self.alpha_mat_ref_sites[self.active_elements] * abs(weights)**2)
            return loss/2

        def get_grad(weights):
            pred = np.exp(K.dot(weights))
            g = self.weightings * pred * (self.exp_amps - pred).conj()
            grad = - K.T.dot(g)
            grad = self.beta * _mpi_sum(grad)
            if self.complex_expand and self.epsilon.dtype==complex:
                grad = grad.real
                grad += self.alpha_mat_ref_sites[self.active_elements]/2 * weights
            else:
                grad += self.alpha_mat_ref_sites[self.active_elements] * weights
            return grad

        def get_hessian(weights):
            pred = np.exp(K.dot(weights))
            b_1 = self.weightings * pred * (self.exp_amps - pred).conj()
            b_2 = self.weightings * pred * pred.conj()

            hess_a = -np.dot(K.T * b_1, K)
            hess_b = np.dot(K.T * b_2, K.conj())

            hessian = self.beta * _mpi_sum(hess_a + hess_b)

            if self.complex_expand and self.epsilon.dtype==complex:
                hessian = (hessian + np.diag(self.alpha_mat_ref_sites[self.active_elements]/2)).real
            else:
                hessian = hessian + np.diag(self.alpha_mat_ref_sites[self.active_elements])

            return hessian

        if minimize_fun is not None:
            min_result = minimize_fun(get_loss, weights, jac=get_grad, hess=get_hessian)
            weights = min_result.x

        self.KtK_alpha = get_hessian(weights)
        self.grad = get_grad(weights)
        get_loss(weights)

        self.cholesky = False
        self.Sinv = np.zeros((np.sum(self.active_elements), np.sum(self.active_elements)), dtype=self.KtK_alpha.dtype)

        update_directions = np.zeros_like(weights)
        if self.active_elements.any():
            if _rank == 0:
                with threadpool_limits(limits=self.max_threads, user_api="blas"):
                    try:
                        L = sp.linalg.cholesky(self.KtK_alpha, lower=True)
                        np.copyto(self.Sinv, sp.linalg.solve_triangular(L, np.eye(self.active_elements.sum()), check_finite=False, lower=True))
                        update_directions = -sp.linalg.cho_solve((L, True), self.grad)
                        self.cholesky = True
                    except:
                        np.copyto(self.Sinv, sp.linalg.pinvh(self.KtK_alpha))
                        update_directions = -self.Sinv.dot(self.grad)

            _MPI_comm.Bcast(self.Sinv, root=0)
            _MPI_comm.Bcast(update_directions, root=0)
            self.cholesky = _MPI_comm.bcast(self.cholesky, root=0)

            if linesearch_fun is not None:
                result = linesearch_fun(get_loss, get_grad, weights, update_directions)
                gamma = result[0]
                if gamma is None:
                    if _rank == 0:
                        print("linesearch not converged, resetting step", self.cholesky)
                    gamma = 0.
            else:
                gamma = 1.
            if self.cholesky:
                np.copyto(weights, weights + gamma * update_directions)
            else:
                if _rank == 0:
                    print("Cholesky fail")

            # This bit is just to emphasize that self.Sinv is not the inverse of sigma but its Cholesky decomposition if self.cholesky==True
            if self.cholesky:
                self.Sinv_L = self.Sinv
                self.Sinv = None

        if self.weights is None:
            if not self.complex_expand and self.epsilon.dtype==complex:
                self.weights = np.zeros(self.alpha_mat_ref_sites.shape[0], dtype=complex)
            else:
                self.weights = np.zeros(self.alpha_mat_ref_sites.shape[0], dtype=float)

        else:
            self.weights.fill(0.0)

        if self.active_elements.any() > 0:
            self.weights[self.active_elements] = weights

    def fit_step(self, confset, target_amplitudes, ref_sites, opt_alpha=True, opt_beta=True, max_alpha_beta_iterations=None, rvm=False, minimize_fun=None, linesearch_fun=None, weightings=None, log_fit_init=False, it_rew_lst_sq_steps=0):
        self.setup_fit(confset, target_amplitudes, ref_sites, minimize_fun=minimize_fun, linesearch_fun=linesearch_fun, weightings=weightings, log_fit_init=log_fit_init, it_rew_lst_sq_steps=it_rew_lst_sq_steps)

        if (opt_alpha or opt_beta) and self.cholesky:
            self.opt_alpha_beta(opt_alpha=opt_alpha, opt_beta=opt_beta, max_iterations=max_alpha_beta_iterations, rvm=rvm)
        if not self.cholesky:
            if _rank == 0:
                print("Warning, update not applied (Cholesky decomposition failed).")

        if not self.precomputed_features:
            self.update_epsilon_with_weights()

        self.noise_tilde = 1/self.beta

    # TODO: fix prefactors, double check, etc.
    def log_marg_lik(self):
        if self.epsilon.dtype==complex:
            log_lik = (np.log(self.beta) - np.log(np.pi)) * self.N_data
        else:
            log_lik = (np.log(self.beta) - np.log(2*np.pi)) * self.N_data

        if self.cholesky:
            if self.complex_expand and self.epsilon.dtype==complex:
                log_lik += 0.5 * np.sum(np.log(0.5 * abs(np.diag(self.Sinv_L))**2))
            else:
                log_lik += 2 * np.sum(np.log(abs(np.diag(self.Sinv_L))))
        else:
            if self.complex_expand and self.epsilon.dtype==complex:
                log_lik += 0.5 * np.linalg.slogdet(0.5 * self.Sinv)[1]
            else:
                log_lik += np.linalg.slogdet(self.Sinv)[1]

        if self.complex_expand and self.epsilon.dtype==complex:
            log_lik += 0.5 * np.sum(np.log(self.alpha_mat_ref_sites[self.active_elements]))
        else:
            log_lik += np.sum(np.log(self.alpha_mat_ref_sites[self.active_elements]))

        if self.complex_expand and self.epsilon.dtype==complex:
            K = np.hstack((self.K, 1.j * self.K))
        else:
            K = self.K

        K = K[:, self.active_elements]
        weights = self.weights[self.active_elements]

        pred = np.exp(K.dot(weights))
        loss = self.beta * _mpi_sum(np.sum(self.weightings * abs(self.exp_amps - pred)**2))

        if self.complex_expand and self.epsilon.dtype==complex:
            loss += np.sum(self.alpha_mat_ref_sites[self.active_elements]/2 * abs(weights)**2)
        else:
            loss += np.sum(self.alpha_mat_ref_sites[self.active_elements] * abs(weights)**2)

        log_lik -= loss

        if self.epsilon.dtype==float:
            log_lik *= 0.5

        return log_lik.real

    # TODO: test this, check prefactors etc.
    def log_marg_lik_beta_der(self):
        log_lik_der = self.N_data/self.beta

        if self.complex_expand and self.epsilon.dtype==complex:
            K = np.hstack((self.K, 1.j * self.K))
        else:
            K = self.K

        K = K[:, self.active_elements]
        weights = self.weights[self.active_elements]

        pred = np.exp(K.dot(weights))

        if self.cholesky:
            diag_Sinv = np.sum(abs(self.Sinv_L) ** 2, 0)
        else:
            diag_Sinv = np.diag(self.Sinv).real
            if _rank == 0:
                print("Warning, update not applied (Cholesky decomposition failed).")

        if self.complex_expand and self.epsilon.dtype==complex:
            diag_Sinv = diag_Sinv * 0.5

        gamma = (1 - (self.alpha_mat_ref_sites[self.active_elements])*diag_Sinv)

        log_lik_der -= np.sum(gamma)/self.beta

        log_lik_der -= _mpi_sum(np.sum(self.weightings * abs(self.exp_amps - pred)**2))

        if self.epsilon.dtype==float:
            log_lik_der *= 0.5

        return log_lik_der.real



class QGPSGenLinModProjSym(QGPSGenLinMod):
    def predict(self, confset):
        assert(confset.size > 0)
        self.set_up_prediction(confset)
        if self.include_bias:
            return np.sum(np.exp(np.einsum("ijk,j->ik", self.K_per_sym, ((self.epsilon[:, self.support_dim_range_ids, self.ref_sites].T).flatten())) + self.bias), axis=-1)
        else:
            return np.sum(np.exp(np.einsum("ijk,j->ik", self.K_per_sym, ((self.epsilon[:, self.support_dim_range_ids, self.ref_sites].T).flatten()))), axis=-1)

    def setup_fit_alpha_dep(self, minimize_fun=None, linesearch_fun=None):
        self.active_elements = self.alpha_mat_ref_sites < self.alpha_cutoff

        if self.precomputed_features:
            assert self.weights is not None
            weights = self.weights
        else:
            if self.weights is None:
                weights = (self.epsilon[:, self.support_dim_range_ids, self.ref_sites]).T.flatten()
                if self.include_bias:
                    weights = np.concatenate((weights, self.bias))
                if self.complex_expand and self.epsilon.dtype==complex:
                    weights = np.concatenate((weights.real, weights.imag))
            else:
                weights = self.weights

        if self.include_bias:
            K = np.hstack((K, np.ones((self.K_per_sym.shape[0], 1, self.K_per_sym.shape[2]))))

        if self.complex_expand and self.epsilon.dtype==complex:
            K = np.hstack((K, 1.j * K))


        self.valid_kern = _mpi_sum(np.sum(abs(K), axis=(0,-1))) > self.kern_cutoff
        self.active_elements = np.logical_and(self.active_elements, self.valid_kern)

        K = K[:, self.active_elements, :]
        weights = weights[self.active_elements]

        def get_loss(weights):
            pred = np.sum(np.exp(np.einsum("ijk,j->ik", K, weights)), axis=-1)
            self.neg_log_likelihood = self.beta * _mpi_sum(np.sum(self.weightings * abs(self.exp_amps - pred)**2))
            if self.complex_expand and self.epsilon.dtype==complex:
                loss = self.neg_log_likelihood + np.sum(self.alpha_mat_ref_sites[self.active_elements]/2 * abs(weights)**2)
            else:
                loss = self.neg_log_likelihood + np.sum(self.alpha_mat_ref_sites[self.active_elements] * abs(weights)**2)
            return loss/2

        def get_grad(weights):
            per_sym_pred = np.exp(np.einsum("ijk,j->ik", K, weights))
            pred = np.sum(per_sym_pred, axis=-1)
            pred_der = np.einsum("ijk,ik->ij", K, per_sym_pred)
            g = self.weightings * (self.exp_amps - pred).conj()
            grad = -np.einsum("ij,i->j", pred_der, g)
            grad = self.beta * _mpi_sum(grad)
            if self.complex_expand and self.epsilon.dtype==complex:
                grad = grad.real
                grad += self.alpha_mat_ref_sites[self.active_elements]/2 * weights
            else:
                grad += self.alpha_mat_ref_sites[self.active_elements] * weights
            return grad

        # TODO: This is quite slow atm, spped this up!
        def get_hessian(weights):
            per_sym_pred = np.exp(np.einsum("ijk,j->ik", K, weights))
            pred = np.sum(per_sym_pred, axis=-1)

            b_1 = per_sym_pred * np.expand_dims(self.weightings * (self.exp_amps - pred).conj(), -1)
            hess_a = -np.einsum("ijk,ik,ilk->jl", K, b_1, K)

            b_2 = np.einsum("i,ij,ik->ijk", self.weightings, per_sym_pred, per_sym_pred.conj())
            tmp = np.einsum("ikm,ilm->ikl", b_2, K.conj(), optimize=True)
            hess_b = np.einsum("ijk,ikl->jl", K, tmp, optimize=True)

            hessian = self.beta * _mpi_sum(hess_a + hess_b)

            if self.complex_expand and self.epsilon.dtype==complex:
                hessian = (hessian + np.diag(self.alpha_mat_ref_sites[self.active_elements]/2)).real
            else:
                hessian = hessian + np.diag(self.alpha_mat_ref_sites[self.active_elements])

            return hessian

        if minimize_fun is not None:
            min_result = minimize_fun(get_loss, weights, jac=get_grad, hess=get_hessian)
            weights = min_result.x

        self.KtK_alpha = get_hessian(weights)
        self.grad = get_grad(weights)
        get_loss(weights)

        self.cholesky = False
        self.Sinv = np.zeros((np.sum(self.active_elements), np.sum(self.active_elements)), dtype=self.KtK_alpha.dtype)

        update_directions = np.zeros_like(weights)
        if self.active_elements.any():
            if _rank == 0:
                with threadpool_limits(limits=self.max_threads, user_api="blas"):
                    try:
                        L = sp.linalg.cholesky(self.KtK_alpha, lower=True)
                        np.copyto(self.Sinv, sp.linalg.solve_triangular(L, np.eye(self.active_elements.sum()), check_finite=False, lower=True))
                        update_directions = -sp.linalg.cho_solve((L, True), self.grad)
                        self.cholesky = True
                    except:
                        np.copyto(self.Sinv, sp.linalg.pinvh(self.KtK_alpha))
                        update_directions = -self.Sinv.dot(self.grad)

            _MPI_comm.Bcast(self.Sinv, root=0)
            _MPI_comm.Bcast(update_directions, root=0)
            self.cholesky = _MPI_comm.bcast(self.cholesky, root=0)

            if linesearch_fun is not None:
                result = linesearch_fun(get_loss, get_grad, weights, update_directions)
                gamma = result[0]
                if gamma is None:
                    if _rank == 0:
                        print("linesearch not converged, resetting step", self.cholesky)
                    gamma = 0.
            else:
                gamma = 1.
            if self.cholesky:
                np.copyto(weights, weights + gamma * update_directions)
            else:
                if _rank == 0:
                    print("Cholesky fail")

            # This bit is just to emphasize that self.Sinv is not the inverse of sigma but its Cholesky decomposition if self.cholesky==True
            if self.cholesky:
                self.Sinv_L = self.Sinv
                self.Sinv = None

        if self.weights is None:
            if not self.complex_expand and self.epsilon.dtype==complex:
                self.weights = np.zeros(self.alpha_mat_ref_sites.shape[0], dtype=complex)
            else:
                self.weights = np.zeros(self.alpha_mat_ref_sites.shape[0], dtype=float)

        else:
            self.weights.fill(0.0)

        if self.active_elements.any() > 0:
            self.weights[self.active_elements] = weights
