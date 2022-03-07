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

class QGPSLearning():
    def __init__(self, epsilon, init_alpha=1.0, complex_expand=False):
        self.epsilon = np.array(epsilon)
        self.K = None
        self.weights = None
        self.site_prod = None
        self.confs = None
        self.ref_site = None
        self.complex_expand = complex_expand
        self.local_dim = epsilon.shape[0]

        if self.complex_expand and self.epsilon.dtype==complex:
            self.alpha_mat = np.ones((epsilon.shape[-1], self.local_dim*2*epsilon.shape[1]))*init_alpha
        else:
            self.alpha_mat = np.ones((epsilon.shape[-1], self.local_dim*epsilon.shape[1]))*init_alpha

        self.alpha_cutoff = 1.e10
        self.kern_cutoff = 1.e-10
        self.alpha_convergence_tol = 1.e-15

    @staticmethod
    @njit()
    def kernel_mat_inner(site_prod, conns, K, ref_site):
        K = K.reshape(site_prod.shape[0], site_prod.shape[1], -1)
        K.fill(0.0)
        for i in range(site_prod.shape[0]):
            K[i, :, conns[i, ref_site]] += site_prod[i, :]
        return K.reshape(site_prod.shape[0], -1)

    @staticmethod
    @njit()
    def compute_site_prod_fast(epsilon, ref_site, conns, site_product):
        site_product.fill(1.0)
        for i in range(conns.shape[0]):
            for w in range(epsilon.shape[1]):
                for j in range(conns.shape[1]):
                    if j != ref_site:
                        site_product[i, w] *= epsilon[conns[i, j], w, j]
        return site_product

    @staticmethod
    @njit()
    def update_site_prod_fast(epsilon, ref_site, ref_site_old, confs, site_product):
        eps = 1.e2 * np.finfo(np.double).eps
        for w in range(epsilon.shape[1]):
            if np.min(np.abs(epsilon[:, w, ref_site])) > eps:
                for i in range(confs.shape[0]):
                    site_product[i, w] /= epsilon[confs[i, ref_site], w, ref_site]
                    site_product[i, w] *= epsilon[confs[i, ref_site_old], w, ref_site_old]
            else:
                for i in range(confs.shape[0]):
                    for j in range(confs.shape[1]):
                        if j != ref_site:
                            site_product[i, w] *= epsilon[confs[i, j], w, j]

        return site_product

    def compute_site_prod(self):
        self.site_prod = np.zeros((self.confs.shape[0], self.epsilon.shape[1]), dtype=self.epsilon.dtype)
        self.site_prod = self.compute_site_prod_fast(self.epsilon, self.ref_site, self.confs, self.site_prod)
        self.site_prod_ref_site = self.ref_site

    def update_site_prod(self):
        if self.site_prod_ref_site != self.ref_site:
            self.site_prod = self.update_site_prod_fast(self.epsilon, self.ref_site, self.site_prod_ref_site, self.confs,
                                                        self.site_prod)
        self.site_prod_ref_site = self.ref_site

    def set_kernel_mat(self, confs, update_K=False):
        assert(self.ref_site is not None)
        if self.site_prod is None or self.confs is None or np.sum(self.confs != confs) != 0:
            self.confs = confs
            self.compute_site_prod()
            update_K = True
        elif self.ref_site != self.site_prod_ref_site:
            self.update_site_prod()
            update_K = True

        if self.K is None:
            self.K = np.zeros((confs.shape[0], self.epsilon.shape[1] * self.local_dim), dtype=self.epsilon.dtype)
            update_K = True

        if update_K:
            self.K = self.kernel_mat_inner(self.site_prod, self.confs, self.K, self.ref_site)

        return self.K

    def reset(self):
        self.site_prod = None
        self.K = None

    def setup_fit_alpha_dep(self):
        self.active_elements = self.alpha_mat[self.ref_site,:] < self.alpha_cutoff
        if self.complex_expand and self.epsilon.dtype==complex:
            self.KtK_alpha = self.KtK + np.diag(self.alpha_mat[self.ref_site,:]/2)
        else:
            self.KtK_alpha = self.KtK + np.diag(self.alpha_mat[self.ref_site,:])
        self.valid_kern = abs(np.diag(self.KtK)) > self.kern_cutoff

        if np.sum(self.active_elements) > 0:
            self.Sinv = sp.linalg.pinvh(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)])
            weights = self.Sinv.dot(self.y[self.active_elements])
        else:
            self.Sinv = np.zeros((0,0))

        if self.weights is None:
            if not self.complex_expand and self.epsilon.dtype==complex:
                self.weights = np.zeros(self.alpha_mat.shape[1], dtype=complex)
            else:
                self.weights = np.zeros(self.alpha_mat.shape[1], dtype=float)

        else:
            self.weights.fill(0.0)

        if np.sum(self.active_elements) > 0:
            # potentially distribute weights across processes
            self.weights[self.active_elements] = weights

    def log_marg_lik_alpha_der(self):
        derivative_alpha = 1/(self.alpha_mat[self.ref_site, :])

        if self.complex_expand and self.epsilon.dtype==complex:
            derivative_alpha[self.active_elements] -= np.diag(0.5 * self.Sinv).real
        else:
            derivative_alpha[self.active_elements] -= np.diag(self.Sinv).real

        derivative_alpha -= (self.weights.conj() * self.weights).real

        if self.complex_expand or self.epsilon.dtype==float:
            derivative_alpha *= 0.5

        return derivative_alpha.real

    def set_up_prediction(self, confset):
        if self.ref_site is None:
            self.ref_site = 0

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

    def update_epsilon_with_weights(self):
        weights = np.where(self.valid_kern, self.weights, (self.epsilon[:, :, self.ref_site]).T.flatten())
        # weights = self.weights
        self.epsilon[:, :, self.ref_site] = weights[:self.local_dim*self.epsilon.shape[1]].reshape(self.epsilon.shape[1], self.local_dim).T

        if self.complex_expand and self.epsilon.dtype==complex:
            self.epsilon[:, :, self.ref_site] += 1.j * weights[self.local_dim*self.epsilon.shape[1]:].reshape(self.epsilon.shape[1], self.local_dim).T



class QGPSLearningExp(QGPSLearning):
    def __init__(self, epsilon, init_alpha = 1.0, init_noise_tilde = 1.e-1, complex_expand=False):
        super().__init__(epsilon, init_alpha=init_alpha, complex_expand=complex_expand)

        self.noise_tilde = init_noise_tilde

    def predict(self, confset):
        assert(confset.size > 0)
        self.set_up_prediction(confset)
        return np.exp(self.K.dot((self.epsilon[:, :, self.ref_site].T).flatten()))

    def setup_fit_noise_dep(self, weightings=None):
        if self.noise_tilde == 0.:
            self.S_diag = np.ones(len(self.exp_amps))
        else:
            self.S_diag = 1/(np.log1p(self.noise_tilde/(abs(self.exp_amps)**2)))
        if weightings is not None:
            self.S_diag *= weightings
        self.weightings = weightings

        self.KtK = _mpi_sum(np.dot(self.K.conj().T, np.einsum("i,ij->ij", self.S_diag, self.K)))

        self.y = _mpi_sum(self.K.conj().T.dot(self.S_diag * self.fit_data))

        if self.complex_expand and self.epsilon.dtype==complex:
            self.KtK = np.block([[self.KtK.real, -self.KtK.imag],[self.KtK.imag, self.KtK.real]])
            self.y = np.concatenate((self.y.real, self.y.imag))

        self.setup_fit_alpha_dep()

    def setup_fit(self, confset, target_amplitudes, ref_site, weightings=None):
        self.ref_site = ref_site
        self.exp_amps = target_amplitudes.astype(self.epsilon.dtype)
        if self.epsilon.dtype == float:
            self.fit_data = np.log(abs(self.exp_amps))
        else:
            self.fit_data = np.log(self.exp_amps)
        self.set_kernel_mat(confset)
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

        if self.complex_expand and self.epsilon.dtype==complex:
            log_lik += 0.5 * np.linalg.slogdet(0.5 * self.Sinv)[1]
        else:
            log_lik += np.linalg.slogdet(self.Sinv)[1]

        if self.complex_expand and self.epsilon.dtype==complex:
            log_lik += 0.5 * np.sum(np.log(self.alpha_mat[self.ref_site, self.active_elements]))
        else:
            log_lik += np.sum(np.log(self.alpha_mat[self.ref_site, self.active_elements]))

        weights = self.weights[self.active_elements]
        log_lik += np.dot(weights.conj(), np.dot(self.KtK_alpha[np.ix_(self.active_elements, self.active_elements)], weights))

        if self.epsilon.dtype==float:
            log_lik *= 0.5

        return log_lik.real

    def log_marg_lik_noise_der(self):
        del_S = 1/((abs(self.exp_amps)**2) * (1 + self.noise_tilde/(abs(self.exp_amps)**2)))
        Delta_S = - (self.S_diag**2 * del_S)

        if self.weightings is not None:
            Delta_S /= self.weightings

        derivative_noise = -np.sum(self.S_diag * del_S)

        K = self.K
        KtK_der = self.K.conj().T.dot(np.einsum("i,ij->ij", Delta_S, self.K))

        if self.complex_expand and self.epsilon.dtype==complex:
            KtK_der = np.block([[KtK_der.real, -KtK_der.imag],[KtK_der.imag, KtK_der.real]])
            K = np.hstack((K, 1.j * K))

        K = K[:,self.active_elements]
        KtK_der = KtK_der[np.ix_(self.active_elements, self.active_elements)]

        if self.complex_expand and self.epsilon.dtype==complex:
            derivative_noise -= 0.5 * np.trace(KtK_der.dot(self.Sinv))
        else:
            derivative_noise -= np.trace(KtK_der.dot(self.Sinv))

        weights = self.weights[self.active_elements]

        derivative_noise -= self.fit_data.conj().dot(Delta_S*self.fit_data)
        derivative_noise -= weights.conj().dot(KtK_der.dot(weights))

        derivative_noise += 2*self.fit_data.conj().dot(Delta_S*K.dot(weights))

        derivative_noise = _MPI_comm.allreduce(derivative_noise)

        if self.epsilon.dtype==float:
            derivative_noise *= 0.5

        return derivative_noise.real

    def opt_alpha(self, max_iterations=None, rvm=False):
        alpha_old = self.alpha_mat[self.ref_site, :].copy()
        converged = False
        j = 0
        if max_iterations is not None:
            if j >= max_iterations:
                converged = True
        while not converged:
            if not np.max(self.active_elements):
                converged = True
            else:
                if self.complex_expand and self.epsilon.dtype==complex:
                    gamma = (1 - (self.alpha_mat[self.ref_site, self.active_elements])*0.5*np.diag(self.Sinv).real)
                else:
                    gamma = (1 - (self.alpha_mat[self.ref_site, self.active_elements])*np.diag(self.Sinv).real)
                if rvm:
                    self.alpha_mat[self.ref_site, self.active_elements] = (gamma/((self.weights.conj()*self.weights)[self.active_elements])).real
                else:
                    self.alpha_mat[self.ref_site, self.active_elements] = ((np.sum(gamma)/(self.weights.conj().dot(self.weights))).real)
                self.setup_fit_alpha_dep()
                j += 1
                if np.sum(abs(self.alpha_mat[self.ref_site, :] - alpha_old)**2) < self.alpha_convergence_tol:
                    converged = True
                np.copyto(alpha_old, self.alpha_mat[self.ref_site, :])
                if max_iterations is not None:
                    if j >= max_iterations:
                        converged = True

    def fit_step(self, confset, target_amplitudes, ref_site, noise_bounds=[(None, None)],
                 opt_alpha=True, opt_noise=True, max_alpha_iterations=None, max_noise_iterations=None, rvm=False,
                 weightings=None):
        self.setup_fit(confset, target_amplitudes, ref_site, weightings=weightings)
        if opt_noise:
            alpha_init = self.alpha_mat.copy()
            def ML(x):
                self.noise_tilde = np.exp(x[0])
                if opt_alpha:
                    np.copyto(self.alpha_mat, alpha_init)
                self.setup_fit_noise_dep(weightings=weightings)
                if opt_alpha:
                    self.opt_alpha(max_iterations=max_alpha_iterations, rvm=rvm)
                return -self.log_marg_lik()

            def derivative(x):
                self.noise_tilde = np.exp(x[0])
                if opt_alpha:
                    np.copyto(self.alpha_mat, alpha_init)
                self.setup_fit_noise_dep(weightings=weightings)
                if opt_alpha:
                    self.opt_alpha(max_iterations=max_alpha_iterations, rvm=rvm)
                der_noise = self.log_marg_lik_noise_der()
                return - der_noise * np.exp(x)

            def update_alpha(x):
                self.noise_tilde = np.exp(x[0])
                if opt_alpha:
                    self.opt_alpha(max_iterations=max_alpha_iterations, rvm=rvm)
                np.copyto(alpha_init, self.alpha_mat)

            if max_noise_iterations is not None:
                opt = sp.optimize.minimize(ML, np.log(self.noise_tilde), options={"maxiter" : max_noise_iterations}, jac=derivative, bounds=noise_bounds, callback=update_alpha)
            else:
                opt = sp.optimize.minimize(ML, np.log(self.noise_tilde), jac=derivative, bounds=noise_bounds, callback=update_alpha)

            self.noise_tilde = np.exp(opt.x)[0]
            if opt_alpha:
                np.copyto(self.alpha_mat, alpha_init)
            self.setup_fit_noise_dep(weightings=weightings)

        if opt_alpha:
            self.opt_alpha(max_iterations=max_alpha_iterations, rvm=rvm)

        self.update_epsilon_with_weights()
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
            qi[self.active_elements] = self.alpha_mat[self.ref_site, self.active_elements] * Qa / (self.alpha_mat[self.ref_site, self.active_elements] - 2*Sa)
            si[self.active_elements] = self.alpha_mat[self.ref_site, self.active_elements] * Sa / (self.alpha_mat[self.ref_site, self.active_elements] - 2*Sa)
        else:
            qi[self.active_elements] = self.alpha_mat[self.ref_site, self.active_elements] * Qa / (self.alpha_mat[self.ref_site, self.active_elements] - Sa)
            si[self.active_elements] = self.alpha_mat[self.ref_site, self.active_elements] * Sa / (self.alpha_mat[self.ref_site, self.active_elements] - Sa)
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
            Qrec, Srec, Arec = Q[recompute], S[recompute], self.alpha_mat[self.ref_site, recompute]/2
            Qdel, Sdel, Adel = Q[delete], S[delete], self.alpha_mat[self.ref_site, delete]/2
        else:
            Qrec, Srec, Arec = Q[recompute], S[recompute], self.alpha_mat[self.ref_site, recompute]
            Qdel, Sdel, Adel = Q[delete], S[delete], self.alpha_mat[self.ref_site, delete]

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

        if theta[feature_index] > 0:
            if self.complex_expand and self.epsilon.dtype==complex:
                self.alpha_mat[self.ref_site, feature_index] = 2 * (s[feature_index] ** 2 / theta[feature_index])
            else:
                self.alpha_mat[self.ref_site, feature_index] = s[feature_index] ** 2 / theta[feature_index]
        else:
            # at least one active features
            if self.active_elements[feature_index] == True and np.sum(self.active_elements) >= 2:
                self.alpha_mat[self.ref_site, feature_index] = np.PINF

        return



    def fit_step_growing_RVM(self, confset, target_amplitudes, ref_site,alpha_iterations=None, multiplication=None, weightings=None):
        self.setup_fit(confset, target_amplitudes, ref_site, multiplication=multiplication, weightings=weightings)

        if np.max(self.active_elements) == 0:
            if np.min(abs(np.diag(self.KtK))) < np.finfo(np.float32).eps:
                self.alpha_mat[self.ref_site, 0] = np.finfo(np.float32).eps
            else:
                projections = (abs(self.y) **2 / np.diag(self.KtK))
                ind = np.argmax(projections)

                alpha_est = (((np.diag(self.KtK))**2 / (abs(self.y)**2 - np.diag(self.KtK))).real)[ind]

                if alpha_est > 0.:
                    self.alpha_mat[self.ref_site, ind] = alpha_est
                    if self.complex_expand and self.epsilon.dtype==complex:
                        self.alpha_mat[self.ref_site, ind] *= 2
                else:
                    self.alpha_mat[self.ref_site, ind] = 1.
                    if self.complex_expand and self.epsilon.dtype==complex:
                        self.alpha_mat[self.ref_site, ind] *= 2
                    print(alpha_est)
            self.setup_fit_alpha_dep()

        for i in range(alpha_iterations):
            s, q, S, Q = self.compute_sparsity_quantities()
            self.update_precisions(s, q, S, Q)
            self.setup_fit_alpha_dep()

        self.update_epsilon_with_weights()

        return