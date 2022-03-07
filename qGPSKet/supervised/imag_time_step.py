import netket as nk
import jax
import jax.numpy as jnp

from functools import partial

class ImagTimeStep():
    def __init__(self, vstate, hamiltonian):
        _, args = nk.vqs.get_local_kernel_arguments(vstate, hamiltonian)
        loc_kernel = partial(nk.vqs.get_local_kernel(vstate, hamiltonian), vstate._apply_fun)
        def loc_en(samples, parameters):
            if len(samples.shape) != 2:
                samples = samples.reshape((-1, samples.shape[-1]))
            return loc_kernel({"params": parameters}, samples, args)
        self.loc_ens = jax.jit(loc_en)
        self.vstate = vstate

    def get_local_energies(self, samples):
        return self.loc_ens(samples, self.vstate.parameters)

    def log_imag_time_step(self, tau, samples):
        samples_reshaped = samples.reshape((-1, samples.shape[-1]))
        log_amps = self.vstate.log_value(samples_reshaped)
        loc_ens = self.get_local_energies(samples_reshaped)
        return log_amps + jnp.log(1 - tau * loc_ens)