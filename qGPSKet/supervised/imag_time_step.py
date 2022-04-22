import netket as nk
import jax
import jax.numpy as jnp

from functools import partial

class ImagTimeStep():
    def __init__(self, vstate, hamiltonian):
        loc_kernel = partial(nk.vqs.get_local_kernel(vstate, hamiltonian), vstate._apply_fun)
        def loc_en(samples, parameters, args):
            if len(samples.shape) != 2:
                samples = samples.reshape((-1, samples.shape[-1]))
            return loc_kernel({"params": parameters}, samples, args)
        self.loc_ens = jax.jit(loc_en)
        self.vstate = vstate
        self.hamiltonian = hamiltonian

    def get_local_energies(self, samples):
        # This is a little bit hacky, the interface should probably be improved at one point
        old_samples = self.vstate._samples
        self.vstate._samples = samples
        _, args = nk.vqs.get_local_kernel_arguments(self.vstate, self.hamiltonian)
        loc_ens = self.loc_ens(samples, self.vstate.parameters, args)
        self.vstate._samples = old_samples
        return loc_ens

    def log_imag_time_step(self, tau, samples):
        samples_reshaped = samples.reshape((-1, samples.shape[-1]))
        self.log_amps = self.vstate.log_value(samples_reshaped)
        self.local_energies = self.get_local_energies(samples_reshaped)
        return self.log_amps + jnp.log(1 - tau * self.local_energies)