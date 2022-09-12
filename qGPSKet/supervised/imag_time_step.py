import netket as nk
import jax
import jax.numpy as jnp

from functools import partial

class ImagTimeStep():
    def __init__(self, vstate, hamiltonian):
        self.vstate = vstate
        self.hamiltonian = hamiltonian

    def get_local_energies(self, samples):
        # This is a little bit hacky, the interface should probably be improved at one point
        old_samples = self.vstate._samples
        self.vstate._samples = samples
        loc_ens = self.vstate.local_estimators(self.hamiltonian, chunk_size=self.vstate.chunk_size)
        self.vstate._samples = old_samples
        return loc_ens

    def log_imag_time_step(self, tau, samples):
        samples_reshaped = samples.reshape((-1, samples.shape[-1]))
        self.log_amps = self.vstate.log_value(samples_reshaped)
        self.local_energies = self.get_local_energies(samples_reshaped)
        return self.log_amps + jnp.log(1 - tau * self.local_energies)