from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp
import netket as nk
from netket.utils.types import PyTree
from netket.stats import Stats
from netket import jax as nkjax
from netket.vqs.mc.mc_state.state import MCState
from qGPSKet.models import qGPS

"""
This is a custom way of evaluating the expectation values for Heisenberg models.
It can make use of the fast update functionality of the qGPS ansatz.
Furthermore it requires less memory than the default netket implementation as connected
configurations are not all created at once but created on the fly.
It can probably at one point also be extended beyond Heisenberg models but at the moment
it explicitly requires that each operator in the Hamiltonian acts on a pair of spins
and connects the test configuration to at most one other different configuration.
"""

tensor_basis_mapping = jnp.array([[0,0], [1,0], [0,1], [1,1]], dtype=jnp.uint8)
off_diag_connected = jnp.array([0,2,1,3], dtype=jnp.uint8)
index_multiplicator = jnp.arange(1,3, dtype=jnp.uint8)

def local_en(logpsi, pars, samples, args, use_fast_update=False):
    operators = args[0]
    acting_on = args[1]
    def vmap_fun(sample):
        if use_fast_update:
            log_amp, workspace = logpsi(pars, sample, mutable="workspace", save_site_prod=True)
            parameters = {**pars, **workspace}
        else:
            log_amp = logpsi(pars, sample)
        def scan_fun(carry, index):
            acting_on_element = acting_on[index]
            operator_element = operators[index]
            rel_occ = sample[acting_on_element]
            basis_index = jnp.sum(rel_occ.astype(jnp.uint8)*index_multiplicator)
            # the way this is set up at the moment is only valid for Heisenberg models where at most one non-zero off-diagonal exists
            def compute_element(connected_index):
                mel = operator_element[basis_index, connected_index]
                new_occ = 2*tensor_basis_mapping[connected_index]-1.
                if use_fast_update:
                    log_amp_connected = logpsi(parameters, new_occ, update_sites=acting_on_element)
                else:
                    updated_config = sample.at[acting_on_element].set(new_occ)
                    log_amp_connected = logpsi(pars, updated_config)
                return jnp.squeeze(mel * jnp.exp(log_amp_connected - log_amp)).astype(complex)
            off_diag_index = off_diag_connected[basis_index]
            off_diag = jax.lax.cond(off_diag_index != basis_index, compute_element, lambda x: jnp.array(0, dtype=complex), off_diag_index)
            value = operator_element[basis_index, basis_index] + off_diag
            return (carry + value, None)
        return jax.lax.scan(scan_fun, 0., jnp.arange(acting_on.shape[0]))[0]

    local_en = jax.vmap(vmap_fun)(samples)
    return local_en

def local_en_fast_update(logpsi, pars, samples, args):
    return local_en(logpsi, pars, samples, args, use_fast_update=True)

def local_en_without_update(logpsi, pars, samples, args):
    return local_en(logpsi, pars, samples, args, use_fast_update=False)

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: nk.operator._hamiltonian.Heisenberg):
    samples = vstate.samples
    operators = jnp.array(op.operators)
    acting_on = jnp.array(op.acting_on)
    return (samples, (operators, acting_on))


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: nk.operator._hamiltonian.Heisenberg):
    use_fast_update = isinstance(vstate.model, qGPS)
    return (local_en_fast_update if use_fast_update else local_en_without_update)


@nk.vqs.expect.dispatch
def expect(vstate: MCState, op: nk.operator._hamiltonian.Heisenberg) -> Stats:  # noqa: F811
    samples, args = get_local_kernel_arguments(vstate, op)
    local_estimator_fun = get_local_kernel(vstate, op)
    return _expect(
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        samples,
        args,
    )


@partial(jax.jit, static_argnums=(0, 1))
def _expect(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    samples: jnp.ndarray,
    local_value_args: PyTree,
) -> Stats:
    samples_shape = samples.shape

    if jnp.ndim(samples) != 2:
        samples = samples.reshape((-1, samples_shape[-1]))

    def log_pdf(w, samples):
        return machine_pow * model_apply_fun({"params": w, **model_state}, samples).real

    _, op_stats = nkjax.expect(
        log_pdf,
        partial(local_value_kernel, model_apply_fun),
        {"params": parameters, **model_state},
        samples,
        local_value_args,
        n_chains=samples_shape[0],
    )

    return op_stats
