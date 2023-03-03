import jax
import jax.numpy as jnp
import netket as nk
import netket.jax as nkjax
from netket.vqs.mc.mc_state.state import MCState
from GPSKet.models import qGPS
import GPSKet.vqs.mc.mc_state.expect

from typing import Optional

# dummy class used if the local energy should be evaluated on the fly (allowing for fast updating)
class HeisenbergOnTheFly(nk.operator.Heisenberg):
    pass

def get_J1_J2_Hamiltonian(Lx, Ly=None, J1=1., J2=0., sign_rule=True, total_sz=0.0, on_the_fly_en=False):
    if J2 != 0.:
        nb_order = 2
    else:
        nb_order = 1

    if Ly is None:
        g = nk.graph.Chain(Lx, max_neighbor_order=nb_order)
    else:
        g = nk.graph.Grid([Lx, Ly], max_neighbor_order=nb_order)

    hilbert = nk.hilbert.Spin(0.5, total_sz=total_sz, N=g.n_nodes)

    """
    This is a slightly hacky way to determine if dispatch rules specified below
    (evaluating the local energy on the fly and for the qGPS applying the fast update)
    are applied or not.
    """
    if on_the_fly_en:
        classtype = HeisenbergOnTheFly
    else:
        classtype = nk.operator.Heisenberg

    if J2 != 0:
        hamiltonian = classtype(hilbert, g, J=[J1/4, J2/4], sign_rule=sign_rule)
    else:
        hamiltonian = classtype(hilbert, g, J=J1/4, sign_rule=sign_rule)

    return hamiltonian


"""
This is a custom way of evaluating the expectation values for Heisenberg models.
It can make use of the fast update functionality of the qGPS ansatz.
Furthermore it can reduce the memory requirements compared to the default netket implementation as connected
configurations are not all created at once but created on the fly.
It can probably at one point also be extended beyond Heisenberg models but at the moment
it explicitly requires that each operator in the Hamiltonian acts on a pair of spins
and connects the test configuration to at most one other different configuration.
"""

def local_en_on_the_fly(states_to_local_indices, logpsi, pars, samples, args, use_fast_update=False, chunk_size=None):
    operators = args[0]
    acting_on = args[1]
    def vmap_fun(sample):
        if use_fast_update:
            log_amp, intermediates_cache = logpsi(pars, jnp.expand_dims(sample, 0), mutable="intermediates_cache", cache_intermediates=True)
            parameters = {**pars, **intermediates_cache}
        else:
            log_amp = logpsi(pars, jnp.expand_dims(sample, 0))
        def inner_vmap(operator_element, acting_on_element):
            rel_occ = sample[acting_on_element]
            basis_index = jnp.sum(states_to_local_indices(rel_occ) * jnp.array([1,2]))
            # the way this is set up at the moment is only valid for Heisenberg models where at most one non-zero off-diagonal exists
            off_diag_connected = jnp.array([0,2,1,3]) # indices of the non-zero off-diagonal element (or the diagonal index if no non-zero off-diagonal element)
            def compute_element(connected_index):
                mel = operator_element[basis_index, connected_index]
                new_occ = 2 * jnp.array([connected_index % 2, connected_index // 2]) - 1. # map back to standard netket representation of spin configurations
                if use_fast_update:
                    log_amp_connected = logpsi(parameters, jnp.expand_dims(new_occ, 0), update_sites=jnp.expand_dims(acting_on_element, 0))
                else:
                    updated_config = sample.at[acting_on_element].set(new_occ)
                    log_amp_connected = logpsi(pars, jnp.expand_dims(updated_config, 0))
                return jnp.squeeze(mel * jnp.exp(log_amp_connected - log_amp))
            # This has a bit of overhead as there is no good way of shortcutting if the non-zero element is the diagonal
            off_diag = jnp.where(off_diag_connected[basis_index] != basis_index, compute_element(off_diag_connected[basis_index]), 0.)
            return off_diag + operator_element[basis_index, basis_index]
        return jnp.sum(jax.vmap(inner_vmap)(operators, acting_on))
    return nkjax.vmap_chunked(vmap_fun, chunk_size=chunk_size)(samples)

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: HeisenbergOnTheFly):
    samples = vstate.samples
    operators = jnp.array(op.operators)
    acting_on = jnp.array(op.acting_on)
    return (samples, (operators, acting_on))

@nk.vqs.get_local_kernel.dispatch(precedence=1)
def get_local_kernel(vstate: nk.vqs.MCState, op: HeisenbergOnTheFly, chunk_size: Optional[int] = None):
    try:
        use_fast_update = vstate.model.apply_fast_update
    except:
        use_fast_update = False
    return nkjax.HashablePartial(local_en_on_the_fly, op.hilbert.states_to_local_indices, use_fast_update=use_fast_update, chunk_size=chunk_size)
