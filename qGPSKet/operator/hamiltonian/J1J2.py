import jax
import jax.numpy as jnp
import netket as nk
import netket.jax as nkjax
from netket.vqs.mc.mc_state.state import MCState
from qGPSKet.models import qGPS
import qGPSKet.vqs.mc.mc_state.expect

from typing import Optional

# dummy class used if the local energy should be evaluated on the fly (allowing for fast updating)
class HeisenbergOnTheFly(nk.operator.Heisenberg):
    pass

def edges_2D(Lx, Ly, next_neighbours=True):
    edges = []
    for i in range(Ly):
        for j in range(Lx):
            edges.append([i * Lx + j, i * Lx + (j+1)%Lx, 0])
            edges.append([i * Lx + j, ((i+1)%Ly) * Lx + j, 0])
            if next_neighbours:
                edges.append([i * Lx + j, ((i+1)%Ly) * Lx + (j+1)%Lx, 1])
                edges.append([i * Lx + j, ((i+1)%Ly) * Lx + (j-1)%Lx, 1])
    return edges

def edges_1D(L, next_neighbours=True):
    edges = []
    for i in range(L):
        edges.append([i, (i + 1)%L, 0])
        if next_neighbours:
            edges.append([i, (i + 2)%L, 1])
    return edges

def get_J1_J2_Hamiltonian(Lx, Ly=None, J1=1., J2=0., sign_rule=True, total_sz=0.0, on_the_fly_en=False):
    if J2 != 0.:
        next_neighbours = True
    else:
        next_neighbours = False

    if Ly is None:
        edges = edges_1D(Lx, next_neighbours=next_neighbours)
    else:
        edges = edges_2D(Lx, Ly, next_neighbours=next_neighbours)

    g = nk.graph.Graph(edges=edges)
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
Furthermore it requires less memory than the default netket implementation as connected
configurations are not all created at once but created on the fly.
It can probably at one point also be extended beyond Heisenberg models but at the moment
it explicitly requires that each operator in the Hamiltonian acts on a pair of spins
and connects the test configuration to at most one other different configuration.
"""

tensor_basis_mapping = jnp.array([[0,0], [1,0], [0,1], [1,1]], dtype=jnp.uint8)
off_diag_connected = jnp.array([0,2,1,3], dtype=jnp.uint8)
index_multiplicator = jnp.arange(1,3, dtype=jnp.uint8)

def local_en_on_the_fly(logpsi, pars, samples, args, use_fast_update=False, chunk_size=None):
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

    return nkjax.vmap_chunked(vmap_fun, chunk_size=chunk_size)(samples)

@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: HeisenbergOnTheFly):
    samples = vstate.samples
    operators = jnp.array(op.operators)
    acting_on = jnp.array(op.acting_on)
    return (samples, (operators, acting_on))

@nk.vqs.get_local_kernel.dispatch(precedence=1)
def get_local_kernel(vstate: nk.vqs.MCState, op: HeisenbergOnTheFly, chunk_size: Optional[int] = None):
    use_fast_update = isinstance(vstate.model, qGPS)
    return nkjax.HashablePartial(local_en_on_the_fly, use_fast_update=use_fast_update, chunk_size=chunk_size)
