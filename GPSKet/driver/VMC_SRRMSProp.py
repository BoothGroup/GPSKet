import jax
import jax.numpy as jnp
import jax.scipy as jsp
from collections.abc import Callable
from functools import partial
from textwrap import dedent
from jax.flatten_util import ravel_pytree
from netket import jax as nkjax
from netket import stats as nkstats
from netket.driver import AbstractVariationalDriver
from netket.operator import AbstractOperator
from netket.utils import mpi, timing
from netket.utils.types import Scalar, ScalarOrSchedule, Optimizer, PyTree
from netket.vqs import MCState


@timing.timed
@partial(jax.jit, static_argnames=("mode", "solver_fn"))
def SRRMSProp(
    O_L, local_energies, ema, step, diag_shift, decay, eps, *, mode, solver_fn, e_mean=None, params_structure
):
    N_mc = O_L.shape[0] * mpi.n_nodes

    # Center local energies
    local_energies = local_energies.flatten()
    if e_mean is None:
        e_mean = nkstats.mean(local_energies)
    local_energies_centered = -2*jnp.conj(local_energies - e_mean) * jnp.sqrt(N_mc)
    local_energies_centered, token = mpi.mpi_gather_jax(local_energies_centered)
    local_energies_centered = local_energies_centered.reshape(-1, *local_energies_centered.shape[2:])

    # Center jacobians
    O_mean = nkstats.mean(O_L, axis=0)
    O_L = (O_L - O_mean) * jnp.sqrt(N_mc)
    O_L, token = mpi.mpi_gather_jax(O_L, token=token)
    O_L = O_L.reshape(-1, *O_L.shape[2:])

    # Convert quantities to dense format
    if mode == "complex":
        O_L = jnp.transpose(O_L, (1, 0, 2)).reshape(-1, O_L.shape[-1])
        local_energies_centered = jnp.concatenate(
            (jnp.real(local_energies_centered), -jnp.imag(local_energies_centered)),
            axis=-1,
            dtype=jnp.float64
        )
    elif mode == "real":
        local_energies_centered = local_energies_centered.real
    else:
        raise NotImplementedError()

    if mpi.rank == 0:
        # Compute gradient and update EMA of squared gradient
        gradient = O_L.T @ local_energies_centered
        ema = decay * ema + (1. - decay) * gradient**2
        ema_hat = ema / (1. - decay**(step+1))

        # Compute S matrix and apply diagonal shift
        S = O_L.T @ O_L
        diag = jnp.diag(jnp.sqrt(ema_hat) + eps)
        S = S + diag_shift * diag

        # Solve linear system
        updates = solver_fn(S, gradient)
        if isinstance(updates, tuple):
            updates, _ = updates
    else:
        updates = jnp.zeros(local_energies_centered.shape[0], dtype=jnp.float64)
    updates, token = mpi.mpi_bcast_jax(updates, token=token, root=0)
    
    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        np = updates.shape[-1] // 2
        updates = updates[:np] + 1j * updates[np:]

    return -updates, ema

linear_solver = lambda A, b: jax.scipy.sparse.linalg.cg(A, b)

@jax.jit
def _flatten_samples(x):
    return x.reshape(-1, x.shape[-1])

class VMC_SRRMSProp(AbstractVariationalDriver):
    r"""
    Energy minimization using Variational Monte Carlo (VMC) and Stochastic Reconfiguration (SR)
    with an RMSProp regularization, as proposed by `Lovato et al. <https://link.aps.org/doi/10.1103/PhysRevResearch.4.043178>`_
    """

    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
        diag_shift: ScalarOrSchedule,
        decay: Scalar = 0.9,
        eps: Scalar = 1e-8,
        initial_scale: Scalar = 0.0,
        linear_solver_fn: Callable[[jax.Array, jax.Array], jax.Array] = linear_solver,
        jacobian_mode: str | None = None,
        variational_state: MCState = None,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the bare energy gradient.
            diag_shift: The diagonal shift of the stochastic reconfiguration matrix. Can be a float or an optax schedule.
            decay: The discount factor for old squared gradients in the regularization term. Defaults to 0.9.
            eps: A small constant for numerical stability in the inversion of the SR matrix. Defaults to 1e-8.
            initial_scale: The initial value of the EMA of the squared gradients. Defaults to 0.0.
            linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters
            jacobian_mode: The mode used to compute the jacobian of the variational state. Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).
            variational_state: The :class:`netket.vqs.MCState` to be optimised. Other variational states are not supported.
        """
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        if variational_state.hilbert != hamiltonian.hilbert:
            raise TypeError(
                dedent(
                    f"""the variational_state has hilbert space {variational_state.hilbert}
                    (this is normally defined by the hilbert space in the sampler), but
                    the hamiltonian has hilbert space {hamiltonian.hilbert}.
                    The two should match.
                    """
                )
            )

        self._ham = hamiltonian.collect()  # type: AbstractOperator
        self._dp: PyTree = None

        self.diag_shift = diag_shift
        self.decay = decay
        self.eps = eps
        self.jacobian_mode = jacobian_mode
        self._linear_solver_fn = linear_solver_fn

        self._params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not nkjax.tree_ishomogeneous(self._params_structure):
            raise ValueError(
                "SRRMSProp only supports neural networks with all real or all complex parameters"
            )
        if nkjax.tree_leaf_iscomplex(self._params_structure):
            n_params = 2*self.state.n_parameters
        else:
            n_params = self.state.n_parameters
        self._ema = jnp.full(n_params, initial_scale, jnp.float64)

        _, unravel_params_fn = ravel_pytree(self.state.parameters)
        self._unravel_params_fn = jax.jit(unravel_params_fn)

    @property
    def jacobian_mode(self) -> str:
        """
        The mode used to compute the jacobian of the variational state. Can be `'real'`
        or `'complex'`.

        Real mode truncates imaginary part of the wavefunction, while `complex` does not.
        This internally uses :func:`netket.jax.jacobian`. See that function for a more
        complete documentation.
        """
        return self._jacobian_mode

    @jacobian_mode.setter
    def jacobian_mode(self, mode: str | None):
        if mode is None:
            mode = nkjax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.samples,
                warn=False,
            )

        if mode not in ["complex", "real"]:
            raise ValueError(
                "`jacobian_mode` only supports 'real' for real-valued wavefunctions and"
                "'complex'.\n\n"
                "`holomorphic` is not yet supported, but could be contributed in the future."
            )
        self._jacobian_mode = mode

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        local_energies = self.state.local_estimators(self._ham)

        self._loss_stats = nkstats.statistics(local_energies)

        samples = _flatten_samples(self.state.samples)
        jacobians = nkjax.jacobian(
            self.state._apply_fun,
            self.state.parameters,
            samples,
            self.state.model_state,
            mode=self.jacobian_mode,
            dense=True,
            center=False,
            chunk_size=self.state.chunk_size
        )  # jacobians is NOT centered

        diag_shift = self.diag_shift
        if callable(self.diag_shift):
            diag_shift = self.diag_shift(self.step_count)

        updates, self._ema = SRRMSProp(
            jacobians,
            local_energies,
            self._ema,
            self.step_count,
            diag_shift,
            self.decay,
            self.eps,
            mode=self.jacobian_mode,
            solver_fn=self._linear_solver_fn,
            e_mean=self._loss_stats.Mean,
            params_structure=self._params_structure,
        )

        self._dp = self._unravel_params_fn(updates)

        return self._dp

    @property
    def energy(self) -> nkstats.Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "VMC_SRRMSProp("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )