import jax
import numpy as np
import netket.jax as nkjax
from textwrap import dedent
from typing import Optional, Tuple
from netket.utils import mpi
from netket.utils.types import SeedT, Array
from netket.vqs import MCState
from netket.operator import AbstractOperator
from netket.driver import AbstractVariationalDriver


class AbstractStateFittingDriver(AbstractVariationalDriver):
    """Abstract base class for State Fitting drivers"""


    def __init__(
        self,
        dataset: Tuple[Array, Array],
        hamiltonian: AbstractOperator,
        optimizer,
        *args,
        variational_state=None,
        mini_batch_size: int = 32,
        seed: Optional[SeedT]=None,
        **kwargs):
        if variational_state is None:
            variational_state = MCState(*args, **kwargs)

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

        super().__init__(variational_state, optimizer, minimized_quantity_name="Loss")
        # TODO: maybe shard the dataset over MPI ranks
        batches = jax.tree_util.tree_map(lambda arr: np.array_split(arr, self._mpi_nodes), dataset)
        batches = jax.tree_util.tree_map(lambda *tup: mpi.mpi_bcast(tup, root=0), *batches)

        self._dataset = batches[mpi.rank]
        self._ham = hamiltonian.collect() 
        self._seed = seed
        self._mini_batch_size = mini_batch_size

        self._size_dataset = self._dataset[0].shape[0]
        self._key = nkjax.mpi_split(nkjax.PRNGKey(self._seed))

    def reset(self):
        self._key = nkjax.mpi_split(nkjax.PRNGKey(self._seed))
        super().reset()

    @property
    def loss(self):
        return self._loss_stats

    @loss.setter
    def loss(self, value):
        self._loss_stats = value

    