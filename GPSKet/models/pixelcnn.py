import jax.numpy as jnp
import netket as nk
import flax.linen as nn
from math import sqrt
from typing import Optional
from jax.nn.initializers import lecun_normal, zeros
from netket.hilbert.homogeneous import HomogeneousHilbert
from netket.utils.types import Array, DType, NNInitFunc, Callable
from netket.models.autoreg import _normalize
from netket.nn import MaskedConv2D
from GPSKet.nn import VerticalStackConv, HorizontalStackConv, CausalConv2d


default_kernel_init = lecun_normal()

class AbstractARNN(nk.models.AbstractARNN):
    """Overrides the abstract class from NetKet in order to allow constrained Hilbert spaces"""

    def __post_init__(self):
        nn.Module.__post_init__(self)

        if not isinstance(self.hilbert, HomogeneousHilbert):
            raise ValueError(
                f"Only homogeneous Hilbert spaces are supported by ARNN, but hilbert is a {type(self.hilbert)}."
            )

class PixelCNN(AbstractARNN):
    """
    Autoregressive wave function Ansatz based on the PixelCNN generative model
    """

    hilbert: HomogeneousHilbert
    """The Hilbert space. Only homogeneous Hilbert spaces are supported."""
    machine_pow: int = 2
    """Exponent required to normalize the output"""
    param_dtype: DType = jnp.float32
    """Type of the variational parameters"""
    kernel_size: int = 3
    """Size of the 2D convolutional kernel"""
    n_channels: int = 32
    """Number of channels in the convolutional filter"""
    depth: int = 10
    """Number of layers in the network"""
    normalize: bool = True
    """Whether the Ansatz should be normalized"""
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the convolutional kernel"""
    bias_init: NNInitFunc = zeros
    """Initializer for the bias"""
    gauge_fn: Optional[Callable] = None
    """Function that computes the value of a gauge symmetry"""
    constraint_fn: Optional[Callable] = None
    """Function that check whether a gauge constraint is broken or not"""
    # TODO: add support for symmetries

    # Dimensions:
    # - B = batch size
    # - D = local dimension
    # - N = number of size, i.e. Hilbert space size
    # - L = number of sites per linear dimension
    # - T = number of symmetries
    
    def setup(self):
        # Set system dimensions
        self._D = self.hilbert.local_size
        self._N = self.hilbert.size
        self._L = int(sqrt(self._N))
        if self._L**2 != self._N:
            raise ValueError(f"Number of sites ({self._N}) is not a square number")
        
        # Setup layers
        self._activation = nk.nn.activation.reim_relu
        self._v_stack_conv = VerticalStackConv(
            self.n_channels,
            self.kernel_size,
            mask_center=True,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        self._h_stack_conv = HorizontalStackConv(
            self.n_channels,
            self.kernel_size,
            mask_center=True,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        self._causal_conv_layers = [
            CausalConv2d(
                n_channels=self.n_channels,
                kernel_size=self.kernel_size,
                activation=self._activation,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init
            ) for _ in range(self.depth)
        ]
        self._final_conv = MaskedConv2D(
            features=self._D,
            kernel_size=(1, 1),
            kernel_dilation=(1, 1),
            exclusive=False,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
    
    def conditionals_log_psi(self, inputs: Array) -> Array:
        # Compute log probabilities by propagating inputs through the network
        batch_size = inputs.shape[0]
        x = jnp.reshape(inputs, (batch_size, self._L, self._L, 1))
        v_stack, h_stack = self._v_stack_conv(x), self._h_stack_conv(x)
        for layer in self._causal_conv_layers:
            v_stack, h_stack = layer(v_stack, h_stack)
        out = self._activation(h_stack)
        out = self._final_conv(out) # (B, L, L, D)
        log_psi = jnp.reshape(out, (batch_size, self._N, self._D)) # (B, N, D)
        
        # Enforce gauge symmetry by setting log probabilities to -inf where gauge is broken
        if self.gauge_fn is not None and self.constraint_fn is not None:
            gauge = self.gauge_fn(inputs)
            is_broken = self.constraint_fn(gauge)
            log_psi = jnp.where(is_broken, -jnp.inf, log_psi)

        # Normalize log probabilities
        if self.normalize:
            log_psi = _normalize(log_psi, self.machine_pow)
        return log_psi
