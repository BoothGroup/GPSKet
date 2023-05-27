import numpy as np
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, zeros
from netket.utils.types import Callable, DType, Array, NNInitFunc


default_kernel_init = lecun_normal()

# Part of the code was inspired by the tutorial on autoregressive image modelling at
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial12/Autoregressive_Image_Modeling.html

class MaskedConv2D(nn.Module):
    features: int
    mask: np.ndarray
    dilation: int = 1
    param_dtype: DType = jnp.float32
    kernel_init: NNInitFunc = default_kernel_init
    bias_init: NNInitFunc = zeros

    @nn.compact
    def __call__(self, x):
        # Flax's convolution module already supports masking
        # The mask must be the same size as kernel
        # => extend over input and output feature channels
        if len(self.mask.shape) == 2:
            mask_ext = self.mask[..., None, None]
            mask_ext = jnp.tile(mask_ext, (1, 1, x.shape[-1], self.features))
        else:
            mask_ext = self.mask
        # Convolution with masking
        x = nn.Conv(
            features=self.features,
            kernel_size=self.mask.shape[:2],
            kernel_dilation=self.dilation,
            mask=mask_ext,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )(x)
        return x

class VerticalStackConv(nn.Module):
    features: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1
    param_dtype: DType = jnp.float32
    kernel_init: NNInitFunc = default_kernel_init
    bias_init: NNInitFunc = zeros

    def setup(self):
        # Mask out all sites on the same row
        mask = np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)
        mask[self.kernel_size//2+1:, :] = 0
        if self.mask_center:
            mask[self.kernel_size//2, :] = 0
        self.conv = MaskedConv2D(
            features=self.features,
            mask=mask,
            dilation=self.dilation,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )

    def __call__(self, x):
        return self.conv(x)

class HorizontalStackConv(nn.Module):
    features: int
    kernel_size: int
    mask_center: bool = False
    dilation: int = 1
    param_dtype: DType = jnp.float32
    kernel_init: NNInitFunc = default_kernel_init
    bias_init: NNInitFunc = zeros

    def setup(self):
        # Mask out all sites on the left of the same row
        mask = np.ones((1, self.kernel_size), dtype=np.float32)
        mask[0, self.kernel_size//2+1:] = 0
        if self.mask_center:
            mask[0, self.kernel_size//2] = 0
        self.conv = MaskedConv2D(
            features=self.features,
            mask=mask,
            dilation=self.dilation,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )

    def __call__(self, x: Array) -> Array:
        return self.conv(x)
    
class CausalConv2d(nn.Module):
    n_channels: int = 32
    kernel_size: int = 3
    activation: Callable = nn.relu
    param_dtype: DType = jnp.float32
    kernel_init: NNInitFunc = default_kernel_init
    bias_init: NNInitFunc = zeros

    def setup(self):
        # Convolutions
        self.conv_v = VerticalStackConv(
            features=self.n_channels,
            kernel_size=self.kernel_size,
            mask_center=False,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        self.conv_h = HorizontalStackConv(
            features=self.n_channels,
            kernel_size=self.kernel_size,
            mask_center=False,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        self.conv_v_to_h = nn.Conv(
            features=self.n_channels,
            kernel_size=(1, 1),
            kernel_dilation=(1, 1),
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )
        self.conv_h_to_1x1 = nn.Conv(
            features=self.n_channels,
            kernel_size=(1, 1),
            kernel_dilation=(1, 1),
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init
        )

    def __call__(self, v_stack: Array, h_stack: Array) -> Array:
        # Vertical stack
        v_features = self.conv_v(v_stack)
        v_out = self.activation(v_features)

        # Horizontal stack
        h_features = self.conv_h(h_stack)
        h_features = h_features + self.conv_v_to_h(v_features)
        h_features = self.activation(h_features)
        h_out = h_stack + self.conv_h_to_1x1(h_features)

        return v_out, h_out
