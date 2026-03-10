"""Denoising Autoencoder model definitions."""
import torch
import torch.nn as nn


class DAE(nn.Module):
    """Single Denoising Autoencoder layer.

    Architecture:
        input (in_dim) -> encoder (Linear + LeakyReLU) -> hidden (out_dim)
                       -> decoder (Linear) -> output (in_dim)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU())
        self.decoder = nn.Linear(self.out_dim, self.in_dim)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.decoder(hidden)

    def encode(self, x):
        """Encode input to hidden representation."""
        return self.encoder(x)


class StackDAE(nn.Module):
    """Stacked Denoising Autoencoder.

    Builds encoder and decoder stacks with geometrically decreasing
    layer sizes from in_dim down to out_dim.

    Args:
        in_dim (int): input dimension
        out_dim (int): feature (bottleneck) dimension
        layer_num (int): number of encoder/decoder layers
    """

    def __init__(self, in_dim, out_dim, layer_num):
        super().__init__()

        # Pre-compute all layer dimensions to avoid float drift
        dims = self._compute_dims(in_dim, out_dim, layer_num)

        self.stack_enc = nn.Sequential()
        self.stack_dec = nn.Sequential()

        # Build stacked encoder
        for i in range(layer_num):
            self.stack_enc.add_module(
                'encoder_%d' % i,
                nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.LeakyReLU()))

        # Build stacked decoder (reverse order)
        for i in range(layer_num):
            dec_idx = layer_num - i
            self.stack_dec.add_module(
                'decoder_%d' % (layer_num - i - 1),
                nn.Linear(dims[dec_idx], dims[dec_idx - 1]))

    @staticmethod
    def _compute_dims(in_dim, out_dim, layer_num):
        """Pre-compute integer dimensions for each layer to avoid float drift.

        Returns:
            list[int]: dimensions from input to bottleneck (length = layer_num + 1)
        """
        stride = pow(in_dim / out_dim, 1 / layer_num)
        dims = [in_dim]
        cur = in_dim
        for i in range(layer_num - 1):
            cur = cur / stride
            dims.append(int(round(cur)))
        dims.append(int(round(out_dim)))  # Force exact target dim at bottleneck
        return dims

    def forward(self, x):
        """Forward pass returning (reconstruction, features) tuple."""
        features = self.stack_enc(x)
        reconstruction = self.stack_dec(features)
        return reconstruction, features

    def extract(self, x):
        """Extract bottleneck features without decoding."""
        return self.stack_enc(x)
