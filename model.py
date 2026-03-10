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
        super(DAE, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.LeakyReLU())
        self.decoder = nn.Linear(self.out_dim, self.in_dim)

    def forward(self, input):
        size = input.size()
        self.hidden_output = self.encoder(input)
        output = self.decoder(self.hidden_output)
        return output.view(size)


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
        super(StackDAE, self).__init__()

        stride = pow(in_dim / out_dim, 1 / layer_num)

        self.stack_enc = nn.Sequential()
        self.stack_dec = nn.Sequential()

        cur_dim = in_dim
        # Build stacked encoder
        for i in range(layer_num):
            next_dim = cur_dim / stride
            self.stack_enc.add_module(
                'encoder_%d' % i,
                nn.Sequential(nn.Linear(int(cur_dim), int(next_dim)), nn.LeakyReLU()))
            cur_dim = next_dim

        # Build stacked decoder (reverse order)
        for i in range(layer_num):
            next_dim = cur_dim * stride
            self.stack_dec.add_module(
                'decoder_%d' % (layer_num - i - 1),
                nn.Linear(int(cur_dim), int(next_dim)))
            cur_dim = next_dim

    def forward(self, input):
        self.hidden_feature = self.stack_enc(input)
        output = self.stack_dec(self.hidden_feature)
        return output

    def extract(self, input):
        """Extract bottleneck features without decoding."""
        feature = self.stack_enc(input)
        return feature
