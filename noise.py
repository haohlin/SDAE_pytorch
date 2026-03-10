"""Noise functions for denoising autoencoders."""
import torch


def salt_and_pepper(X, prop):
    """Salt-and-pepper noise: randomly set elements to min or max value.

    Args:
        X (Tensor): input tensor
        prop (float): proportion of elements to corrupt (0.0 to 1.0)

    Returns:
        Tensor: corrupted copy of X
    """
    X_clone = X.clone()
    mask = torch.rand_like(X) < prop
    salt = torch.rand_like(X) < 0.5
    X_clone[mask & salt] = X.min()
    X_clone[mask & ~salt] = X.max()
    return X_clone


def gaussian(X, prop):
    """Additive Gaussian noise.

    Args:
        X (Tensor): input tensor
        prop (float): noise scale (standard deviation as fraction of data range)

    Returns:
        Tensor: corrupted copy of X
    """
    std = prop * (X.max() - X.min())
    noise = torch.randn_like(X) * std
    return X + noise


def masking(X, prop):
    """Masking noise: randomly zero out elements.

    Args:
        X (Tensor): input tensor
        prop (float): proportion of elements to zero out (0.0 to 1.0)

    Returns:
        Tensor: corrupted copy of X
    """
    mask = torch.bernoulli(torch.full_like(X, 1.0 - prop))
    return X * mask


NOISE_FUNCTIONS = {
    'salt_and_pepper': salt_and_pepper,
    'gaussian': gaussian,
    'masking': masking,
}


def get_noise_fn(name):
    """Get a noise function by name.

    Args:
        name (str): one of 'salt_and_pepper', 'gaussian', 'masking'

    Returns:
        callable: noise function(X, prop) -> Tensor
    """
    if name not in NOISE_FUNCTIONS:
        raise ValueError(f"Unknown noise type '{name}'. Choose from: {list(NOISE_FUNCTIONS.keys())}")
    return NOISE_FUNCTIONS[name]
