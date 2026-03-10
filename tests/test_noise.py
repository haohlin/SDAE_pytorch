"""Tests for noise functions."""
import torch
import numpy as np

from noise import salt_and_pepper, gaussian, masking, get_noise_fn, NOISE_FUNCTIONS


class TestSaltAndPepper:

    def test_output_shape(self):
        x = torch.randn(10, 48)
        assert salt_and_pepper(x, 0.1).shape == x.shape

    def test_zero_noise(self):
        x = torch.randn(5, 48)
        assert torch.allclose(salt_and_pepper(x, 0.0), x)

    def test_noise_modifies_values(self):
        torch.manual_seed(42)
        np.random.seed(42)
        x = torch.randn(100, 48)
        assert not torch.allclose(salt_and_pepper(x, 0.5), x)


class TestGaussian:

    def test_output_shape(self):
        x = torch.randn(10, 48)
        assert gaussian(x, 0.1).shape == x.shape

    def test_zero_noise(self):
        x = torch.randn(5, 48)
        assert torch.allclose(gaussian(x, 0.0), x)

    def test_noise_modifies_values(self):
        torch.manual_seed(42)
        x = torch.randn(100, 48)
        assert not torch.allclose(gaussian(x, 0.5), x)


class TestMasking:

    def test_output_shape(self):
        x = torch.randn(10, 48)
        assert masking(x, 0.1).shape == x.shape

    def test_zero_noise(self):
        x = torch.ones(5, 48) * 3.0
        result = masking(x, 0.0)
        assert torch.allclose(result, x)

    def test_full_noise(self):
        x = torch.ones(5, 48) * 3.0
        result = masking(x, 1.0)
        assert torch.allclose(result, torch.zeros_like(x))

    def test_partial_masking(self):
        torch.manual_seed(42)
        x = torch.ones(100, 48) * 5.0
        result = masking(x, 0.5)
        zero_frac = (result == 0).float().mean().item()
        assert 0.3 < zero_frac < 0.7


class TestGetNoiseFn:

    def test_all_names(self):
        for name in NOISE_FUNCTIONS:
            fn = get_noise_fn(name)
            x = torch.randn(4, 48)
            assert fn(x, 0.1).shape == x.shape

    def test_invalid_name(self):
        try:
            get_noise_fn('invalid')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
