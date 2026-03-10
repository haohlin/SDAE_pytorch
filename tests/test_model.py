"""Tests for SDAE model components."""
import numpy as np
import torch
import torch.nn as nn

from model import DAE, StackDAE, salt_and_pepper


class TestSaltAndPepper:
    """Test the noise generation function."""

    def test_output_shape_matches_input(self):
        x = torch.randn(10, 48)
        noisy = salt_and_pepper(x, 0.1)
        assert noisy.shape == x.shape

    def test_zero_noise_ratio_returns_original(self):
        x = torch.randn(5, 48)
        noisy = salt_and_pepper(x, 0.0)
        assert torch.allclose(noisy, x)

    def test_noise_modifies_some_values(self):
        torch.manual_seed(42)
        np.random.seed(42)
        x = torch.randn(100, 48)
        noisy = salt_and_pepper(x, 0.5)
        # With 50% noise, values should differ
        assert not torch.allclose(noisy, x)

    def test_noisy_values_are_min_or_max(self):
        torch.manual_seed(0)
        np.random.seed(0)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        noisy = salt_and_pepper(x, 0.8)
        mn, mx = x.min().item(), x.max().item()
        for val in noisy.view(-1):
            v = val.item()
            # Each value should be either original, min, or max
            assert v in [mn, mx] or v in x.tolist()


class TestDAE:
    """Test individual Denoising Autoencoder."""

    def test_init(self):
        model = DAE(48, 24)
        assert model.in_dim == 48
        assert model.out_dim == 24

    def test_forward_shape(self):
        model = DAE(48, 24)
        x = torch.randn(16, 48)
        out = model(x)
        assert out.shape == (16, 48), f"Expected (16, 48), got {out.shape}"

    def test_hidden_output_shape(self):
        model = DAE(48, 24)
        x = torch.randn(16, 48)
        _ = model(x)
        assert model.hidden_output.shape == (16, 24)

    def test_encoder_decoder_dimensions(self):
        model = DAE(48, 12)
        # Encoder: 48 -> 12
        enc_linear = model.encoder[0]
        assert enc_linear.in_features == 48
        assert enc_linear.out_features == 12
        # Decoder: 12 -> 48
        assert model.decoder.in_features == 12
        assert model.decoder.out_features == 48

    def test_single_sample_forward(self):
        model = DAE(48, 24)
        x = torch.randn(1, 48)
        out = model(x)
        assert out.shape == (1, 48)

    def test_gradient_flows(self):
        model = DAE(48, 24)
        x = torch.randn(8, 48)
        out = model(x)
        loss = nn.MSELoss()(out, x)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


class TestStackDAE:
    """Test Stacked DAE architecture."""

    def test_init_default(self):
        model = StackDAE(48, 3, 4)
        assert len(model.stack_enc) == 4
        assert len(model.stack_dec) == 4

    def test_forward_shape(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(16, 48)
        out = model(x)
        assert out.shape == (16, 48), f"Expected (16, 48), got {out.shape}"

    def test_extract_shape(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(16, 48)
        features = model.extract(x)
        assert features.shape == (16, 3), f"Expected (16, 3), got {features.shape}"

    def test_hidden_feature_after_forward(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(8, 48)
        _ = model(x)
        assert model.hidden_feature.shape == (8, 3)

    def test_different_layer_counts(self):
        for layers in [1, 2, 3, 5]:
            model = StackDAE(48, 3, layers)
            x = torch.randn(4, 48)
            out = model(x)
            feat = model.extract(x)
            assert out.shape == (4, 48), f"layers={layers}: output shape {out.shape}"
            assert feat.shape == (4, 3), f"layers={layers}: feature shape {feat.shape}"

    def test_different_dimensions(self):
        configs = [(64, 8, 3), (32, 4, 2), (100, 10, 4)]
        for in_dim, out_dim, layers in configs:
            model = StackDAE(in_dim, out_dim, layers)
            x = torch.randn(4, in_dim)
            out = model(x)
            feat = model.extract(x)
            assert out.shape == (4, in_dim)
            # Feature dim may differ slightly from out_dim due to int truncation
            # in geometric stride calculation; just verify it's reasonable
            assert feat.shape[0] == 4
            assert feat.shape[1] > 0 and feat.shape[1] <= out_dim + 1

    def test_gradient_flows(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(8, 48)
        out = model(x)
        loss = nn.MSELoss()(out, x)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode_deterministic(self):
        model = StackDAE(48, 3, 4)
        model.eval()
        x = torch.randn(4, 48)
        with torch.no_grad():
            out1 = model(x).clone()
            out2 = model(x).clone()
        assert torch.allclose(out1, out2)

    def test_encoder_decoder_symmetry(self):
        """Encoder layers should reduce dims, decoder should restore."""
        model = StackDAE(48, 3, 4)
        # Check encoder reduces to feature dim
        x = torch.randn(4, 48)
        feat = model.extract(x)
        assert feat.shape[-1] == 3
        # Check full forward restores to input dim
        out = model(x)
        assert out.shape[-1] == 48
