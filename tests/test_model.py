"""Tests for SDAE model components."""
import torch
import torch.nn as nn

from model import DAE, StackDAE


class TestDAE:

    def test_init(self):
        model = DAE(48, 24)
        assert model.in_dim == 48
        assert model.out_dim == 24

    def test_forward_shape(self):
        model = DAE(48, 24)
        x = torch.randn(16, 48)
        assert model(x).shape == (16, 48)

    def test_encode_shape(self):
        model = DAE(48, 24)
        x = torch.randn(16, 48)
        assert model.encode(x).shape == (16, 24)

    def test_gradient_flows(self):
        model = DAE(48, 24)
        x = torch.randn(8, 48)
        loss = nn.MSELoss()(model(x), x)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


class TestStackDAE:

    def test_forward_returns_tuple(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(16, 48)
        result = model(x)
        assert isinstance(result, tuple)
        assert len(result) == 2
        reconstruction, features = result
        assert reconstruction.shape == (16, 48)
        assert features.shape == (16, 3)

    def test_extract_shape(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(16, 48)
        assert model.extract(x).shape == (16, 3)

    def test_compute_dims(self):
        dims = StackDAE._compute_dims(48, 3, 4)
        assert len(dims) == 5
        assert dims[0] == 48
        assert dims[-1] == 3
        # All dims should be decreasing
        for i in range(len(dims) - 1):
            assert dims[i] > dims[i + 1]

    def test_different_layer_counts(self):
        for layers in [1, 2, 3, 5]:
            model = StackDAE(48, 3, layers)
            x = torch.randn(4, 48)
            reconstruction, features = model(x)
            assert reconstruction.shape == (4, 48)
            assert features.shape[0] == 4

    def test_exact_bottleneck_dim(self):
        """Bottleneck should match out_dim exactly (no float drift)."""
        model = StackDAE(48, 3, 4)
        x = torch.randn(4, 48)
        _, features = model(x)
        assert features.shape == (4, 3)

        model2 = StackDAE(100, 10, 5)
        _, features2 = model2(torch.randn(4, 100))
        assert features2.shape == (4, 10)

    def test_eval_deterministic(self):
        model = StackDAE(48, 3, 4)
        model.eval()
        x = torch.randn(4, 48)
        with torch.no_grad():
            r1, f1 = model(x)
            r2, f2 = model(x)
        assert torch.allclose(r1, r2)
        assert torch.allclose(f1, f2)

    def test_gradient_flows(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(8, 48)
        reconstruction, _ = model(x)
        loss = nn.MSELoss()(reconstruction, x)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
