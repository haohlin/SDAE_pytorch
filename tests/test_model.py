"""Tests for SDAE model components."""
import numpy as np
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

    def test_hidden_output_shape(self):
        model = DAE(48, 24)
        x = torch.randn(16, 48)
        _ = model(x)
        assert model.hidden_output.shape == (16, 24)

    def test_gradient_flows(self):
        model = DAE(48, 24)
        x = torch.randn(8, 48)
        loss = nn.MSELoss()(model(x), x)
        loss.backward()
        for param in model.parameters():
            assert param.grad is not None


class TestStackDAE:

    def test_forward_shape(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(16, 48)
        assert model(x).shape == (16, 48)

    def test_extract_shape(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(16, 48)
        assert model.extract(x).shape == (16, 3)

    def test_different_layer_counts(self):
        for layers in [1, 2, 3, 5]:
            model = StackDAE(48, 3, layers)
            x = torch.randn(4, 48)
            assert model(x).shape == (4, 48)
            assert model.extract(x).shape[0] == 4

    def test_eval_deterministic(self):
        model = StackDAE(48, 3, 4)
        model.eval()
        x = torch.randn(4, 48)
        with torch.no_grad():
            assert torch.allclose(model(x), model(x))

    def test_gradient_flows(self):
        model = StackDAE(48, 3, 4)
        x = torch.randn(8, 48)
        loss = nn.MSELoss()(model(x), x)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
