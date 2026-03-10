"""Tests for utility functions."""
import numpy as np
import pytest
import torch

from utils import StateData, select_device, clean_output
from model import StackDAE


class TestSelectDevice:
    """Test device selection."""

    def test_returns_device(self):
        device = select_device()
        assert isinstance(device, torch.device)

    def test_force_cpu(self):
        device = select_device(force_cpu=True)
        assert device.type == 'cpu'

    def test_cpu_on_ci(self):
        """CI runners typically don't have GPU."""
        device = select_device()
        # Should not crash regardless of GPU availability
        assert device.type in ('cpu', 'cuda')


class TestStateData:
    """Test dataset loading and normalization."""

    def test_from_numpy_data(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(100)]
        dataset = StateData(data=data, data_size=100)
        assert len(dataset) == 100

    def test_getitem_returns_tensor(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(10)]
        dataset = StateData(data=data, data_size=10)
        item = dataset[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.float32

    def test_getitem_shape(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(10)]
        dataset = StateData(data=data, data_size=10)
        item = dataset[0]
        assert item.shape == (48,)

    def test_normalization_range(self):
        """Normalized values should be in [0, 1]."""
        data = [np.array([1.0, 5.0, 3.0, -2.0] * 12) for _ in range(10)]
        dataset = StateData(data=data, data_size=10)
        item = dataset[0]
        assert item.min() >= -1e-6, f"Min {item.min()} below 0"
        assert item.max() <= 1.0 + 1e-6, f"Max {item.max()} above 1"

    def test_normalization_constant_input(self):
        """Constant input should not cause div-by-zero."""
        data = [np.ones(48) * 5.0 for _ in range(5)]
        dataset = StateData(data=data, data_size=5)
        item = dataset[0]
        # Should not be NaN or Inf
        assert torch.isfinite(item).all()

    def test_dataloader_integration(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(100)]
        dataset = StateData(data=data, data_size=100)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        batch = next(iter(loader))
        assert batch.shape == (16, 48)
        assert batch.dtype == torch.float32


class TestCleanOutput:
    """Test encoder output generation for layer-wise training."""

    def test_output_dataset_type(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(64)]
        dataset = StateData(data=data, data_size=64)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        model = StackDAE(48, 3, 4)
        encoder = model.stack_enc[:1]  # First encoder layer only
        device = torch.device('cpu')

        result = clean_output(loader, encoder, device)
        assert isinstance(result, StateData)
        assert len(result) == 64

    def test_output_dimension_reduced(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(32)]
        dataset = StateData(data=data, data_size=32)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16)

        model = StackDAE(48, 3, 4)
        encoder = model.stack_enc[:1]  # First encoder layer
        device = torch.device('cpu')

        result = clean_output(loader, encoder, device)
        item = result[0]
        # Output dim should be less than input dim
        assert item.shape[0] < 48
