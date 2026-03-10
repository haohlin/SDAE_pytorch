"""Tests for utility functions."""
import numpy as np
import torch

from utils import StateData, select_device, clean_output, split_dataset
from model import StackDAE


class TestSelectDevice:

    def test_returns_device(self):
        assert isinstance(select_device(), torch.device)

    def test_force_cpu(self):
        assert select_device(force_cpu=True).type == 'cpu'


class TestStateData:

    def test_from_numpy_data(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(100)]
        assert len(StateData(data=data, data_size=100)) == 100

    def test_getitem_returns_float_tensor(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(10)]
        item = StateData(data=data, data_size=10)[0]
        assert isinstance(item, torch.Tensor)
        assert item.dtype == torch.float32
        assert item.shape == (48,)

    def test_normalization_range(self):
        data = [np.array([1.0, 5.0, 3.0, -2.0] * 12) for _ in range(10)]
        item = StateData(data=data, data_size=10)[0]
        assert item.min() >= -1e-6
        assert item.max() <= 1.0 + 1e-6

    def test_constant_input_no_nan(self):
        data = [np.ones(48) * 5.0 for _ in range(5)]
        item = StateData(data=data, data_size=5)[0]
        assert torch.isfinite(item).all()

    def test_normalize_false(self):
        """When normalize=False, data should pass through as-is."""
        data = [np.array([10.0, -5.0, 3.0] * 16) for _ in range(5)]
        item = StateData(data=data, data_size=5, normalize=False)[0]
        assert item[0].item() == 10.0
        assert item[1].item() == -5.0

    def test_dataloader_integration(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(100)]
        loader = torch.utils.data.DataLoader(
            StateData(data=data, data_size=100), batch_size=16, shuffle=True)
        batch = next(iter(loader))
        assert batch.shape == (16, 48)


class TestSplitDataset:

    def test_split_sizes(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(100)]
        dataset = StateData(data=data, data_size=100)
        train_ds, val_ds = split_dataset(dataset, val_ratio=0.2)
        assert len(train_ds) == 80
        assert len(val_ds) == 20

    def test_split_zero_ratio(self):
        data = [np.random.randn(48).astype(np.float64) for _ in range(50)]
        dataset = StateData(data=data, data_size=50)
        train_ds, val_ds = split_dataset(dataset, val_ratio=0.0)
        assert len(train_ds) == 50
        assert len(val_ds) == 0


class TestCleanOutput:

    def test_output_not_renormalized(self):
        """clean_output should return data without re-normalization."""
        data = [np.random.randn(48).astype(np.float64) for _ in range(64)]
        loader = torch.utils.data.DataLoader(
            StateData(data=data, data_size=64), batch_size=16)
        model = StackDAE(48, 3, 4)
        encoder = model.stack_enc[:1]
        result = clean_output(loader, encoder, torch.device('cpu'))
        assert isinstance(result, StateData)
        assert result.normalize is False
        assert len(result) == 64
