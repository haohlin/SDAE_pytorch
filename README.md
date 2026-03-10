# SDAE_pytorch

Stacked Denoising Autoencoder (SDAE) implementation in PyTorch with layer-wise pretraining.

## Features

- **Layer-wise pretraining** with greedy training of individual DAE layers
- **Configurable noise types**: salt-and-pepper, gaussian, masking
- **Train/validation split** with configurable ratio
- **Early stopping** with patience and minimum delta
- **Learning rate warmup** (optional linear warmup)
- **TensorBoard logging** for loss curves and learning rates
- **3D visualization** of feature space and reconstructions

## Training

Place training data (`observation.data`) in the `dataset/` folder.

```bash
# Basic training
python train.py

# With all features enabled
python train.py \
    --noise_type gaussian \
    --noise_r 15 \
    --val_ratio 0.1 \
    --patience 5 \
    --warmup_epochs 3 \
    --tensorboard \
    --log_dir runs/experiment_1

# View TensorBoard logs
tensorboard --logdir runs/
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epoch` | 15 | Training epochs per layer |
| `--data_size` | 200000 | Size of training data |
| `--lr` | 0.0002 | Learning rate |
| `--batchSize` | 640 | Batch size |
| `--in_dim` | 48 | Input dimension |
| `--out_dim` | 3 | Feature (bottleneck) dimension |
| `--stack_num` | 4 | Number of encoder/decoder layers |
| `--noise_type` | salt_and_pepper | Noise type: `salt_and_pepper`, `gaussian`, `masking` |
| `--noise_r` | 10 | Noise ratio (percent) |
| `--val_ratio` | 0.1 | Validation split ratio (0 to disable) |
| `--patience` | 0 | Early stopping patience (0 to disable) |
| `--min_delta` | 0.0001 | Minimum improvement for early stopping |
| `--warmup_epochs` | 0 | Linear LR warmup epochs (0 to disable) |
| `--tensorboard` | off | Enable TensorBoard logging |
| `--log_dir` | runs/sdae_training | TensorBoard log directory |
| `--save_path` | model/chekp.pt | Model save path |

## Visualization

```bash
python visualize.py --data_size 10000
```

## Feature Extraction

```python
from SDAE_pytorch.extract import Autoencoder
import numpy as np

sdae = Autoencoder()
features = sdae.extract(np.random.randn(48))
print(features)
```

## CI

- **Lint**: flake8 (fatal errors block, style warnings informational)
- **Tests**: pytest on Python 3.11
- **AI Review**: CodeRabbit on all PRs
