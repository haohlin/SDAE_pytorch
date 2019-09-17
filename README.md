# SDAE_pytorch

## Training and visualization
For training and visualization, go into folder `SDAE_pytorch`. Place the training data (`observation.data`) in `dataset` folder, and use following commands for training. SDAE model is trained and stored as `chept.pt` in folder `model`, and will be used for visualization and feature extraction.

Training:
```
usage: python train.py [-h] [--epoch EPOCH] [--data_size DATA_SIZE] [--lr LR]
                [--workers WORKERS] [--batchSize BATCHSIZE] [--in_dim IN_DIM]
                [--out_dim OUT_DIM] [--stack_num STACK_NUM]
                [--noise_r NOISE_R]

optional arguments:
  -h, --help              show this help message and exit
  --epoch EPOCH           number of training epochs
  --data_size DATA_SIZE   size of training data
  --lr LR                 learning rate, default=0.0002
  --workers WORKERS       number of data loading workers
  --batchSize BATCHSIZE   input batch size
  --in_dim IN_DIM         input dimension
  --out_dim OUT_DIM       output(feature) dimension
  --stack_num STACK_NUM   number of hidden layers
  --noise_r NOISE_R       Ratio of noise
```
Visualization:
```
usage: python visualize.py [-h] [--data_size DATA_SIZE]

optional arguments:
  -h, --help              show this help message and exit
  --data_size DATA_SIZE   size of data used for visualization

```

## Feature extraction
Place the module in the root folder of the project. Use `from SDAE_pytorch.extract import Autoencoder` to import the feature extraction class. 

Example:
```python
from SDAE_pytorch.extract import Autoencoder
import numpy as np

SDAE = Autoencoder()

for i in range(10):
	dumb_data = np.random.randn(48)
	dumb_feature = SDAE.extract(dumb_data)

	print(dumb_feature)
```
