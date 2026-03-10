"""Data loading and utility functions."""
import random
import pickle
import torch
from torch.utils.data import Dataset


class StateData(Dataset):
    """Dataset for observation state vectors.

    Loads data from a pickle file or accepts pre-loaded data.
    Normalizes each sample to [0, 1] range unless normalize=False.

    Args:
        data (list, optional): pre-loaded list of numpy arrays
        data_size (int): number of samples to use
        data_path (str): path to pickle file (used if data is None)
        normalize (bool): whether to normalize samples to [0, 1]
    """

    def __init__(self, data=None, data_size=100000, data_path='dataset/observation.data',
                 normalize=True):
        self.observation = data
        self.normalize = normalize
        if self.observation is None:
            with open(data_path, 'rb') as f:
                self.observation = pickle.load(f, encoding='bytes')
                self.observation = random.sample(self.observation, data_size)

    def __len__(self):
        return len(self.observation)

    def __getitem__(self, idx):
        sample = self.observation[idx]
        state = torch.from_numpy(sample).float() if not isinstance(sample, torch.Tensor) else sample.float()
        if self.normalize:
            max_val = state.max()
            min_val = state.min()
            state = (state - min_val) / (max_val - min_val + 1e-8)
        return state


def select_device(force_cpu=False):
    """Select training device (CPU or CUDA).

    Args:
        force_cpu (bool): force CPU even if CUDA is available

    Returns:
        torch.device: selected device
    """
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print("Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (x[0].name, x[0].total_memory / c))
        for i in range(1, ng):
            print("           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')
    print('')
    return device


def clean_output(train_loader, stacked_net, device):
    """Generate clean encoder outputs for training next layer.

    Passes data through the current encoder stack and collects outputs.
    Returns a dataset with normalize=False to avoid double normalization.

    Args:
        train_loader: DataLoader with input data
        stacked_net: current encoder stack (nn.Sequential)
        device: torch device

    Returns:
        StateData: dataset of encoder outputs (not re-normalized)
    """
    output_stack = []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            output = stacked_net(data).cpu().numpy()
            output_stack.extend(output)
    return StateData(output_stack, normalize=False)


def split_dataset(dataset, val_ratio=0.1):
    """Split a dataset into train and validation sets.

    Args:
        dataset: a torch Dataset
        val_ratio (float): fraction of data for validation (0.0 to 1.0)

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    total = len(dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])
