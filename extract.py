"""Feature extraction using a trained Stacked DAE model."""
import torch
import numpy as np
from .model import StackDAE
from . import utils


class Autoencoder(object):
    """Wrapper for loading a trained SDAE and extracting features.

    Args:
        model_path (str): path to saved checkpoint
    """

    def __init__(self, model_path='SDAE_pytorch/model/chekp.pt'):
        self.device = utils.select_device()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.reconstruct_dim = checkpoint['in_dim']
        feature_dim = checkpoint['out_dim']
        stack_num = checkpoint['stack_num']

        self.model = StackDAE(self.reconstruct_dim, feature_dim, stack_num)
        self.model.to(self.device).load_state_dict(checkpoint['model'])
        self.model.eval()

    def extract(self, data):
        """Extract features from input data.

        Args:
            data (numpy.ndarray): input array matching model input dimension

        Returns:
            numpy.ndarray: extracted features
        """
        data = torch.from_numpy(data).float().to(self.device)
        with torch.no_grad():
            feature = self.model.extract(data)
        return feature.cpu().numpy()
