from . import utils
import torch
import argparse
import numpy as np
from .model import StackDAE

class Autoencoder(object):
	def __init__(self):
		self.device = utils.select_device()
		chekp = torch.load('SDAE_pytorch/model/chekp.pt')
		self.reconstruct_dim = chekp['in_dim']
		feature_dim = chekp['out_dim']
		chekp_model = chekp['model']
		stack_num = chekp['stack_num']

		self.model = StackDAE(self.reconstruct_dim, feature_dim, stack_num)
		self.model.to(self.device).load_state_dict(chekp_model)


	def extract(self, data):
		''' Input/Output: numpy.ndarray

			Data shape must match autoencoder input shape'''
		"""
		if isinstance(data, np.ndarray):
			pass
		else:
			raise(TypeError("numpy.ndarray is required for input."))
	
		if data.size != self.reconstruct_dim:
			raise(RuntimeError('Input size (%d) does not meet autoencoder input requirement (%d)' % (data.shape[0], self.reconstruct_dim)))
		"""
		data = torch.from_numpy(data).float().to(self.device)
	
		feature = self.model.extract(data)
		feature = feature.detach().cpu().numpy()
		return feature

