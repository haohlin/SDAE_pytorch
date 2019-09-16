from . import utils
import torch
import argparse
import numpy as np
from .model import StackDAE

def extract(data):
	device = utils.select_device()
	chekp = torch.load('SDAE_pytorch/model/chekp.pt')
	reconstruct_dim = chekp['in_dim']
	feature_dim = chekp['out_dim']
	chekp_model = chekp['model']
	stack_num = chekp['stack_num']

	if isinstance(data, np.ndarray):
		pass
	else:
		raise(TypeError("numpy.ndarray is required for input."))
	
	if data.size != reconstruct_dim:
		raise(RuntimeError('Input size (%d) does not meet autoencoder input requirement (%d)' % (data.shape[0], reconstruct_dim)))
	
	model = StackDAE(reconstruct_dim, feature_dim, stack_num)

	data = torch.from_numpy(data).float().to(device)
	model.to(device).load_state_dict(chekp_model)
	
	feature = model.extract(data)
	feature = feature.detach().cpu().numpy()
	return feature
