from model import StackDAE
import utils
import torch
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(data_size = 10000):
	'''
	Plot 3-dimensional feature space and reconstructed result. 
	Images are stored in 'result_SDAE'
	* Have to used autoencoder models with 3-dimensional output
	 Can only be used on training data *

	'''
	batchSize = 100
	device = utils.select_device()
	Test_dataset = utils.StateData(data_size=data_size)
	visu_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=batchSize, shuffle=True)
	
	chekp = torch.load('model/chekp.pt')
	reconstruct_dim = chekp['in_dim']
	feature_dim = chekp['out_dim']
	chekp_model = chekp['model']
	stack_num = chekp['stack_num']
	
	model = StackDAE(reconstruct_dim, feature_dim, stack_num)

	model.to(device).load_state_dict(chekp_model)

	reconstruct_stack = torch.FloatTensor(reconstruct_dim).to(device).unsqueeze(0)
	feature_stack = torch.FloatTensor(feature_dim).to(device).unsqueeze(0)
	for i, data in enumerate(visu_loader):
		data = data.to(device)
		reconstruct = model.forward(data)
		reconstruct_stack = torch.cat((reconstruct_stack, reconstruct))
		feature_stack = torch.cat((feature_stack, model.hidden_feature))
		

	reconstruct_stack = reconstruct_stack[1:].detach().cpu().numpy()
	feature_stack = feature_stack[1:].detach().cpu().numpy()

	print('%d data points, input size: %d, feature size: %d' % (feature_stack.shape[0], reconstruct_dim, feature_stack.shape[1]))

	fig1 = plt.figure()
	ax = Axes3D(fig1)
	ax.scatter(feature_stack[:,0], feature_stack[:,1], feature_stack[:,2])
	fig1.savefig('result_SDAE/result_feature.png')

	fig2 = plt.figure()
	ax = Axes3D(fig2)
	ax.scatter(reconstruct_stack[:,-8], reconstruct_stack[:,-7], reconstruct_stack[:,-6])
	fig2.savefig('result_SDAE/result_reconstruct.png')

	return feature_stack

def visualize_orig(data_size = 10000):
	'''
	Plot positions(x, y, z) of the last joint of original data(observation). 
	* Can only be used on training data *

	'''
	batchSize = 100
	device = utils.select_device()
	Test_dataset = utils.StateData(data_size)
	visu_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=batchSize, shuffle=True)

	output_stack = torch.FloatTensor(48).unsqueeze(0)

	for i, data in enumerate(visu_loader):
		output_stack = torch.cat((output_stack, data))

	output_stack = output_stack[1:].numpy()
	fig = plt.figure()

	ax = Axes3D(fig)
	ax.scatter(output_stack[:,-8],output_stack[:,-7],output_stack[:,-6])
	fig.savefig('result_SDAE/result_orig.png')

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_size', type=int, default=10000, help='size of data used for visualization')

	opt = parser.parse_args()
	visualize(opt.data_size)