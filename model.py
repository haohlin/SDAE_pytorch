import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

def salt_and_pepper(X, prop):
	'''Noise generator'''
	X_clone=X.clone().view(-1, 1)
	num_feature=X_clone.size(0)
	mn=X_clone.min()
	mx=X_clone.max()
	indices=np.random.randint(0, num_feature, int(num_feature*prop))
	for elem in indices :
		if np.random.random() < 0.5 :
			X_clone[elem]=mn
		else :
			X_clone[elem]=mx
	return X_clone.view(X.size())

class DAE(nn.Module):
	'''
	Denoising auto-encoder net structure (default): 
	input (in: in_dim) --> 
	--> hidden layer (Linear(in: in_dim, out: out_dim), activation: LeakyReLU) --> 
	--> output (Linear(in: out_dim, out: in_dim))

	'''
	def __init__(self, in_dim, out_dim):
		super(DAE, self).__init__()
		'''
		Args: 
			in_dim(int): input size
			out_dim(int): output(feature) size
		
		Activation func: LeakyReLU (other functions remain to be tested)
		'''
		self.in_dim = int(in_dim)
		self.out_dim = int(out_dim)
		self.encoder = nn.Sequential(
			nn.Linear(self.in_dim, self.out_dim),
			nn.LeakyReLU())
		self.decoder = nn.Linear(self.out_dim, self.in_dim)

	def forward(self, input):
		'''
		Forward calculation of the network
		Args: 
			input(FloatTensor): training data in a batch (batch_size, data_size)
		'''
		size=input.size()
		self.hidden_output=self.encoder(input)
		output = self.decoder(self.hidden_output)

		return output.view(size)

	def train_DAE(self, train_loader, device, learning_rate, loss_fn=nn.MSELoss(), epoch = 20, noise_r=20, layer=1):
		'''
		Training individual DAE. Salt and pepper noise is applied to raw training data. Adam optimization 
		strategy and step-decay learning rate is used for now. Further experiments on training hyperparameters 
		will be carried out.

		Args:
			train_loader(Dataloader): iterable dataloader, pack data in batches and feed them to the network. 
									  See torch.utils.data.DataLoader for details.
			device(CPU/GPU): device used for training. 
			learning_rate(float): learning rate
			loss_fn(pytorch loss class): loss function used for optimization
			epoch(int): training epochs
			noise_r(float): Ratio of noise
			layer(int): current layer being trained

		
		
		'''
		noise_fn = salt_and_pepper # noise type

		optimizer = optim.Adam(self.parameters(), lr=learning_rate, betas=(0.5, 0.999))
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.2)

		for epoch in range(epoch):
			
			for i, data in enumerate(train_loader):
				data = data.to(device) # Unlike nn.module, .cuda() on tensor is not in-place
				noise_data = noise_fn(data, float(noise_r)/100)
				noise_data = noise_data.to(device)
				
				output=self.forward(noise_data)
				error=loss_fn(output, data)
				if i%10 == 0:
					print('Layer: %d, Epoch : %d, Error: %f' % (layer, epoch+1, error))
				error.backward()
				optimizer.step()
				optimizer.zero_grad()
			scheduler.step()
	
		return output
		

class StackDAE(nn.Module):
	'''Stacked DAE network initializer'''
	def __init__(self, in_dim, out_dim, layer_num):
		super(StackDAE, self).__init__()

		stride = pow(in_dim / out_dim, 1 / (layer_num))

		self.stack_enc = nn.Sequential()
		self.stack_dec = nn.Sequential()
		# build stacked encoder
		for i in range(layer_num):
			out_dim = in_dim / stride
			self.stack_enc.add_module('encoder_%d' % i, nn.Sequential(nn.Linear(int(in_dim), int(out_dim)),
																	 nn.LeakyReLU()))
			in_dim /= stride
		# build stacked decoder
		for i in range(layer_num):
			out_dim = in_dim * stride
			self.stack_dec.add_module('decoder_%d' % (layer_num - i-1), nn.Linear(int(in_dim), int(out_dim)))
			in_dim *= stride
			
	def forward(self, input):
		self.hidden_feature = self.stack_enc(input)
		output = self.stack_dec(self.hidden_feature)
		return output

	def extract(self, input):
		feature = self.stack_enc(input)
		return feature