import torch
from utils import *
from model import *

import pickle
import argparse
import torch.nn as nn
import torch.optim as optim

from collections import OrderedDict
from torch.utils.data import Dataset
import numpy as np


def train():
	'''
	Training a Stacked DAE. Generate 'observation.data', put it in dataset folder and use it for training
	
	'''
	device = select_device()
	stack_num = opt.stack_num
	in_dim = opt.in_dim
	end_dim = opt.out_dim
	training_data_size = opt.data_size
	final_net = StackDAE(in_dim, end_dim, stack_num)

	# Set Data Loader(input pipeline)
	Test_dataset = StateData(training_data_size=training_data_size)
	train_loader_raw = torch.utils.data.DataLoader(Test_dataset, batch_size=opt.batchSize,
												shuffle=True)#, num_workers=4, pin_memory=True
	
	
	out_dim = in_dim/2
	stacked_enc_net = nn.Sequential()
	stacked_dec_net = nn.Sequential()
	for i in range(stack_num):
		model=DAE(in_dim, out_dim).to(device)

		if i==0:
			model.train_DAE(train_loader_raw,device, learning_rate=opt.lr, epoch=opt.epoch, noise_r=opt.noise_r, layer=i+1)
		else:
			model.train_DAE(train_loader,device, learning_rate=opt.lr, epoch=opt.epoch, noise_r=opt.noise_r, layer=i+1)

		stacked_enc_net.add_module("encoder_%d" % i, model.encoder)
		stacked_dec_net.add_module("decoder_%d" % i, model.decoder)

		Test_dataset = clean_output(train_loader_raw, stacked_enc_net)
		train_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=opt.batchSize,
												shuffle=True)
		
		in_dim /= 2
		out_dim = in_dim/2

	stacked_enc_dict = stacked_enc_net.state_dict()
	stacked_dec_dict = stacked_dec_net.state_dict()
	new_enc_dict = OrderedDict()
	new_dec_dict = OrderedDict()

	for k, v in stacked_enc_dict.items():
		name = 'stack_enc.' + k # add `stack_net.` to encoder model dict
		new_enc_dict[name] = v

	for k, v in reversed(stacked_dec_dict.items()):
		name = 'stack_dec.' + k # add `stack_net.` to encoder model dict
		new_dec_dict[name] = v

	new_enc_dict.update(new_dec_dict)
	new_state_dict = new_enc_dict

	final_net.load_state_dict(new_state_dict)

	# Save trained model in 'model' folder
	chekp = {'in_dim': opt.in_dim,
			 'out_dim': opt.out_dim,
			 'stack_num': opt.stack_num,
			 'model': final_net.state_dict()}
	torch.save(chekp, 'model/chekp.pt')

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', type=int, default=2, help='number of training epochs')
	parser.add_argument('--data_size', type=int, default=100000, help='size of training data')
	parser.add_argument('--lr', type=float, default=0.0004, help='learning rate, default=0.0002')
	parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
	parser.add_argument('--batchSize', type=int, default=640, help='input batch size')
	parser.add_argument('--in_dim', type=int, default=48, help='input dimension')
	parser.add_argument('--out_dim', type=int, default=3, help='output(feature) dimension')
	parser.add_argument('--stack_num', type=int, default=4, help='number of hidden layers')
	parser.add_argument('--noise_r', type=int, default=10, help='Ratio of noise')

	opt = parser.parse_args()

	train()