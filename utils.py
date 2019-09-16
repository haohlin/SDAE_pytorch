import numpy as np
import random
import torch
import pickle
from torch.utils.data import Dataset


# Fetch Data
class StateData(Dataset):
	def __init__(self, data=None, training_data_size=100000):
		self.observation = data
		if not self.observation:
			with open('dataset/observation.data', 'rb') as f:
				self.observation = pickle.load(f, encoding='bytes')
				self.observation = random.sample(self.observation, training_data_size)
			f.close()
			#self.observation = torch.load('dataset/observation_clusterd.pt')[320000:330000]

	def __len__(self):
		return len(self.observation)
	def __getitem__(self, idx):
        # normalize 0-1
		max_ob = self.observation[idx].max()
		min_ob = self.observation[idx].min()
		state = (self.observation[idx] + min_ob) / (max_ob - min_ob)
        # np_array to tensor
		state = torch.from_numpy(state).float()
		return state


def select_device(force_cpu=False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    if not cuda:
        print('Using CPU')
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print("Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
              (x[0].name, x[0].total_memory / c))
        if ng > 0:
            # torch.cuda.set_device(0)  # OPTIONAL: Set GPU ID
            for i in range(1, ng):
                print("           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                      (i, x[i].name, x[i].total_memory / c))

    print('')  # skip a line
    return device

def clean_output(train_loader, stacked_net):
	batchSize = 1
	device = select_device()

	output_stack = []
	for i, data in enumerate(train_loader):
		data = data.cuda()
		output=stacked_net(data).detach().cpu().numpy()
		for j in range(output.shape[0]):
			output_stack.append(output[j])
	Test_dataset = StateData(output_stack)
	
	return Test_dataset

