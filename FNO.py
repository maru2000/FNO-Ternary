## Imports

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import operator
from functools import reduce
from functools import partial

from sklearn.metrics import mean_squared_error 

from timeit import default_timer
import scipy.io

from FNO.Layers import Net2d, SimpleBlock2d, SpectralConv2d_fast

torch.manual_seed(0)
np.random.seed(0)

def load_model(model):

	model = torch.load(model, map_location=torch.device('cpu'))

	return model

class Fourier_Neural_Operator(object):

	def __init__(self, data, model, train_size, modes, width, epochs, learning_rate):

		self.train_size = train_size
		self.modes = modes
		self.width = width
		self.epochs = epochs
		self.learning_rate = learning_rate

		self.data = np.load(data)
		self.model = torch.load(model, map_location=torch.device('cpu'))


	def process_data(self):

		sub = 1
		S = 64
		T_in = 10
		self.T = 10
		self.step = 1

		self.batch_size = 20

		D = self.data 
		ntrain = self.train_size
		ntest = D.shape[0] - ntrain

		# Training dataset
		# train_a = torch.tensor(mat['u'][:ntrain, ::sub, ::sub, :T_in])
		# train_u = torch.tensor(mat['u'][:ntrain, ::sub, ::sub, T:T+T_in])
		train_a = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, :T_in])
		train_u = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, self.T:self.T+T_in])

		# Test dataset
		# test_a = torch.tensor(mat['u'][-ntest:,::sub,::sub,:T_in])
		# test_u = torch.tensor(mat['u'][-ntest:,::sub,::sub,T_in:T+T_in])
		test_a = torch.tensor(D[-ntest:,::sub,::sub,::sub,:T_in])
		test_u = torch.tensor(D[-ntest:,::sub,::sub,::sub,T_in:self.T+T_in])

		print(train_u.shape)
		print(test_u.shape)
		# assert (S == train_u.shape[-2])
		# assert (T == train_u.shape[-1])

		train_a = train_a.reshape(ntrain,S,S,2,T_in)
		test_a = test_a.reshape(ntest,S,S,2,T_in)

		# pad the location (x,y)
		gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
		gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
		self.gridx = gridx.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])
		gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
		gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])
		self.gridy = gridy.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])

		train_a = torch.cat((train_a, self.gridx.repeat([ntrain,1,1,1,1]), self.gridy.repeat([ntrain,1,1,1,1])), dim=-1)
		test_a = torch.cat((test_a, self.gridx.repeat([ntest,1,1,1,1]), self.gridy.repeat([ntest,1,1,1,1])), dim=-1)

		self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=self.batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=self.batch_size, shuffle=False)

	def predict(self, n):

		# n : batch number in the test data
		# model : pretrained_model 
		# Net2d(self.modes, self.width)
		# SimpleBlock2d(self.modes, self.modes, self.width)
		# SpectralConv2d_fast(self.width, self.width, self.modes, self.modes)
		model = self.model

		xx = []
		yy = []
		yh = []
		for x, y in self.test_loader:
		    xx.append(x)
		    yy.append(y)

		x = xx[n]

		t1 = default_timer()
		for t in range(0, self.T, self.step):
		    
		    
		    im = model(x.float())
		    
		    if t ==0:
		        pred = im
		    else:
		        pred = torch.cat((pred, im), -1)
		        
		    x = torch.cat((x[..., self.step:-2], im, self.gridx.repeat([self.batch_size, 1, 1, 1, 1]), self.gridy.repeat([self.batch_size, 1, 1, 1, 1])), dim=-1)
		t2 = default_timer()
		print(t2-t1)

		self.input = x
		self.true_output = yy[n]
		self.pred_output = pred

		print(self.input.shape, self.true_output.shape, self.pred_output.shape)

		return pred, yy[n]


	def print_images(self, pred = None):

		if pred == None:
			pred_data = self.pred_output
		else:
			pred_data = pred

		fig = plt.figure(figsize=(40., 40.))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
		                nrows_ncols=(1, 10),  # creates 2x2 grid of axes
		                axes_pad=0.1,  # pad between axes in inch.
		                )
		for i, ax in zip(range(0, 10),grid):
		    # Iterating over the grid returns the Axes. #shape of arr is (nx, ny, time, channel)
		    ax.imshow(np.array(pred_data.detach())[19,:,:,0,i])
		    ax.set_title("Timestep: '{0}'".format(i))


	def compare_images(self, batch, channel, timestep):

		fig = plt.figure(figsize=(40., 40.))
		grid = ImageGrid(fig, 111,  # similar to subplot(111)
		                nrows_ncols=(1, 2),  # creates 2x2 grid of axes
		                axes_pad=0.1,  # pad between axes in inch.
		                )
		img_arr = [self.true_output[batch,:,:,channel,timestep], self.pred_output[batch,:,:,channel,timestep]]
		for i, ax in zip(img_arr, grid):
		    # Iterating over the grid returns the Axes. #shape of arr is (nx, ny, time, channel)
		    ax.imshow(i.detach())

	def print_images(self, pred = None):

		if pred == None:
			pred_data = self.pred_output
		else:
			pred_data = pred





	# def show_error(self, batch, channel):

	# 	y_true = np.array(self.true_output.detach())[batch,:,:,channel,:]
	# 	y_pred = np.array(self.pred_output.detach())[batch,:,:,channel,:]

	# 	# print(mean_squared_error(y_true, y_pred, multioutput = 'raw_values'), mean_squared_error(y_true, y_pred))
	# 	print(mean_squared_error(y_true, y_pred))

	def show_error(self, batch, channel, timestep, s_avg = True, red = True, reduction = 'mean'):

		y_true = self.true_output[batch,:,:,channel,timestep]
		y_pred = self.pred_output[batch,:,:,channel,timestep]

		loss = nn.MSELoss(size_average = s_avg, reduce = red, reduction = reduction)

		return loss(y_true, y_pred)





 
  