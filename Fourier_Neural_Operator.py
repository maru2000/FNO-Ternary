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

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul2d(a, b):
    # print(a.type(), b.type())
    return torch.einsum('...bixyz,...ioxyz->...boxyz', a, b)
#     op = partial(torch.einsum, "bctq,dctq->bdtq")
#     return torch.stack([
#         op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
#         op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
#     ], dim=-1)

## Fourier Layer

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype = torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype = torch.complex64))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x, dim = (-3, -2), norm = "forward")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2)//2 + 1, 2, dtype = torch.complex64)
        out_ft[:, :, :self.modes1, :self.modes2, :] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2, :], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2, :], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, dim = (-3, -2), norm = "forward")
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        self.w2 = nn.Conv3d(self.width, self.width, 1)
        self.w3 = nn.Conv3d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        # print(x1.size(), x2.size())
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width)


    def forward(self, x):
        x = self.conv1(x)
        return x


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


class Fourier_Neural_Operator(object):

	def __init__(self, data, model, train_size = 800, modes = 12, width = 20):

		# SpectralConv2d_fast(modes, modes, width)
		# SimpleBlock2d()

		self.train_size = train_size
		self.modes = modes
		self.width = width
		# self.epochs = epochs
		# self.learning_rate = learning_rate

		self.data = np.load(data)
		checkpoint = torch.load(model, map_location=torch.device('cpu'))
		self.model = Net2d(self.modes, self.width)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		# self.model = torch.load(model, map_location=torch.device('cpu'))

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
		# train_a = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, :T_in])
		# train_u = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, self.T:self.T+T_in])

		# Test dataset
		# test_a = torch.tensor(mat['u'][-ntest:,::sub,::sub,:T_in])
		# test_u = torch.tensor(mat['u'][-ntest:,::sub,::sub,T_in:T+T_in])
		test_a = torch.tensor(D[-ntest:,::sub,::sub,::sub,:T_in])
		test_u = torch.tensor(D[-ntest:,::sub,::sub,::sub,T_in:self.T+T_in])

		# print(train_u.shape)
		# print(test_u.shape)
		# assert (S == train_u.shape[-2])
		# assert (T == train_u.shape[-1])

		# train_a = train_a.reshape(ntrain,S,S,2,T_in)
		test_a = test_a.reshape(ntest,S,S,2,T_in)

		# pad the location (x,y)
		gridx = torch.tensor(np.linspace(0, S, S), dtype=torch.float)
		gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
		self.gridx = gridx.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])
		gridy = torch.tensor(np.linspace(0, S, S), dtype=torch.float)
		gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])
		self.gridy = gridy.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])

		# train_a = torch.cat((train_a, self.gridx.repeat([ntrain,1,1,1,1]), self.gridy.repeat([ntrain,1,1,1,1])), dim=-1)
		test_a = torch.cat((test_a, self.gridx.repeat([ntest,1,1,1,1]), self.gridy.repeat([ntest,1,1,1,1])), dim=-1)

		# self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=self.batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=self.batch_size, shuffle=False)

	def predict(self, n, final_ts = None):

		# n : batch number in the test data
		# model : pretrained_model 
		# Net2d(self.modes, self.width)
		# SimpleBlock2d(self.modes, self.modes, self.width)
		# SpectralConv2d_fast(self.width, self.width, self.modes, self.modes)
		model = self.model

		if final_ts == None:
			final_ts = self.T
		else:
			final_ts = final_ts

		xx = []
		yy = []
		yh = []
		for x, y in self.test_loader:
		    xx.append(x)
		    yy.append(y)

		x = xx[n]

		t1 = default_timer()
		for t in range(0, final_ts, self.step):
		    
		    
		    im = model(x.float())
		    
		    if t ==0:
		        pred = im
		    else:
		        pred = torch.cat((pred, im), -1)
		        # pred = im
		        
		    x = torch.cat((x[..., self.step:-2], im, self.gridx.repeat([self.batch_size, 1, 1, 1, 1]), self.gridy.repeat([self.batch_size, 1, 1, 1, 1])), dim=-1)
		t2 = default_timer()
		# print(t2-t1)

		self.input = x
		self.true_output = yy[n]
		self.pred_output = pred

		# print(self.input.shape, self.true_output.shape, self.pred_output.shape)

		return pred, yy[n], xx[n]


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
		    ax.imshow(np.array(self.pred_output.detach())[19,:,:,0,i])
		    ax.set_title("Timestep: '{0}'".format(i))

		return fig

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
 




 
