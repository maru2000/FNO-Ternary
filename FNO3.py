## Imports

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv3d(nn.Module):
	def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
		super(SpectralConv3d, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.modes1 = modes1
		self.modes2 = modes2
		self.modes3 = modes3

		self.scale = (1 / (in_channels * out_channels))
		self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.complex64))
		self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.complex64))
		self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.complex64))
		self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype=torch.complex64))

	def compl_mul3d(self, inputs, weights):

		return torch.einsum("bixyzn, ioxyzn -> boxyzn", inputs, weights)

	def forward(self, x):

		batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
		x_ft = torch.fft.rfftn(x, dim=[-4,-3,-2])

        # Multiply relevant Fourier modes
		out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2)//2 + 1, 2, dtype=torch.complex64, device=x.device)
		out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :], self.weights1)
		out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :], self.weights2)
		out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :], self.weights3)
		out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :], self.weights4)

        #Return to physical space
		x = torch.fft.irfftn(out_ft, dim=[-4,-3,-2])

		return x

class SimpleBlock3d(nn.Module):

	def __init__(self, modes1, modes2, modes3, width):
		super(SimpleBlock3d, self).__init__()

		self.modes1 = modes1
		self.modes2 = modes2
		self.modes3 = modes3
		self.width = width
		self.fc0 = nn.Linear(13, self.width)

		self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
		self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
		self.conv2 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
		self.conv3 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
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
		size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

		x = self.fc0(x)
		x = x.permute(0, 5, 1, 2, 3, 4)

		xp = self.conv0(x)
		x1 = self.w0(x[...,0])
		x1 = self.bn0(x1)
		x2 = self.w0(x[...,1])
		x2 = self.bn0(x2)
		xc = torch.cat([x1.unsqueeze(5),x2.unsqueeze(5)], dim = -1)
		x = F.relu(xp+xc)


		xp = self.conv1(x)
		x1 = self.w1(x[...,0])
		x1 = self.bn1(x1)
		x2 = self.w1(x[...,1])
		x2 = self.bn1(x2)
		xc = torch.cat([x1.unsqueeze(5),x2.unsqueeze(5)], dim = -1)
		x = F.relu(xp+xc)


		xp = self.conv2(x)
		x1 = self.w2(x[...,0])
		x1 = self.bn2(x1)
		x2 = self.w2(x[...,1])
		x2 = self.bn2(x2)
		xc = torch.cat([x1.unsqueeze(5),x2.unsqueeze(5)], dim = -1)
		x = F.relu(xp+xc)

		x = x.permute(0, 2, 3, 4, 5, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)

		return x

class Net3d(nn.Module):
	def __init__(self, modes, width):
		super(Net3d, self).__init__()

		"""
		A wrapper function
		"""

		self.conv1 = SimpleBlock3d(modes, modes, modes, width)


	def forward(self, x):
		x = self.conv1(x)
		return x.squeeze()


	def count_params(self):
		c = 0
		for p in self.parameters():
			c += reduce(operator.mul, list(p.size()))

		return c



# ## Initial Params

# ntrain = 900
# ntest = 200

# modes = 8
# width = 10

# batch_size = 20
# batch_size2 = batch_size

# epochs = 500
# learning_rate = 0.0025
# scheduler_step = 100
# scheduler_gamma = 0.5

# print(epochs, learning_rate, scheduler_step, scheduler_gamma)

# # path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
# # path_model = 'model/'+path
# # path_train_err = 'results/'+path+'train.txt'
# # path_test_err = 'results/'+path+'test.txt'
# # path_image = 'image/'+path

# runtime = np.zeros(2, )
# t1 = default_timer()

# sub = 1
# S = 64
# T_in = 10
# T = 20
# step = 1

# ## Load data
# D = np.load('Data_dt5_3d.npy')

# # Training dataset
# # train_a = torch.tensor(mat['u'][:ntrain, ::sub, ::sub, :T_in])
# # train_u = torch.tensor(mat['u'][:ntrain, ::sub, ::sub, T:T+T_in])
# train_a = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, :T_in])
# train_u = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, T_in:T+T_in])

# # Test dataset
# # test_a = torch.tensor(mat['u'][-ntest:,::sub,::sub,:T_in])
# # test_u = torch.tensor(mat['u'][-ntest:,::sub,::sub,T_in:T+T_in])
# # test_a = torch.tensor(D[-ntest:,::sub,::sub,::sub,:T_in])
# # test_u = torch.tensor(D[-ntest:,::sub,::sub,::sub,T_in:T+T_in])

# print(train_u.shape)
# # print(test_u.shape)
# # assert (S == train_u.shape[-2])
# # assert (T == train_u.shape[-1])

# train_a = train_a.reshape(ntrain,S,S,1,2,T_in).repeat([1,1,1,T,1,1])
# train_u = train_u.reshape(ntrain,S,S,T,2)
# # test_a = test_a.reshape(ntest,S,S,2,T_in)

# # pad the location (x,y)
# gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
# gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
# gridx = gridx.reshape(1, S, S, 1, 1, 1).repeat([1, 1, 1, T, 2, 1])
# gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
# gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])
# gridy = gridy.reshape(1, S, S, 1, 1, 1).repeat([1, 1, 1, T, 2, 1])
# gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
# gridt = gridt.reshape(1, 1, 1, T, 1, 1).repeat([1, S, S, 1, 2, 1])

# print(train_a.shape)
# print(gridx.shape, gridy.shape, gridt.shape)
# train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1,1]), gridt.repeat([ntrain,1,1,1,1,1]), train_a), dim=-1)
# # test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1])), dim=-1)

# print(train_a.shape)
# # print(test_a.shape)

# train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
# # test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

# t2 = default_timer()

# print('preprocessing finished, time used:', t2-t1)
# device = torch.device('cuda')

# ## Model train

# model = Net3d(modes, width).cuda()
# # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

# print(model.count_params())
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# myloss = nn.L1Loss(size_average=False)

# for ep in range(epochs):
# 	model.train()
# 	t1 = default_timer()
# 	train_mse = 0
# 	train_l2 = 0
# 	for x, y in train_loader:
# 		x, y = x.cuda(), y.cuda()

# 		optimizer.zero_grad()
# 		out = model(x.float())

# 		mse = F.mse_loss(out, y, reduction='mean')
# 		# mse.backward()

# 		# y = y_normalizer.decode(y)
# 		# out = y_normalizer.decode(out)
# 		l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
# 		l2.backward()

# 		optimizer.step()
# 		train_mse += mse.item()
# 		train_l2 += l2.item()

# 	train_mse /= len(train_loader)
# 	train_l2 /= ntrain
# 	# test_l2 /= ntest

# 	t2 = default_timer()
# 	print(ep, t2-t1, train_mse, train_l2)

# 	torch.save({
# 	'epoch': ep,
# 	'model_state_dict':model.state_dict(),
# 	'optimizer_state_dict': optimizer.state_dict(),
# 	'loss':loss,
# 	}, 'model_ts5_3d.pt')
