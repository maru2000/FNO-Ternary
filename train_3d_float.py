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

from utilities import *

torch.manual_seed(0)
np.random.seed(0)

#### Model

## Spectral Layer

#Complex multiplication
def compl_mul3d(a, b):
    # print(a.type(), b.type())
    return torch.einsum('...bixytc,...ioxytc->...boxytc', a, b)
#     op = partial(torch.einsum, "bctq,dctq->bdtq")
#     return torch.stack([
#         op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
#         op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
#     ], dim=-1)

## Fourier Layer

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex64))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex64))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex64))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim = (-4, -3, -2), norm = "forward")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2)//2 + 1, 2, dtype = torch.complex64, device = x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :] = \
                compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :] = \
                compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :] = \
                compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :] = \
                compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, dim = (-4, -3, -2), norm = "forward")
        return x

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width):
        super(SimpleBlock3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)


        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_c, size_t = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        #x = x.float()
        #print('xtype', x.dtype)

        x = self.fc0(x)
        x = x.permute(0,5,1,2,4,3)
        #print('xtype', x.dtype)
        x1 = self.conv0(x)
        x2 = self.w0(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = self.bn0((x1+x2).reshape(batchsize, self.width, size_x, size_y, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = F.relu(x)
        #print('xtype', x.dtype)
        x1 = self.conv1(x)
        x2 = self.w1(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = self.bn1((x1+x2).reshape(batchsize, self.width, size_x, size_y, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = self.bn2((x1+x2).reshape(batchsize, self.width, size_x, size_y, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = self.bn3((x1+x2).reshape(batchsize, self.width, size_x, size_y, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)


        x = x.permute(0, 2, 3, 5, 4, 1)
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


## Params

ntrain = 800
ntest = 200

modes = 4
width = 20

batch_size = 10
batch_size2 = batch_size

epochs = 100
learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)


runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64 // sub
T_in = 10
T = 20

## Data loading

D = np.load('Data_dt1_ts50.npy')
print(D.dtype)
# Data Preparation


train_a = torch.tensor(D[:ntrain,::sub,::sub,::sub,:T_in])
train_u = torch.tensor(D[:ntrain,::sub,::sub,::sub,T_in:T+T_in])

test_a = torch.tensor(D[-ntest:,::sub,::sub,::sub,:T_in])
test_u = torch.tensor(D[-ntest:,::sub,::sub,::sub,T_in:T+T_in])

print(train_u.shape)
print(test_u.shape)

train_a = train_a.reshape(ntrain,S,S,2,1,T_in).repeat([1,1,1,1,T,1])
test_a = test_a.reshape(ntest,S,S,2,1,T_in).repeat([1,1,1,1,T,1])

print(train_a.shape, test_a.shape)

# pad locations (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1, 1).repeat([1, 1, S, 1, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1, 1).repeat([1, S, 1, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, 1, T, 1).repeat([1, S, S, 1, 1, 1])

train_a = torch.cat((gridx.repeat([ntrain,1,1,2,1,1]), gridy.repeat([ntrain,1,1,2,1,1]),
    gridt.repeat([ntrain,1,1,2,1,1]), train_a), dim=-1)
test_a = torch.cat((gridx.repeat([ntest,1,1,2,1,1]), gridy.repeat([ntest,1,1,2,1,1]),
    gridt.repeat([ntest,1,1,2,1,1]), test_a), dim=-1)
print(train_a.shape, test_a.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda:1')


################################################################
# training and evaluation
################################################################

model = Net3d(modes, width).cuda()
# model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

myloss = nn.L1Loss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)


print(model.count_params())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

loss_list = []

for ep in range(epochs):
    print('epoch:', ep)
    model.train()
    t1 = default_timer()

    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda().float(), y.cuda().float()

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out, y, reduction='mean')
        # mse.backward()

        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    # model.eval()
    # test_l2 = 0.0
    # with torch.no_grad():
    #     for x, y in test_loader:
    #         x, y = x.cuda(), y.cuda()

    #         out = model(x)
    #         out = y_normalizer.decode(out)
    #         test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    # test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2-t1, train_mse, train_l2)

    loss_list.append(l2.detach().cpu())

    torch.save({
        'epoch': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': np.array(loss_list)
        }, 'model_fno3_dt1_10_20_ep50_w20_test.pt')
