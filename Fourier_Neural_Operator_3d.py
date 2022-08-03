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
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex128))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex128))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex128))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2, dtype = torch.complex128))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim = (-4, -3, -2), norm = "forward")

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2)//2 + 1, 2, dtype = torch.complex128, device = x.device)
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
        self.fc0 = nn.Linear(13, self.width).double()
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)


        self.conv0 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = SpectralConv3d_fast(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.w0 = nn.Conv1d(self.width, self.width, 1).double()
        self.w1 = nn.Conv1d(self.width, self.width, 1).double()
        self.w2 = nn.Conv1d(self.width, self.width, 1).double()
        self.w3 = nn.Conv1d(self.width, self.width, 1).double()
        self.bn0 = torch.nn.BatchNorm3d(self.width).double()
        self.bn1 = torch.nn.BatchNorm3d(self.width).double()
        self.bn2 = torch.nn.BatchNorm3d(self.width).double()
        self.bn3 = torch.nn.BatchNorm3d(self.width).double()


        self.fc1 = nn.Linear(self.width, 128).double()
        self.fc2 = nn.Linear(128, 1).double()

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_c, size_t = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        x = self.fc0(x)
        x = x.permute(0,5,1,2,4,3)

        x1 = self.conv0(x)
        x2 = self.w0(x.reshape(batchsize, self.width, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = self.bn0((x1+x2).reshape(batchsize, self.width, size_x, size_y, -1)).reshape(batchsize, self.width, size_x, size_y, size_t, size_c)
        x = F.relu(x)
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