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

## Model

#Complex multiplication
def compl_mul2d(a, b):
    # print(a.type(), b.type())
    return torch.einsum('...bixyz,...ioxyz->...boxyz', a, b)
#     op = partial(torch.einsum, "bctq,dctq->bdtq")
#     return torch.stack([
#         op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
#         op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
#     ], dim=-1)


################################################################
# fourier layer
################################################################

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
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2)//2 + 1, 2, dtype = torch.complex64, device = x.device)
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

## Initial Params

ntrain = 800
ntest = 200

modes = 12
width = 10

batch_size = 20
batch_size2 = batch_size

epochs = 500
learning_rate = 0.005
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

# path = 'ns_fourier_2d_rnn_V10000_T20_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
# path_model = 'model/'+path
# path_train_err = 'results/'+path+'train.txt'
# path_test_err = 'results/'+path+'test.txt'
# path_image = 'image/'+path

runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64
T_in = 10
T = 10
step = 1

## Load data
D = np.load('Data_dt5_(0.25,0.25).npy')

# Training dataset
# train_a = torch.tensor(mat['u'][:ntrain, ::sub, ::sub, :T_in])
# train_u = torch.tensor(mat['u'][:ntrain, ::sub, ::sub, T:T+T_in])
train_a = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, :T_in])
train_u = torch.tensor(D[:ntrain, ::sub, ::sub, ::sub, T:T+T_in])

# Test dataset
# test_a = torch.tensor(mat['u'][-ntest:,::sub,::sub,:T_in])
# test_u = torch.tensor(mat['u'][-ntest:,::sub,::sub,T_in:T+T_in])
test_a = torch.tensor(D[-ntest:,::sub,::sub,::sub,:T_in])
test_u = torch.tensor(D[-ntest:,::sub,::sub,::sub,T_in:T+T_in])

print(train_u.shape)
print(test_u.shape)
# assert (S == train_u.shape[-2])
# assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,2,T_in)
test_a = test_a.reshape(ntest,S,S,2,T_in)

# pad the location (x,y)
gridx = torch.tensor(np.linspace(0, S, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
gridx = gridx.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])
gridy = torch.tensor(np.linspace(0, S, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])
gridy = gridy.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])

train_a = torch.cat((train_a, gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1])), dim=-1)
test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1])), dim=-1)

print(train_a.shape)
print(test_a.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')


'''

MODEL TRAIN

'''

model = Net2d(modes, width).cuda()

myloss = nn.L1Loss(size_average=False)
gridx = gridx.to(device)
gridy = gridy.to(device)
i = 0

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

loss_list = []

for ep in range(epochs):
    print('epoch:', ep)
    model.train()
    t1 = default_timer()

    train_l2_step = 0
    train_l2_full = 0
    for xx, yy in train_loader:
        loss = 0
        # print(xx.shape, yy.shape)0
        xx = xx.float().to(device)
        yy = yy.float().to(device)
        
        for t in range(0, T, step):
            y = yy[..., t:t+step]
            
            im = model(xx)

            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

                xx = torch.cat((xx[..., step:-2], im, gridx.repeat([batch_size, 1, 1, 1, 1]), gridy.repeat([batch_size, 1, 1, 1, 1])), dim=-1)
            # xx = torch.cat((xx[...,step:], im), dim = -1)
            # print(loss)

        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()
        # print(train_l2_full)
        optimizer.zero_grad()
        loss.backward()
        # l2_full.backward()
        optimizer.step()

    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain)


    loss_list.append(loss)
 
    torch.save({
        'epoch': ep,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':np.array(loss_list),
        }, 'model_ts5_(0.25,0.25).pt')


