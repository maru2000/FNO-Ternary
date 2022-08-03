from Fourier_Neural_Operator_3d import SpectralConv3d_fast, SimpleBlock3d, Net3d
# from mpl_toolkits.aes_grid1 import ImageGrid
import matplotlib.pyplot as plt 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

from utilities import *


checkpoint = torch.load('model_fno3_dt1.pt')
model = Net3d(4,20).cuda()
model.load_state_dict(checkpoint['model_state_dict'])

## Params

ntest = 2

modes = 4
width = 20

batch_size = 10
batch_size2 = batch_size

learning_rate = 0.0025
scheduler_step = 100
scheduler_gamma = 0.5

print(learning_rate, scheduler_step, scheduler_gamma)


runtime = np.zeros(2, )
t1 = default_timer()

sub = 1
S = 64 // sub
T_in = 10
T = 1000

## Data loading

D = np.load('Data_dt1_retrain_lt.npy')
print(D.shape)

# Data Preparation

test_a = torch.tensor(D[-ntest:,::sub,::sub,::sub,:T_in])
test_u = torch.tensor(D[-ntest:,::sub,::sub,::sub,T_in:T+T_in])

print(test_u.shape)

a_normalizer = UnitGaussianNormalizer(test_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(test_u)
test_u = y_normalizer.encode(test_u)

test_a = test_a.reshape(ntest,S,S,2,1,T_in).repeat([1,1,1,1,T,1])

print(test_a.shape)

# pad locations (x,y,t)
gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1, 1, 1).repeat([1, 1, S, 1, T, 1])
gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1, 1, 1).repeat([1, S, 1, 1, T, 1])
gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
gridt = gridt.reshape(1, 1, 1, 1, T, 1).repeat([1, S, S, 1, 1, 1])

test_a = torch.cat((gridx.repeat([ntest,1,1,2,1,1]), gridy.repeat([ntest,1,1,2,1,1]),
    gridt.repeat([ntest,1,1,2,1,1]), test_a), dim=-1)
print(test_a.shape)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

pred = torch.zeros(test_u.shape)
index = 0
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
y_normalizer.cuda()
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.cuda(), y.cuda()

        out = model(x)
        out = y_normalizer.decode(out)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        print(index, test_l2)
        index = index + 1

np.save('prediction_fno3.npy', pred)
