from Fourier_Neural_Operator import Fourier_Neural_Operator as FNO
from Fourier_Neural_Operator import SpectralConv2d_fast, SimpleBlock2d, Net2d
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import numpy as np
import torch

# def plot_images(data):
#     fig = plt.figure(figsize = (40., 40.))
#     grid = ImageGrid(fig, 111, nrows_ncols = (1,10), axes_pad=0.1)
#     for i, ax in zip(range(0, 10), grid):
#         ax.imshow(np.array(data.detach())[19,:,:,0,i])
#         ax.set_title("Timestep: '{0}'".format(i))
#     fig.savefig('prediction')

# solver = FNO('Data_dt5.npy', 'model_ts5.pt', width = 10)
# solver.process_data()
# prediction5, y, x = solver.predict(5, 1000)


# np.save('prediction.npy', prediction5.detach())
# plot_images(prediction5)

# checkpoint = torch.load('model_ts5.pt', map_location=torch.device('cpu'))
checkpoint = torch.load('model_ts5.pt')
model = Net2d(12, 10)
model.load_state_dict(checkpoint['model_state_dict'])

D = np.load('Data_dt5_time_jump.npy')

sub = 1
S = 64
T_in = 10
T = 10
step = 1

ntest = 2

batch_size = 20

test_a = torch.tensor(D[-ntest:,::sub,::sub,::sub,:T_in])
test_u = torch.tensor(D[-ntest:,::sub,::sub,::sub,T_in:])

gridx = torch.tensor(np.linspace(0, 64, S), dtype=torch.float)
gridx = gridx.reshape(1, S, 1, 1).repeat([1, 1, S, 1])
gridx = gridx.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])
gridy = torch.tensor(np.linspace(0, 64, S), dtype=torch.float)
gridy = gridy.reshape(1, 1, S, 1).repeat([1, S, 1, 1])
gridy = gridy.reshape(1, S, S, 1, 1).repeat([1, 1, 1, 2, 1])

test_a = torch.cat((test_a, gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1])), dim=-1)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

xx = []
yy = []
yh = []
for x, y in test_loader:
    xx.append(x)
    yy.append(y)

x = xx[0]

for t in range(0, 1500, step):


    im = model(x.float())

    if t ==0:
        pred = im
    else:
        pred = torch.cat((pred, im), -1)
        # pred = im

    x = torch.cat((x[..., step:-2], im, gridx.repeat([ntest, 1, 1, 1, 1]), gridy.repeat([ntest, 1, 1, 1, 1])), dim=-1)

np.save('prediction_jump.npy', pred)