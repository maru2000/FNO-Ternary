Model indexes: 
model.pt - Model trained on Data.npy - normal FNO: gridsize - linspace(0,1,64) - epochs - 500
model-ts1.py - Model trained on D1.npy - "" - epochs - 500
model-new.py - Model trained on D_new.npy - epochs - 500

model_mod.py - Model trained on D_new.npy - FNO grid: linspace(1,64,64) - epochs - 250, width = 10
model_mod500.py - Model trained on D_new.npy - "" - epochs - 500, width = 10

Datasets:
Data.py - 1000 datapoints - time difference - 200
D_new.py - 1000 datapoints - time difference - 20
D1.py - 1000 datapoints - time difference - 1

Training file - train.py: In the system: ~vir/neuralpde

------------------------------------------------------------------------------

Data_Format:

[batch_size, gridx, gridy, channel, timestep]


Using FNO function test jupyter notebook:

First cell: All required imports from the script files in the folder

Second cell: Plot function to give images of all the test and predict data for T:T+10 tiemsteps

Third cell onwards: Get Results using the following code:
	solver = FNO(Data_file, Model_file, train_size = 800, modes = 12, width = 20) # Initialize solver with the pretrained model file and dataset
	solver.process_data() # Does dataprocessing by appending grid size
	prediction, y = solver.predict(batch_num) # batch number in the test data: Currently the batches in test data are 20 - use one of them to predict
	for i in range(0, 10):
		print(solver.show_error(19, 0, i)) # Prints MSELoss for any one batch entry in a batch of 20 for all the timesteps given a channel
						   # solver.show_error(batch_entry, channel, timestep)
	plot_images(prediction) # Plots for predicted data
	plot_images(y) # Plots for ground truth