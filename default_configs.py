import numpy as np
#### INPUTS
history = 'data/input_files/bregenz_ver5.his'
all_params = 'data/input_files/training_params_310.npy'
history_samples = 'data/input_files/bregenz_samples.his'
samples = 'data/input_files/bregenz_data.csv'
noddy_exe = "/rwthfs/rz/cluster/home/ho640525/projects/pynoddy/pynoddy/noddyapp/noddy"
save_each = False
save_overall = True
#### Modeling params
cubesize = 200
lith = [11] #lith ID for the indicator layer.

zdim = 65

#### OUTPUTS
output_folder = "data/outputs"
model_name = "transalp"

maps = [[21,-1000,-11000],
       [22,2000,-9500],
       [23,-2000,-5000]]
synthetic_data = [[21, 3400],[22, 3300],[24,5400]]
sigma = 800

#INPUTS FOR MCMC (without indicator layer)
event = np.arange(25,30) #events to modify
prop = ['Slip'] #properties to modify
std = [2000,800,1000,600] #
lith_list = np.arange(16,28) #lith IDs for the samples being analyzed.
sample_num = [4] #sample indices

#TRANSALP INPUTS
history_transalp = 'data/input_files/trans004.his'
synth_samples = 'data/input_files/synth_samples.csv'
avg_conv_factor = ((5872/2211.25) + (10999/4193.75) + (8729/3355) + (2698/991.25) + (9999/3812.5)) / 5
geo_gradient = 25
