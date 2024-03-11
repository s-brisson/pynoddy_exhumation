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
sample_num = [4] #sample indices

#TRANSALP INPUTS
history_transalp = 'data/input_files/trans006.his'
synth_samples = 'data/input_files/synth_samples_model2.csv'
avg_conv_factor = ((7500/3372.5) + (4602/2041.25)+ (9999/4437.5)) / 3
geo_gradient = 25
prop = ['X', 'Z', 'Amplitude','Slip'] #properties to modify
event = np.arange(22,31) #events to modify
std = [2000,800,1000,600] #
lith_list = np.arange(16,28) #lith IDs for the samples being analyzed.
