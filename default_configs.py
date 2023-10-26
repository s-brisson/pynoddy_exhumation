#### INPUTS
history = 'data/input_files/bregenz_ver5.his'
all_params = 'data/input_files/training_params_320.npy'
history_samples = 'data/input_files/bregenz_samples.his'
samples = 'data/input_files/bregenz_data.csv'
noddy_exe = "/rwthfs/rz/cluster/home/ho640525/projects/pynoddy/pynoddy/noddyapp/noddy"
save_each = False
save_overall = True
#### Modeling params
cubesize = 100
lith = [11] #lith ID for the indicator layer.

zdim = 65

#### OUTPUTS
output_folder = "data/outputs"
model_name = "bregenz"

maps = [[21,-1000,-11000],
       [22,2000,-9500],
       [23,-2000,-5000]]
synthetic_data = [[21, 3400],[22, 3300],[24,5400]]
sigma = 800

#INPUTS FOR MCMC (without indicator layer)
event = [21,22,24] #events to modify
prop = ['Z','Slip'] #properties to modify
std = [150,400] #
lith_list = [13,15,17] #lith IDs for the samples being analyzed.
sample_num = [2,4,6] #sample indices
