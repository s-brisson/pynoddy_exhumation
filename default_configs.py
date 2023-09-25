#### INPUTS
history = 'data/input_files/bregenz_ver5.his'
all_params = 'data/input_files/validation_params_20.npy'
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

maps = [[21,29387,-744.58,-10283.4,4536.86]]

#INPUTS FOR MCMC (without indicator layer)
event = [20,21,22,24] #events to modify
prop = ['Z','Slip'] #properties to modify
std = [500,1000] #
lith_list = [12,13,15,17] #lith IDs for the samples being analyzed.
sample_num = [1,2,4,6] #sample indices
