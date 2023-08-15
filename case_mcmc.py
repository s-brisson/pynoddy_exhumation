
from exh_functions import *
from exh_processing import *
from default_configs import * 
import numpy as np
import pandas as pd
import copy, pickle, os
from os import makedirs
import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.output

created_parser = parser()
args = created_parser.parse_args()
n_draws = args.ndraws
res = args.resolution #samples coord every res voxels
interval = args.interval #calculates exhumation every interval m 

label = generate_unique_label()

#model_coords_folder = f"{output_folder}/{model_name}/model_coords/{args.folder}/"
#model_blocks_folder = f"{output_folder}/{model_name}/model_blocks/{args.folder}/"
model_scores_folder = f"{output_folder}/{model_name}/model_scores/{args.folder}/"
model_params_folder = f"{output_folder}/{model_name}/model_params/{args.folder}/"
model_samples_folder = f"{output_folder}/{model_name}/model_samples/{args.folder}/"
#makedirs(model_coords_folder,exist_ok=True)
#makedirs(model_blocks_folder,exist_ok=True)
makedirs(model_scores_folder,exist_ok=True)
makedirs(model_params_folder,exist_ok=True)
makedirs(model_samples_folder,exist_ok=True)

print(f"[{time_string()}] {'Simulating based on file':<40} {history}")
print(f"[{time_string()}] {'Simulating based on observation':<40} {samples}")
print(f"[{time_string()}] {'Number of simulations':<40} {args.ndraws}")
print(f"[{time_string()}] {'Indicator layer z-axis step size':<40} {interval}")
print(f"[{time_string()}] {'Voxels resampling resolution':<40} {res}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
print()

#LOAD THE NODDY MODEL
print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history, output_name, 
                      noddy_path = noddy_exe,
                      verbose=True)
hist = pynoddy.history.NoddyHistory(history)
hist.change_cube_size(cubesize)
hist_hd = f'{output_folder}/history/hist_hd_{label}.his'
out_hd = f'{output_folder}/noddy/out_hd_{label}'
hist.write_history(hist_hd)
print(f"[{time_string()}] Running the HD model")
pynoddy.compute_model(hist_hd, out_hd, noddy_path = noddy_exe)
out_hd = pynoddy.output.NoddyOutput(out_hd)

#DEFINE IMPORTANT VALUES
samples = pd.read_csv(samples, delimiter = ',')
upperlim = out_hd.zmax
for event_name, event in hist.events.items():
    if isinstance(event, pynoddy.events.Dyke):
        ori_depth = event.properties['Z'] #original depth of the indicator
res_z = np.floor((np.abs(ori_depth) + out_hd.zmax) / interval) #resolution in z
min_depth = (res_z * interval) + ori_depth #minimum depth the indicator gets to

xdim,ydim,zdim = calc_dims(out_hd, res, zdim)
x = np.arange(0, xdim+1)
y = np.arange(0, ydim+1)
z = np.arange(0, zdim+1)
grid = (x,y,z)

#STARTING PARAMS AND EXHUMATION
current_hist = copy.deepcopy(hist)

original_params = []
for event_name, event in hist.events.items():
    if isinstance(event, pynoddy.events.Fault):
        slip = event.properties['Slip']
        amplitude = event.properties['Amplitude']
        x = event.properties['X']
        dipdir = event.properties['Dip Direction']
        
        original_params.append((event_name, slip, amplitude, x, dipdir))
std_list = [400,100,50,5]

print(f"[{time_string()}] Calculating original exhumation")
current_coords, current_out,_ = exhumationComplex(0,current_hist, lith, res, interval, upperlim)
current_exh_block, _ = exhumation_grid_single(current_coords, current_out, res, zdim+1)
samples['respected'] = 0
model_score, current_exhumation = interp_and_score(current_exh_block, samples, cubesize, res, zdim, min_depth, grid)

#ACTUAL SIMULATION
scores = []
accepted = 0
all_params = pd.DataFrame(columns = ['Event', 'Slip','Amplitude', 'X','Dip Direction', 'n_draw'])

current_params = original_params

for i in range(n_draws):
    while accepted < n_draws:

        #a random model is proposed and its exhumation block is calclated
        proposed_params,_ = disturb(current_hist, std_list, i)

        proposed_coords, proposed_out, proposed_hist = exhumationComplex(current_hist, lith, res, interval, upperlim)
        exh_block, _ = exhumation_grid_single(proposed_coords, out_hd, res, zdim+1)
        #interpolation step
        model_score, proposed_exhumation = interp_and_score(exh_block, samples, cubesize, res, zdim, min_depth, grid)
        print(f"ndraw {i} model_score {model_score}")
        #accept or reject step
        acceptance_ratio = (prior_dist(original_params,proposed_params,std_list) * likelihood(proposed_exhumation)) / (prior_dist(original_params,current_params,std_list) * likelihood(current_exhumation))
        
        if acceptance_ratio > np.random.rand(1):
            #accept
            current_params = proposed_params
            current_exhumation = proposed_exhumation
            current_hist = proposed_hist

            #store
            scores.append([model_score, i])
            all_params = pd.concat([all_params, current_params], ignore_index = True)
            accepted += 1
        #otherwise reject

scores = pd.DataFrame(scores, columns = ['Score', 'iteration'])
all_params.to_csv(f'{model_params_folder}/params_{label}.csv', index = False)

print(f"[{time_string()}] Complete")
clean(label)
