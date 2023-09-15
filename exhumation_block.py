#SETUP
import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
import copy 
from exh_functions import *
from exh_processing import * 
from default_configs import *
from os import makedirs

created_parser = parser()
args = created_parser.parse_args()
n_draws = args.ndraws
res = args.resolution #samples coord every res voxels
interval = args.interval #calculates exhumation every interval m 
label = generate_unique_label()

model_exhumation_folder = f"{output_folder}/{model_name}/model_exhumation/{args.folder}/"
makedirs(model_exhumation_folder,exist_ok=True)

print(f"[{time_string()}] {'Simulating based on file':<40} {history}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
print()

#LOAD NODDY MODEL
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
fault_list = [11,12,13,15]
prop_list = ['Slip', 'Amplitude']

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

all_params = np.load(all_params,allow_pickle = True)

#EXHUMATION BLOCK CALCULATION
print(f"[{time_string()}] Starting exhumation calculation")
hist_copy = copy.deepcopy(hist)

for i in range(len(all_params)):
    print(f"Processing #{i} out of {len(all_params)}")
    for f, fault in enumerate(fault_list):
        for p, prop in enumerate(prop_list):
            hist_copy.events[fault].properties[prop] = all_params[i][p+f*2]
            
    model_coords,_,_ = exhumationComplex(i,hist_copy, lith, res, interval, upperlim, label) 
    exh_block,_ = exhumation_grid_single(model_coords, out_hd, res, zdim)

    np.save(f"{model_exhumation_folder}/exh_block_{label}.npy", exh_block)

print(f"[{time_string()}] Complete")
clean(label)