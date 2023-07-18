import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import pickle
from os import makedirs
from exh_functions import *
from exh_processing import *
from default_configs import* 

created_parser = parser()
args = created_parser.parse_args()
n_draws = args.ndraws
res = args.resolution #samples coord every res voxels
interval = args.interval #calculates exhumation every interval m 

label = generate_unique_label()

model_coords_folder = f"{output_folder}/{model_name}/model_coords/{args.folder}/"
model_blocks_folder = f"{output_folder}/{model_name}/model_blocks/{args.folder}/"
model_scores_folder = f"{output_folder}/{model_name}/model_scores/{args.folder}/"
model_params_folder = f"{output_folder}/{model_name}/model_params/{args.folder}/"
model_samples_folder = f"{output_folder}/{model_name}/model_samples/{args.folder}/"
makedirs(model_coords_folder,exist_ok=True)
makedirs(model_blocks_folder,exist_ok=True)
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

print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history, output_name, 
                      noddy_path = noddy_exe,
                      verbose=True)

hist = pynoddy.history.NoddyHistory(history)
out = pynoddy.output.NoddyOutput(output_name)
upperlim = out.zmax - interval

hist.change_cube_size(cubesize)
hist_hd = f'{output_folder}/history/hist_hd_{label}.his'
out_hd = f'{output_folder}/noddy/out_hd_{label}'
hist.write_history(hist_hd)
print(f"[{time_string()}] Running the HD model")
pynoddy.compute_model(hist_hd, out_hd, noddy_path = noddy_exe)
out_hd = pynoddy.output.NoddyOutput(out_hd)


#other important values
for event_name, event in hist.events.items():
    if isinstance(event, pynoddy.events.Dyke): 
        ori_depth = event.properties['Z']
res_z = np.floor((np.abs(ori_depth) + out_hd.zmax) / interval)
min_depth = (res_z * interval) + ori_depth #depth up until the indicator actually gets

#setting up grid for interpolation
xdim,ydim,zdim = calc_dims(out_hd, res, zdim)
x = np.arange(0, xdim+1)
y = np.arange(0, ydim+1)
z = np.arange(0, zdim+1)
grid = (x,y,z)

#for storage
all_coords = []
all_blocks = np.ndarray((n_draws, out_hd.nx, out_hd.ny, out_hd.nz), dtype = 'int')
all_params = pd.DataFrame(columns = ['Event', 'New Dip', 'New Dip Direction', 'New Pitch', 'New Slip', 'New Amplitude', 'New X', 'New Z','nDraw'])
all_scores = []
samples_df = pd.read_csv(samples, delimiter = ',')
sections = np.empty((n_draws, out_hd.nx, out_hd.nz))


#starting counter 
samples_df['respected'] = 0

#simulation starting
for i in tqdm(range(n_draws), desc = 'Lets score em all'):
    hist_copy = copy.deepcopy(hist) 
    
    #add uncertainty and save model parameters
    new_params = disturb(hist_copy,i)
    
    #calculate exhumation for the new model
    coords, output = exhumationComplex(i, hist_copy, lith, res, interval, upperlim, unique_label = label)
    
    #interpolate exhumation for each sample position
    #value to interpolate
    E,_ = exhumation_grid_single(coords, out_hd, res, zdim)
    
    #interpolate and score the model based on how many samples are respected.
    model_score, samples_df = interp_and_score(E, samples_df, cubesize, res, zdim, min_depth, grid)
    print(f"ndraw {i} model_score {model_score}")
    all_scores.append([model_score,i])
    
    #store other important stuff
    all_coords.append(coords)
    all_blocks[i,:,:,:] = output.block
    all_params = pd.concat([all_params, new_params], ignore_index = True)
    sections[i,:,:] = output.block[:,10,:]
    
    if save_each == True:
        np.save('scoring/coords/coords_%04d.npy'%i, coords)
        np.save('scoring/blocks/blocks_%04d.npy'%i, output.block)
        new_params.to_csv('scoring/params/params_%04d.csv'%i, index = False)
        
all_scores = pd.DataFrame(all_scores,columns=["Score","nDraw"])

#calculate entropy so I don't have to save all the blocks.
total_entropy, slice_entropy = calc_entropy(all_blocks, out_hd, n = 1)
pickle.dump(total_entropy, open(f'{model_samples_folder}/entropy_{label}.pkl','wb'))
np.save(f'{model_samples_folder}/slice_entropy_{label}.npy', slice_entropy)

if save_overall == True:
    pickle.dump(all_coords, open(f'{model_coords_folder}/coords_{label}.pkl', 'wb'))
    pickle.dump(sections, open(f'{model_blocks_folder}/sections_{label}.pkl', 'wb')
#    pickle.dump(all_blocks, open(f'{model_blocks_folder}/blocks_{label}.pkl', 'wb'))
    all_params.to_csv(f'{model_params_folder}/params_{label}.csv', index = False)
    all_scores.to_csv(f'{model_scores_folder}/scores_{label}.csv',  index = False)
    samples_df.to_csv(f'{model_samples_folder}/samples_{label}.csv', index = False)
print(f"[{time_string()}] Complete")
clean(label)
