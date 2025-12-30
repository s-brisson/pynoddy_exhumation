import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
import pickle

history = '../Lechtal model/bregenz_ver5.his'
output_name = 'outputs/noddy_out'
pynoddy.compute_model(history, output_name, 
                      noddy_path = r'C:\Users\Sofia\pynoddy\noddyapp\noddy_win64.exe')

hist = pynoddy.history.NoddyHistory(history)
out = pynoddy.output.NoddyOutput(output_name)

hist.change_cube_size(100)
hist_hd = 'hist_hd.his'
out_hd = 'outputs/out_hd'
hist.write_history(hist_hd)
pynoddy.compute_model(hist_hd, out_hd, noddy_path = r'C:\Users\Sofia\pynoddy\noddyapp\noddy_win64.exe')
out_hd = pynoddy.output.NoddyOutput(out_hd)

#Start with defining the modeling parameters
cubesize = 100
res = 16
interval = 100
upperlim = out.zmax - interval
lith = [11] #has to be checked when model is changed. Lith ID for the indicator layer

n_draws = 2
zdim = 65
samples = pd.read_csv('../bregenz model/bregenz_data.csv', delimiter = ',')

#other important values
ori_depth = hist.events[8].properties['Z'] #original depth of indicator layer
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
all_params = pd.DataFrame(columns = ['Event', 'New Dip', 'New Dip Direction', 'New Pitch', 'New Slip', 'New Amplitude', 'New X', 'New Z'])
all_scores = []

#starting counter 
samples['respected'] = 0

save_each = False
save_overall = True

#simulation starting
for i in tqdm(range(n_draws), desc = 'Its happening'):
    hist_copy = copy.deepcopy(hist) 
    
    #add uncertainty and save model parameters
    new_params = disturb(hist_copy)
    
    #calculate exhumation for the new model
    coords, output = exhumationComplex(hist_copy, lith, res, interval, upperlim)
    
    #interpolate exhumation for each sample position
    #value to interpolate
    E,_ = exhumation_grid_single(coords, out_hd, res, zdim)
    
    #interpolate and score the model based on how many samples are respected.
    model_score, samples = interp_and_score(E, samples, cubesize, res, zdim, min_depth, grid)
    print(model_score)
    all_scores.append(model_score)
    
    #store other important stuff
    all_coords.append(coords)
    all_blocks[i,:,:,:] = output.block
    all_params = pd.concat([all_params, new_params], ignore_index = True)
    
    if save_each == True:
        np.save('scoring/coords/coords_%04d.npy'%i, coords)
        np.save('scoring/blocks/blocks_%04d.npy'%i, output.block)
        new_params.to_csv('scoring/params/params_%04d.csv'%i, index = False)
        
if save_overall == True:
    pickle.dump(all_coords, open('scoring/coords/all_coords.pkl', 'wb'))
    pickle.dump(all_coords, open('scoring/blocks/all_blocks.pkl', 'wb'))
    all_params.to_csv('scoring/params/all_params.csv', index = False)
    all_scores.to_csv('scoring/scores/all_scores.csv', index = False)
    samples.to_csv('scoring/samples/samples.csv', index = False)