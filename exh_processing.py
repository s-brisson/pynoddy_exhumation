import numpy as np
from scipy import interpolate
import sys
import importlib
import copy
import measures
# making coordinate transformation code generic
# first we have to select the first non-zero value of each coord x,y,z and this will correspond to the "res" value
#that we chose
def first_unique_vals(array):
    unique_vals = set()  # initialize an empty set to store unique non-zero values
    first_unique_val = None
    for i in array:
        if i != 0:
            if i not in unique_vals:
                unique_vals.add(i)
                if first_unique_val is None:
                    first_unique_val = i
                else:
                    return first_unique_val, i
    return None

def calc_dims(output, res, zdim):
    xdim = np.round(output.nx/res, decimals = 0).astype(int) #get the extent by checking how many voxels were actually sampled.
    ydim = np.round(output.ny/res, decimals = 0).astype(int)
    #zdim = np.round((np.abs(ori_depth) - np.abs(upperlim))/interval, decimals = 0).astype(int)
    zdim = zdim
    return xdim, ydim, zdim

def exhumation_grid(array, output, res, zdim): #code to convert bullshit coords back into voxels and then calculates the exhumation grid
    
    unit_x,_ = first_unique_vals(array[...,0]) #select first non-zero value
    unit_y,_ = first_unique_vals(array[...,1]) #idem
    unit_z,unit_z2 = first_unique_vals(array[...,2]) # select first two unique non-zero values
    
    results_array = array[:,:5] #grab only xyz
    
    #conversion to voxels
    xx = (results_array[...,0]) / unit_x
    yy = (results_array[...,1]) / unit_y 
    zz = np.round((results_array[...,2]) / (unit_z2 - unit_z)) 
    zzz,_ = first_unique_vals(zz)
    zz = zz - zzz
    exh = np.round((results_array[...,3]) / (unit_z2 - unit_z)) 
    std = np.round((results_array[...,4]) / (unit_z2 - unit_z)) 
    
    voxel_coords = []
    for j in range(len(xx)):
        voxel_coords.append([xx[j],yy[j],zz[j]])
    voxel_coords = np.array(voxel_coords).astype(int)
    
    #setting up the grids
    xdim, ydim, zdim = calc_dims(output, res, zdim)
    print(xdim,ydim,zdim)
    
    E = np.zeros((xdim+1, ydim+1, zdim+1))
    SD = np.zeros((xdim+1, ydim+1, zdim+1))
    
    for i, point in enumerate(voxel_coords):
        x,y,z = point
        E[x][y][z] = exh[i] #fills in empty grid with exhumation values
        SD[x][y][z] = std[i] #fills in empty grid with stdev values
    
    return E, SD, voxel_coords

def exhumation_grid_single(array, output, res, zdim): #code to convert bullshit coords back into voxels and then calculates the exhumation grid
    
    x_col = [row[1] for row in array]
    y_col = [row[2] for row in array]
    z_col = [row[3] for row in array]
    e_col = [row[4] for row in array]
    
    #prepare the units
    z_col1 = [row[3] for row in array if row[0] == 1] #first sample
    z_col2 = [row[3] for row in array if row[0] == 2] #second sample
    unit_x = next((value for value in x_col if value != 0), None)
    unit_y = next((value for value in y_col if value != 0), None)
    unit_z = next((value for value in z_col1 if value != 0), None)
    unit_z2 = next((value for value in z_col2 if value != 0), None)
    
    #conversion to voxels
    xx = x_col / unit_x
    yy = y_col / unit_y 
    zz = np.round((z_col) / (unit_z2 - unit_z)) 
    zzz = next((value for value in zz if value != 0), None)
    zz = zz - zzz
    raw_exh = (e_col) / (unit_z2 - unit_z)
    exh = np.round(raw_exh) 
    
    voxel_coords = []
    for j in range(len(xx)):
        voxel_coords.append([xx[j],yy[j],zz[j]])
    voxel_coords = np.array(voxel_coords).astype(int)
    
    #setting up the grids
    xdim, ydim, zdim = calc_dims(output, res, zdim)
    print(xdim,ydim,zdim)
    
    E = np.zeros((xdim+1, ydim+1, zdim+1))
    E_raw = np.zeros((xdim+1, ydim+1, zdim+1))
    
    for i, point in enumerate(voxel_coords):
        x,y,z = point
        E[x][y][z] = exh[i] #fills in empty grid with exhumation values
        E_raw[x][y][z] = raw_exh[i]
    return E, voxel_coords, E_raw

def calc_entropy(block_array, output, n):
    
    """Calculate the full model entropy.
        block_array = the block outputs from each model run,
        output = the NoddyOutput file from the original model,
        n = steps with which to calculate the entropy for each y slice (1 = each y slice's entropy is calculated, but it's super slow"""
    
    ie_mean_array = []
    slice_entropy = []
    
    for i in np.arange(0,output.ny,n):
        ie = np.empty_like(block_array[0,:,i,:])
    
        for j in np.arange(0,block_array.shape[1],n):
            for k in np.arange(0,block_array.shape[3],n):
                ie[j,k] = measures.joint_entropy(block_array[:,j,i,k])
        
        slice_entropy.append(ie)
        ie_mean_array.append(ie.mean())
        
    total_entropy = sum(ie_mean_array) / len(ie_mean_array)
        
    return total_entropy, slice_entropy

def interp_and_score(E,samples,cubesize,res,zdim,min_depth,grid):    
    
    sample_coord = []
    exhumation = []
    model_score = 0
    
    for i in range(len(samples)):
        samples_copy = copy.deepcopy(samples)
        
        sample_xy = (samples_copy.iloc[i,:2] / cubesize) / res 
        sample_z = (zdim - (min_depth / cubesize)) + (samples_copy.iloc[i,2] / cubesize) 
        sample_coord.append(sample_xy.tolist()+[sample_z])
        
        #and now interpolate
        mypoint = sample_coord[-1]
        exhumation.append(interpolate.interpn(grid,E,mypoint))
        
        samples_copy.loc[i, 'exhumation'] = exhumation[i]
    
        #assign a score to the model
        if samples_copy.iloc[i]['group'] in ['a'] and samples_copy.iloc[i]['exhumation'] < 30:
            model_score += 1
            samples.loc[i,'respected'] += 1
        elif samples_copy.iloc[i]['group'] in ['b'] and samples_copy.iloc[i]['exhumation'] > 48:
            model_score += 1
            samples.loc[i,'respected'] += 1
        elif samples_copy.iloc[i]['group'] in ['c'] and samples_copy.iloc[i]['exhumation'] > 32 and samples_copy.iloc[i]['exhumation'] < 48:
            model_score += 1
            samples.loc[i,'respected'] += 1
        elif samples_copy.iloc[i]['group'] in ['d'] and samples_copy.iloc[i]['exhumation'] > 32:
            model_score += 1
            samples.loc[i,'respected'] += 1
        else:
            model_score += 0
    samples['exhumation'] = exhumation
    return model_score, samples

def score(samples, exhumation):
    model_score = 0
    for i in range(len(samples)):
        samples_copy = copy.deepcopy(samples)
        samples_copy.loc[i, 'exhumation'] = exhumation.loc[i,'exhumation']
    #assign a score to the model
        if samples_copy.iloc[i]['group'] in ['a'] and samples_copy.iloc[i]['exhumation'] < 4500:
            model_score += 1
            samples.loc[i,'respected'] += 1
        elif samples_copy.iloc[i]['group'] in ['b'] and samples_copy.iloc[i]['exhumation'] > 4800:
            model_score += 1
            samples.loc[i,'respected'] += 1
        elif samples_copy.iloc[i]['group'] in ['c'] and samples_copy.iloc[i]['exhumation'] > 3200 and samples_copy.iloc[i]['exhumation'] < 4800:
            model_score += 1
            samples.loc[i,'respected'] += 1
        elif samples_copy.iloc[i]['group'] in ['d'] and samples_copy.iloc[i]['exhumation'] > 3200:
            model_score += 1
            samples.loc[i,'respected'] += 1
        else:
            model_score += 0
    samples['exhumation'] = exhumation['exhumation']
    return model_score, samples
