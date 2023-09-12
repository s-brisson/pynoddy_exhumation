import sys, os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D 
import argparse
from default_configs import noddy_exe,output_folder
rcParams['font.size'] = 15

#Determine the path of the repository to set paths correctly below.

# repo_path = r'C:\Users\Sofia\Documents\Sofia\Noddy'
# sys.path.insert(0, repo_path)

import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.experiment
importlib.reload(pynoddy.experiment)
rcParams.update({'font.size': 15})
import pandas as pd
import pickle
import copy
from tqdm import tqdm
import time 
import random
import string

# function to parse arguments to the script
def parser():
    parser = argparse.ArgumentParser(description="stochastic simulation of exhumation from a kinematic modeling")
    parser.add_argument("ndraws", help="Number of simulations to be run",type=int)
    parser.add_argument("interval", type=int, help="Indicator layer z-axis step size")
    parser.add_argument("resolution", help="samples coord every res voxels",  choices=[8,16,32,64],type=int)
    parser.add_argument("--folder", help="folder where to store the output files", type=str, required=False, default="test")
    return parser

def parser_new():
    parser = argparse.ArgumentParser(description="stochastic simulation of exhumation from a kinematic modeling")
    parser.add_argument("ndraws", help="Number of simulations to be run",type=int)
    parser.add_argument("events", help="Event to modify", type = int)
    parser.add_argument("property", help="property to change", nargs = '+', type = str)
    parser.add_argument("--standard_deviation", help = "uncertainty added to each property", nargs = '+', type = int)
    parser.add_argument("--folder", help="folder where to store the output files", type=str, required=False, default="test")
    return parser

# function to clean temporary files
def clean(label):
    os.system(f"rm {output_folder}/noddy/*{label}*")
    os.system(f"rm {output_folder}/history/*{label}*")

# function to generate unique label string to avoid output files to be overwritten
def generate_unique_label():
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{timestamp}_{random_string}"

## generate the current time string i.e. June 25 - 20:41:26
def time_string():
    return time.strftime("%B %d - %H:%M:%S", time.localtime())

#function for getting noddy model coordinates by Kaifeng Gao
def ExtractCoords(hist_moment, lith, res, unique_label):
    
    # Compute noddy model for history file
    temp_hist = f'{output_folder}/history/temp_hist_{unique_label}.his'
    temp_out = f'{output_folder}/noddy/temp_out_{unique_label}'
    hist_moment.write_history(temp_hist)
    pynoddy.compute_model(temp_hist, temp_out, 
                          noddy_path = noddy_exe)
    N1 = pynoddy.output.NoddyOutput(temp_out)
    #N1.plot_section('y', litho_filter = lith)
    
    #num_rock = N1.n_rocktypes   #number of rock types
    sum_node = []   #stack the nodes in each interface
    num_list = []   #output the node number in each interface
    
    if isinstance(lith, list):
        #litho_list = list(range(1, num_rock+1))  ##the lithoID in noddy begin at 2, lithoID could be a lis
        for i in lith:
            rockID = i*2  #because points = np.absolute(points/2), multiply 2 in advance
            out = N1.get_surface_grid(lithoID=[i], res=res)  #here can change the sampling number by changing res
            listx = out[0][0][i]   #out[0]：get_surface_grid, only choose first direction；out[0][0]: x values in x-grid
            listy = out[0][1][i]   #y values
            listz = out[0][2][i]
            xx = sum(listx, [])  #multi-rows to one row
            yy = sum(listy, [])
            zz = sum(listz, [])
            num_node = 0
            for j in range(len(xx)):
                #if xx[j] <-1 and yy[j]<-1 and zz[j]<-1:
                sum_node.append([xx[j],yy[j],zz[j],rockID])
                #num_node +=1
            #num_list.append(num_node)

        
    #get_surface_grid function will get negative value, and twice the value, don't know the reason
    points = np.array(sum_node)
    points = np.absolute(points/2)
    np.set_printoptions(precision=2) #two decimal places
    np.set_printoptions(suppress=True) #don't use scientific notation

    return points, num_list, N1

def disturb_percent(event, prop, percent = 30):
    """Disturb the property of an event by a given percentage, assuming a normal distribution"""
    ori_val = event.properties[prop]
    new_val = np.random.randn() * percent/100. * ori_val + ori_val
    event.properties[prop] = new_val
    
    return new_val
    
def disturb_value(event, prop, stdev):
    """Disturb the property of an event by a given stdev, assuming a normal distribution"""
    ori_val = event.properties[prop]
    new_val = np.random.normal(ori_val, stdev)
    event.properties[prop] = new_val
    
    return new_val


def disturb(PH_local, std_list, ndraw):
    data = []
    for event_name, event in PH_local.events.items():
        if isinstance(event, pynoddy.events.Fault):
            new_slip = disturb_value(event, 'Slip', std_list[0])
            new_amp = disturb_value(event, 'Amplitude', std_list[1])
            new_x = disturb_value(event, 'X', std_list[2])
            #new_dip = disturb_percent(event, 'Dip', percent=5)
            new_dipdir = disturb_percent(event, 'Dip Direction', std_list[3])
            #new_pitch = disturb_percent(event, 'Pitch', percent=5)
            #new_z = disturb_value(event, 'Z', 75)
            data.append([event_name, new_slip, new_amp, new_x, new_dipdir, ndraw])
    
    columns = ['Event', 'New Slip', 'New Amplitude', 'New X', 'New Dip Direction','nDraw']
    df = pd.DataFrame(data, columns=columns)
    return data, df


def exhumationComplex(ndraw, history, lith, res = 8, interval = 50, upperlim = 0, unique_label="20235555555555_AAAAAA"):
    
    """function for estimating the exhumation (vertical movement) from a noddy history. Arguments:
            lith: lith id of the dyke or item used to track the movement
            res: 
            interval: sampling interval in the z direction 
            upperlim: limit up to which sampling is performed.
            """
    
    new_z = upperlim - 1
    n_sample = 0
    
    coords = []
    hist_copy = copy.deepcopy(history)
    
    while new_z < upperlim:
        n_sample += 1
        
        for event in hist_copy.events.values():
            if isinstance(event, pynoddy.events.Dyke):
                old_z = event.properties['Z']
                new_z = old_z + interval
                event.properties['Z'] = new_z
                print(f"[{time_string()}] Processing indicator at z = {new_z} ...")
    
                points,_,N1 = ExtractCoords(hist_copy, lith, res, unique_label) #make sure that the history is at least cube size 100.
                
                try:
                    x = points[...,0]
                    y = points[...,1]
                    z = points[...,2]
                except IndexError:
                    continue
                
                #correct for weird noddy coordinates
                #real_z = points[0][2] #select the Z value of the first row.
                real_z = points[...,2].min() #select the minimum z value - that's the original depth
                exhumation =  z - real_z
                print(f"[{time_string()}] Processing indicator at z = {new_z} ... Done")
        
        for j in range(len(x)):    
            coords.append([n_sample,x[j],y[j],z[j],exhumation[j],ndraw])
        
        #coords = np.array(coords)
            
    return coords, N1, hist_copy

#INVERSION FUNCTIONS
def create_pdf(mean, std_dev):
    def pdf(x):
        coeff = 1 / (std_dev * np.sqrt(2 * np.pi))
        exponent = - ((x - mean) ** 2) / (2 * std_dev ** 2)
        return coeff * np.exp(exponent)
    return pdf

def prior_dist(og_params,proposed_params,std_list):
    prior_prob = 1.0
    for i in range(len(og_params)):
        for j in range(len(std_list)-1):
            pdf = create_pdf(og_params[i][j+1], std_list[j])
            prior_prob *= pdf(proposed_params[i][j+1])
    return prior_prob

def likelihood(samples_df):
    likelihood = 1.0
    
    for i in range(len(samples_df)):
        if samples_df.iloc[i]['group'] in ['a']:
            if samples_df.iloc[i]['exhumation'] < 30: #non reset AFT sample (B60, always accepted) strict
                likelihood *= 10
                
            else:
                proximity = (samples_df.iloc[i]['exhumation'][0] - 30) / 30
                rf = np.exp(-25 * proximity)
                likelihood *= rf
                
        
        elif samples_df.iloc[i]['group'] in ['b']:
            if samples_df.iloc[i]['exhumation'] > 48: #reset AFT sample (B10, never accepted) not strict
                likelihood *= 30
            else:
                #proximity = (48 - samples_df.iloc[i]['exhumation'][0]) / 48
                #rf = np.exp(-2 * proximity)
                likelihood *= 5
                
                
        elif samples_df.iloc[i]['group'] in ['c']: #this should be a strict criteria #reset AHe, partially reset AFT
            if samples_df.iloc[i]['exhumation'] > 32 and samples_df.iloc[i]['exhumation'] < 48:
                likelihood *= 10
            else:
                likelihood *= 0.05
                
        elif samples_df.iloc[i]['group'] in ['d']: #reset AHe samples
            if samples_df.iloc[i]['exhumation'] > 32:
                likelihood *= 10
            else:
                proximity = (32 - samples_df.iloc[i]['exhumation'][0]) / 32
                rf = np.exp(-10 * proximity)
                likelihood *= rf
        else:
            print('this is not a group')
    return likelihood

#MCMC USING INDEPENDENT PARAMETERS - FUNCTIONS
def calc_new_position(hist, diff, og_depths, lith_list,samples,unique_label):
    samples_noddy_pos = []
    for i in lith_list:
        p,_,out = ExtractCoords(hist, lith = [i], res = 1,unique_label = unique_label)
        t = p[...,2].min()
        z = (t*1000) / 3681.39
        samples_noddy_pos.append(z)
    
    if len(lith_list) > 1:
        proposed_exhumation = [x - y - z for x,y,z in zip(samples_noddy_pos, diff, og_depths)]
    else:
        proposed_exhumation = samples_noddy_pos - diff - og_depths
    samples['exhumation'] = proposed_exhumation
    return samples, samples_noddy_pos 

def disturb_property(PH_local, event_list, prop_list, std_list):
    data = []
    for i in event_list:
        event_data = [i]
        for j, prop in enumerate(prop_list):
            new_param = disturb_value(PH_local.events[i], prop_list[j], std_list[j])
            event_data.append(new_param)
            
        data.append(event_data)
    col = ['event_name'] + prop_list
    df = pd.DataFrame(data, columns = col)
    
    return data, df

def likelihood_and_score(samples_df):
    
    likelihood = 1.0
    model_score = 0
    
    for i in range(len(samples_df)):
        if samples_df.iloc[i]['group'] in ['a']:
            if samples_df.iloc[i]['exhumation'] < 3000: #non reset AFT sample (B60, always accepted) strict
                likelihood *= 2
                model_score += 1
                samples_df.loc[i,'respected'] += 1
                
            else:
                proximity = (samples_df.iloc[i]['exhumation'] - 3000) / 3000
                rf = np.exp(-proximity)
                likelihood *= rf
                
        
        elif samples_df.iloc[i]['group'] in ['b']:
            if samples_df.iloc[i]['exhumation'] > 4800: #reset AFT sample (B10, never accepted) not strict
                likelihood *= 2
                model_score += 1
                samples_df.loc[i,'respected'] += 1
            else:
                proximity = (4800 - samples_df.iloc[i]['exhumation']) / 4800
                rf = np.exp(-proximity)
                likelihood *= rf
                
        elif samples_df.iloc[i]['group'] in ['c']: #this should be a strict criteria #reset AHe, partially reset AFT
            if samples_df.iloc[i]['exhumation'] > 3200 and samples_df.iloc[i]['exhumation'] < 4800:
                likelihood *= 4
                model_score += 1
                samples_df.loc[i,'respected'] += 1
            else:
                likelihood *= 0.05
                
        elif samples_df.iloc[i]['group'] in ['d']: #reset AHe samples
            if samples_df.iloc[i]['exhumation'] > 3200:
                likelihood *= 2
                model_score += 1
                samples_df.loc[i,'respected'] += 1
            else:
                proximity = (3200 - samples_df.iloc[i]['exhumation']) / 3200
                rf = np.exp(-proximity)
                likelihood *= rf
    return likelihood, model_score, samples_df

def simple_likelihood(samples_df):
    likelihood = 1.0
    model_score = 0
    
    
    if samples_df.loc['group'] in ['a']:
        if samples_df.loc['exhumation'] < 3000: #non reset AFT sample (B60, always accepted) strict
            likelihood *= 2
            model_score += 1
            samples_df.loc['respected'] += 1
            
        else:
            proximity = (samples_df.loc['exhumation'] - 3000) / 3000
            rf = np.exp(-proximity)
            likelihood *= rf
            
    
    elif samples_df.loc['group'] in ['b']:
        if samples_df.loc['exhumation'] > 4800: #reset AFT sample (B10, never accepted) not strict
            likelihood *= 2
            model_score += 1
            samples_df.loc['respected'] += 1
        else:
            proximity = (4800 - samples_df.loc['exhumation']) / 4800
            rf = np.exp(-proximity)
            likelihood *= rf
            
    elif samples_df.loc['group'] in ['c']: #this should be a strict criteria #reset AHe, partially reset AFT
        if samples_df.loc['exhumation'] > 3200 and samples_df.loc['exhumation'] < 4800:
            likelihood *= 4
            model_score += 1
            samples_df.loc['respected'] += 1
        else:
            likelihood *= 0.05
            
    elif samples_df.loc['group'] in ['d']: #reset AHe samples
        if samples_df.loc['exhumation'] > 3200:
            likelihood *= 100
            model_score += 1
            samples_df.loc['respected'] += 1
        else:
            proximity = abs(3200 - samples_df.loc['exhumation'])
            if proximity <= 200:
                rf = np.exp(-proximity/ 3200)
            else:
                rf = np.exp(-20*proximity / 3200)
            likelihood *= (rf) 
    
    return likelihood, model_score, samples_df
            
            
