import sys, os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D 
import argparse
from default_configs import noddy_exe
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

# function to clean temporary files
def clean(label):
    os.system(f"rm ./data/outputs/noddy/*{label}*")
    os.system(f"rm ./data/outputs/history/*{label}*")

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
    temp_hist = f'data/outputs/history/temp_hist_{unique_label}.his'
    temp_out = f'data/outputs/noddy/temp_out_{unique_label}'
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


def disturb(PH_local, ndraw):
    data = []
    for event_name, event in PH_local.events.items():
        if isinstance(event, pynoddy.events.Fault):
            new_dip = disturb_percent(event, 'Dip', percent=5)
            new_dipdir = disturb_percent(event, 'Dip Direction', percent=5)
            new_pitch = disturb_percent(event, 'Pitch', percent=5)
            new_slip = disturb_value(event, 'Slip', 400)
            new_amp = disturb_value(event, 'Amplitude', 100)
            new_x = disturb_value(event, 'X', 50)
            new_z = disturb_value(event, 'Z', 75)
            data.append([event_name, new_dip, new_dipdir, new_pitch, new_slip, new_amp, new_x, new_z, ndraw])
    
    columns = ['Event', 'New Dip', 'New Dip Direction', 'New Pitch', 'New Slip', 'New Amplitude', 'New X', 'New Z','nDraw']
    df = pd.DataFrame(data, columns=columns)
    return df


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
            
    return coords, N1

  
