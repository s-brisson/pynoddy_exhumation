import numpy as np
import matplotlib.pyplot as plt
from time import time as time_ex
import sys
import os
from os import makedirs
from exh_functions import *
from default_configs import *
import OnlineNIRB as online
online = online.OnlineNIRB()
from SALib.sample import saltelli
from SALib.analyze import sobol
plt.rcParams['font.size'] = 16
from joblib import load
from tensorflow import keras

created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws

qoi_folder = f"{output_folder}/{model_name}/qoi/{args.folder}/"

###LOAD THE SURROGATE MODEL
print("Loading the surrogate model")
# Files for the physics based ML surrogate model
model_file = 'data/input_files/Alps_KinematicModelAlpsFault1-4.joblib'
basis_file = 'data/input_files/BasisFctsKinematicModelAlpsFault1-4_e-0'

# Data-based surrogate
model_file_nn = '../Surrogates/Kinematic_nn_model/'

# Training and Validation Parameters
train_param = np.load('data/input_files/TrainingsParametersKinematicModelAlpsFault1-4_depth_rounded_unique.npy')
val_param = np.load('data/input_files/ValidationParametersKinematicModelAlpsFault1-4_depth_rounded.npy')
        
# The training and validation parameters need to be scaled    
mean = np.mean(train_param,axis=0)
var = np.var(train_param,axis=0)

train_param_scaled = (train_param-mean)/np.sqrt(var)   
val_param_scaled = (val_param-mean)/np.sqrt(var)   

# Load the surrogate (ML part)
model_gpr = load(model_file)
# Load the basis functions
bfs=np.load(basis_file+'.npy')

# # Load the data-based surrogate
model_nn = keras.models.load_model(model_file_nn)

###GENERATE THE SAMPLES FOR THE GSA
print("Generating samples for GSA")
# Parameter Ranges
fault_1_slip = np.array([-11000, -9000])
fault_1_z = np.array([-959, 241])
fault_2_slip =np.array([-10200, -8200])
fault_2_z = np.array([-1194, -394])
fault_3_slip =np.array([-10000, -8000])
fault_3_z = np.array([1980, 2780])
fault_4_slip =np.array([-13000, -11000])
fault_4_z = np.array([-2464, -1664])

t = time_ex()
# Define the model inputs
problem = {
    'num_vars': 8,
    'names': ['fault_1_slip','fault_1_z','fault_2_slip','fault_2_z','fault_3_slip','fault_3_z','fault_4_slip',
              'fault_4_z'],
    'bounds': [[fault_1_slip[0], fault_1_slip[-1]],
               [fault_1_z[0], fault_1_z[-1]],
               [fault_2_slip[0], fault_2_slip[-1]],
               [fault_2_z[0], fault_2_z[-1]],
               [fault_3_slip[0], fault_3_slip[-1]],
               [fault_3_z[0], fault_3_z[-1]],
               [fault_4_slip[0], fault_4_slip[-1]],
               [fault_4_z[0], fault_4_z[-1]]]
}

# Generate samples
param_values = saltelli.sample(problem, 2**17) #2**17=131072
t1 = time_ex()- t

param_values=(param_values-mean)/np.sqrt(var)

###CALCULATING THE SENSITIVITY INDICES
print("Calculating the QOI")
t = time_ex()
Y_0 = np.zeros(len(param_values))

# Calculate the coefficients for all parameter values
RB_outputs = model_gpr.predict(param_values)

for i in range(len(param_values)):
    # Calculate the full-dimensional solution for every parameter
    RB_full = np.matmul(RB_outputs[i,:], bfs[0,:,:]) 
    
    # Clip negative values for the exhumation rate for the RB surrogate
    indices_negative = np.where(RB_full<0)
    RB_full[indices_negative] = 0
    
    # Define the quantitiy of interest (at the moment just an example, this needs to be adjusted)
    Y_0[i] = np.mean(RB_full)
    
t2 = time_ex()- t

print("saving...")
np.save(f"{qoi_folder}/qoi.npy", Y_0)

print("Complete")
