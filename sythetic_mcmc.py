import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
from default_configs import *
from exh_processing import *
from exh_functions import *
import os
from os import makedirs

#SET UP ARGUMENT PARSERS AND OUTPUT FOLDERS
created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws

label = generate_unique_label()
current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv"
model_params_folder = f"{output_folder}/{model_name}/model_params/{args.folder}/"
model_histories_folder = f"{output_folder}/{model_name}/model_histories/{args.folder}/"
model_exhumation_folder = f"{output_folder}/{model_name}/model_exhumation/{args.folder}/"

makedirs(model_params_folder,exist_ok=True)
makedirs(model_histories_folder,exist_ok=True)
makedirs(model_exhumation_folder,exist_ok=True)

#LOAD THE MODEL STEP MISSING
print(f"[{time_string()}] Running the base model")
history_synth = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/hist_initial.his"
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history_synth, output_name, 
                      noddy_path = noddy_exe,
                      verbose=True)
hist = pynoddy.history.NoddyHistory(history_synth)

#EXTRACT CURRENT EXHUMATION
og_depths = []
for event_name, evento in hist.events.items():
    if isinstance(evento, pynoddy.events.Plug):
        z = evento.properties['Z']
        
        og_depths.append(z)

if os.path.exists(current_exh_path):
    print(f"[{time_string()}] Found existing data.")
    samples = pd.read_csv(current_exh_path)
    diff = np.load("/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/diff.npy")

diff = [diff[i] for i in sample_num]
og_depths = [og_depths[i] for i in sample_num]
samples = samples.iloc[sample_num]

print(f"[{time_string()}] Calculating current exhumation.")
current_exhumation,_,_ = calc_new_position(hist, diff,
                                        og_depths, lith_list, samples.copy(), label)
samples.reset_index(drop = True, inplace = True)
samples['exhumation'] = current_exhumation['exhumation']

#DEFINING LAST IMPORTANT PARAMS
og_params = [[21, -860.6186270541134, -12981.20232166832],[22, 1449.6491377332402, -10209.185549843101],[24, -1171.2056049491198, -3163.6213015919775]]

current_params = og_params

accepted = 0
rejected = 0
total_runs = 0

#SIMULATION
print(f"[{time_string()}] Starting MCMC.")
for i in range(n_draws):
    while accepted < n_draws:
        
        current_hist = copy.deepcopy(hist)
        proposed_params, proposed_params_df = disturb_property(current_hist, event, prop, std)
        try:
            proposed_exhumation,_,new_hist = calc_new_position(current_hist, diff, og_depths, lith_list, samples.copy(), label)
        except IndexError:
            continue
        
        #Likelihood and prior probabilities
        current_prior = prior_dist(og_params, current_params, std)
        proposed_prior = prior_dist(og_params, proposed_params, std)
        current_likelihood = synthetic_likelihood(current_exhumation, synthetic_data, sigma)
        proposed_likelihood = synthetic_likelihood(proposed_exhumation, synthetic_data, sigma)
        
        #acceptance ratio and thresholds
        acceptance_ratio = (proposed_prior * proposed_likelihood) / (current_prior * current_likelihood)
        threshold = np.random.rand(1)
        
        if acceptance_ratio > threshold:
            current_params_df = proposed_params_df
            current_params = proposed_params
            current_exhumation = proposed_exhumation
            hist = current_hist
            
            accepted += 1
            print(f"Accepted model number {accepted} out of {total_runs+1}")
            #storage

            #store stuff for each run
            np.save(f"{model_exhumation_folder}/accepted_exh_{label}_draw{accepted}.npy", current_exhumation['exhumation'])
            np.save(f"{model_params_folder}/accepted_params_{label}_draw{accepted}.npy", current_params)
            accepted_hist = f'{model_histories_folder}/acc_hist_{label}_draw{accepted}.his'
            os.rename(new_hist, accepted_hist)
            
        else:
            #store stuff for each run
            np.save(f"{model_exhumation_folder}/rejected_exh_{label}_draw{rejected}.npy", proposed_exhumation['exhumation'])
            np.save(f"{model_params_folder}/rejected_params_{label}_draw{rejected}.npy", proposed_params)
            rej_hist = f'{model_histories_folder}/rej_hist_{label}_draw{rejected}.his'
            os.rename(new_hist, rej_hist)
        
        total_runs += 1
        print(f"Total runs: {total_runs}")
    
print(f"[{time_string()}] Complete")       
clean(label) 
