import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
import copy 
from exh_functions import *
from default_configs import *
from os import makedirs

created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws
if isinstance(args.events, list):
    event = args.events  # Events as a list
elif isinstance(args.events, int):
    event = [args.events]  # Convert single integer to a list
else:
    raise ValueError("Invalid input for 'events' argument")
prop = args.property #property that will be disturbed
std = args.standard_deviation #uncertainty assigned to the property

label = generate_unique_label()
current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv"
model_scores_folder = f"{output_folder}/{model_name}/model_scores/{args.folder}/"
model_params_folder = f"{output_folder}/{model_name}/model_params/{args.folder}/"
model_samples_folder = f"{output_folder}/{model_name}/model_samples/{args.folder}/"

makedirs(model_scores_folder,exist_ok=True)
makedirs(model_params_folder,exist_ok=True)
makedirs(model_samples_folder,exist_ok=True)

print(f"[{time_string()}] {'Simulating based on file':<40} {history_samples}")
print(f"[{time_string()}] {'Simulating based on observation':<40} {samples}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
print()

#LOAD NODDY MODEL
print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history_samples, output_name, 
                      noddy_path = noddy_exe,
                      verbose=True)
hist = pynoddy.history.NoddyHistory(history_samples)
hist.change_cube_size(cubesize)
hist_hd = f'{output_folder}/history/hist_hd_{label}.his'
out_hd = f'{output_folder}/noddy/out_hd_{label}'
hist.write_history(hist_hd)
print(f"[{time_string()}] Running the HD model")
pynoddy.compute_model(hist_hd, out_hd, noddy_path = noddy_exe)
out_hd = pynoddy.output.NoddyOutput(out_hd)

samples = pd.read_csv(samples,delimiter = ',')

#DEFINE IMPORTANT VALUES
og_depths = []
for event_name, evento in hist.events.items():
    if isinstance(evento, pynoddy.events.Plug):
        z = evento.properties['Z']  
        og_depths.append(z)

samples_z = []
for i in range(len(samples)):
    z = samples.iloc[i]['Z']
    samples_z.append(z)

 #CALCULATING ORIGINAL EXHUMATION
print(f"[{time_string()}] Calculating original exhumation")

if os.path.exists(current_exh_path):
    print(f"[{time_string()}] Found existing data.")
    samples = pd.read_csv(current_exh_path)
    diff = np.load("/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/diff.npy")
  
else:
    print(f"[{time_string()}] Calculating from scratch.")
    all_liths = np.arange(11,21)
    _,samples_noddy_pos = calc_new_position(hist, og_depths, og_depths, all_liths,samples,label)   
    diff = [x - y for x, y in zip(samples_noddy_pos, samples_z)]
    current_exhumation = [x - y - z for x,y,z in zip(samples_noddy_pos, diff, og_depths)]
    samples['exhumation'] = current_exhumation
    samples['respected'] = 0

og_params = []
for i in event:
    print(f"[{time_string()}] event = {i}")
    event_data = [i]
    for j, props in enumerate(prop):
        print(f"[{time_string()}] analyzing prop {props}")
        propert = hist.events[i].properties[props]
        event_data.append(propert)
    og_params.append(event_data)
    
col = ['event_name'] + prop
og_params_df = pd.DataFrame(og_params, columns = col)

#SIMULATION
accepted = 0
total_runs = 0
current_params = og_params
current_exhumation = samples.loc[sample_num].copy()
score = []
accepted_params = pd.DataFrame(columns = ['Event'] + prop + ['n_draw'])
accepted_exhumation = []
rejected_params = pd.DataFrame(columns = ['Event'] + prop + ['n_draw'])

print(f"[{time_string()}] Starting MCMC")
for i in range(n_draws):

    while accepted < n_draws:
        hist_copy = copy.deepcopy(hist)

        proposed_params, proposed_params_df = disturb_property(hist_copy,event,prop,std)
        proposed_exhumation,_ = calc_new_position(hist_copy, diff[sample_num], 
                                                  og_depths[sample_num],lith_list, samples.loc[sample_num].copy(),label)

        #calculate likelihood and priors
        current_likelihood,current_score,current_samples = simple_likelihood(current_exhumation)
        proposed_likelihood,proposed_score,proposed_samples = simple_likelihood(proposed_exhumation)
        current_prior = prior_dist(og_params, current_params, std)
        proposed_prior = prior_dist(og_params, proposed_params, std)

        print(f"Model score: {proposed_score}")
        print(f"proposed exhumation {proposed_exhumation.loc['exhumation']}")
        
        #accept or reject
        acceptance_ratio = (proposed_prior*proposed_likelihood) / (current_prior*current_likelihood)
        print(f"Acceptance ratio: {acceptance_ratio}")

        random_n = np.random.rand(1)
        print(f"Random threshold: {random_n}")

        if acceptance_ratio > random_n:
            current_params_df = proposed_params_df
            current_params = proposed_params
            current_exhumation = proposed_exhumation
            accepted += 1
            print(f"accepted model number {accepted}")

            #store stuff
            score.append([proposed_score, i])
            accepted_params = pd.concat([accepted_params, current_params_df], ignore_index=True)
            accepted_exhumation.append(current_exhumation.loc['exhumation'])
        else:
            rejected_params = pd.concat([rejected_params, proposed_params_df], ignore_index=True)

        total_runs += 1

print(f"The acceptance rate was: {accepted / total_runs}")
#SAVE THE STUFF TO THE CLUSTER
print(f"[{time_string()}] Saving all the important shit")
scores = pd.DataFrame(score, columns = ['score', 'iteration'])
accepted_params.to_csv(f"{model_params_folder}/acc_params_{label}.csv", index = False)
rejected_params.to_csv(f"{model_params_folder}/rej_params_{label}.csv", index = False)
np.save(f"{model_params_folder}/accepted_exhumation_{label}.npy", accepted_exhumation)

print(f"[{time_string()}] Complete")
clean(label)

