import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
import copy 
import os
from exh_functions import *
from exh_processing import *
from default_configs import *
from os import makedirs

created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws

label = generate_unique_label()
current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv"
model_scores_folder = f"{output_folder}/{model_name}/model_scores/{args.folder}/"
model_params_folder = f"{output_folder}/{model_name}/model_params/{args.folder}/"
model_samples_folder = f"{output_folder}/{model_name}/model_samples/{args.folder}/"
model_histories_folder = f"{output_folder}/{model_name}/model_histories/{args.folder}/"

makedirs(model_scores_folder,exist_ok=True)
makedirs(model_params_folder,exist_ok=True)
makedirs(model_samples_folder,exist_ok=True)
makedirs(model_histories_folder,exist_ok=True)

print(f"[{time_string()}] {'Simulating based on file':<40} {history_samples}")
print(f"[{time_string()}] {'Simulating based on observation':<40} {samples}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
print()

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


if os.path.exists(current_exh_path):
    print(f"[{time_string()}] Found existing data.")
    samples = pd.read_csv(current_exh_path)
    diff = np.load("/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/diff.npy")
else:
    print(f"[{time_string()}] No data available.")

og_depths = []
for event_name, evento in hist.events.items():
    if isinstance(evento, pynoddy.events.Plug):
        z = evento.properties['Z']  
        og_depths.append(z)

samples_z = []
for i in range(len(samples)):
    z = samples.iloc[i]['Z']
    samples_z.append(z)


scores = []
all_params = pd.DataFrame(columns = ['Event'] + prop + ['n_draw'])
exhumation = []
all_liths = np.arange(11,21)
events = [18,19,20,21,22,24,25,26]

print(f"[{time_string()}] Starting simulation")
for i in range(n_draws):
    print(f"run {i}")
    hist_copy = copy.deepcopy(hist)
    new_params, new_params_df = disturb_property(hist_copy,events,prop,std)
    try:
        new_exhumation,_,new_hist = calc_new_position(hist_copy,diff,og_depths,all_liths,samples.copy(),label)
    except IndexError:
        continue
    
    model_score, samples = score(samples, new_exhumation)
    print(f"model score: {model_score}")
    #store per run
    scores.append([i, model_score])
    exhumation.append(new_exhumation['exhumation'])
    all_params = pd.concat([all_params, new_params_df], ignore_index=True)
    new_hist_name = f'{model_histories_folder}/hist_{label}_draw{i}.his'
    os.rename(new_hist, new_hist_name)

#store overall
all_params.to_csv(f'{model_params_folder}/params_{label}.csv', index = False)
samples.to_csv(f'{model_samples_folder}/samples_{label}.csv', index = False)
np.save('f{model_samples_folder}/exhumation_{label}.npy', exhumation)
