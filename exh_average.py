#SETUP
import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
import copy 
import pickle
from exh_functions import *
from exh_processing import * 
from default_configs import *
from os import makedirs

created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws

label = generate_unique_label()
current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv"
model_exhumation_folder = f"{output_folder}/{model_name}/model_exhumation/{args.folder}/"
makedirs(model_exhumation_folder,exist_ok=True)

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

print(f"[{time_string()}] Calculating original exhumation")

if os.path.exists(current_exh_path):
    print(f"[{time_string()}] Found existing data.")
    samples = pd.read_csv(current_exh_path)
    diff = np.load("/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/diff.npy")
  
else:
    print(f"[{time_string()}] Calculating from scratch.")
    all_liths = np.arange(11,21)
    _,samples_noddy_pos,_ = calc_new_position(hist, og_depths, og_depths, all_liths,samples,label)   
    diff = [x - y for x, y in zip(samples_noddy_pos, samples_z)]
    current_exhumation = [x - y - z for x,y,z in zip(samples_noddy_pos, diff, og_depths)]
    samples['exhumation'] = current_exhumation
    samples['respected'] = 0
diff = [diff[i] for i in sample_num]
og_depths = [og_depths[i] for i in sample_num]
samples = samples.iloc[sample_num]

exhumation = []

print(f"[{time_string()}] Start calculation.")

params = np.load(all_params, allow_pickle=True)

hist_copy = copy.deepcopy(hist)

for i in range(len(params)):
    print(f"[{time_string()}] Calculating param row {i}.")
    for f, fault in enumerate(event):
        for p, props in enumerate(prop):
            hist_copy.events[fault].properties[props] = params[i][f][p+1]

    exh,_,_ = calc_new_position(hist_copy, diff, og_depths, lith_list, samples.copy(), label)
    np.save(f"{model_exhumation_folder}/accepted_exh_{label}_row{i}.npy", exh['exhumation'])

print(f"[{time_string()}] Completadowwwww")
clean(label)
