import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
import copy
from exh_functions import*
from default_configs import*
from exh_processing import*
from os import makedirs

created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws
label = generate_unique_label()

model_exhumation_folder = f"{output_folder}/{model_name}/model_exhumation/{args.folder}/"
makedirs(model_exhumation_folder,exist_ok=True)

print(f"[{time_string()}] {'Simulating based on file':<40} {history_transalp}")
print(f"[{time_string()}] {'Number of simulations':<40} {args.ndraws}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
#current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv"

### RUNNING THE INITIAL MODEL
print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history_transalp, output_name, 
                      noddy_path = noddy_exe,
                      verbose=True)

hist = pynoddy.history.NoddyHistory(history_transalp)
hist.change_cube_size(cubesize)
hist_hd = f'{output_folder}/history/hist_hd_{label}.his'
out_hd = f'{output_folder}/noddy/out_hd_{label}'
hist.write_history(hist_hd)
print(f"[{time_string()}] Running the HD model")
pynoddy.compute_model(hist_hd, out_hd, noddy_path = noddy_exe)
out_hd = pynoddy.output.NoddyOutput(out_hd)

synth_samples = pd.read_csv(synth_samples, delimiter = ',')

### DEFINE IMPORTANT VALUES
### Extract the depth of the sample
og_depths = []
for event_name, evento in hist.events.items():
    if isinstance(evento, pynoddy.events.Plug):
        z = evento.properties['Z']  
        og_depths.append(z)
#og_depths = [og_depths[i] for i in sample_num] #but only for the samples used

### Extract the altitude of the samples
#samples_z = []
#for i in range(len(samples)):
#    z = samples.iloc[i]['Z']
#    samples_z.append(z)

### Initial exhumation
#if os.path.exists(current_exh_path):
#    print(f"[{time_string()}] Found existing data.")
#    samples = pd.read_csv(current_exh_path)
#    diff = np.load("/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/diff.npy")
#diff = [diff[i] for i in sample_num]
#samples = samples.iloc[sample_num]

### SIMULATION
starting_params = []

for param in prop:
    hist_copy = copy.deepcopy(hist)
    
    for i, ev in enumerate(event):
        change_unit = 0
        starting_param = hist_copy.events[ev].properties[param] - (26 * 200)
        print(f"starting param for event {ev} and param {param}: {starting_param}")
        
        p = []
        e = []
        exh = pd.DataFrame(columns = ['Parameter', 'Exhumation'])
        
        for j in range(50):
            
            print(f"Run {j} of 50")
            
            new_param = starting_param + change_unit
            hist_copy.events[ev].properties[param] = new_param
            
            #calculate new model with the changed parameter
            temp_hist = f'{output_folder}/history/temp_hist_{label}.his'
            temp_out = f'{output_folder}/noddy/temp_out_{label}'
            hist_copy.write_history(temp_hist)
            pynoddy.compute_model(temp_hist, temp_out, 
                                  noddy_path = noddy_exe)
            output = pynoddy.output.NoddyOutput(temp_out)
            
            new_exhumation = calc_exhumation(output, avg_conv_factor, synth_samples.copy())
            new_exhumation.reset_index(drop = True, inplace = True)
            
            p.append(new_param)
            e.append(new_exhumation['exhumation'].tolist())
            
            change_unit += 200
            
        exhumation['Parameter'] = p
        exhumation['Exhumation'] = e
        exhumation.to_csv(f"{model_exhumation_folder}/exhumation_{param}_{ev}.csv", index = False)

print(f"[{time_string()}] Complete")
clean(label)
