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


print(f"[{time_string()}] {'Simulating based on file':<40} {history_samples}")
print(f"[{time_string()}] {'Number of simulations':<40} {args.ndraws}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv"

### RUNNING THE INITIAL MODEL
print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history, output_name, 
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

samples = pd.read_csv(samples, delimiter = ',')

### DEFINE IMPORTANT VALUES
### Extract the depth of the sample
og_depths = []
for event_name, evento in hist.events.items():
    if isinstance(evento, pynoddy.events.Plug):
        z = evento.properties['Z']  
        og_depths.append(z)
og_depths = [og_depths[i] for i in sample_num] #but only for the samples used

### Extract the altitude of the samples
samples_z = []
for i in range(len(samples)):
    z = samples.iloc[i]['Z']
    samples_z.append(z)

### Initial exhumation
if os.path.exists(current_exh_path):
    print(f"[{time_string()}] Found existing data.")
    samples = pd.read_csv(current_exh_path)
    diff = np.load("/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/diff.npy")
diff = [diff[i] for i in sample_num]
samples = samples.iloc[sample_num]



### SIMULATION
for param in prop:
    change_unit = 0
    hist_copy = copy.deepcopy(hist)
    #starting_param = hist_copy.events[21].properties[param] - (26*200)
    starting_param = 0   #for angle values
    
    e = []
    p = []
    exhumation = pd.DataFrame(columns = ['Parameters', 'Exhumation'])
    for i in range(50):
        print(f"This is run number {i}")
        new_param = starting_param + change_unit
        hist_copy.events[21].properties[param] = new_param
        
        #calculate the exhumation
        exh,_,_ = calc_new_position(hist_copy,diff, og_depths, lith, samples, label)
        p.append(new_param)
        e.append(exh['exhumation'])
        
        change_unit += 2
        
    exhumation['Parameters'] = p
    exhumation['Exhumation'] = e
    exhumation.to_csv(f"{model_exhumation_folder}/exhumation_{param}.csv", index = False)


print(f"[{time_string()}] Complete")
clean(label)
