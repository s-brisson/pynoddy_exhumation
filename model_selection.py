import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output
import pandas as pd
import numpy as np
import copy
from exh_functions import *
from exh_processing import *
from default_configs import *
from os import makedirs

created_parser = parser_new()
args = created_parser.parse_args()
n_draws = args.ndraws
label = generate_unique_label()

model_scores_folder = f"{output_folder}/{model_name}/model_scores/{args.folder}/"
model_params_folder = f"{output_folder}/{model_name}/model_params/{args.folder}/"
model_samples_folder = f"{output_folder}/{model_name}/model_samples/{args.folder}/"
model_exhumation_folder = f"{output_folder}/{model_name}/model_exhumation/{args.folder}/"

makedirs(model_scores_folder,exist_ok=True)
makedirs(model_params_folder,exist_ok=True)
makedirs(model_samples_folder,exist_ok=True)
makedirs(model_exhumation_folder,exist_ok=True)


print(f"[{time_string()}] {'Simulating based on file':<40} {history_transalp}")
print(f"[{time_string()}] {'Number of simulations':<40} {args.ndraws}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
#current_exh_path = "/rwthfs/rz/cluster/home/ho640525/projects/Exhumation/data/input_files/bregenz_exh.csv"

### RUNNING THE INITIAL MODEL
print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history_samples, output_name, 
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

### Original parameters
og_params = []
for i in event:
    event_data = [i]
    for j, props in enumerate(prop):
        propert = hist.events[i].properties[props]
        event_data.append(propert)
    og_params.append(event_data)
col = ['event_name'] + prop
og_params_df = pd.DataFrame(og_params, columns = col)

### PREPARING OUTPUTS
scores = []
params = pd.DataFrame(columns=['Event'] + prop + ['n_draw'])
exhumations = []

### SIMULATION
print(f"[{time_string()}] Starting simulation")
for i in range(n_draws):
  
    hist_copy = copy.deepcopy(hist)

    ### Disturb the model
    _, new_params_df, output = disturb_property(hist_copy, event, prop, std, recompute = True, label)
  
    ### Calculate the exhumation with the new parameters
    try:
        new_exhumation = calc_exhumation(output, avg_conv_factor, synth_samples.copy(), og_depths)
    except IndexError:
        print("IndexError")
        continue
    new_exhumation.reset_index(drop = True, inplace = True)
    
    ### Score the model based on the new exhumation values
    #_, model_score, samples_df = likelihood_and_score(new_exhumation)
    #samples = samples_df # redefine samples so that the respected count is preserved
    synth_samples_updated, model_score = score(new_exhumation, geo_gradient)
    synth_samples = synth_samples_updated
    print(f"Model score: {model_score}")

    ### Save outputs
    print(f"[{time_string()}] Saving results of run {i+1}")
    exhumations.append(new_exhumation['exhumation'])
    params = pd.concat([params, new_params_df], ignore_index=True)
    scores.append([model_score, i])

np.save(f"{model_exhumation_folder}/exhumation_{label}.npy", exhumations) 
synth_samples.to_csv(f"{model_samples_folder}/samples_{label}.csv", index = False) # Tells me which of the samples were respected
scores_df = pd.DataFrame(scores, columns = ['score', 'iteration']) # Tells me how many of the samples were respected
params.to_csv(f"{model_params_folder}/params_{label}.csv", index = False) # Saves all of the random parameters
scores_df.to_csv(f"{model_scores_folder}/scores_{label}.csv", index = False)

print(f"[{time_string()}] Complete")
clean(label)
