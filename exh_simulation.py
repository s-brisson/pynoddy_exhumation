from exh_functions import *
from os import makedirs
import pickle

created_parser = parser()
args = created_parser.parse_args()

label = generate_unique_label()
history = 'data/input_files/bregenz_ver5.his'
output_folder = "data/outputs"
model_coordinate_folder = f"{output_folder}/model_coords/bregenz_coords/{args.folder}/"
model_block_folder = f"{output_folder}/model_block/bregenz_coords/{args.folder}/"
makedirs(model_coordinate_folder,exist_ok=True)
makedirs(model_block_folder,exist_ok=True)

print(f"[{time_string()}] {'Simulating based on file':<40} {history}")
print(f"[{time_string()}] {'Number of simulations':<40} {args.ndraws}")
print(f"[{time_string()}] {'Indicator layer z-axis step size':<40} {args.interval}")
print(f"[{time_string()}] {'Voxels resampling resolution':<40} {args.resolution}")
print(f"[{time_string()}] {'Model output files folder':<40} {args.folder}")
print()



print(f"[{time_string()}] Running the base model")
output_name = f'{output_folder}/noddy/noddy_out_{label}'
pynoddy.compute_model(history, output_name, 
                      noddy_path = noddy_exe,
                      verbose=True)

hist = pynoddy.history.NoddyHistory(history)
out = pynoddy.output.NoddyOutput(output_name)

hist.change_cube_size(100)
hist_hd = f'{output_folder}/history/hist_hd_{label}.his'
out_hd = f'{output_folder}/noddy/out_hd_{label}'
hist.write_history(hist_hd)
print(f"[{time_string()}] Running the HD model")
pynoddy.compute_model(hist_hd, out_hd, noddy_path = noddy_exe)
out_hd = pynoddy.output.NoddyOutput(out_hd)

lith = [11]
save = True

#Make code faster or slower
n_draws = args.ndraws
res = args.resolution #samples coord every res voxels
interval = args.interval #calculates exhumation every interval m 
upperlim = out.zmax - interval #upplimit to which intervals get to (zmax is the highest, also the slowest)
all_coords = []
all_blocks = np.ndarray((n_draws, out_hd.nx, out_hd.ny, out_hd.nz), dtype = 'int')

for i in tqdm(range(n_draws), desc = "This code actually works"):
    hist_copy = copy.deepcopy(hist) #hist = the ORIGINAL hist, to reset the modele after each run
    
    #change parameters - in this case only the fault geometry
    for event in hist_copy.events.values():
        if isinstance(event, pynoddy.events.Fault):
            disturb_percent(event, 'Dip', percent = 5)
            disturb_percent(event, 'Dip Direction', percent = 5)
            disturb_percent(event, 'Pitch', percent = 5)
            disturb_value(event, 'X', 50)
            disturb_value(event, 'Z', 75)
    
    coords = exhumationComplex(hist_copy,lith=lith,res=res,interval=interval, upperlim=upperlim, unique_label = label)
    coords,output = exhumationComplex(hist_copy,lith=lith,res=res,interval=interval, upperlim=upperlim, unique_label = label)
    
    all_coords.append(coords)
    all_blocks[i,:,:,:] = output.block
    
    if save == True:
        np.save(f'{model_coordinate_folder}/coords_{i:04d}_{label}.npy', coords)
        print(f"[{time_string()}] Model {i} saved with label {label}")

pickle.dump(all_blocks, open(f'{model_block_folder}/all_blocks_{args.ndraws}_{label}.pkl', 'wb'))
print(f"[{time_string()}] Complete")
clean(label)
