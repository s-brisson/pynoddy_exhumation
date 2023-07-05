import pickle
import pandas as pd

history = '../Lechtal model/bregenz_ver5.his'
output_name = 'outputs/noddy_out'
pynoddy.compute_model(history, output_name, 
                      noddy_path = r'C:\Users\Sofia\pynoddy\noddyapp\noddy_win64.exe')

hist = pynoddy.history.NoddyHistory(history)
out = pynoddy.output.NoddyOutput(output_name)

hist.change_cube_size(100)
hist_hd = 'hist_hd.his'
out_hd = 'outputs/out_hd'
hist.write_history(hist_hd)
pynoddy.compute_model(hist_hd, out_hd, noddy_path = r'C:\Users\Sofia\pynoddy\noddyapp\noddy_win64.exe')
out_hd = pynoddy.output.NoddyOutput(out_hd)

n_draws = 50
lith = [11]
save = True

#Make code faster or slower 
res = 16 #samples coord every res voxels
interval = 100 #calculates exhumation every interval m 
upperlim = out.zmax - interval #upplimit to which intervals get to (zmax is the highest, also the slowest)

#storing stuff
all_coords = []
all_blocks = np.ndarray((n_draws, out_hd.nx, out_hd.ny, out_hd.nz), dtype = 'int')
all_params = pd.DataFrame(columns = ['Event', 'New Dip', 'New Dip Direction', 'New Pitch', 'New Slip', 'New Amplitude', 'New X', 'New Z'])

for i in tqdm(range(n_draws), desc = "This code actually works"):
    hist_copy = copy.deepcopy(hist) #hist = the ORIGINAL hist, to reset the modele after each run
    
    #change parameters - in this case only the fault geometry
    new_params = disturb(hist_copy)
    
    coords, output = exhumationComplex(hist_copy,lith=lith,res=res,interval=interval, upperlim=upperlim)
    
    all_coords.append(coords)
    all_blocks[i,:,:,:] = output.block
    all_params = pd.concat([all_params, new_params], ignore_index = True)
    
    #Save outputs
    if save == True:
        np.save('model_coords/bregenz_coords/coords_%04d.npy'%i, coords)
pickle.dump(all_blocks, open('model_blocks/all_blocks_%d.pkl'%(n_draws), 'wb'))
all_params.to_csv('outputs/params.csv', index = False)