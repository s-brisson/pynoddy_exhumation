import pynoddy
import importlib
importlib.reload(pynoddy)
import pynoddy.history
import pynoddy.output

noddy_exe_path = "/rwthfs/rz/cluster/home/ho640525/projects/pynoddy/pynoddy/noddyapp/noddy"
history = 'twofaults_translation.his'
output_name = 'noddy_out'
pynoddy.compute_model(history, output_name, 
                      noddy_path = noddy_exe_path)

hist = pynoddy.history.NoddyHistory(history)
out = pynoddy.output.NoddyOutput(output_name)
print("Francesco is the best")
