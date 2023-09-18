#!/bin/bash

#cd /home/ho640525/projects/Exhumation
#source /home/ho640525/.cache/pypoetry/virtualenvs/exhumation-IrWm4u2d-py3.8/bin/activate
#export PYTHONPATH=$PYTHONPATH:/home/ho640525/projects/pynoddy/pynoddy
#python exhumation_block.py $1 $2 $3 --folder $4 

#!/bin/bash

cd /home/ho640525/projects/Exhumation
source /home/ho640525/.cache/pypoetry/virtualenvs/exhumation-IrWm4u2d-py3.8/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/ho640525/projects/pynoddy/pynoddy

input_file="home/ho640525/projects/Exhumation/data/input_files/TrainingsParametersKinematicModelAlpsFault1-4.npy"
total_params=$(python -c "import numpy as np; arr = np.load("$input_file"); total_params = len(arr); print(total_params)")

params_per_node=10
total_nodes=$((total_params / params_per_node))

for node_id in $(seq 1 $total_nodes); do
    start_param=$((($node_id - 1) * $params_per_node))
    end_param=$(($node_id * $params_per_node))

    python exhumation_block.py $1 $2 $3 --folder $4 $start_param $end_param &
done

wait

echo "finished"
