#!/bin/bash

std_values = (500 500 1000 300)

for std in "${std_values[@]}"; do
  cd /home/ho640525/projects/Exhumation
  source /home/ho640525/.cache/pypoetry/virtualenvs/exhumation-IrWm4u2d-py3.8/bin/activate
  export PYTHONPATH=$PYTHONPATH:/home/ho640525/projects/pynoddy/pynoddy
  python case_mcmc_new.py "$1" "$2" --property "$3" --standard_deviation "$std" --folder "$5" 
done
