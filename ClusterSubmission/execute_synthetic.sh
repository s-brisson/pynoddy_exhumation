#!/bin/bash

cd /home/ho640525/projects/Exhumation
source /home/ho640525/.cache/pypoetry/virtualenvs/exhumation-IrWm4u2d-py3.8/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/ho640525/projects/pynoddy/pynoddy
python sythetic_mcmc.py "$1" --folder "$2"
