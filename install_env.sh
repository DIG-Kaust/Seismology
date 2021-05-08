#!/bin/bash
# 
# Installer for Seismology course
# 
# Run: ./install_env.sh
# 
# M. Ravasi, 08/05/2021

echo 'Creating Seismology Course environment'

# create conda env
conda env create -f environment.yml
source activate seismologycourse
conda env list
echo 'Created and activated environment:' $(which python)

# check numpy works as expected
echo 'Checking numpy version and running a command...'
python -c 'import numpy; print(numpy.__version__); print(numpy.ones(10))'

echo 'Done!'

