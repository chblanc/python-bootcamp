#!/bin/bash

# make sure that anaconda/miniconda is install before proceeding and update
# conda, if missing, install from: https://www.continuum.io/downloads
#
# NOTE: when prompted to append the Anaconda path to your systems PATH, it is
# strongly advised that you say *NO*, unless you intend to allow Anaconda to
# overwrite your system version of Python

# change dir to where this script lives
CURR_DIR=$(pwd)
SH_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SH_DIR

# create env
conda create --name python-bootcamp python=3.7

# activating the env:
#   in windows: activate chill
#   in os/linux: source activate chill
#
# deactivating the env:
#   in windows: deactivate
#   in os/linux: source deactivate

source activate python-bootcamp

# install python packages -----------------------------------------------------
conda install numpy
conda install pandas
conda install scikit-learn
conda install matplotlib

# install R -------------------------------------------------------------------

# R, RSudio, and useful packages
conda install -c r r-base r=3.5.1 --force
conda install -c rdonnelly rstudio=1.2.502
conda install -c r r-essentials
conda install -c r r-devtools
conda install -c r r-e1071
conda install -c r r-matrix 
conda install -c r r-reticulate


# add .Rprofile ---------------------------------------------------------------

PYPATH="$(which python)"
echo "Sys.setenv(RETICULATE_PYTHON = '$PYPATH')" > ../.Rprofile

# save yml --------------------------------------------------------------------

# note: the output here is OS specific (so will not work cross-platform)
# conda env export > ./env/chill_environment.yml

# return to original pwd
cd $CURR_DIR
