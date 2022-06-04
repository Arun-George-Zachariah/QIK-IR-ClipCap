#!/usr/bin/env bash

# Constants.
HOME=${HOME}
QIK_HOME=${PWD}/../..

# Installing FR-CNN.
cd $QIK_HOME/ML_Models/DeepVision && bash setup.sh

# Activating the conda environment.
source activate deepvision

# Installing additional requirements
pip install opencv-python==4.2.0.32

# Setting python path after installation.
export PYTHONPATH=$QIK_HOME/ML_Models/DeepVision/py-faster-rcnn/caffe-fast-rcnn/python:$PYTHONPATH

# Extracting features. (Note features needs to be extraced with a GPU)
# python read_data.py && python features.py
wget https://mailmissouri-my.sharepoint.com/:u:/g/personal/az2z7_umsystem_edu/EfA4JX6_9CNGpUJo9EombI4Bc47B8o2aqt153Xejprqjuw?download=1 -O $QIK_HOME/ML_Models/DeepVision/data
cd $QIK_HOME/ML_Models/DeepVision/data && tar -xvf FR_CNN_Features.tar

# Starting the FR-CNN web app.
python qik_search.py &>> /dev/null &