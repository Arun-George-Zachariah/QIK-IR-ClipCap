#!/usr/bin/env bash

# Constants.
HOME=${HOME}
QIK_HOME=${PWD}/../..

# Obtaining the model.
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -O $QIK_HOME/ML_Models/CroW/vgg/VGG_ILSVRC_16_layers.caffemodel

# Setting up the conda environment.
conda create -y --name crow_env python=2.7
conda activate crow_env
conda install caffe
pip install numpy
pip install scipy==0.14.0
pip install scikit-learn==0.15.2
pip install pillow

# Setting python path after installation.
export PYTHONPATH=$QIK_HOME/ML_Models/DeepVision/py-faster-rcnn/caffe-fast-rcnn/python:$PYTHONPATH

# Extracting the features.
# To construct CroW features: cd $QIK_HOME/ML_Models/CroW && python extract_features.py --images $HOME/apache-tomcat/webapps/QIK_Image_Data --out out
