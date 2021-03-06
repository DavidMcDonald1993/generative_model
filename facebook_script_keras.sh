#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --time 5-00:00:00
#SBATCH --mem-per-cpu 8gb

set -e

module purge; module load bluebear
module load apps/python2/2.7.11
module load apps/tensorflow/1.3.1-python-2.7.11
module load apps/keras/2.0.8-python-2.7.11

python src/generative_model_keras.py data/facebook_graph.gml data/facebook_attributes.csv 193 -c data/facebook_circles.csv --gamma 2.51017 --T 0.3497 --lamb_F 1e-2 --lamb_W 1e-2 --plot plots/facebook
