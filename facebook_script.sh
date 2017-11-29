#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --time 1:00:00
#SBATCH --mem-per-cpu 8gb

set -e

module purge; module load bluebear
module load apps/python2/2.7.11


python src/generative_model.py data/facebook_graph.gml data/facebook_attributes.csv 193 -c data/facebook_circles.csv --gamma 2.51017 --T 0.3497 --plot plots/facebook 
