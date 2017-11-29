#!/bin/bash

#SBATCH --ntasks 6
#SBATCH --time 1-00:00:00
#SBATCH --mem-per-cpu 8gb

set -e

module purge; module load bluebear
module load apps/python2/2.7.11


python src/generative_model.py data/0_facebook_graph.gml data/0_facebook_attributes.csv 24 -c data/0_facebook_circles.csv --gamma 2.51017 --T 0.3497 --plot plots/par_0_facebook -p 5
