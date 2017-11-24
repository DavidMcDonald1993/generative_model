#!/bin/bash

#SBATCH --ntasks 25
#SBATCH --time 1-00:00:00
#SBATCH --mem 64000

module purge; module load bluebear

python src/generative_model.py data/facebook_graph.gml data/facebook_attributes.csv 193 -c data/facebook_circles.csv -p 25 --gamma 2.51017 --T 0.3497
