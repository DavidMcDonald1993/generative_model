#!/bin/bash
python src/generative_model.py data/facebook_graph.gml data/facebook_attributes.csv 193 -c data/facebook_circles.csv --gamma 2.51017 --T 0.3497 -e 1
