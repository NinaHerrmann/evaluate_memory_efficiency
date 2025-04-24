#!/bin/bash

# Ensure you're using bash
#datasets=("magic" "spambase" "statlog" "cycle" "concrete" "superconductor") #"gas"
#types=("binary" "binary" "multiclass" "regression" "regression" "regression" ) #"multiclass"

# Loop through each dataset and its corresponding type
python3 Experiments.py --iterations 5 --nrs 5  --train > data/$dataset/$dataset.out
