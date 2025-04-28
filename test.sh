#!/bin/bash

# Ensure you're using bash
#datasets=("magic" "spambase" "statlog" "cycle" "concrete" "superconductor") #"gas"
#types=("binary" "binary" "multiclass" "regression" "regression" "regression" ) #"multiclass"

# Loop through each dataset and its corresponding type
python3 Experiments.py --iterations 100 --nrs 10  --train > /scratch/tmp/n_herr03/ecai/outdata/$dataset.out
