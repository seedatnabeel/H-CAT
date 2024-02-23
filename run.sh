#!/bin/bash

# EXAMPLE USAGE BELOW: 
# run this script from the root directory of the project
# bash run.sh

# Hardness types:
# - "uniform": Uniform mislabeling
# - "asymmetric": Asymmetric mislabeling
# - "adjacent" : Adjacent mislabeling
# - "instance": Instance-specific mislabeling
# - "ood_covariate": Near-OOD Covariate Shift
# - "domain_shift": Specific type of Near-OOD
# - "far_ood": Far-OOD shift (out-of-support)
# - "zoom_shift": Zoom shift  - type of Atypical for images
# - "crop_shift": Crop shift  - type of Atypical for images

# --fix_seed for conistency: where the seed is fixed and we assess model randomness

# Set the parameterizable arguments
total_runs=3
epochs=10
seed=0

# uniform mnist 
hardness="uniform"
dataset="mnist"
model_name="LeNet"
fuser -v /dev/nvidia0 -k
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.1 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.2 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.3 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.4 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.5 --epochs $epochs


# uniform cifar
hardness="uniform"
dataset="cifar"
model_name="ResNet"
fuser -v /dev/nvidia0 -k
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.1 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.2 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.3 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.4 --epochs $epochs
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.5 --epochs $epochs

