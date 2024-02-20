#!/bin/bash

# EXAMPLE USAGE BELOW: 
# run this script from the root directory of the project

# Hardness types:
# - "uniform": Uniform mislabeling
# - "asymmetric": Asymmetric mislabeling
# - "adjacent" : Adjacent mislabeling
# - "instance": Instance-specific mislabeling
# - "ood_covariate": Near-OOD Covariate Shift
# - "domain_shift": Specific type of Near-OOD
# - "far_ood": Far-OOD shift (out-of-support)
# - "atypical": Marginal atypicality (for tabular data ONLY)


# Set the parameterizable arguments
total_runs=3
epochs=10
seed=0

# examples for instance and atypical hardness

model_name="MLP"
hardness="instance"
dataset="diabetes"
fuser -v /dev/nvidia0 -k
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.1 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.2 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.3 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.4 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.5 --epochs $epochs

dataset="cover"
fuser -v /dev/nvidia0 -k
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.1 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.2 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.3 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.4 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.5 --epochs $epochs


hardness="atypical"
dataset="diabetes"
fuser -v /dev/nvidia0 -k
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.1 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.2 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.3 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.4 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.5 --epochs $epochs

dataset="cover"
fuser -v /dev/nvidia0 -k
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.1 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.2 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.3 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.4 --epochs $epochs
python run_experiment_tabular.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.5 --epochs $epochs


