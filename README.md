# Hardness Characterization Analysis Toolkit (H-CAT)

![GitHub top language](https://img.shields.io/github/languages/top/seedatnabeel/H-CAT)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)]()

---

## 📝 What is H-CAT? 

Data and hardness characterization are crucial in Data-Centric AI. 

Many methods have been developed for this purpose. H-CAT is a unified framework and API interface for 13 state-of-the-art hardness and data characterization methods --- making them easy to use and/or evaluate. 

We also include a benchmark capability that allows these hardness characterization methods (HCMs) to be evaluated on 9 different types of hardness.

![image](pipeline.png "H-CAT framework")


## 🚀 Installation

To install H-CAT, follow the steps below:

1. Clone the repository

2. Create a new virtual environment or conda environment with Python >3.7: 

    ```bash
    virtualenv hcat_env 
    ```
    OR
    ```bash
    conda create --name hcat_env
    ```

3. With the environment activated, run the following command from the repository directory:

    ```bash
    pip install -r requirements.txt
    ```

4. Link the venv or conda env to the kernel:

    ```bash
    python -m ipykernel install --user --name=hcat_env
    ```

## 🛠️ Usage of H-CAT

There are two ways to get started with H-CAT:

1. Have a look at the tutorial notebook: `tutorial.ipynb` which shows you step by step how to use the different H-CAT modules.

2. Using H-CAT on your own data --- you could follow the steps in the tutorial notebook.

3. Running a benchmarking evaluation as in our paper. `run_experiment.py` runs different experiements. These can be triggered by bash scripts. We provide examples in `run.sh` or `run_tabular.sh`.

Below is a simple example of how to use H-CAT:

```bash
# Set the parameterizable arguments
total_runs=3
epochs=10
seed=0
hardness="uniform"
dataset="mnist"
model_name="LeNet"
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop 0.1 --epochs $epochs
```


Detailed commands:
```
# Usage: python run_experiment.py [OPTIONS]

or 

python run_experiment_tabular.py [OPTIONS]

# Options:
#   --total_runs INTEGER          Number of independent runs
#   --seed INTEGER                Seed
#   --prop FLOAT                  Proportion of samples perturbed (0.1-0.5)
#   --epochs FLOAT                Training epochs to get training dynamics
#   --hardness [uniform|asymmetric|adjacent|instance|ood_covariate|zoom_shift|crop_shift|far_ood| atypical]   Hardness types
#   --dataset [mnist|cifar|diabetes|cover|eye|xray]  *run_experiment.py if image datasets [cifar, mnist] and run_experiment_tabular.py if tabular datasets [diabetes, cover, eye]
#   --model_name [LeNet|ResNet|MLP]   LeNet: LeNet Model (images), ResNet: ResNet-18 (Images), MLP: Multi-layer perceptron (tabular)     
```

**Analysis and plots:** 
Results from the benchmarking can be visualized using `analysis_plots.ipynb` (all other results) and/or `stability_plot.ipynb` (stability/consistency results). The values are pulled from wandb logs (see below).

**Hardness types:**
- "uniform": Uniform mislabeling
- "asymmetric": Asymmetric mislabeling
- "adjacent" : Adjacent mislabeling
- "instance": Instance-specific mislabeling
- "ood_covariate": Near-OOD Covariate Shift
- "domain_shift": Specific type of Near-OOD
- "far_ood": Far-OOD shift (out-of-support)
- "zoom_shift": Zoom shift  - type of Atypical for images
- "crop_shift": Crop shift  - type of Atypical for images
- "atypical": Marginal atypicality (for tabular data ONLY)



## 🔎 Logging
Outputs from experimental scripts are logged to [Weights and Biases - wandb](https://wandb.ai). An account is required and your WANDB_API_KEY and Entity need to be set in wandb.yaml file provided.


## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for more details.

---
# Citing

If you find this repository useful in your research, please cite the following paper:

```
@inproceedings
{seedat2024hardness,
title={Dissecting sample hardness: Fine-grained analysis of Hardness Characterization Methods},
author={Seedat, Nabeel and Imrie, Fergus and van der Schaar, Mihaela},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024}
}
```

---

<div align="center">
    <strong>Give a ⭐️ if this project was useful!</strong>
</div>

