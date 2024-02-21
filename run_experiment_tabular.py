import argparse
import io
import os
import pickle
import tempfile
import time

import numpy as np
import openml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from torchvision import datasets, transforms

from src.dataloader import MultiFormatDataLoader
from src.evaluator import Evaluator
from src.models import *
from src.trainer import PyTorchTrainer
from src.utils import seed_everything


def main(args):
    # Load the WANDB YAML file
    with open("./wandb.yaml") as file:
        wandb_data = yaml.load(file, Loader=yaml.FullLoader)

    os.environ["WANDB_API_KEY"] = wandb_data["wandb_key"]
    wandb_entity = wandb_data["wandb_entity"]

    total_runs = args.total_runs
    hardness = args.hardness
    dataset = args.dataset
    model_name = args.model_name
    epochs = args.epochs
    seed = args.seed
    p = args.prop

    assert dataset in ["diabetes", "eye", "cover", "jannis"], "Invalid dataset!"

    for i in range(total_runs):

        ####################
        #
        # SET UP EXPERIMENT
        #
        ####################

        print(f"Running {i+1}/{total_runs} for {p}")
        seed_everything(seed)
        print(f"{hardness}_{dataset}_{model_name}_{epochs}")
        dir_to_delete = None

        # new wandb run
        run = wandb.init(
            project=f"test_tabular_{hardness}_{dataset}_{model_name}",
            entity=wandb_entity,
        )

        if hardness == "instance":
            if dataset == "cover":
                rule_matrix = {4: [5], 1: [0], 0: [6], 6: [0], 2: [3], 5: [2], 3: [2]}
            if dataset == "diabetes":
                rule_matrix = {2: [0], 1: [0], 0: [1]}
            if dataset == "eye":
                rule_matrix = {0: [1], 2: [1], 1: [0]}

        else:
            rule_matrix = None

        # diabetes - 4541 (3)
        # eye movement 1044 - (3) https://www.openml.org/search?type=data&sort=runs&id=1044&status=active
        # cover 1596 (7) https://www.openml.org/search?type=data&sort=runs&id=1596&status=active
        # https://huggingface.co/datasets/inria-soda/tabular-benchmark
        # 41168 - Jannis (4)
        print("loading dataset...")
        if dataset == "diabetes":
            id = 4541
        elif dataset == "eye":
            id = 1044
        elif dataset == "cover":
            id = 1596
        elif dataset == "jannis":
            id = 41168
        else:
            raise ValueError("Invalid dataset!")

        dataset_openml = openml.datasets.get_dataset(id)

        X, y, categorical_indicator, attribute_names = dataset_openml.get_data(
            dataset_format="array", target=dataset_openml.default_target_attribute
        )
        df = pd.DataFrame(X, columns=attribute_names)
        df["y"] = y

        print(df.shape)

        if df.shape[0] > 50000:
            df = df.sample(50000, random_state=0)

        total_samples = len(df)
        num_classes = np.unique(y)

        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ####################
        #
        # DATALOADER MODULE
        #
        ####################
        print("loading dataloader...")
        metadata = {
            "p": p,
            "hardness": hardness,
            "dataset": dataset,
            "model": model_name,
            "run": i,
            "seed": seed,
        }

        wandb.log(metadata)

        X = df.drop(columns="y").to_numpy()
        y = df["y"].values

        if hardness == "atypical":
            if dataset == "cover":
                feat = "Elevation"

            if dataset == "diabetes":
                feat = "num_medications"

            if dataset == "eye":
                feat = "timePrtctg"

            marginal = df[feat].values
            index = df.columns.get_loc(feat)
            atypical_marginal = (marginal, index)
        else:
            atypical_marginal = None

        data = (X, y)

        # Allows importing data in multiple formats
        dataloader_class = MultiFormatDataLoader(
            data=data,
            target_column=None,
            data_type="numpy",
            data_modality="tabular",
            batch_size=64,
            shuffle=True,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=hardness,
            p=p,
            rule_matrix=rule_matrix,
            atypical_marginal=atypical_marginal,
        )

        dataloader, dataloader_unshuffled = dataloader_class.get_dataloader()
        flag_ids = dataloader_class.get_flag_ids()

        ####################
        #
        # TRAINER MODULE
        #
        ####################

        # Instantiate the neural network
        model = MLP(input_size=X.shape[1], nlabels=len(np.unique(y))).to(device)

        characterization_methods = [
            "aum",
            "data_uncert",
            "el2n",
            "grand",
            "cleanlab",
            "forgetting",
            "vog",
            "prototypicality",
            "loss",
            "conf_agree",
            "detector",
        ]

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_classes = len(np.unique(y))
        # Instantiate the PyTorchTrainer class
        print("training_model...")
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr=0.001,
            epochs=epochs,
            total_samples=total_samples,
            num_classes=num_classes,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            characterization_methods=characterization_methods,
        )

        # Train the model
        trainer.fit(dataloader, dataloader_unshuffled)
        print("computing...")
        hardness_dict = trainer.get_hardness_methods()

        ####################
        #
        # EVALUATOR MODULE
        #
        ####################

        eval = Evaluator(hardness_dict=hardness_dict, flag_ids=flag_ids, p=p)

        eval_dict, raw_scores_dict = eval.compute_results()
        # add sleep in case of machine latency
        time.sleep(10)
        print(eval_dict)
        wandb.log(eval_dict)

        scores_dict = {
            "metadata": metadata,
            "scores": raw_scores_dict,
            "flag_ids": flag_ids,
        }
        # add sleep in case of machine latency
        time.sleep(30)
        metainfo = f"{dataset}_{hardness}_{p}_{seed}_{i}"
        # log overall_result_dicts to wandb as a pickle
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pickle.dump(scores_dict, temp_file)
            temp_file_path = temp_file.name

        # Log the pickle as a wandb artifact
        artifact = wandb.Artifact(f"scores_dict_{metainfo}", type="pickle")
        artifact.add_file(temp_file_path, name=f"scores_dict_{metainfo}.pkl")
        wandb.run.log_artifact(artifact)
        # Clean up the temporary file
        os.remove(temp_file_path)

        # add sleep in case of machine latency
        time.sleep(30)

        wandb.finish()

        seed += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program.")
    # Add command-line arguments
    parser.add_argument("--total_runs", type=int, default=3, help="Total runs")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--prop", type=float, default=0.1, help="prop")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--hardness", type=str, default="uniform", help="hardness type")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar", "diabetes", "cover", "eye", "jannis"],
        help="Dataset",
    )
    parser.add_argument("--model_name", type=str, default="MLP", help="Model name")

    args = parser.parse_args()

    main(args)
