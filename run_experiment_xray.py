import argparse
import io
import os
import pickle
import tempfile
import time

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from torchvision import datasets, transforms

from src.dataloader_xray import MultiFormatDataLoader
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

    assert dataset == "xray", "Invalid dataset!"

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
            project=f"{hardness}_{dataset}_{model_name}",
            entity=wandb_entity,
        )

        rule_matrix = None

        if dataset == "xray":
            # Paper: https://pubs.rsna.org/doi/10.1148/radiol.2019191293
            # Dataset link: https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
            # Usage: https://mlmed.org/torchxrayvision/datasets.html#torchxrayvision.datasets.NIH_Google_Dataset
            imgpath = "./data/images/images-224"
            transform = torchvision.transforms.Compose([xrv.datasets.XRayResizer(32)])
            train_dataset = xrv.datasets.NIH_Google_Dataset(
                imgpath,
                csvpath="USE_INCLUDED_FILE",
                transform=transform,
                data_aug=None,
                nrows=None,
                seed=0,
                unique_patients=True,
            )
            num_classes = 10
        else:
            raise ValueError("Invalid dataset!")

        total_samples = len(train_dataset)

        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ####################
        #
        # DATALOADER MODULE
        #
        ####################
        metadata = {
            "p": p,
            "hardness": hardness,
            "dataset": dataset,
            "model": model_name,
            "run": i,
            "seed": seed,
        }

        wandb.log(metadata)

        # Allows importing data in multiple formats
        dataloader_class = MultiFormatDataLoader(
            data=train_dataset,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            batch_size=64,
            shuffle=True,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=hardness,
            p=p,
            rule_matrix=rule_matrix,
        )

        dataloader, dataloader_unshuffled = dataloader_class.get_dataloader()
        flag_ids = dataloader_class.get_flag_ids()

        ####################
        #
        # TRAINER MODULE
        #
        ####################

        # Instantiate the neural network
        if model_name == "LeNet":
            model = LeNetMNIST(num_classes=2).to(device)
        if model_name == "ResNet":
            model = ResNet18MNIST().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Instantiate the PyTorchTrainer class
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr=0.001,
            epochs=epochs,
            total_samples=total_samples,
            num_classes=num_classes,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Train the model
        trainer.fit(dataloader, dataloader_unshuffled)

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
        choices=["mnist", "cifar", "xray"],
        help="Dataset",
    )
    parser.add_argument("--model_name", type=str, default="LeNet", help="Model name")

    args = parser.parse_args()

    main(args)
