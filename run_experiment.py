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

    assert dataset in ["cifar", "mnist"], "Invalid dataset!"

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

        if hardness == "instance":
            if dataset == "mnist":
                rule_matrix = {
                    1: [7],
                    2: [7],
                    3: [8],
                    4: [4],
                    5: [6],
                    6: [5],
                    7: [1, 2],
                    8: [3],
                    9: [7],
                    0: [0],
                }
            if dataset == "cifar":

                rule_matrix = {
                    0: [2],  # airplane (unchanged)
                    1: [9],  # automobile -> truck
                    2: [9],  # bird (unchanged)
                    3: [5],  # cat -> automobile
                    4: [5, 7],  # deer (unchanged)
                    5: [3, 4],  # dog -> cat
                    6: [6],  # frog (unchanged)
                    7: [5],  # horse -> dog
                    8: [7],  # ship (unchanged)
                    9: [9],  # truck -> horse
                }

        else:
            rule_matrix = None

        if dataset == "mnist":
            # Define transforms for the dataset
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )
            # Load the MNIST dataset
            train_dataset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            )
            num_classes = 10

        elif dataset == "cifar":
            # Define transforms for the dataset
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Load the CIFAR-10 dataset
            train_dataset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
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
        if dataset == "cifar":
            if model_name == "LeNet":
                model = LeNet(num_classes=10).to(device)
            if model_name == "ResNet":
                model = ResNet18().to(device)
        elif dataset == "mnist":
            if model_name == "LeNet":
                model = LeNetMNIST(num_classes=10).to(device)
            if model_name == "ResNet":
                model = ResNet18MNIST().to(device)
        elif dataset == "xray":
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

        if not args.fix_seed:
            seed += 1


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        choices=["mnist", "cifar"],
        help="Dataset",
    )
    parser.add_argument("--model_name", type=str, default="LeNet", help="Model name")
    parser.add_argument(
        "--fix_seed",
        type=str2bool,
        default="false",
        help="fix the seed for consistency exps",
    )

    args = parser.parse_args()

    main(args)
