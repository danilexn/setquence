import argparse
import os
import pprint
import shutil
from pathlib import Path

import torch
import wandb
from mpi4py import MPI

from setquence.base import Config, Environment
from setquence.data import get_dataset_loader
from setquence.distributed import get_distributer
from setquence.models import get_model, load_model_from_file
from setquence.utils import get_optimizer, ns_to_dict
from setquence.utils.metrics import classification_metrics
from setquence.utils.slurm import slurm_config_to_dict

__all__ = ["MPI"]
SUPPORTED_TRACE = ["scorep", "nsys"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SetQuence training, testing and inference")
    parser.add_argument(
        "-c", "--config", help="Path to the JSON file containing the configuration", required=True,
    )
    parser.add_argument("-e", "--experiment", help="Path for the experiment (save, load checkpoints)", default=None)
    args = parser.parse_args()

    # Program general configuration
    config = Config(Path(args.config))
    slurm_env = slurm_config_to_dict()
    env = Environment(slurm_env)

    if env.rank == 0 and "SETQUENCE_LOG_WANDB" in os.environ:
        wandb.init(
            project=os.environ["SETQUENCE_LOG_WANDB_PROJECT"],
            entity=os.environ["SETQUENCE_LOG_WANDB_ENTITY"],
            name=os.environ["SETQUENCE_LOG_WANDB_RUNNAME"],
            settings=wandb.Settings(show_emoji=False),
            config=ns_to_dict(Config(Path(args.config))),
        )

    # Copy the configuration file to the experiment directory
    if env.rank == 0:
        if "SETQUENCE_LOG_WANDB" in os.environ:
            args.experiment = wandb.run.dir
        elif args.experiment is not None and not Path(args.experiment).exists():
            Path(args.experiment).mkdir(parents=True, exist_ok=True)
            shutil.copy(args.config, args.experiment)

    EXPERIMENT_DIR = args.experiment

    # Create the model, or load from an already trained model
    # If finetune option is enabled, will load all possible weights and biases
    # and will not return an error if something fails to load.
    # See models.__init__.load_model_from_file
    if config.model.pretrained_route is not None:
        if config.model.finetune:
            pretrained_model = load_model_from_file(config.model.pretrained_route, env=env)
            model = get_model(config.model.name)(config=config.model.config, env=env)
            model.from_pretrained(pretrained_model)
        else:
            model = load_model_from_file(config.model.pretrained_route, env=env)
    else:
        model = get_model(config.model.name)(config=config.model.config, env=env)

    # Write model representation to a file
    if EXPERIMENT_DIR is not None and env.rank == 0:
        model_repr_route = Path(EXPERIMENT_DIR) / Path("model_repr").with_suffix(".txt")
        with open(model_repr_route, "w") as text_file:
            text_file.write(repr(model))

        if "SETQUENCE_LOG_WANDB" in os.environ:
            wandb.save(str(model_repr_route))

    # Create the distribution context
    dist = get_distributer(config.model.distributer)(env).init(config.model.distribution)
    model.distribute(dist)

    # Loading the data
    train_dataset = get_dataset_loader(config.data.dataloader)(config.data.train, env)

    train_sampler = dist.get_sampler(train_dataset, env, shuffle=config.data.train.shuffle)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.data.train.batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    if config.model.config.decoder.loss.weighted:
        n_classes = config.model.config.decoder.n_classes
        config.model.config.decoder.loss.weights = train_dataset.calculate_class_weight()
        if len(config.model.config.decoder.loss.weights) != n_classes and n_classes > 2:
            raise ValueError("The specified number of classes does not match the number of classes in the dataset")

    # Set scheduling for training data
    dist.scheduled = config.data.train.load_balance

    # Loading the testing data
    # Test does not work with distributed sampler
    test_dataset = get_dataset_loader(config.data.dataloader)(config.data.test, env)

    # Put a DistributedSampler for the test data, too!
    test_sampler = dist.get_sampler(test_dataset, env)

    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config.data.test.batch_size,
        num_workers=1,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    # Creating the optimizer and training
    optimizer = get_optimizer(model, config.training.optimizer)

    if config.training.enabled:
        if config.testing.enabled:

            def callback_fn(model, *args, **kwargs):
                """
                This tests the model with the testing data, and then saves the checkpoint
                """
                prediction_list, label_list = model.predict(test_dataloader)
                if env.rank == 0:
                    output = torch.tensor([], dtype=torch.float32)
                    for prediction in prediction_list:
                        output = torch.cat((output, prediction.to("cpu")), dim=0)

                    labels = torch.tensor([], dtype=torch.float32)
                    for label in label_list:
                        labels = torch.cat((labels, label.to("cpu")), dim=0)
                else:
                    return

                with torch.no_grad():
                    if config.model.config.decoder.n_classes <= 2:
                        pred_labels = torch.where(torch.nn.Sigmoid()(output).view(-1) > 0.5, 1, 0)
                        probs = torch.nn.Sigmoid()(output).view(-1)
                    else:
                        pred_labels = torch.argmax(output, dim=1)
                        probs = torch.nn.Softmax()(output)

                # Using .tensors instead of [:], error slicing!
                # Also, select only those that have been calculated!
                # (drop_last option is enabled)
                test_metrics = classification_metrics(labels, pred_labels, probs)
                pprint.pprint(test_metrics)
                if "SETQUENCE_LOG_WANDB" in os.environ:
                    wandb.log(test_metrics)

                if EXPERIMENT_DIR is not None:
                    # Also, add model saving to a specified path (as checkpoints)
                    # at the moment, checkpoints are overwritten
                    model_state_dict = model.state_dict()
                    optimizer_state_dict = optimizer.state_dict()
                    checkpoint_route = Path(EXPERIMENT_DIR) / Path(f"checkpoint_{model.model_name}").with_suffix(
                        ".chk"
                    )
                    torch.save(
                        {"model_state_dict": model_state_dict, "optimizer_state_dict": optimizer_state_dict},
                        checkpoint_route,
                    )
                    if "SETQUENCE_LOG_WANDB" in os.environ:
                        wandb.save(str(checkpoint_route))

        else:

            def callback_fn(model, *args, **kwargs):
                """
                This saves model checkpoint at the end of epochs
                """

                if env.rank == 0 and EXPERIMENT_DIR is not None:
                    # Also, add model saving to a specified path (as checkpoints)
                    # at the moment, checkpoints are overwritten
                    model_state_dict = model.state_dict()
                    optimizer_state_dict = optimizer.state_dict()
                    checkpoint_route = Path(EXPERIMENT_DIR) / Path(f"checkpoint_{model.model_name}").with_suffix(
                        ".chk"
                    )
                    torch.save(
                        {"model_state_dict": model_state_dict, "optimizer_state_dict": optimizer_state_dict},
                        checkpoint_route,
                    )
                    if "SETQUENCE_LOG_WANDB" in os.environ:
                        wandb.save(str(checkpoint_route))

        model.fit(
            config.training.config, train_dataloader=train_dataloader, optimizer=optimizer, callback_fn=callback_fn,
        )

        if env.rank == 0 and EXPERIMENT_DIR is not None:
            # Save model after training at the specified experiment path
            checkpoint_route = Path(EXPERIMENT_DIR) / Path(f"{model.model_name}")
            model.save_model(checkpoint_route)
            if "SETQUENCE_LOG_WANDB" in os.environ:
                wandb.save(str(checkpoint_route))
