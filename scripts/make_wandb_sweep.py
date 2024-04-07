#!/usr/bin/env python

"""Creates a hyperparameter sweep in the wandb API.

Manually modify functions below with different Dicts to change
the hyperparameters, or sweep method (defaults to random).

See here for details:
https://docs.wandb.ai/guides/sweeps/define-sweep-configuration"""

import argparse
from typing import Dict
import os

import wandb

# Turn off the wandb logs, so we can log only the sweep id.
os.environ['WANDB_SILENT']="true"


def get_hparams(arch: str) -> Dict:
    """Gets the dictionary of hyperparams to sweep.

    Args:
        arch (str): name of the architecture.
            Used for choosing architecture specific hyperparameters.

    Returns:
        Dict: Dictionary of hyperparameter names and value distributions.
    """
    # Optimization params
    optim_hparams = {
        "batch_size": {
            "distribution": "q_uniform",
            "q": 16,
            "min": 16,
            "max": 2048,
        },
        # "learning_rate": {
        #     "distribution": "log_uniform_values",
        #     "min": 0.0000001,
        #     "max": 0.1,
        # },
        # {1e-6, ..., .01}
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.000001,
            "max": 0.01,
        },
        "beta1": {"distribution": "uniform", "min": 0.8, "max": 0.999},
        "beta2": {"distribution": "uniform", "min": 0.98, "max": 0.999},
        "label_smoothing": {"distribution": "uniform", "min": 0.0, "max": 0.2},
        "scheduler": {"values": ["reduceonplateau", "warmupinvsqrt", None]},
        "num_warmup_samples": {
            "distribution": "q_uniform",
            "q": 100,
            "min": 0,
            "max": 5000000,
        },
        "factor": {"distribution": "uniform", "min": 0.1, "max": 0.9},
        "reduce_lr_patience": {
            "distribution": "q_uniform",
            "q": 1,
            "min": 1,
            "max": 5,
        },
        "min_lr": {"distribution": "uniform", "min": 10e-7, "max": 0.001},
        "reduceonplateau_mode": {"values": ["min"]},
    }

    # Hyperparameters that impact the actual architecture.
    arch_hparams = {
        "embedding_size": {
            "distribution": "q_uniform",
            "q": 16,
            "min": 16,
            "max": 512,
        },
        "hidden_size": {
            "distribution": "q_uniform",
            "q": 64,
            "min": 64,
            "max": 2048,
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.5,
        },
    }

    if arch == "transformer":
        arch_hparams.update(
            {
                "attention_heads": {"values": [2, 4, 8]},
                "encoder_layers": {"values": [2, 4, 6, 8]},
                "decoder_layers": {"values": [2, 4, 6, 8]},
            }
        )
    elif arch == "attentive_lstm":
        arch_hparams.update(
            {
                "attention_heads": {"values": [1]},
                "encoder_layers": {"values": [1, 2]},
                "decoder_layers": {"values": [1]},
            }
        )

    # Combines the optimization and architectural params into one dict.
    return optim_hparams | arch_hparams


def make_sweep(project: str, sweep: str, arch: str) -> int:
    """Creates the sweep in the wandb API, according to the hyperparameter
    ranges in `HPARAMS`.

    Args:
        project (str): Name of the wandb project.
        sweep (str): Name of the wandb sweep.
        arch (str): The architecture for this sweep.

    Returns:
        int: The wandb sweep ID for this configuration.
    """
    # TODO: Change search method or metric to maximize/minimize
    sweep_configuration = {
        "method": "random",
        "name": sweep,
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": get_hparams(arch),
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    return sweep_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="Name of the project")
    parser.add_argument("--sweep", required=True, help="Name of the sweep")
    parser.add_argument(
        "--arch", required=True, help="Architecture for the sweep"
    )
    parser.add_argument(
        "--outpath", required=True, help="Path to append sweep info to."
    )
    args = parser.parse_args()
    sweep_id = make_sweep(args.project, args.sweep, args.arch)

    with open(args.outpath, "a") as out:
        print(f"{args.project},{args.sweep},{sweep_id}", file=out)


if __name__ == "__main__":
    main()
