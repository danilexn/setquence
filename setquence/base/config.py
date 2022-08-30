import json
import pprint
import warnings
from pathlib import Path
from typing import Any, Dict, Union

from setquence.utils import dict_to_namespace

DEFAULT_CONFIG = {
    "trace": {"enabled": False, "type": "scorep", "n_steps": 20},
    "data": {
        "dataloader": "SetQuenceDataset",
        "train": {
            "synthetic": False,
            "head": -1,
            "route": "data/dual/train.torch",
            "batch_size": 1,
            "shuffle": False,
            "kmer_file": None,
        },
        "test": {
            "synthetic": False,
            "head": -1,
            "route": "data/dual/dev.torch",
            "batch_size": 1,
            "shuffle": False,
            "kmer_file": None,
        },
    },
    "training": {
        "enabled": True,
        "config": {"epochs": 1, "steps": -1, "max_grad_norm": -1, "amp": False, "grad_acc_steps": 1, "loss_factor": 1},
        "optimizer": {"name": "sgd", "distributed": False, "config": {"lr": 0.000025},},
    },
    "testing": {"enabled": True, "evaluator": "classification"},
    "model": {
        "name": "setquence",
        "distribution": "DistributedDataParallel",
        "distributer": "base",
        "finetune": False,
        "pretrained_route": None,
        "config": {
            "encoder": {
                "seq_len": 64,
                "max_seq": 400,
                "seq_split": 400,
                "hidden_size": 768,
                "embedding_size": 768,
                "bert": True,
                "bert_config": "configs/config_bert.json",
                "bert_pretrained": True,
                "bert_route": "/home/dale016c/PerformanceAnalysis/DNABERT_6",
                "lr_times": 1,
                "max_gradient": 400,
                "selector": None,
                "temperature": 3,
                "alpha": 0.1,
                "teacher_bert_config": "/home/dale016c/PerformanceAnalysis/DNABERT_6/config.json",
                "teacher_bert_route": "/home/dale016c/PerformanceAnalysis/DNABERT_6",
            },
            "pooler": {
                "p_dropout": 0.3,
                "n_heads": 12,
                "k_seeds": 1,
                "hidden_size": 768,
                "max_sequences_pool": 5000,
                "n_layers": 1,
                "pairing_method": "sum,diff,or",
            },
            "decoder": {
                "n_classes": 33,
                "p_dropout": 0.3,
                "hidden_size": 768,
                "loss": {"module": "crossentropyloss", "weighted": True, "weights": None},
                "lr_times": 1,
                "noisy_gating": True,
                "num_experts": 10,
                "k": 4,
                "n_blocks": 2,
            },
            "reconstructor": {"dim_out": 768, "p_dropout": 0.3, "hidden_size": 768},
        },
    },
}


class Config(object):
    def __init__(self, config: Union[Dict, Path], verbose=True, *args, **kwargs):
        if isinstance(config, Path):
            with open(config) as json_file:
                self.config = json.load(json_file)
        elif isinstance(config, Dict):
            self.config = config
            for k, v in self.config.items():
                setattr(self, k, v)
        else:
            raise ValueError("Configuration argument needs to be either Path or Dict type")

        self.mandatory_keys_default = DEFAULT_CONFIG
        self._check_mandatory(self.mandatory_keys_default, self.config)

        self.config = dict_to_namespace(self.config)
        if verbose:
            print("Loaded configuration:")
            pprint.pprint(self.config)

    def __getattr__(self, __name: str) -> Any:
        return self.config.__getattribute__(__name)

    def __repr__(self) -> str:
        return pprint.pformat(self.config)

    def _check_mandatory(self, mandatory_dict, against_dict):
        for k, v in mandatory_dict.items():
            if k not in against_dict.keys():
                warnings.warn(f"'{k}' was not specified in Environment. Setting default to {v}")
                against_dict[k] = v
            else:
                if isinstance(v, dict):
                    self._check_mandatory(v, against_dict[k])


class Environment:
    def __init__(self, env: Dict, verbose=False, *args, **kwargs):
        self.mandatory_keys_default = {"rank": 0, "local_rank": 0, "size": 1}
        self.env = env
        for k, v in self.env.items():
            setattr(self, k, v)
        self._check_mandatory()

        self.env = dict_to_namespace(self.env)
        if verbose:
            pprint.pprint(self.env)

    def __getattr__(self, __name: str) -> Any:
        return self.env.__getattribute__(__name)

    def __repr__(self) -> str:
        return pprint.pformat(self.env)

    def _check_mandatory(self) -> bool:
        for k, v in self.mandatory_keys_default.items():
            if k not in self.env.keys():
                warnings.warn(f"'{k}' was not specified in Environment. Setting default to {v}")
                self.env[k] = v


EMPTY_ENVIRONMENT = Environment(
    {
        "rank": 0,
        "local_rank": 0,
        "size": 1,
        "nodes": 1,
        "nodename": "localhost",
        "tasks_per_node": 1,
        "local_device_rank": 0,
        "hostnames": ["localhost"],
        "device": "cpu",
        "device_ids": [0],
        "local_device_ids": 0,
    }
)
