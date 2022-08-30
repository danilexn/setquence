import warnings
from typing import Dict, List, Union

import torch
from torch import distributed as dist
from torch import nn as nn
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from setquence.base.config import EMPTY_ENVIRONMENT, Environment
from setquence.data.dataset import SetQuenceDataset


class NoDistributed(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


COMM_SEND = "send"
COMM_RECV = "recv"

SUPPORTED_DISTRIBUTERS = (NoDistributed, DDP, DataParallel)
STR_2_DISTR_STRATEGIES = {"DistributedDataParallel": DDP, "DataParallel": DataParallel, "NoDistributed": NoDistributed}
COMPATIBLE_DEVICES_STRATEGIES = {NoDistributed: ["cpu", "cuda"], DDP: ["cpu", "cuda"], DataParallel: ["cuda"]}


class Distributed:
    def __init__(self, env: Environment = EMPTY_ENVIRONMENT, *args, **kwargs):
        self.env = env
        self.distribution_strategy = NoDistributed

    def _init_ddp(self):
        try:
            if self.env.device == "cpu":
                dist.init_process_group(
                    backend="gloo", world_size=self.env.size, rank=self.env.rank,
                )
                self.env.device = torch.device("cpu")
            elif self.env.device == "cuda":
                dist.init_process_group(
                    backend="nccl", init_method="env://", world_size=self.env.size, rank=self.env.rank,
                )
                torch.cuda.set_device(self.env.local_device_rank)
                self.env.device = f"cuda:{self.env.local_device_rank}"
            else:
                raise ValueError(f"Device of type {self.env.device} cannot be distributed")
        except KeyError as e:
            raise KeyError(f"Could not set up a Distributed context\n{e}")

    def _init_fsdp(self):
        warnings.warn("FSDP uses DDP backend for communication")
        self._init_ddp()

    def _init_dp(self):
        warnings.warn("DataParallel does not need initialisation. Continuing.")

    def _init_none(self):
        warnings.warn("No distribution strategy chosen. Continuing.")

    def init(self, strategy: Union[DDP, NoDistributed, DataParallel, str]):
        DISTR_STRATEGIES = {
            DDP: self._init_ddp,
            DataParallel: self._init_dp,
            NoDistributed: self._init_none,
        }

        if isinstance(strategy, str):
            if strategy not in STR_2_DISTR_STRATEGIES:
                raise ValueError(f"Strategy {strategy} cannot be used")
            else:
                strategy = STR_2_DISTR_STRATEGIES[strategy]

        if strategy in SUPPORTED_DISTRIBUTERS:
            if self.env.device not in COMPATIBLE_DEVICES_STRATEGIES[strategy]:
                warnings.warn(f"Cannot use {strategy} with device {self.env.device}")
                strategy = NoDistributed  # no distribution at all, compatible with all

            DISTR_STRATEGIES[strategy]()
            self.distribution_strategy = strategy
        else:
            raise ValueError(f"Strategy {strategy} cannot be used")

        return self

    def available_strategies(self):
        return {
            "FullyShardedDataParallel",
            "DistributedDataParallel",
            "DataParallel",
            "NoDistributed",
        }

    def get_sampler(
        self, dataset: Union[SetQuenceDataset, Dataset], env: Environment = EMPTY_ENVIRONMENT, shuffle=False
    ):
        DISTR_STRATEGIES = {
            DDP: DistributedSampler(dataset, num_replicas=env.size, rank=env.rank, shuffle=shuffle),
            DataParallel: SequentialSampler(dataset),
            NoDistributed: SequentialSampler(dataset),
        }

        return DISTR_STRATEGIES[self.distribution_strategy]

    def distribute(self, data_in, data_out):
        raise NotImplementedError()

    def undo_distribute(self, data_in: torch.Tensor, data_out: torch.Tensor, schedule: List[Dict]) -> torch.Tensor:
        raise NotImplementedError()

    def allocate_communicators(self):
        _rank = self.env.rank
        for i in range(self.env.size):
            if i != _rank and max(i, _rank) == i:
                _send_tensor = torch.ones((10, 10), device=self.env.device)
                dist.send(_send_tensor, dst=i)
            elif i != _rank and max(i, _rank) == _rank:
                _recv_tensor = torch.ones((10, 10), device=self.env.device)
                dist.recv(_recv_tensor, src=i)
