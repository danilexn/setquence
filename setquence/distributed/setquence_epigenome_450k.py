from typing import Dict

import torch
from torch import distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from setquence.base.config import EMPTY_ENVIRONMENT, Environment
from setquence.distributed.distribution import Distributed, NoDistributed

COMM_RECV = "recv"
COMM_SEND = "send"


class AllGather(torch.autograd.Function):
    """
    all_gather with gradient back-propagation

    Credits to: https://github.com/vlkit/vlkit/blob/master/vlkit/ops/distributed.py
    """

    @staticmethod
    def forward(ctx, tensor_list, tensor):
        dist.all_gather(tensor_list, tensor)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_list):
        grad_list = list(grad_list)
        rank = dist.get_rank()

        dist_ops = [dist.reduce(grad_list[i], i, async_op=True) for i in range(dist.get_world_size())]

        for op in dist_ops:
            op.wait()

        return None, grad_list[rank]


all_gather = AllGather.apply


def find_last(data):
    return torch.where(data > 0, 1, 0).sum(dim=2).bool().sum().item()


class SetQuenceEpigenome450kDistributed(Distributed):
    def __init__(self, env: Environment = EMPTY_ENVIRONMENT, scheduled=False, *args, **kwargs):
        self.env = env
        self.distribution_strategy = NoDistributed
        self.scheduled = scheduled

    def distribute(self, dataset_in, data_out):
        data_in = dataset_in["input"]
        if self.scheduled:
            return (data_in, dataset_in["schedule"])

        DISTR_STRATEGIES = {
            DDP: self._distribute_ddp,
            DataParallel: self._distribute_dp,
        }

        try:
            _dist_fn = DISTR_STRATEGIES[self.distribution_strategy]
        except KeyError:
            raise KeyError(f"Distribution strategy '{self.distribution_strategy}' not supported for redistribution")

        return (_dist_fn(data_in, data_out, dataset_in["schedule"]), dataset_in["schedule"])

    def undo_distribute(self, data_in, data_out, schedule: Dict) -> torch.tensor:
        if not isinstance(data_in, torch.Tensor):
            raise TypeError("data_in must be torch.tensor type")

        if not isinstance(schedule, Dict):
            raise TypeError("A schedule must be specified!")

        UNDO_DISTR_STRATEGIES = {
            DDP: self._undo_distribute_ddp,
            DataParallel: self._undo_distribute_dp,
        }

        try:
            _dist_fn = UNDO_DISTR_STRATEGIES[self.distribution_strategy]
        except KeyError:
            raise KeyError(f"Distribution strategy {self.distribution_strategy} not supported for redistribution")

        return _dist_fn(data_in, data_out, schedule)

    def _undo_distribute_ddp(self, data_in, data_out, schedule: Dict) -> torch.tensor:
        all_gather(data_out, data_in)
        return torch.cat(data_out, dim=1)

    def _distribute_ddp(self, data_in: torch.tensor, data_out: torch.tensor, schedule: Dict = None) -> torch.tensor:
        raise NotImplementedError()

    def _distribute_dp(
        self, data_in: torch.tensor, data_out: torch.tensor, size: int = 1, nodes: int = 1,
    ) -> torch.tensor:
        raise NotImplementedError("Schedule needs to be precalculated!")

    def _undo_distribute_dp(self, data_in: torch.tensor, data_out: torch.tensor, schedule: Dict) -> torch.tensor:
        raise NotImplementedError()
