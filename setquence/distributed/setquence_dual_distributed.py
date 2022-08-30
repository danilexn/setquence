import warnings
from typing import Dict

import torch
from torch import distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from setquence.base.config import EMPTY_ENVIRONMENT, Environment
from setquence.distributed.distribution import Distributed, NoDistributed

COMM_RECV = "recv"
COMM_SEND = "send"


def find_last(data):
    return data.bool().sum(dim=2).bool().sum().item()


class SetQuenceDualDistributed(Distributed):
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

    def undo_distribute(self, data_in: torch.tensor, data_out: torch.tensor, schedule: Dict) -> torch.tensor:
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

    def _undo_distribute_ddp(self, data_in: torch.tensor, data_out: torch.tensor, schedule: Dict) -> torch.tensor:
        if len(schedule) == 0:
            data_out = data_in
            return data_out

        # In DDP, each GPU manages one patient!
        data_in = data_in[0]
        n_curr_seq = len(data_in)
        n_recv_seq = 0
        n_send_seq = 0

        if not isinstance(schedule, Dict):
            warnings.warn("A schedule should be specified. Will not be redistributed.")
            return data_in

        for _tag, ops in dict(reversed(list(schedule.items()))).items():
            for comm_op in ops:
                _host = comm_op["host"] if isinstance(comm_op["host"], int) else comm_op["host"].item()
                _partner = comm_op["partner"] if isinstance(comm_op["partner"], int) else comm_op["partner"].item()
                _length = comm_op["length"] if isinstance(comm_op["length"], int) else comm_op["length"].item()
                if COMM_RECV in comm_op["event"] and self.env.rank == _host:
                    _s_bert_out = data_in[-(1 + n_send_seq + _length) : -(1 + n_send_seq)]
                    dist.send(_s_bert_out, dst=_partner)
                    n_send_seq += _length

                elif COMM_SEND in comm_op["event"] and self.env.rank == _host:
                    _r_poolr_in = torch.zeros((_length, data_out.shape[-1]), device=data_out.device)
                    dist.recv(_r_poolr_in, src=_partner)
                    data_out[(n_curr_seq + n_recv_seq) : (n_curr_seq + n_recv_seq + _length)] = _r_poolr_in

                    n_recv_seq += _length

        if n_send_seq != 0:
            data_out = data_in[0:-(n_send_seq)]
        elif n_recv_seq != 0:
            data_out[0:n_curr_seq] = data_in[0:n_curr_seq]
            data_out = data_out[0 : (n_curr_seq + n_recv_seq)]
        elif n_send_seq * n_recv_seq != 0:
            raise RuntimeError("Same GPU send and receive is not supported!")
        elif n_send_seq == 0 and n_recv_seq == 0:
            data_out = data_in

        return data_out

    def _distribute_ddp(self, data_in: torch.tensor, data_out: torch.tensor, schedule: Dict = None) -> torch.tensor:
        if len(schedule) == 0:
            data_out = data_in
            return data_out

        data_in = data_in[0]
        n_curr_seq = find_last(data_in)
        n_recv_seq = 0
        n_send_seq = 0

        if not isinstance(schedule, Dict):
            warnings.warn("A schedule should be specified. Will not be redistributed.")
            return data_in

        for _tag, ops in schedule.items():
            for comm_op in ops:
                _host = comm_op["host"] if isinstance(comm_op["host"], int) else comm_op["host"].item()
                _partner = comm_op["partner"] if isinstance(comm_op["partner"], int) else comm_op["partner"].item()
                _length = comm_op["length"] if isinstance(comm_op["length"], int) else comm_op["length"].item()
                if COMM_RECV in comm_op["event"] and self.env.rank == _host:
                    _r_poolr_in = torch.zeros(
                        (1, _length, data_out.shape[-1]), dtype=data_in.dtype, device=data_out.device
                    )
                    dist.recv(_r_poolr_in, src=_partner)
                    data_out[:, (n_curr_seq + n_recv_seq) : (n_curr_seq + n_recv_seq + _length)] = _r_poolr_in

                    n_recv_seq += _length

                elif COMM_SEND in comm_op["event"] and self.env.rank == _host:
                    _s_bert_out = data_in[:, (n_curr_seq - n_send_seq - _length) : (n_curr_seq - n_send_seq)]
                    dist.send(_s_bert_out, dst=_partner)
                    n_send_seq += _length

        if n_send_seq != 0:
            data_out[:, 0 : (n_curr_seq - n_send_seq)] = data_in[:, 0 : (n_curr_seq - n_send_seq)]
        elif n_recv_seq != 0:
            data_out[:, 0:(n_curr_seq)] = data_in[:, 0:(n_curr_seq)]
        elif n_send_seq * n_recv_seq != 0:
            raise RuntimeError("Same GPU send and receive is not supported!")
        elif n_send_seq == 0 and n_recv_seq == 0:
            data_out = data_in

        return data_out

    def _distribute_dp(
        self, data_in: torch.tensor, data_out: torch.tensor, size: int = 1, nodes: int = 1,
    ) -> torch.tensor:
        raise NotImplementedError("Schedule needs to be precalculated!")

    def _undo_distribute_dp(self, data_in: torch.tensor, data_out: torch.tensor, schedule: Dict) -> torch.tensor:
        raise NotImplementedError()

    def _find_last(self, data):
        raise NotImplementedError()
