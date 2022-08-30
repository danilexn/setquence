import collections
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import Dataset

from setquence.utils.dasud import dasud

TORCH = ".torch"
HDF5 = ".h5"
COMM_RECV = "recv"
COMM_SEND = "send"
BACKEND_COMM = "mpi"
MIN_TO_DROPOUT = 20
DASUD_ITER = 1
NEIGHBORS = 64


class SetQuenceDataset(Dataset):
    def __init__(self, config, env):
        """
        Args:
            root_dir (string): route to the torch/hdf5 file containing the dataset
        """
        self.config = config
        self.filetype = Path(self.config.route).suffix
        self.head = config.head
        self.dataset_comm_group = None
        self.env = env

        if self.filetype == TORCH:
            self.tensors = torch.load(self.config.route)
        elif self.filetype == HDF5:
            self.tensors = h5py.File(self.config.route, "r")
        else:
            raise ValueError(f"Cannot read '{self.filetype}' files; only .torch or .h5 files are supported!")

        try:
            self.max_seq = self.config.max_seq
        except AttributeError:
            raise AttributeError("Could not find the property 'max_seq' in the configuration file")

        try:
            self.seq_len = self.config.seq_len
        except AttributeError:
            raise AttributeError("Could not find the property 'seq_len' in the configuration file")

        try:
            self.load_balance = self.config.load_balance
        except AttributeError:
            self.load_balance = True
            warnings.warn("Could not find a default value for 'load_balance'; setting to True")

        try:
            self.crop_preprocess = self.config.crop_preprocess
        except AttributeError:
            self.crop_preprocess = True
            warnings.warn("Could not find a default value for 'crop_preprocess'; setting to True")

        try:
            self.p_dropout = self.config.p_dropout
        except AttributeError:
            self.p_dropout = 0
            warnings.warn("Could not find a default value for 'p_dropout'; setting to '0'")

        try:
            self.shuffle = self.config.shuffle
        except AttributeError:
            self.shuffle = False
            warnings.warn("Could not find a default value for 'shuffle'; setting to False")

        self.build_dataset()

    def find_last(self, data):
        return data.bool().view(-1, self.max_seq, self.seq_len).sum(dim=2).bool().sum()

    def calculate_class_weight(self):
        _output = self.output
        if self.binarize != -1:
            _output = torch.where(_output == self.binarize, 1, 0)
        class_sample_count = np.unique(_output, return_counts=True)[1]
        weight = 1.0 / class_sample_count
        weight = torch.from_numpy(weight).float()
        return weight

    def build_dataset(self):
        try:
            self.input = self.tensors["input_ids"]
            self.output = self.tensors["labels"][:]

            if self.head > 0:
                self.input = self.input[0 : self.head]
                self.output = self.output[0 : self.head]

        except KeyError as e:
            raise KeyError(e)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        _input = torch.tensor(self.input[idx], dtype=torch.long).view(-1, self.max_seq, self.seq_len)
        _output = torch.tensor(self.output[idx], dtype=torch.long)

        seq_count = self.find_last(_input).view(-1)

        if seq_count > MIN_TO_DROPOUT and self.p_dropout != 0:
            _input_dropout = torch.arange(0, int(seq_count))[torch.randperm(int(seq_count))][
                0 : int(seq_count * (1 - self.p_dropout))
            ].view(-1)
            _input[:, 0 : len(_input_dropout)] = _input[:, _input_dropout, :]
            _input[:, len(_input_dropout) :, :] = 0
            seq_count = self.find_last(_input).view(-1)

        seq_count = seq_count.int()

        if self.load_balance and self.env.size > 1:
            if self.dataset_comm_group is None:
                self.dataset_comm_group = dist.new_group(backend=BACKEND_COMM)

            load = [torch.zeros(seq_count.shape, dtype=torch.int) for _ in range(self.env.size)]
            dist.all_gather(load, seq_count, group=self.dataset_comm_group)
            load = torch.cat(load).numpy()

            schedule_messages = {}
            for i_s in range(self.env.size):
                schedule_messages[i_s] = 0

            for _ in range(DASUD_ITER):
                neighbors = np.arange(
                    max(0, self.env.rank - NEIGHBORS - 1), min(self.env.size, self.env.rank + NEIGHBORS + 1)
                )
                dasud(self.env.rank, neighbors, load, schedule_messages)

            schedule_array = np.zeros(self.env.size, dtype="i")
            for i in schedule_messages.keys():
                schedule_array[i] = schedule_messages[i]

            gather_array = [torch.zeros((1, self.env.size), dtype=torch.int) for _ in range(self.env.size)]
            dist.all_gather(gather_array, torch.tensor(schedule_array).view(1, -1), group=self.dataset_comm_group)
            gather_array = torch.cat(gather_array, dim=0)
            schedule = self._build_schedule_from_gather_array(gather_array)
        else:
            indices = list(range(self.env.size))
            schedule = {}

        indices = torch.arange(self.max_seq)
        if self.shuffle:
            permuted_indices = torch.randperm(seq_count.item())
            indices[0 : seq_count.item()] = permuted_indices

        _input = _input.view(-1, self.max_seq, self.seq_len)[:, indices]
        seq_count = self.find_last(_input)

        if self.load_balance:
            _input = self._distribute_ddp(_input, torch.zeros_like(_input), seq_count, schedule=schedule)

        seq_count = self.find_last(_input)
        if self.crop_preprocess:
            _input = _input[:, 0:seq_count].view(-1, seq_count, self.seq_len)
        else:
            _input = _input.view(-1, self.max_seq, self.seq_len)

        _attention = _input.bool().long()

        return {
            "input": _input,
            "attention": _attention,
            "output": _output,
            "schedule": schedule,
        }

    def _build_schedule_from_gather_array(self, gather_array: np.array):
        tag_comm = 0
        schedule = collections.defaultdict(list)

        for i in range(self.env.size):
            _send_n = gather_array[self.env.rank][i]
            _recv_n = gather_array[i][self.env.rank]
            if i == self.env.rank:
                continue
            if _send_n != 0:
                schedule[tag_comm].append({"host": self.env.rank, "event": "send", "length": _send_n, "partner": i})
                tag_comm += 1
            if _recv_n != 0:
                schedule[tag_comm].append({"host": self.env.rank, "event": "recv", "length": _recv_n, "partner": i})
                tag_comm += 1

        return schedule

    def _distribute_ddp(
        self, data_in: torch.tensor, data_out: torch.tensor, n_curr_seq, schedule=None
    ) -> torch.tensor:

        if schedule is None or len(schedule) == 0:
            data_out = data_in
            return data_out

        n_recv_seq = 0
        n_send_seq = 0

        handles_send = [None] * self.env.size
        handles_recv = [None] * self.env.size

        recv_tensors = []

        for _, ops in schedule.items():
            for comm_op in ops:
                _host = comm_op["host"] if isinstance(comm_op["host"], int) else comm_op["host"].item()
                _partner = comm_op["partner"] if isinstance(comm_op["partner"], int) else comm_op["partner"].item()
                _length = comm_op["length"] if isinstance(comm_op["length"], int) else comm_op["length"].item()

                if COMM_RECV in comm_op["event"] and self.env.rank == _host:
                    recv_tensors += [torch.zeros((1, _length, data_out.shape[-1]), dtype=data_in.dtype)]
                    handles_recv[_partner] = dist.irecv(recv_tensors[-1], src=_partner, group=self.dataset_comm_group)
                    n_recv_seq += _length

                elif COMM_SEND in comm_op["event"] and self.env.rank == _host:
                    _s_bert_out = data_in[:, (n_curr_seq - n_send_seq - _length) : (n_curr_seq - n_send_seq)]
                    handles_send[_partner] = dist.isend(_s_bert_out, dst=_partner, group=self.dataset_comm_group)
                    n_send_seq += _length

        finished = np.zeros(self.env.size)
        finished[self.env.rank] = 1

        for i in range(self.env.size):
            if i == self.env.rank:
                continue
            if handles_send[i] is not None:
                handles_send[i].wait()
            if handles_recv[i] is not None:
                handles_recv[i].wait()

        data_out[:, 0 : (n_curr_seq - n_send_seq)] = data_in[:, 0 : (n_curr_seq - n_send_seq)]

        if n_recv_seq != 0:
            recv_tensors = torch.cat(recv_tensors, dim=1)
            data_out[:, (n_curr_seq - n_send_seq) : (n_curr_seq - n_send_seq + n_recv_seq), :] = recv_tensors

        return data_out
