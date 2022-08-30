import collections
import copy
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import Dataset

TORCH = ".torch"
HDF5 = ".h5"
COMM_RECV = "recv"
COMM_SEND = "send"
BACKEND_COMM = "mpi"
MAX_WORKERS = 100


class SetQuenceDatasetEpigenome(Dataset):
    def __init__(self, config, env):
        """
        Args:
            root_dir (string): route to the torch/hdf5 file containing the dataset
        """
        self.config = config
        self.filetype = Path(self.config.route).suffix
        self.torch_dir = self.config.route
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

        self._build_dataset()

    def _build_dataset(self):
        try:
            self.input = self.tensors["input_ids"]
            self.epigenome_ids = self.tensors["epigenome_ids"]
            self.epigenome_level = self.tensors["epigenome_level"]
            self.output = self.tensors["labels"]

            if self.head > 0:
                self.input = self.input[0 : self.head]
                self.epigenome_ids = self.epigenome_ids[0 : self.head]
                self.epigenome_level = self.epigenome_level[0 : self.head]
                self.output = self.output[0 : self.head]

        except KeyError as e:
            raise KeyError(e)

    def calculate_class_weight(self):
        class_sample_count = np.unique(self.output[:], return_counts=True)[1]
        weight = 1.0 / class_sample_count
        weight = torch.from_numpy(weight).float()
        return weight

    def __len__(self):
        return len(self.input)

    def _aggregate_ranks_sequence_lengths_schedule(self, seq_count):
        if self.dataset_comm_group is None:
            self.dataset_comm_group = dist.new_group(backend=BACKEND_COMM)

        _seq_counts = [torch.zeros(seq_count.shape, dtype=torch.long) for _ in range(self.env.size)]
        if self.env.rank == 0:
            dist.gather(seq_count, _seq_counts, group=self.dataset_comm_group)

            seq_counts = torch.cat(_seq_counts)
            schedule, partition, indices = self.create_sequence_schedule(seq_counts)
            dist.broadcast_object_list([schedule, partition, indices], src=0, group=self.dataset_comm_group)
        else:
            dist.gather(seq_count, group=self.dataset_comm_group)

            ret_broadcast = [None, None, None]
            dist.broadcast_object_list(ret_broadcast, src=0, group=self.dataset_comm_group)
            schedule, partition, indices = ret_broadcast

        dist.barrier(group=self.dataset_comm_group)
        return schedule

    def _permute_sequences(self, input_sequences, seq_count):
        permuted_indices = torch.randperm(seq_count.item())
        indices = torch.arange(self.max_seq)
        indices[0 : seq_count.item()] = permuted_indices
        _out = input_sequences.view(-1, self.max_seq, self.seq_len)[:, indices]
        seq_count = self.find_last(_out)

        return _out, seq_count

    def __getitem__(self, idx):
        _input = torch.tensor(self.input[idx], dtype=torch.long)
        _epigenome_input = torch.tensor(self.epigenome_ids[idx], dtype=torch.long)
        _epigenome_level = torch.tensor(self.epigenome_level[idx], dtype=torch.float)
        _output = torch.tensor(self.output[idx], dtype=torch.long)
        seq_count_input = self.find_last(_input).view(-1)
        seq_count_epigenome = self.find_last(_epigenome_input).view(-1)

        if self.load_balance:
            schedule_input = self._aggregate_ranks_sequence_lengths_schedule(seq_count_input)
            schedule_epigenome = self._aggregate_ranks_sequence_lengths_schedule(seq_count_epigenome)
        else:
            schedule_input, schedule_epigenome = {}, {}

        # Permute input sequences, not epigenome! Would need to permute levels, too
        _input, seq_count_input = self._permute_sequences(_input, seq_count_input)
        _epigenome_input = _epigenome_input.view(-1, self.max_seq, self.seq_len)

        if self.load_balance:
            dist.barrier(group=self.dataset_comm_group)
            _input = self._distribute_ddp(_input, torch.zeros_like(_input), seq_count_input, schedule=schedule_input)
            _epigenome_input = self._distribute_ddp(
                _epigenome_input, torch.zeros_like(_epigenome_input), seq_count_epigenome, schedule=schedule_epigenome
            )

        seq_count_input = self.find_last(_input)
        seq_count_epigenome = self.find_last(_epigenome_input)
        _input = _input[:, 0:seq_count_input].view(-1, seq_count_input, self.seq_len)
        _epigenome_input = _epigenome_input[:, 0:seq_count_epigenome].view(-1, seq_count_epigenome, self.seq_len)
        _attention_input = _input.bool().long()
        _attention_epigenome = _epigenome_input.bool().long()

        return {
            "genome_input": _input,
            "genome_attention": _attention_input,
            "epigenome_input": _epigenome_input,
            "epigenome_attention": _attention_epigenome,
            "epigenome_level": _epigenome_level,  # is not rescheduled
            "output": _output,
            "schedule_genome": schedule_input,
            "schedule_epigenome": schedule_epigenome,
        }

    def find_last(self, data):
        return torch.where(data > 0, 1, 0).view(-1, self.max_seq, self.seq_len).sum(dim=2).bool().sum()

    def _distribute_ddp(
        self, data_in: torch.tensor, data_out: torch.tensor, n_curr_seq, schedule=None
    ) -> torch.tensor:
        if len(schedule) == 0:
            data_out = data_in
            return data_out

        n_recv_seq = 0
        n_send_seq = 0

        for _, ops in schedule.items():
            for comm_op in ops:
                _host = comm_op["host"] if isinstance(comm_op["host"], int) else comm_op["host"].item()
                _partner = comm_op["partner"] if isinstance(comm_op["partner"], int) else comm_op["partner"].item()
                _length = comm_op["length"] if isinstance(comm_op["length"], int) else comm_op["length"].item()

                if COMM_RECV in comm_op["event"] and self.env.rank == _host:
                    _r_poolr_in = torch.zeros((1, _length, data_out.shape[-1]), dtype=data_in.dtype)
                    dist.recv(_r_poolr_in, src=_partner, group=self.dataset_comm_group)
                    data_out[:, (n_curr_seq + n_recv_seq) : (n_curr_seq + n_recv_seq + _length)] = _r_poolr_in

                    n_recv_seq += _length

                elif COMM_SEND in comm_op["event"] and self.env.rank == _host:
                    _s_bert_out = data_in[:, (n_curr_seq - n_send_seq - _length) : (n_curr_seq - n_send_seq)]
                    dist.send(_s_bert_out, dst=_partner, group=self.dataset_comm_group)
                    n_send_seq += _length

        if n_send_seq != 0:
            data_out[:, 0 : (n_curr_seq - n_send_seq)] = data_in[:, 0 : (n_curr_seq - n_send_seq)]
        elif n_recv_seq != 0:
            data_out[:, 0:(n_curr_seq)] = data_in[:, 0:(n_curr_seq)]
        elif n_send_seq * n_recv_seq != 0:
            raise RuntimeError("Same CPU send and receive is not supported!")
        elif n_send_seq == 0 and n_recv_seq == 0:
            data_out = data_in

        return data_out

    def create_sequence_schedule(self, seq_counts):
        batch_count = copy.deepcopy(seq_counts.tolist())
        partition, indices = self.split(batch_count)

        schedule = self.build_schedule(partition)
        return schedule, partition, indices

    def build_schedule(self, seq_buffer):
        tag_comm = 0
        schedule = collections.defaultdict(list)

        seq_buffer = torch.tensor(seq_buffer).flatten()
        GPU_buffer = seq_buffer.tolist()
        num, div = np.array(seq_buffer).sum(), self.env.size
        divide_seqs = torch.tensor([num // div + (1 if x < num % div else 0) for x in range(div)])

        send = torch.where((seq_buffer > divide_seqs.max()) & (seq_buffer != divide_seqs.max()), 1, 0)

        GPU_divide = divide_seqs.tolist()

        for i in range(self.env.size):
            for j in range(self.env.size):

                s_op = send[i]

                if s_op and len(GPU_divide) > 0:

                    n_send = (GPU_divide[0] - GPU_buffer[j]) * s_op
                    rest = GPU_buffer[i] - divide_seqs.min()

                    if n_send > 0 and rest > 0:
                        n_send = min(n_send, rest)
                        GPU_divide.pop(0)
                        GPU_buffer[i] -= n_send
                        GPU_buffer[j] += n_send

                        schedule[tag_comm].append({"host": i, "event": "send", "length": n_send.item(), "partner": j})
                        schedule[tag_comm].append({"host": j, "event": "recv", "length": n_send.item(), "partner": i})

                    tag_comm += 1

        return schedule
