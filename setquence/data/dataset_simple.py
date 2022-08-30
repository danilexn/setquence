import collections
import copy

import numpy as np
import torch
from torch import distributed as dist

from setquence.data.dataset import SetQuenceDataset

TORCH = ".torch"
HDF5 = ".h5"
COMM_RECV = "recv"
COMM_SEND = "send"
BACKEND_COMM = "mpi"
MIN_TO_DROPOUT = 20


class SetQuenceSimpleDataset(SetQuenceDataset):
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

        if self.load_balance:
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
            dist.barrier(group=self.dataset_comm_group)
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

    def split(self, a, iterations=32):
        values = 0
        indice = 0
        cost = 10e20
        c_indice = list(range(len(a)))

        for _ in range(iterations):
            k, m = divmod(len(a), self.env.size)
            _a = np.array(a)[c_indice]
            c_values = [_a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(self.env.size)]
            c_cost = np.linalg.norm(np.sum(c_values, axis=1))

            if cost > c_cost:
                cost = copy.deepcopy(c_cost)
                values = copy.deepcopy(c_values)
                indice = copy.deepcopy(c_indice)

            np.random.shuffle(c_indice)

        return values, indice

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
