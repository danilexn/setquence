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
BINS = 3
MAX_WORKERS = 100


class SetQuenceDatasetEpigenome450k(Dataset):
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
        self.bins = np.linspace(0, 1, BINS)

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
            self.datatype = self.config.datatype
        except AttributeError:
            self.datatype = "train"
            warnings.warn("Could not find a default value for 'datatype'; setting to 'Train'")

        try:
            self.shuffle = self.config.shuffle
        except AttributeError:
            self.shuffle = True
            warnings.warn("Could not find a default value for 'shuffle'; setting to True")

        self.build_dataset()

    def build_dataset(self):
        try:
            self.epigenome_ids = torch.tensor(self.tensors["epigenome_ids"][:], dtype=torch.long)
            self.attention_epigenome = self.epigenome_ids.bool().long()
            self.epigenome_level = self.tensors["epigenome_level"]
            self.output = self.tensors["labels"]

            if self.head > 0:
                self.epigenome_ids = self.epigenome_ids[0 : self.head]
                self.epigenome_level = self.epigenome_level[0 : self.head]
                self.output = self.output[0 : self.head]

            all_n_samples = len(self.epigenome_level)
            self.indices = np.random.RandomState(seed=2022).permutation(all_n_samples)
            self.indices_epigenome = np.random.RandomState(seed=2022).permutation(len(self.epigenome_ids))
            if self.datatype == "train":
                self.indices = self.indices[0 : int(all_n_samples * 0.7)]
                self.indices_epigenome = self.indices_epigenome[0 : int(len(self.epigenome_ids) * 0.5)]
            else:
                self.indices = self.indices[int(all_n_samples * 0.7) :]
                self.indices_epigenome = self.indices_epigenome[int(len(self.epigenome_ids) * 0.5) :]

            self.epigenome_ids = self.epigenome_ids[self.indices_epigenome, :]

        except KeyError as e:
            raise KeyError(e)

    def __len__(self):
        return len(self.indices)

    def _synchronize_ranks_permutation(self, n_sequences):
        if self.dataset_comm_group is None:
            self.dataset_comm_group = dist.new_group(backend=BACKEND_COMM)

        if self.env.rank == 0:
            permutation_list = torch.randperm(n_sequences).type(torch.long)
            dist.broadcast(permutation_list, src=0, group=self.dataset_comm_group)
        else:
            permutation_list = torch.zeros(n_sequences, dtype=torch.long)
            dist.broadcast(permutation_list, src=0, group=self.dataset_comm_group)

        dist.barrier(group=self.dataset_comm_group)
        return permutation_list

    def __getitem__(self, idx):
        _epigenome_input = self.epigenome_ids
        _epigenome_level = self.epigenome_level[self.indices[idx]][:][self.indices_epigenome]
        _epigenome_level = np.where(_epigenome_level != np.nan, _epigenome_level, -100)
        _epigenome_level = np.where(_epigenome_level != 0, _epigenome_level, -100)
        _epigenome_level = np.digitize(_epigenome_level, self.bins)
        _epigenome_level = torch.tensor(_epigenome_level, dtype=torch.int16)
        _output = torch.tensor(self.output[self.indices[idx]], dtype=torch.long)

        # Detect which sequences are associated to NaN
        all_indices = torch.arange(len(_epigenome_input))

        # Random permutation of sequences that are not NaN
        if self.shuffle:
            _shuffled = self._synchronize_ranks_permutation(len(all_indices))
            all_indices = all_indices[_shuffled]

        # Permute epigenome sequences and levels
        _epigenome_input = _epigenome_input[all_indices].chunk(self.env.size)[self.env.rank]
        _attention_mask = self.attention_epigenome[all_indices].chunk(self.env.size)[self.env.rank]
        _epigenome_level = _epigenome_level[all_indices]

        return {
            "epigenome_input": _epigenome_input.view(-1, _epigenome_input.shape[0], _epigenome_input.shape[1]),
            "epigenome_attention": _attention_mask.view(-1, _epigenome_input.shape[0], _epigenome_input.shape[1]),
            "epigenome_level": _epigenome_level,
            "output": _output,
            "schedule_epigenome": {},
        }
