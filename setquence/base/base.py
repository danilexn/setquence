import copy
import os
import warnings
from pathlib import Path
from typing import Callable, Dict, Union

import torch
import torch.onnx
import wandb
from torch import distributed as pytorch_dist
from torch import jit as jit
from torch import nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import BertModel

from setquence.base.optimizer import SAM
from setquence.data import SetQuenceDataset
from setquence.distributed.distribution import Distributed, NoDistributed
from setquence.utils import dict_to_namespace, ns_to_dict

from .config import EMPTY_ENVIRONMENT, Config, Environment


def empty_fn(*args, **kwargs):
    pass


class BaseModule(nn.Module):
    def __init__(self, config: Config, env=None, *args, **kwargs):
        super().__init__()
        self.config = config
        self.env = env

    def zeros_like_input(self) -> torch.Tensor:
        raise NotImplementedError()

    def zeros_like_output(self) -> torch.Tensor:
        raise NotImplementedError()


class BaseJITModule(jit.ScriptModule):
    def __init__(self, config: Config, env=None, *args, **kwargs):
        super().__init__()
        self.config = config
        self.env = env

    def zeros_like_input(self) -> torch.Tensor:
        raise NotImplementedError()

    def zeros_like_output(self) -> torch.Tensor:
        raise NotImplementedError()


class Base:
    model_name = None
    model_basetype = "Base"

    def __init__(
        self, config: Config, model: BaseModule, env: Environment = EMPTY_ENVIRONMENT,
    ):
        self.config = config
        self._model = model
        self.original_parameters = None
        self._setup_environment(env)
        self._setup_attributes()
        self._setup_model(config, env)

    def _setup_model(self, config, env):
        self.model = self._model(config, env)
        if self.config.encoder.bert and self.config.encoder.bert_pretrained:
            self.load_bert_pretrained(self.config.encoder.bert_route)

    def to_device(self):
        self.model.to(self.environment.device)

    def _setup_attributes(self):
        self.epoch = 0
        self.acc_epochs = 0
        self.n_step = 0

    def _setup_environment(self, env):
        self.environment = env
        self.distribution_strategy = NoDistributed
        self.distribution = Distributed(self.environment)
        if isinstance(self.environment, Environment):
            self.device = self.environment.device
        else:
            raise NotImplementedError("An Environment class must be specified")

    def distribute(self, dist):
        try:
            self.distribution = dist
            self.distribution_strategy = dist.distribution_strategy
            # Send the model to the GPUs once the device has been configured!
            self.to_device()
        except KeyError:
            raise ValueError("The distribution strategy is not valid or not implemented")

    def _dist_alloc(self):
        self.model = (
            self.distribution_strategy(self.model, device_ids=self.environment.local_device_ids)
            if not isinstance(self.model, self.distribution_strategy) and self.distribution_strategy != NoDistributed
            else self.model
        )

    def _dist_dealloc(self):
        self.model = self.model.module if isinstance(self.model, self.distribution_strategy) else self.model

    def __call__(self, dataset, *args, **kwargs):
        raise NotImplementedError("Needs to implement the model call")

    def dist_alloc(self):
        STRATEGY_2_INIT_RELEASE = {
            NoDistributed: {"init": empty_fn, "release": empty_fn},
            DistributedDataParallel: {"init": self._dist_alloc, "release": self._dist_dealloc},
            DataParallel: {"init": self._dist_alloc, "release": self._dist_dealloc},
        }

        STRATEGY_2_INIT_RELEASE[self.distribution_strategy]["init"]()

    def parameters(self, optimizer=False):
        self.dist_alloc()

        return self.model.parameters()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def fit(
        self,
        config: Union[Dict, Config],
        train_dataloader: Union[SetQuenceDataset, Dataset],
        optimizer: Optimizer,
        callback_fn: Callable = empty_fn,
        callback_args: Dict = {"None": None},
        callback_step_fn: Callable = empty_fn,
        callback_step_args: Dict = {"None": None},
        freq_step_callback: int = 1,
    ):
        freq_log_callback = int(os.environ.get("SETQUENCE_LOG_WANDB_FREQ", 25))
        config = dict_to_namespace(config) if isinstance(config, Dict) else config
        self.epoch = 0
        rank = self.environment.rank
        n_epochs = config.epochs

        # Memory preallocation across all ranks
        # Create a tensor with preallocated data
        if not isinstance(optimizer, SAM):
            self._preallocate(optimizer)

        if rank == 0:
            epoch_iter = tqdm(range(n_epochs), position=0, leave=True)
        else:
            epoch_iter = range(n_epochs)

        for _ in epoch_iter:
            # Just in case the callback function calls eval
            self.train()

            # Needs to be set for PyTorch == 1.9.0; why? idk
            if isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(self.epoch)

            if rank == 0:
                steps_iter = tqdm(train_dataloader, position=0, leave=False)
            else:
                steps_iter = train_dataloader

            self.n_step = 0
            for batch in steps_iter:
                for k, _ in batch.items():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.environment.device)

                # Will gradients be updated? Otherwise, accumulation!
                sync = (self.n_step + 1) % config.grad_acc_steps == 0
                output = self(batch, sync=sync)

                if "recon" in batch.keys():
                    loss = self.loss(output, batch["output"], batch["recon"]) * self.environment.size
                else:
                    # To get same results between DDP and DP
                    loss = self.loss(output, batch["output"]) * self.environment.size

                if isinstance(optimizer, SAM):
                    self.backward(loss, sync=False)
                    optimizer.first_step(zero_grad=True)

                    if "recon" in batch.keys():
                        loss = self.loss(self(batch, sync=sync), batch["output"], batch["recon"]) * config.loss_factor
                    else:
                        # To get same results between DDP and DP
                        loss = self.loss(self(batch, sync=sync), batch["output"]) * config.loss_factor

                    self.backward(loss)
                    optimizer.second_step(zero_grad=True)
                else:
                    self.backward(loss, sync=sync)

                    # Gradient clipping
                    if config.max_grad_norm != -1:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)

                    # Parameter update if gradient accumulation finished
                    if sync:
                        optimizer.step()
                        optimizer.zero_grad()

                self.n_step += 1

                if freq_step_callback == 0:
                    callback_step_fn(**callback_step_args)

                if rank == 0 and "SETQUENCE_LOG_WANDB" in os.environ and self.n_step % freq_log_callback == 0:
                    wandb.log({"loss": loss.item()})

                if self.n_step > config.steps and config.steps > -1:
                    break

            callback_fn(self, **callback_args)

            self.epoch += 1
            self.acc_epochs += 1

    def backward(self, loss, sync=True):
        if sync:
            loss.backward()
        else:
            with self.model.no_sync():
                loss.backward()

    def _preallocate(self, *args, **kwargs):
        warnings.warn("Base does not support preallocation")

    def _wandb_model_watch(self):
        wandb.watch(self.model)

    def predict(self, pred_dataloader: Union[SetQuenceDataset, Dataset],) -> torch.tensor:
        rank = self.environment.rank
        output = []
        labels = []
        self.eval()

        if rank == 0:
            steps_iter = tqdm(pred_dataloader, position=0, leave=False)
        else:
            steps_iter = pred_dataloader

        with torch.no_grad():
            for batch in steps_iter:
                for k, _ in batch.items():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.environment.device, non_blocking=True)

                _out = self(batch)

                if isinstance(_out, tuple):
                    output += [_out[0]]
                else:
                    output += [_out]

                if "output" in batch.keys():
                    labels += [batch["output"]]

        output_tensor = torch.cat(output)
        destination_tensor = [torch.zeros_like(output_tensor) for _ in range(self.environment.size)]
        pytorch_dist.all_gather(destination_tensor, output_tensor)

        label_tensor = torch.cat(labels)
        label_destination_tensor = [torch.zeros_like(label_tensor) for _ in range(self.environment.size)]
        pytorch_dist.all_gather(label_destination_tensor, label_tensor)
        return destination_tensor, label_destination_tensor

    def __repr__(self) -> str:
        return repr(self.model)

    def loss(self, output: torch.tensor, objective: torch.tensor, *args, **kwargs) -> torch.tensor:
        raise NotImplementedError()

    def _save_prepare(self, f: Union[Path, str]):
        self._dist_dealloc()
        if not Path(f).parents[0].exists():
            Path(f).parents[0].mkdir(parents=True, exist_ok=True)

    def save_model(self, f: Union[Path, str]):
        self._save_prepare(f)
        _save_path = Path(f).with_suffix(".pth")
        model_dict = self.model_dict()
        torch.save(
            model_dict, _save_path,
        )

    def model_dict(self):
        return {
            "model_name": self.model_name,
            "model_basetype": self.model_basetype,
            "model_weights": self.state_dict(),
            "env": ns_to_dict(self.environment.env),
            "config": ns_to_dict(self.config),
        }

    def state_dict(self):
        return self.model.state_dict()

    def save_onnx(self, x, f: Union[Path, str]):
        raise NotImplementedError("ONNX saving is not implemented, yet")

    def load_bert_pretrained(self, bert_dir):
        bert_model = BertModel.from_pretrained(bert_dir)
        num_hidden_layers = self.model.Encoder.config.num_hidden_layers

        for i in range(num_hidden_layers):
            self.model.Encoder.encoder.layer[i].load_state_dict(bert_model.encoder.layer[i].state_dict())

        self.model.Encoder.embeddings.load_state_dict(bert_model.embeddings.state_dict())
        self.model.Encoder.pooler.load_state_dict(bert_model.pooler.state_dict())

    def from_pretrained(self, pretrained_model, skip: bool = True):
        r"""Copies parameters and buffers from :attr:`pretrained_model` into
        this module and its descendants. If :attr:`skip` is ``True``, then
        the keys of :attr:`state_dict` do not need to exactly match the keys
        returned by this module's :meth:`~setquence.base.Base.state_dict` function.

        Args:
            state_dict (Union[Base, BaseDual]): a pre-trained module
            skip (bool, optional): whether to allow that the keys and value sizes
                in :attr:`state_dict`  do notmatch this module's function
                :meth:`~setquence.base.Base.state_dict`. Default: ``True``
        """
        if skip:
            pretrained_state_dict = self.match_state_dict(pretrained_model.state_dict(), self.state_dict())
        else:
            pretrained_state_dict = pretrained_model.state_dict()

        self.model.load_state_dict(pretrained_state_dict, strict=not skip)

    def match_state_dict(self, source_state_dict: Dict, model_state_dict: Dict) -> Dict:
        dest_state_dict = {}
        for k in source_state_dict:
            if k in model_state_dict:
                if source_state_dict[k].shape == model_state_dict[k].shape:
                    dest_state_dict[k] = source_state_dict[k]

        return dest_state_dict

    def init_network_noise(self, all_noises, alpha, beta):
        if self.original_parameters is None:
            self.original_parameters = copy.deepcopy(self.parameters())

        with torch.no_grad():
            for param, noises, original_param in zip(self.parameters(), all_noises, self.original_parameters):
                delta, nu = noises
                # the scaled noises added to the current filter
                new_value = original_param + alpha * delta + beta * nu
                param.copy_(new_value)

    def loss_iter(self, loss_dataloader: Union[SetQuenceDataset, Dataset], max_steps=100) -> torch.tensor:
        rank = self.environment.rank
        self.eval()

        loss = 0
        step = 0

        if rank == 0:
            steps_iter = tqdm(loss_dataloader, position=0, leave=False)
        else:
            steps_iter = loss_dataloader

        with torch.no_grad():
            for batch in steps_iter:
                for k, _ in batch.items():
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.environment.device, non_blocking=True)

                output = self(batch)

                if isinstance(output, tuple):
                    loss += self.loss(output, batch["output"], batch["recon"])
                else:
                    loss += self.loss(output, batch["output"])

                if step > max_steps and max_steps > -1:
                    break

                step += 1

        loss_tensor = torch.tensor(loss)
        destination_tensor = [torch.zeros_like(loss_tensor) for _ in range(self.environment.size)]
        pytorch_dist.all_gather(destination_tensor, loss_tensor)

        destination_tensor = [d.view(-1) for d in destination_tensor]
        destination_tensor = torch.cat(destination_tensor)
        acc_loss = destination_tensor.sum() / (step * self.environment.size)
        return acc_loss


class BaseDual(Base):
    model_basetype = "BaseDual"

    def __init__(
        self,
        config: Config,
        encoder: Union[BaseModule, nn.Module],
        decoder: Union[BaseModule, nn.Module],
        env: Environment = EMPTY_ENVIRONMENT,
    ):
        self.config = config
        self._encoder = encoder
        self._decoder = decoder
        self.original_parameters = None

        self._setup_environment(env)
        self._setup_attributes()
        self._setup_model(config, env)

    def __repr__(self) -> str:
        return repr(self.encoder) + "\n" + repr(self.decoder)

    def _setup_model(self, config, env):
        self.encoder = self._encoder(config, env)
        self.decoder = self._decoder(config, env)
        if self.config.encoder.bert and self.config.encoder.bert_pretrained:
            self.load_bert_pretrained(self.config.encoder.bert_route)

    def to_device(self):
        self.encoder.to(self.environment.device)
        self.decoder.to(self.environment.device)

    def __call__(self, dataset, *args, **kwargs):
        raise NotImplementedError("Needs to specifically implement in dual models")

    def train(self):
        self.decoder.train()
        self.encoder.train()

    def eval(self):
        self.decoder.eval()
        self.encoder.eval()

    def dist_alloc(self):
        STRATEGY_2_INIT_RELEASE = {
            NoDistributed: {"init": empty_fn, "release": empty_fn},
            DistributedDataParallel: {"init": self._dist_alloc, "release": self._dist_dealloc},
            DataParallel: {"init": self._dist_alloc, "release": self._dist_dealloc},
        }

        STRATEGY_2_INIT_RELEASE[self.distribution_strategy]["init"]()

    def parameters(self, optimizer=False):
        self.dist_alloc()

        if optimizer:
            parameters = [
                {"params": self.encoder.parameters(), "lr": self.config.encoder.lr_times},
                {"params": self.decoder.parameters(), "lr": self.config.decoder.lr_times},
            ]
            return parameters
        else:
            parameters = []
            for net_ in [self.encoder, self.decoder]:
                parameters += net_.parameters()

        return parameters

    def _dist_alloc(self):
        self.encoder = (
            self.distribution_strategy(self.encoder, device_ids=self.environment.local_device_ids)
            if not isinstance(self.encoder, self.distribution_strategy) and self.distribution_strategy != NoDistributed
            else self.encoder
        )

        self.decoder = (
            self.distribution_strategy(self.decoder, device_ids=self.environment.local_device_ids)
            if not isinstance(self.decoder, self.distribution_strategy) and self.distribution_strategy != NoDistributed
            else self.decoder
        )

    def _dist_dealloc(self):
        self.encoder = self.encoder.module if isinstance(self.encoder, self.distribution_strategy) else self.encoder

        self.decoder = self.decoder.module if isinstance(self.decoder, self.distribution_strategy) else self.decoder

    def state_dict(self):
        return (self.encoder.state_dict(), self.decoder.state_dict())

    def load_bert_pretrained(self, bert_dir: Union[Path, str]):
        if not hasattr(self.encoder, "bert"):
            raise ValueError(f"'{self.model_name}' does not implement BERT")
        else:
            bert_model = BertModel.from_pretrained(bert_dir)
            num_hidden_layers = self.encoder.bert.config.num_hidden_layers

            for i in range(num_hidden_layers):
                self.encoder.bert.encoder.layer[i].load_state_dict(bert_model.encoder.layer[i].state_dict())

            self.encoder.bert.embeddings.load_state_dict(bert_model.embeddings.state_dict())
            self.encoder.bert.pooler.load_state_dict(bert_model.pooler.state_dict())

    def from_pretrained(self, pretrained_model: Base, skip: bool = True):
        r"""Copies parameters and buffers from :attr:`pretrained_model` into
        this module and its descendants. If :attr:`skip` is ``True``, then
        the keys of :attr:`state_dict` do not need to exactly match the keys
        returned by this module's :meth:`~setquence.base.Base.state_dict` function.

        Args:
            state_dict (Union[Base, BaseDual]): a pre-trained module
            skip (bool, optional): whether to allow that the keys and value sizes
                in :attr:`state_dict`  do notmatch this module's function
                :meth:`~setquence.base.Base.state_dict`. Default: ``True``
        """
        if skip:
            encoder_dict = self.match_state_dict(pretrained_model.state_dict()[0], self.state_dict()[0])
            decoder_dict = self.match_state_dict(pretrained_model.state_dict()[1], self.state_dict()[1])
        else:
            encoder_dict, decoder_dict = pretrained_model.state_dict()

        self.encoder.load_state_dict(encoder_dict, strict=not skip)
        self.decoder.load_state_dict(decoder_dict, strict=not skip)

    def _wandb_model_watch(self):
        wandb.watch(self.encoder)
        wandb.watch(self.decoder)

    def _preallocate(self, optimizer):
        zeros = (
            self.encoder.zeros_like_input()
            if isinstance(self.encoder, BaseModule)
            else self.encoder.module.zeros_like_input()
        )

        random_prealloc = torch.randint(1, 10, zeros.shape, device=self.environment.device)[None]
        random_prealloc_dict = {"input": random_prealloc, "schedule": {}}
        out_prealloc = self(random_prealloc_dict)

        random_decoder_prealloc = torch.zeros(1, dtype=torch.long, device=self.environment.device)
        loss_prealloc = self.loss(out_prealloc, random_decoder_prealloc)
        loss_prealloc.backward()
        optimizer.zero_grad()

    def backward(self, loss, sync=True):
        if sync:
            loss.backward()
        else:
            with self.encoder.no_sync(), self.decoder.no_sync():
                loss.backward()
