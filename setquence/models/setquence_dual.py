from typing import Union

import torch
from torch import nn as nn

from setquence.base.base import BaseDual, BaseModule
from setquence.base.config import Config, Environment
from setquence.base.downstream import DownTaskClassification
from setquence.base.encoder import SetQuenceBERT
from setquence.base.pooling import PoolingPMA


class SetQuencePool(BaseModule):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.Pooler = PoolingPMA(self.config.pooler)
        self.Decoder = DownTaskClassification(self.config.decoder, self.env)
        self.loss = self.Decoder.loss

    def forward(self, encoded):
        encoded = encoded.view(1, -1, self.config.pooler.hidden_size)
        if self.training:
            encoded = encoded + torch.randn(encoded.shape, device=self.env.device) * 0.01

        H = self.Pooler(encoded)
        logits = self.Decoder(H)

        try:
            if self.config.pooler.return_pooled:
                return logits, H, encoded
        except AttributeError:
            return logits

    def zeros_like_input(self) -> torch.tensor:
        return torch.zeros((1, self.config.pooler.max_seq, self.config.pooler.hidden_size), device=self.env.device,)

    def zeros_like_output(self) -> torch.tensor:
        return torch.zeros((1, self.config.decoder.n_classes), device=self.env.device)


class SetQuenceDual(BaseDual):
    model_name = "setquence_dual"

    def __init__(
        self,
        config: Config,
        encoder: Union[BaseModule, nn.Module] = SetQuenceBERT,
        decoder: Union[BaseModule, nn.Module] = SetQuencePool,
        env: Environment = None,
    ):
        super().__init__(config, encoder, decoder, env)
        self.loss = self.decoder.loss
        self.allocated = False

    def __call__(self, dataset, sync=True, *args, **kwargs):
        self.dist_alloc()
        if not self.allocated:
            self.distribution.allocate_communicators()
            self.allocated = True
        # return self.decoder(self.encoder(*args, **kwargs))
        if "attention" not in dataset.keys():
            zeros = (
                self.encoder.zeros_like_input()
                if isinstance(self.encoder, BaseModule)
                else self.encoder.module.zeros_like_input()
            )
            tensor, schedule = self.distribution.distribute(dataset, zeros)
            attention = None
        else:
            tensor, attention, schedule = dataset["input"], dataset["attention"], dataset["schedule"]

        if sync:
            encoder_out = self.encoder(tensor, attention)
        else:
            with self.encoder.no_sync():
                encoder_out = self.encoder(tensor, attention)

        zeros = (
            self.encoder.zeros_like_output()
            if isinstance(self.encoder, BaseModule)
            else self.encoder.module.zeros_like_output()
        )
        encoder_out = self.distribution.undo_distribute(encoder_out, zeros, schedule)

        if sync:
            decoder_out = self.decoder(encoder_out)
        else:
            with self.decoder.no_sync():
                decoder_out = self.decoder(encoder_out)

        return decoder_out
