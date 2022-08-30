import copy
from typing import Union

import torch
from torch import nn as nn

from setquence.base.base import BaseModule
from setquence.base.config import Config, Environment
from setquence.base.downstream import DownTaskClassification
from setquence.base.encoder import SetQuenceBERTLarge
from setquence.base.pooling import PoolingPMA
from setquence.distributed.distribution import NoDistributed
from setquence.models import SetQuenceDual

BINS = 3


class SetQuencePoolEpigenome(BaseModule):
    def __init__(self, config, env):
        super().__init__(config, env)
        # General configuration for pooler
        # self.PoolerArgs = {"topk": 128, "Q_chunk_size": 128}

        # Configuration for pooling epigenome data
        _config_epigenome = copy.deepcopy(self.config.pooler)
        # self.PoolerEpigenome = PoolingPMA(_config_epigenome)
        self.PoolerEpigenomeLow = PoolingPMA(_config_epigenome)
        self.PoolerEpigenomeHigh = PoolingPMA(_config_epigenome)
        self.LinearEpigenome = nn.Linear(_config_epigenome.hidden_size * 2, _config_epigenome.hidden_size)
        self.ActivationEpigenome = nn.ReLU()
        # self.EmbedLevel = nn.Embedding(BINS + 1, self.config.pooler.hidden_size, padding_idx=0)

        # Configuration for decoding both informations
        _config_decoder = copy.deepcopy(self.config.decoder)
        _config_decoder.hidden_size = _config_epigenome.hidden_size
        self.Decoder = DownTaskClassification(_config_decoder, self.env)
        self.loss = self.Decoder.loss

    def forward(self, epigenome_encoder, epigenome_levels):
        epigenome_encoder = epigenome_encoder.view(1, -1, self.config.pooler.hidden_size)
        epigenome_levels = epigenome_levels.view(1, -1).long()
        # epigenome_levels = self.EmbedLevel(epigenome_levels)
        epigenome_encoder_LOW = epigenome_encoder[epigenome_levels == 1]
        epigenome_encoder_HIGH = epigenome_encoder[epigenome_levels == 2]
        # epigenome_encoder = epigenome_encoder + epigenome_levels

        # H_epigenome = self.PoolerEpigenome(epigenome_encoder)#, self.PoolerArgs)
        H_epigenome_LOW = torch.mean(self.PoolerEpigenomeLow(epigenome_encoder_LOW), dim=1, keepdim=True)
        H_epigenome_HIGH = torch.mean(self.PoolerEpigenomeHigh(epigenome_encoder_HIGH), dim=1, keepdim=True)
        H_epigenome = torch.cat((H_epigenome_LOW, H_epigenome_HIGH), dim=2)
        H_epigenome = self.ActivationEpigenome(self.LinearEpigenome(H_epigenome))
        # H_epigenome = torch.mean(H_epigenome, dim=1, keepdim=True)

        logits = self.Decoder(H_epigenome)
        return logits

    def zeros_like_input(self) -> torch.tensor:
        return torch.zeros((1, self.config.pooler.max_seq, self.config.pooler.hidden_size), device=self.env.device,)

    def zeros_like_output(self) -> torch.tensor:
        return torch.zeros((1, self.config.decoder.n_classes), device=self.env.device)


class SetQuenceEpigenome450k(SetQuenceDual):
    model_name = "setquence_epigenome_450k"
    saved_embeddings = None

    def __init__(
        self,
        config: Config,
        encoder: Union[BaseModule, nn.Module] = SetQuenceBERTLarge,
        decoder: Union[BaseModule, nn.Module] = SetQuencePoolEpigenome,
        env: Environment = None,
    ):
        super().__init__(config, encoder, decoder, env)

    def _dist_alloc(self):
        """
        This is overriden wrt SetQuenceDual to avoid the DDP error
        when trying to update self.encoder (BERT) pooler modules
        """
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

    def __call__(self, dataset, sync=True, *args, **kwargs):
        self.dist_alloc()

        if not self.encoder.training and self.saved_embeddings is None:
            epigenome_encoder_out = self.encoder(dataset["epigenome_input"], dataset["epigenome_attention"])
            self.saved_embeddings = None
        elif self.encoder.training:
            self.saved_embeddings = None
            if sync:
                epigenome_encoder_out = self.encoder(dataset["epigenome_input"], dataset["epigenome_attention"])
            else:
                with self.encoder.no_sync():
                    epigenome_encoder_out = self.encoder(dataset["epigenome_input"], dataset["epigenome_attention"])
        elif self.saved_embeddings is not None:
            epigenome_encoder_out = self.saved_embeddings

        zeros = (
            self.encoder.zeros_like_output().view
            if isinstance(self.encoder, BaseModule)
            else self.encoder.module.zeros_like_output()
        )
        zeros = zeros.view(1, zeros.shape[0], zeros.shape[1])
        zeros = list(zeros.chunk(self.environment.size, dim=1))
        epigenome_encoder_out = self.distribution.undo_distribute(
            epigenome_encoder_out, zeros, dataset["schedule_epigenome"]
        )

        if sync:
            decoder_out = self.decoder(epigenome_encoder_out, dataset["epigenome_level"])
        else:
            with self.decoder.no_sync():
                decoder_out = self.decoder(epigenome_encoder_out, dataset["epigenome_level"])

        return decoder_out

    def _preallocate(self, *args, **kwargs):
        return
