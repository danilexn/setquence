import copy
from typing import Union

import torch
from torch import nn as nn

from setquence.base.base import BaseModule
from setquence.base.config import Config, Environment
from setquence.base.downstream import DownTaskClassification
from setquence.base.encoder import SetQuenceBERTLarge
from setquence.base.pooling import PoolingPMATopKSelf
from setquence.distributed.distribution import NoDistributed
from setquence.models import SetQuenceDual


class SetQuencePoolEpigenome(BaseModule):
    def __init__(self, config, env):
        super().__init__(config, env)
        # General configuration for pooler
        self.PoolerArgs = {"topk": -1, "Q_chunk_size": 1024}

        # Configuration for pooling genome data
        _config_genome = copy.deepcopy(self.config.pooler)
        # self.PoolerGenome = PoolingPMATopKSelf(_config_genome)

        # Configuration for pooling epigenome data
        _config_epigenome = copy.deepcopy(self.config.pooler)
        self.PoolerEpigenome = PoolingPMATopKSelf(_config_epigenome)
        # self.EmbedHighLow = nn.Embedding(10, self.config.pooler.hidden_size)

        # Configuration for decoding both informations
        _config_decoder = copy.deepcopy(self.config.decoder)
        _config_decoder.hidden_size = _config_genome.hidden_size  # + _config_epigenome.hidden_size
        self.Decoder = DownTaskClassification(_config_decoder, self.env)
        self.loss = self.Decoder.loss

    def forward(self, genome_encoded, epigenome_encoder, epigenome_levels):
        genome_encoded = genome_encoded.view(1, -1, self.config.pooler.hidden_size)
        epigenome_encoder = epigenome_encoder.view(1, -1, self.config.pooler.hidden_size)
        # epigenome_levels = epigenome_levels.view(1, -1)[:, 0:epigenome_encoder.shape[1]].long()

        # epigenome_levels = self.EmbedHighLow(epigenome_levels)
        epigenome_encoder = epigenome_encoder  # + epigenome_levels

        # H_genome = self.PoolerGenome(genome_encoded, self.PoolerArgs)
        H_epigenome = self.PoolerEpigenome(epigenome_encoder, self.PoolerArgs)

        # H_genome = torch.mean(H_genome, dim=1, keepdim=True)
        H_epigenome = torch.mean(H_epigenome, dim=1, keepdim=True)

        # H = torch.cat([H_genome, H_epigenome], dim=2)
        logits = self.Decoder(H_epigenome)  # H)

        return logits

    def zeros_like_input(self) -> torch.tensor:
        return torch.zeros((1, self.config.pooler.max_seq, self.config.pooler.hidden_size), device=self.env.device,)

    def zeros_like_output(self) -> torch.tensor:
        return torch.zeros((1, self.config.decoder.n_classes), device=self.env.device)


class SetQuenceEpigenome(SetQuenceDual):
    model_name = "setquence_epigenome"

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
        # This assumes that the data has already been redistributed!
        self.dist_alloc()
        self.distribution.allocate_communicators()

        if sync:
            genome_encoder_out = self.encoder(dataset["genome_input"], dataset["genome_attention"])
            epigenome_encoder_out = self.encoder(dataset["epigenome_input"], dataset["epigenome_attention"])
        else:
            with self.encoder.no_sync():
                genome_encoder_out = self.encoder(dataset["genome_input"], dataset["genome_attention"])
                epigenome_encoder_out = self.encoder(dataset["epigenome_input"], dataset["epigenome_attention"])

        zeros = (
            self.encoder.zeros_like_output()
            if isinstance(self.encoder, BaseModule)
            else self.encoder.module.zeros_like_output()
        )
        genome_encoder_out = self.distribution.undo_distribute(genome_encoder_out, zeros, dataset["schedule_genome"])
        epigenome_encoder_out = self.distribution.undo_distribute(
            epigenome_encoder_out, zeros, dataset["schedule_epigenome"]
        )

        if sync:
            decoder_out = self.decoder(genome_encoder_out, epigenome_encoder_out, dataset["epigenome_level"])
        else:
            with self.decoder.no_sync():
                decoder_out = self.decoder(genome_encoder_out, epigenome_encoder_out, dataset["epigenome_level"])

        return decoder_out

    def _preallocate(self, optimizer):
        pass
