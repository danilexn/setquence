from pathlib import Path
from typing import Union

import torch
from torch import nn as nn
from transformers import BertModel

from setquence.base import Base, BaseModule
from setquence.base.config import Config, Environment
from setquence.base.downstream import DownTaskClassification
from setquence.base.pooling import PoolingPMA
from setquence.utils.loader import BERT_from_json


class SetQuence(BaseModule):
    def __init__(self, config, env):
        super().__init__(config, env)
        bert_config = BERT_from_json(Path(config.encoder.bert_config))
        self.Encoder = BertModel(bert_config)
        self.Pooler = PoolingPMA(self.config.pooler)
        self.BertNorm = nn.LayerNorm(self.config.encoder.hidden_size)
        self.Decoder = DownTaskClassification(self.config.decoder)
        self.loss = self.Decoder.loss

    def forward(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.view(-1, self.config.encoder.seq_len)
        batch_size = int(input_ids.flatten().shape[0] / (self.config.encoder.seq_len * self.config.encoder.seq_split))

        if attention_mask is None:
            attention_mask = torch.where(input_ids > 0, 1, 0)
        else:
            attention_mask = attention_mask.view(-1, self.config.encoder.seq_len)

        _attention = (
            attention_mask.view(batch_size, self.config.encoder.seq_split, self.config.encoder.seq_len)
            .max(dim=2)[0]
            .view(batch_size, self.config.encoder.seq_split, 1)
        )

        max_length = torch.where(_attention.flatten() == 0)[0]
        if max_length.nelement() == 0 or max_length[0] > self.config.encoder.max_seq:
            max_length = self.config.encoder.max_seq
        else:
            max_length = max_length[0]

        input_ids = input_ids[0:max_length, :]
        attention_mask = attention_mask[0:max_length, :]

        # Calculate the pooled BERT representations
        _, pooled_out = self.Encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None, head_mask=None,
        )

        # Change dimensionality of BERT output
        pooled_output = pooled_out.view(batch_size, -1, self.config.encoder.hidden_size)

        H = self.BertNorm(pooled_output)
        H = self.Pooler(H)
        logits = self.Decoder(H)

        return logits


class SetQuenceOriginal(Base):
    model_name = "setquence"

    def __init__(
        self, config: Config, model: Union[BaseModule, nn.Module] = SetQuence, env: Environment = None,
    ):
        super().__init__(config, model, env)
        self.loss = self.model.loss

    def __call__(self, dataset, sync=True, *args, **kwargs):
        self.dist_alloc()
        if sync:
            return self.model(dataset["input"])
        else:
            with self.model.no_sync():
                return self.model(dataset["input"])
