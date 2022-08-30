from pathlib import Path

import torch
from transformers import BertModel

from setquence.base.base import BaseModule
from setquence.utils.loader import BERT_from_json


class SetQuenceBERT(BaseModule):
    def __init__(self, config, env=None):
        super().__init__(config, env)
        bert_config = BERT_from_json(Path(config.encoder.bert_config))
        self.bert = BertModel(bert_config)
        self.max_seq = config.encoder.max_seq
        self.seq_len = config.encoder.seq_len
        self.seq_split = config.encoder.seq_split
        self.hidden_size = config.encoder.hidden_size

    def forward(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.view(-1, self.seq_len)
        batch_size = int(input_ids.flatten().shape[0] / (self.seq_len * self.seq_split))

        if attention_mask is None:
            attention_mask = torch.where(input_ids > 0, 1, 0)
        else:
            attention_mask = attention_mask.view(-1, self.seq_len)

        _attention = (
            attention_mask.view(batch_size, self.seq_split, self.seq_len)
            .max(dim=2)[0]
            .view(batch_size, self.seq_split, 1)
        )

        max_length = torch.where(_attention.flatten() == 0)[0]
        if max_length.nelement() == 0 or max_length[0] > self.max_seq:
            max_length = self.max_seq
        else:
            max_length = max_length[0] - 1

        input_ids = input_ids[0:max_length, :]
        attention_mask = attention_mask[0:max_length, :]

        _, pooled_out = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None, head_mask=None,
        )

        # Change dimensionality of BERT output
        pooled_output = pooled_out.view(batch_size, -1, self.hidden_size)

        return pooled_output

    def zeros_like_input(self) -> torch.tensor:
        return torch.zeros((1, self.seq_split, self.seq_len), dtype=torch.int32, device=self.env.device)

    def zeros_like_output(self) -> torch.tensor:
        return torch.zeros((self.max_seq, self.hidden_size), device=self.env.device)


class SetQuenceBERTLarge(SetQuenceBERT):
    def __init__(self, config, env=None):
        super().__init__(config, env)
        self.max_gradient = config.encoder.max_gradient
        self.pooled_out_nograd = None

    def forward(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.view(-1, self.seq_len)

        if attention_mask is None:
            attention_mask = torch.where(input_ids > 0, 1, 0)
        else:
            attention_mask = attention_mask.view(-1, self.seq_len)

        max_length = input_ids.shape[0]

        if self.pooled_out_nograd is None:
            self.pooled_out_nograd = torch.zeros((self.seq_split, self.hidden_size), device=self.env.device)

        n_sequences_gradient = min(max_length, self.max_gradient)
        permuted_indices = torch.arange(max_length)
        gradient_indices = permuted_indices[0:n_sequences_gradient]
        non_grad_indices = permuted_indices[n_sequences_gradient:]

        _, pooled_out_grad = self.bert(input_ids[gradient_indices], attention_mask=attention_mask[gradient_indices])

        with torch.no_grad():
            for i in range(0, len(non_grad_indices), self.max_seq):
                _i_n = min(i + self.max_seq, max_length - pooled_out_grad.shape[0])
                _, self.pooled_out_nograd[i:_i_n] = self.bert(
                    input_ids[non_grad_indices[i:_i_n]], attention_mask=attention_mask[non_grad_indices[i:_i_n]],
                )

        pooled_out = torch.cat([pooled_out_grad, self.pooled_out_nograd[0 : len(non_grad_indices)]], dim=0)
        pooled_output = pooled_out.view(-1, max_length, self.hidden_size)

        return pooled_output

    def zeros_like_output(self) -> torch.tensor:
        return torch.zeros((self.seq_split, self.hidden_size), device=self.env.device)
