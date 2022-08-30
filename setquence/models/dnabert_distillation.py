from pathlib import Path
from typing import Union

import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from transformers import BertForMaskedLM, BertModel

from setquence.base import Base, BaseModule
from setquence.base.config import Config, Environment
from setquence.distributed.distribution import NoDistributed
from setquence.utils.loader import BERT_from_json

ACTIVATIONS = {"log_softmax": F.log_softmax, "softmax": F.softmax}


class DNABERTMaskedPretraining(BertForMaskedLM):
    def forward(self, input_ids=None):
        attention_mask = torch.where(input_ids > 0, 1, 0)

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=None, position_ids=None, head_mask=None,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        return prediction_scores


class DNABERTEncoderDecoder(BaseModule):
    def __init__(self, config, env):
        super().__init__(config, env)
        self._setup_teacher(config)
        self.bert_config = BERT_from_json(Path(config.encoder.bert_config))
        self.Encoder = DNABERTMaskedPretraining(self.bert_config)
        self.teacher_predictions = None

    def forward(self, input_ids=None):
        prediction_scores = self.Encoder(input_ids)

        with torch.no_grad():
            self.teacher_predictions = self.teacher_model(input_ids)

        return prediction_scores

    def loss(self, prediction_scores, masked_lm_labels):
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        student_loss = loss_fct(prediction_scores.view(-1, self.bert_config.vocab_size), masked_lm_labels.view(-1))

        rescaling = (self.config.encoder.temperature ** 2) / prediction_scores.size(0)
        # Distillation loss with KL-Divergence wrt teacher
        distillation_loss = (
            F.kl_div(
                ACTIVATIONS[self.config.encoder.loss_act_a](
                    self.teacher_predictions / self.config.encoder.temperature, dim=2
                ),
                ACTIVATIONS[self.config.encoder.loss_act_b](
                    prediction_scores / self.config.encoder.temperature, dim=2
                ),
                reduction="batchmean",
                log_target=True,
            )
        ) * rescaling

        loss = self.config.encoder.alpha * student_loss + (1 - self.config.encoder.alpha) * distillation_loss

        return loss

    def _setup_teacher(self, config):
        bert_config = BERT_from_json(Path(config.encoder.teacher_bert_config))
        self.teacher_model = DNABERTMaskedPretraining(bert_config)
        if self.config.encoder.teacher_bert_route is not None:
            self.load_teacher_bert(self.config.encoder.teacher_bert_route)

    def load_teacher_bert(self, bert_dir):
        bert_model = BertModel.from_pretrained(bert_dir)
        num_hidden_layers = self.teacher_model.config.num_hidden_layers

        for i in range(num_hidden_layers):
            self.teacher_model.bert.encoder.layer[i].load_state_dict(bert_model.encoder.layer[i].state_dict())

        self.teacher_model.bert.embeddings.load_state_dict(bert_model.embeddings.state_dict())
        self.teacher_model.bert.pooler.load_state_dict(bert_model.pooler.state_dict())


class DNABERTDistillation(Base):
    model_name = "dnabert_distillation"

    def __init__(
        self, config: Config, model: Union[BaseModule, nn.Module] = DNABERTEncoderDecoder, env: Environment = None,
    ):
        super().__init__(config, model, env)
        self.loss = self.model.loss

    def __call__(self, dataset, *args, **kwargs):
        self.dist_alloc()
        return self.model(dataset["input"])

    def _dist_alloc(self):
        self.model = (
            self.distribution_strategy(
                self.model, device_ids=self.environment.local_device_ids, find_unused_parameters=True
            )
            if not isinstance(self.model, self.distribution_strategy) and self.distribution_strategy != NoDistributed
            else self.model
        )

    def _save_prepare(self, f: Union[Path, str]):
        self._dist_dealloc()
        if not Path(f).exists():
            Path(f).mkdir(parents=True, exist_ok=True)

    def save_model(self, f: Union[Path, str]):
        self._save_prepare(f)
        _save_path = Path(f)  # saves model into directory, not a pth file because of hf's transformers
        self.model.Encoder.bert.save_pretrained(_save_path) if not isinstance(
            self.model, self.distribution_strategy
        ) else self.model.module.Encoder.bert.save_pretrained(_save_path)

    def load_bert_pretrained(self, bert_dir):
        bert_model = BertModel.from_pretrained(bert_dir)
        num_hidden_layers = self.model.Encoder.config.num_hidden_layers

        for i in range(num_hidden_layers):
            self.model.Encoder.bert.encoder.layer[i].load_state_dict(bert_model.encoder.layer[i].state_dict())

        self.model.Encoder.bert.embeddings.load_state_dict(bert_model.embeddings.state_dict())
        self.model.Encoder.bert.pooler.load_state_dict(bert_model.pooler.state_dict())
