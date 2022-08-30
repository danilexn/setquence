from pathlib import Path

from transformers import BertConfig


def BERT_from_json(json_file: Path):
    return BertConfig.from_json_file(json_file)


def BERT_to_distilBERT(n_transformer: int = 6):
    pass
