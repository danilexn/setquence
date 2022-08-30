from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import nn as nn

from setquence.base.base import Base, BaseDual
from setquence.base.config import EMPTY_ENVIRONMENT
from setquence.models.dnabert_distillation import DNABERTDistillation
from setquence.models.setquence_dual import SetQuenceDual
from setquence.models.setquence_dual_large import SetQuenceDualLarge
from setquence.models.setquence_epigenome import SetQuenceEpigenome
from setquence.models.setquence_epigenome_450k import SetQuenceEpigenome450k
from setquence.models.setquence_original import SetQuenceOriginal
from setquence.utils import dict_to_namespace

MODELS_STR = {
    "distil_dnabert": DNABERTDistillation,
    "setquence": SetQuenceOriginal,
    "setquence_dual": SetQuenceDual,
    "setquence_dual_large": SetQuenceDualLarge,
    "setquence_epigenome": SetQuenceEpigenome,
    "setquence_epigenome_450k": SetQuenceEpigenome450k,
}


def available_models() -> List:
    return MODELS_STR.keys()


def get_model(name: str) -> Union[Base, nn.Module]:
    try:
        return MODELS_STR[name]
    except KeyError:
        raise KeyError("Could not find a model named '{name}' in the current version of SetQuence")


def load_model_from_file(f: Union[Path, str], env=EMPTY_ENVIRONMENT) -> Union[Base, BaseDual]:
    if not Path(f).exists():
        raise FileNotFoundError(f"The specified route {f} does not exist. Cannot load a model.")

    model_dict = torch.load(f, map_location="cpu")
    model = load_model(model_dict, env)
    return model


def load_model(state_dict: Dict, env=EMPTY_ENVIRONMENT) -> Union[Base, BaseDual]:
    _config = dict_to_namespace(state_dict["config"])
    model = get_model(state_dict["model_name"])(config=_config, env=env)
    if state_dict["model_basetype"] == "Base":
        model.model.load_state_dict(state_dict["model_weights"])
    elif state_dict["model_basetype"] == "BaseDual":
        model.encoder.load_state_dict(state_dict["model_weights"][0], strict=False)
        model.decoder.load_state_dict(state_dict["model_weights"][1], strict=False)
    else:
        raise ValueError(f"Model base type '{state_dict['model_basetype']}' is not available")

    return model


def validate_state_dicts(model_state_dict_1, model_state_dict_2):
    if len(model_state_dict_1) != len(model_state_dict_2):
        return False

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()}

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()}

    for ((k_1, v_1), (k_2, v_2)) in zip(model_state_dict_1.items(), model_state_dict_2.items()):
        if k_1 != k_2:
            return False
        # convert both to the same CUDA device
        if str(v_1.device) != "cuda:0":
            v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(v_2.device) != "cuda:0":
            v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        if not torch.allclose(v_1, v_2):
            return False

    return True
