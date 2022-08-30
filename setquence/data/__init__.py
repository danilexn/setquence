from typing import List

from torch.utils.data import Dataset

from setquence.data.dataset import SetQuenceDataset
from setquence.data.dataset_epigenome import SetQuenceDatasetEpigenome
from setquence.data.dataset_epigenome_450k import SetQuenceDatasetEpigenome450k
from setquence.data.dataset_simple import SetQuenceSimpleDataset

__all__ = [
    "SetQuenceDataset",
]

DATASET_STR = {
    "SetQuenceDataset": SetQuenceDataset,
    "SetQuenceSimpleDataset": SetQuenceSimpleDataset,
    "SetQuenceDatasetEpigenome": SetQuenceDatasetEpigenome,
    "SetQuenceDatasetEpigenome450k": SetQuenceDatasetEpigenome450k,
}


def available_dataset_loader() -> List:
    return DATASET_STR.keys()


def get_dataset_loader(name: str) -> Dataset:
    try:
        return DATASET_STR[name]
    except KeyError:
        raise KeyError("Could not find the dataset loader '{name}'")
