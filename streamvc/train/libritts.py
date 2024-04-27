from typing import Iterable

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datasets import Audio

DATASET_PATH = "blabble-io/libritts"
SAMPLE_RATE = 16000


def get_libritts_dataloader(split,  batch_size, num_workers=0) -> DataLoader:
    """
    Get a dataloader for the LibriTTS dataset.
    :param split: The split of the dataset to load.
    :param batch_size: The batch size.
    :param num_workers: The number of workers to use for data loading.
    :return: A pytorch dataloader for the LibriTTS dataset.
    """
    dataset = load_dataset(DATASET_PATH, "all", split=split, streaming=True)
    dataset = dataset.select_columns('audio')
    dataset = dataset.cast_column('audio', Audio(sampling_rate=SAMPLE_RATE))
    dataset = dataset.with_format('torch')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda s: concat_and_pad_tensors(
            [x['audio']['array'] for x in s])
    )
    return dataloader


def concat_and_pad_tensors(tensors: Iterable[torch.Tensor], pad_to_divisible_by: int = 1) -> torch.Tensor:
    """
    Concatenate tensors with variable length by padding with zeros at the end.
    :param tensors: A list of 1 dimension tensors.
    :param pad_to_divisible_by: Add additional padding to ensure that the last dimension of the output tensor is
           divisible by this parameter.
    :return: A tensor of the concatenated input tensors padded with zeros at the end to match the length of the largest
             input tensor.
    Example:
      Input: list([1],
                  [2, 3],
                  [4, 5, 6]),
             pad_to_divisible_by=2
      Output: torch.Tensor([[1, 0, 0, 0],
                            [2, 3, 0, 0],
                            [4, 5, 6, 0]])
    """
    max_len = max(tensor.shape[0] for tensor in tensors)
    if pad_to_divisible_by is not None:
        max_len = int(np.ceil(max_len / pad_to_divisible_by) * pad_to_divisible_by)
    padded_vectors = [
        nn.functional.pad(
            vec,
            (0, max_len - vec.shape[0]),
            mode='constant', value=0
        )
        for vec in tensors
    ]
    concatenated_vectors = torch.stack(padded_vectors, dim=0)
    return concatenated_vectors
