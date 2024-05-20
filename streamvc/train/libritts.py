from math import ceil
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from datasets import Audio

DATASET_PATH = "blabble-io/libritts"
SAMPLE_RATE = 16000


def get_libritts_dataloader(split,  batch_size, num_workers=0, limit_samples=None) -> DataLoader:
    """
    Get a dataloader for the LibriTTS dataset.
    :param split: The split of the dataset to load.
    :param batch_size: The batch size.
    :param num_workers: The number of workers to use for data loading.
    :param limit_samples: The number of samples in a batch (length of audio)
        with padding.
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
        collate_fn=lambda samples: concat_and_pad_tensors([
            cap(sample['audio']['array'], limit_samples)
            for sample in samples])
    )
    return dataloader


def cap(x: torch.Tensor, max_len: int) -> torch.Tensor:
    if (max_len is None) or (x.shape[0] <= max_len):
        return x
    else:
        return x[:max_len]


def concat_and_pad_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate tensors with variable length by padding with zeros at the end.
    :param tensors: A list of 1 dimension tensors.
    :return: A tensor of the concatenated input tensors padded with zeros at the end to match the length of the largest
             input tensor.
    Example:
      Input: list([1],
                  [2, 3],
                  [4, 5, 6])
      Output: torch.Tensor([[1, 0, 0],
                            [2, 3, 0],
                            [4, 5, 6]])
    """
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded_vectors = [
        nn.functional.pad(
            vec,
            (0, max_len - vec.shape[0]),
            mode='constant', value=0
        )
        for vec in tensors
    ]
    concatenated_vectors = torch.stack(padded_vectors, dim=0)
    mask = torch.zeros_like(concatenated_vectors, dtype=torch.bool)

    for i, vec in enumerate(tensors):
        mask[i, :vec.shape[0]] = True

    return concatenated_vectors, mask
