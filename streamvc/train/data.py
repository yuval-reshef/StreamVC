from functools import reduce
import os
from typing import Iterable
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from datasets import Audio
import soundfile as sf
import numpy as np

DATASET_PATH = "blabble-io/libritts"
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 320


def get_libritts_dataloader(split,  batch_size, num_workers=0, limit_samples=None, streaming=True) -> DataLoader:
    """
    Get a dataloader for the LibriTTS dataset.
    :param split: The split of the dataset to load.
    :param batch_size: The batch size.
    :param num_workers: The number of workers to use for data loading.
    :param limit_samples: The number of samples in a batch (length of audio)
        with padding.
    :param streaming: Whether to stream the dataset.
    :param select: A function to select the rows to load.
    :return: A pytorch dataloader for the LibriTTS dataset.
    """
    dataset = load_dataset(DATASET_PATH, "all",
                           split=split, streaming=streaming)
    dataset = dataset.select_columns('audio')
    dataset = dataset.cast_column('audio', Audio(sampling_rate=SAMPLE_RATE))
    dataset = dataset.with_format('torch')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=lambda samples: trunc_pad_concat_tensors(
            (sample['audio']['array'] for sample in samples),
            max_len=limit_samples))
    return dataloader


class DataShard(Dataset):
    def __init__(self, path):
        self.path = path
        self.ids = sorted(set(file.split('.')[0] for file in os.listdir(path)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        audio_path = os.path.join(self.path, f"{id}.ogg")
        labels_path = os.path.join(self.path, f"{id}.npy")
        audio = torch.tensor(sf.read(audio_path)[0], dtype=torch.float32)
        labels = torch.tensor(np.load(labels_path), dtype=torch.int8)
        return audio, labels


class PreprocessedDataset(Dataset):

    def __init__(self, path):
        self.path = path

        directories = sorted(filter(os.path.isdir,
                                    (os.path.join(path, dirname) for dirname in os.listdir(path))))
        self.shards = [DataShard(shard_dir) for shard_dir in directories]

    def __len__(self):
        return reduce(int.__add__, map(len, self.shards))

    def _get_relative_idx(self, idx):
        shard_idx = 0
        while idx >= len(self.shards[shard_idx]):
            idx -= len(self.shards[shard_idx])
            shard_idx += 1
        return shard_idx, idx

    def __getitem__(self, idx):
        shard_idx, rel_idx = self._get_relative_idx(idx)
        return self.shards[shard_idx][rel_idx]

    @staticmethod
    def collate_fn(samples, limit_samples=None):
        audios, labels = zip(*samples)
        audio_batch, audio_mask = trunc_pad_concat_tensors(
            audios, max_len=limit_samples)
        labels_batch, labels_mask = trunc_pad_concat_tensors(
            labels,
            max_len=limit_samples // SAMPLES_PER_FRAME if limit_samples is not None else None,
            pad_value=-1)
        frame_mask = audio_mask.unfold(
            dimension=-1, size=SAMPLES_PER_FRAME, step=SAMPLES_PER_FRAME).all(dim=-1)
        assert torch.all(frame_mask == labels_mask)
        return audio_batch, labels_batch, audio_mask

    def get_dataloader(self, batch_size, num_workers=0, limit_samples=None):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=lambda samples: self.collate_fn(
                samples, limit_samples=limit_samples)
        )


def trunc_pad_concat_tensors(tensors: Iterable[torch.Tensor], max_len=None, pad_value=0) -> torch.Tensor:
    """
    Concatenate tensors in a batch, padding them to the same length.
    :param tensors: The tensors to concatenate.
    :param max_len: The maximum length to pad to. Tensors are truncated to this length.
    :param pad_value: The value to use for padding.
    Example:
      Input: list([1],
                  [2, 3],
                  [4, 5, 6])
      Output: torch.Tensor([[1, 0, 0],
                            [2, 3, 0],
                            [4, 5, 6]])
    """
    def trunc(x: torch.Tensor, max_len: int) -> torch.Tensor:
        if (max_len is None) or (x.shape[0] <= max_len):
            return x
        else:
            return x[:max_len]

    tensors = [trunc(tensor, max_len) for tensor in tensors]
    padded_len = max(tensor.shape[0] for tensor in tensors)
    padded_vectors = [
        nn.functional.pad(
            vec,
            (0, padded_len - vec.shape[0]),
            mode='constant', value=pad_value
        )
        for vec in tensors
    ]
    concatenated_vectors = torch.stack(padded_vectors, dim=0)
    mask = torch.zeros_like(concatenated_vectors, dtype=torch.bool)

    for i, vec in enumerate(tensors):
        mask[i, :vec.shape[0]] = True

    return concatenated_vectors, mask
