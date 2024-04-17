import argparse

import torch
from datasets import load_dataset
import torch.nn as nn
import torchaudio.transforms as T
from typing import Iterator, Optional

TRAIN_DATASET = "blabble-io/libritts"
DATASET_SAMPLE_RATE = 24000
SAMPLE_RATE = 16000
BATCH_SIZE = 128


parser = argparse.ArgumentParser(
    prog='StreamVC Training Script',
    description='Training script for StreamVC model, using the LibriTTS dataset')


args = parser.parse_args()


def concat_and_pad_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    max_len = max([tensor.shape[0] for tensor in tensors])
    padded_vectors = [nn.functional.pad(vec, (0, max_len - vec.shape[0]), mode='constant', value=0) for vec in tensors]
    concatenated_vectors = torch.stack(padded_vectors, dim=0)
    return concatenated_vectors


def get_batch(samples_iterator: Iterator, batch_size: int, resampler: T.Resample) -> Optional[torch.Tensor]:
    samples = []
    for _ in range(batch_size):
        try:
            sample = next(samples_iterator)
            audio = torch.from_numpy(sample["audio"]["array"])
            resampled_audio = resampler(audio)
            samples.append(resampled_audio)
        except StopIteration:
            return None
    return concat_and_pad_tensors(samples)


def train(batch_size: int) -> None:
    data_iter = iter(load_dataset(TRAIN_DATASET, "clean", split="train.clean.100", streaming=True))
    resampler = T.Resample(DATASET_SAMPLE_RATE, SAMPLE_RATE)
    resampler.to(torch.float64)
    while True:
        batch = get_batch(data_iter, batch_size, resampler)
        if batch is None:
            return
        print(batch)


if __name__ == '__main__':
    train(BATCH_SIZE)
