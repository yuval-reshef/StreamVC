# %%
import argparse

from datasets import load_dataset
import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Iterator, Optional
import ssl
from streamvc.encoder_decoder import Encoder
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context


TRAIN_DATASET = "blabble-io/libritts"
DATASET_SAMPLE_RATE = 24000
SAMPLE_RATE = 16000
BATCH_SIZE = 128

#
# parser = argparse.ArgumentParser(
#     prog='StreamVC Training Script',
#     description='Training script for StreamVC model, using the LibriTTS dataset')
#
#
# args = parser.parse_args()

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
            audio = torch.from_numpy(sample["audio"]["array"].astype(np.float32))
            resampled_audio = resampler(audio)
            samples.append(resampled_audio)
        except StopIteration:
            return None
    return concat_and_pad_tensors(samples)


def train(batch_size: int) -> Optional[torch.Tensor]:
    data_iter = iter(load_dataset(TRAIN_DATASET, "clean", split="train.clean.100", streaming=True))
    resampler = T.Resample(DATASET_SAMPLE_RATE, SAMPLE_RATE)
    resampler.to(torch.float32)
    while True:
        batch = get_batch(data_iter, batch_size, resampler)
        if batch is None:
            return None
        # TODO instead of returning batch, train.
        return batch

first_batch = train(BATCH_SIZE)
# %%
content_encoder = Encoder(scale=64, embedding_dim=64)

print(first_batch.shape)
thin_batch = first_batch[:2, :]
print(f"{thin_batch.shape=}")
print(type(content_encoder))
output = content_encoder(thin_batch)
print(output.shape)
# %%
hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
out = hubert.units(first_batch)

