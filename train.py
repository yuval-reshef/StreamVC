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
single_sample_batch = first_batch[0].unsqueeze(0)
print(f"{single_sample_batch.shape=}")
output = content_encoder(single_sample_batch)
print(output.shape)
# %%
# Apply hubert on a wav file

import torchaudio

hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
wav, sr = torchaudio.load("/Users/yonicohen/Documents/university/Masters/yearA/AUDIO-83091/Exercises/Ex2/code/recordings/male-4-4.wav")
assert sr == 16000
wav = wav.unsqueeze(0)
print(wav.shape)
units = hubert.units(wav)
print(units.shape)
print(type(hubert))

# Extract speech units


# %%
# Apply hubert on libritts

import torchaudio

hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
simple_batch = first_batch[0].unsqueeze(0).unsqueeze(0)
# simple_batch = simple_batch.unsqueeze(1)[0, :, :]
print(simple_batch.shape)
units = hubert.units(simple_batch)
print(units.shape)
# print(type(hubert))

