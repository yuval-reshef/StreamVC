# %%
import argparse

from datasets import load_dataset
import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Iterator, Optional
import ssl
import numpy as np
import torch.optim as optim
from streamvc.model import StreamVC
from streamvc.train.encoder_classifier import EncoderClassifier
ssl._create_default_https_context = ssl._create_unverified_context

DATASET_SAMPLE_RATE = 24000
SAMPLE_RATE = 16000
NUM_CLASSES = 100
# TODO replace batch size to 128.
BATCH_SIZE = 4
# TODO maybe get embedding dims from other file.
EMBEDDING_DIMS = 64


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


def get_first_batch(batch_size: int) -> Optional[torch.Tensor]:
    data_iter = iter(load_dataset("blabble-io/libritts", "clean", split="train.clean.100", streaming=True))
    resampler = T.Resample(DATASET_SAMPLE_RATE, SAMPLE_RATE)
    resampler.to(torch.float32)
    return get_batch(data_iter, batch_size, resampler)


def batch_generator(batch_size: int) -> Optional[torch.Tensor]:
    dataset = load_dataset("blabble-io/libritts", "clean", split="train.clean.100", streaming=True)
    resampler = T.Resample(DATASET_SAMPLE_RATE, SAMPLE_RATE)
    resampler.to(torch.float32)
    for batch in dataset.iter(batch_size=batch_size):
        audios_data = batch["audio"]
        audio_waveforms = [resampler(torch.from_numpy(audio_data["array"].astype(np.float32))) for audio_data in
                           audios_data]
        tensor_batch = concat_and_pad_tensors(audio_waveforms)
        yield tensor_batch


# %%
def streamvc_encoder_example(batch: Optional[torch.Tensor] = None):
    if batch is None:
        batch = get_first_batch(BATCH_SIZE)
    streamvc_model = StreamVC()
    content_encoder = streamvc_model.content_encoder
    print(f"{batch.shape=}")
    output = content_encoder(batch)
    print(f"{output.shape}")


# %%
# Apply hubert on libritts
def hubert_example(batch: Optional[torch.Tensor] = None):
    if batch is None:
        batch = get_first_batch(BATCH_SIZE)
    hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
    simple_batch = batch[0].unsqueeze(0).unsqueeze(0)
    print(simple_batch.shape)
    units = hubert.units(simple_batch)
    print(units.shape)
    print(units)

# %%
def get_batch_labels(hubert_model, batch: torch.Tensor) -> torch.Tensor:
    labels = []
    for sample in batch:
        single_sample_batch = sample.unsqueeze(0).unsqueeze(0)
        labels.append(hubert_model.units(single_sample_batch))
    return torch.stack(labels, dim=0)


def train_content_encoder(encoder_classifier: nn.Module):
    # TODO add epochs
    hubert_model = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(encoder_classifier.parameters(), lr=0.001, momentum=0.9)
    for batch in batch_generator(BATCH_SIZE):
        optimizer.zero_grad()
        labels = get_batch_labels(hubert_model, batch)
        outputs = encoder_classifier(batch)
        print(outputs.shape)
        print(labels.shape)
        outputs_flat = outputs.view(-1, NUM_CLASSES)
        labels_flat = labels.view(-1)
        loss = criterion(outputs_flat, labels_flat)
        loss.backward()
        optimizer.step()
        print(loss.item())
        # TODO print loss divided by samples num.


def main():
    streamvc_model = StreamVC()
    content_encoder = streamvc_model.content_encoder
    wrapped_content_encoder = EncoderClassifier(content_encoder, EMBEDDING_DIMS, NUM_CLASSES)
    train_content_encoder(wrapped_content_encoder)

if __name__ == '__main__':
    main()





