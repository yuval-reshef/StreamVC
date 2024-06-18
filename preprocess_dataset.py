import argparse
import os
from datasets import load_dataset, Dataset, IterableDataset, Audio
import torch
import soundfile as sf
import numpy as np
from einops import rearrange
import tqdm as tqdm

DATASET_HF = "blabble-io/libritts"
SAMPLE_RATE = 16000


def get_libritts_dataset(split, streaming=True) -> (Dataset | IterableDataset):
    dataset = load_dataset(DATASET_HF, "all",
                           split=split, streaming=streaming)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=SAMPLE_RATE))
    dataset = dataset.with_format('torch')
    return dataset


class Hubert:
    def __init__(self):
        self.model = torch.hub.load("bshall/hubert:main", "hubert_discrete",
                                    trust_repo=True).eval()

    @torch.no_grad()
    def get_labels(self, audio):
        single_sample_batch = rearrange(audio, 's -> 1 1 s')
        labels = self.model.units(single_sample_batch)
        return labels


def write_audio_and_labels(id, audio, labels, save_path):
    audio_path = os.path.join(save_path, f"{id}.ogg")
    labels_path = os.path.join(save_path, f"{id}.npy")
    sf.write(audio_path, audio.detach().cpu().numpy(), SAMPLE_RATE)
    np.save(labels_path, labels.detach().cpu().numpy().astype(np.uint8))


def main(args):
    save_path = os.path.join(args.path, args.split)
    os.makedirs(save_path, exist_ok=True)
    dataset = get_libritts_dataset(args.split)
    hubert = Hubert()
    pbar = None
    for i, sample in enumerate(dataset):
        if i % args.shard_length == 0:
            if pbar:
                pbar.close()
            print(f"Processing shard {i // args.shard_length}")
            pbar = tqdm.tqdm(total=args.shard_length)
        pbar.update(1)

        shard_number = i // args.shard_length
        shard_path = os.path.join(save_path, f"shard_{shard_number}")
        os.makedirs(shard_path, exist_ok=True)

        write_audio_and_labels(
            id=sample['id'],
            audio=sample['audio']['array'],
            labels=hubert.get_labels(sample['audio']['array']),
            save_path=shard_path
        )
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train.clean.100")
    parser.add_argument("--shard_length", type=int, default=5000)
    parser.add_argument("--path", type=str, default="./dataset")
    args = parser.parse_args()
    main(args)
