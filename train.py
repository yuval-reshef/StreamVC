""" StreamVC training script.
TODO: Complete after we decide where we keep the output model.

Example usage:
    python train.py --ce_lr=0.0001
"""
import argparse

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
import torch
import torch.nn as nn
import torchaudio.transforms as T
from typing import Iterable, Optional
import ssl
import numpy as np
import torch.optim as optim
from streamvc.model import StreamVC
from streamvc.train.encoder_classifier import EncoderClassifier
from streamvc.train.discriminator import Discriminator
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
import yaml
from torch.utils.tensorboard import SummaryWriter
import time
import torch.nn.functional as F

DATASET_SAMPLE_RATE = 24000
SAMPLE_RATE = 16000
SAMPLES_PER_FRAME = 320
NUM_CLASSES = 100
# TODO: Change batch size to 128 if we manage to run it faster.
BATCH_SIZE = 4
EMBEDDING_DIMS = 64
BETAS = (0.9, 0.98)
EPS = 1e-06
WEIGHT_DECAY = 1e-2
MEL_BINS = 64
DATASET_PATH = "blabble-io/libritts"
# TODO: Change to 500.
TRAIN_SPLIT = "train.clean.100"
TEST_SPLIT = "test.clean"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAMBDA_ADV = 1
LAMBDA_FEAT = 100
LAMBDA_REC = 1


def save_net(net: nn.Module, path: str) -> None:
    """
    Saves the model in the given path.
    :param net: The model to save.
    :param path: Path to save the model.
    """
    torch.save(net.state_dict(), path)


def load_net(net, path, eval_mode: bool) -> None:
    """
    Loads the model from the given path.
    :param net: The uninitialized model to be loaded.
    :param path: The path to the parameters to load the model.
    :param eval_mode: Whether to load the model in eval mode.
    :return:
    """
    net.load_state_dict(torch.load(path))
    if eval_mode:
        net.eval()


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


def get_first_batch(batch_size: int) -> Optional[torch.Tensor]:
    # TODO: Delete if we don't use.
    """
    Returns the first batch from LibriTTS.

    :param batch_size: The batch size.
    :return: The first batch from LibriTTS.
    """
    data_iter = iter(load_dataset("blabble-io/libritts", "clean", split="train.clean.100", streaming=True))
    resampler = T.Resample(DATASET_SAMPLE_RATE, SAMPLE_RATE)
    resampler.to(torch.float32)
    samples = []
    for _ in range(batch_size):
        try:
            sample = next(data_iter)
            audio = torch.from_numpy(sample["audio"]["array"].astype(np.float32))
            resampled_audio = resampler(audio)
            samples.append(resampled_audio)
        except StopIteration:
            return None
    return concat_and_pad_tensors(samples, SAMPLES_PER_FRAME)


def batch_generator(iterable_dataset: IterableDataset, batch_size: int) -> Optional[torch.Tensor]:
    """
    Generates the next batch from LibriTTS.

    :param iterable_dataset: Iterable dataset of type datasets.iterable_dataset.IterableDataset.
    :param batch_size: The batch size.
    :return: The next batch from LibriTTS.
    """
    resampler = T.Resample(DATASET_SAMPLE_RATE, SAMPLE_RATE)
    resampler.to(torch.float32)
    for batch in iterable_dataset.iter(batch_size=batch_size):
        audios_data = batch["audio"]
        audio_waveforms = [resampler(torch.from_numpy(audio_data["array"].astype(np.float32))) for audio_data in
                           audios_data]
        tensor_batch = concat_and_pad_tensors(audio_waveforms, SAMPLES_PER_FRAME)
        yield tensor_batch


def streamvc_encoder_example(batch: Optional[torch.Tensor] = None):
    # TODO: Delete function when we finish with the training script.
    if batch is None:
        batch = get_first_batch(BATCH_SIZE)
    streamvc_model = StreamVC(SAMPLES_PER_FRAME)
    content_encoder = streamvc_model.content_encoder
    output = content_encoder(batch)


def hubert_example(batch: Optional[torch.Tensor] = None):
    # TODO: Delete function when we finish with the training script.
    if batch is None:
        batch = get_first_batch(BATCH_SIZE)
    hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True)
    simple_batch = batch[0].unsqueeze(0).unsqueeze(0)
    units = hubert.units(simple_batch)


@torch.no_grad()
def get_batch_labels(hubert_model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """
    Get hubert output labels for a given audio samples batch.

    :param hubert_model: Hubert model with discrete output labels.
    :param batch: A batch of audio samples.
    :return: The output predictions generated by the Hubert model for the input batch.
    """
    labels = []
    for sample in batch:
        single_sample_batch = sample.unsqueeze(0).unsqueeze(0)
        labels.append(hubert_model.units(single_sample_batch))
    return torch.stack(labels, dim=0)


def train_content_encoder(content_encoder: nn.Module, hubert_model: nn.Module, lr: float,
                          num_epochs: int) -> EncoderClassifier:
    """
    Train a content encoder as a classifier to predict the same labels as a discrete hubert model.

    :param content_encoder: A content encoder wrapped with a linear layer to
    :param hubert_model: Hubert model with discrete output labels.
    :param lr: Learning rate.
    :param num_epochs: Number of epochs.
    :return: The trained content encoder wrapped with a linear layer for classification.
    """
    # TODO: add epochs or number of steps when we know how much time it takes to train the model.
    wrapped_content_encoder = EncoderClassifier(content_encoder, EMBEDDING_DIMS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        wrapped_content_encoder.parameters(),
        lr=lr,
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )
    for epoch in range(1, num_epochs + 1):
        step = 0
        running_loss = 0.0
        running_loss_samples_num = 0
        dataset = load_dataset(DATASET_PATH, "all", split=TRAIN_SPLIT, streaming=True)
        for batch in batch_generator(dataset, BATCH_SIZE):
            batch = batch.to(DEVICE)
            step += 1
            optimizer.zero_grad()

            labels = get_batch_labels(hubert_model, batch)
            outputs = wrapped_content_encoder(batch)
            outputs_flat = outputs.view(-1, NUM_CLASSES)
            labels_flat = labels.view(-1)

            loss = criterion(outputs_flat, labels_flat)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_loss_samples_num += labels_flat.shape[0]
            # TODO save checkpoints
            if step % 5 == 0:  # print every 5 mini-batches
                print('[%d, %5d] loss: %.4f' %
                      (epoch, step, running_loss / running_loss_samples_num))
                running_loss = 0.0
                running_loss_samples_num = 0
    return wrapped_content_encoder


def compute_content_encoder_accuracy(encoder_classifier: EncoderClassifier, hubert_model: nn.Module):
    """
    Computes the accuracy of the wrapped content encoder as a classifier.
    :param encoder_classifier: A EncoderClassifier model to compute its accuracy.
    :param hubert_model: A pretrained hubert model for labels creation.
    :return: The accuracy of the wrapped content encoder.
    """
    correct = 0
    total = 0
    with torch.no_grad():
        dataset = load_dataset(DATASET_PATH, "clean", split=TEST_SPLIT, streaming=True)
        for batch in batch_generator(dataset, BATCH_SIZE):
            labels = get_batch_labels(hubert_model, batch)
            outputs = encoder_classifier(batch)
            outputs_flat = outputs.view(-1, NUM_CLASSES)
            labels_flat = labels.view(-1)
            _, predicted = torch.max(outputs_flat.data, 1)
            total += labels_flat.size(0)
            correct += (predicted == labels_flat).sum().item()

    print('Accuracy of the network: %d %%' % (
            100 * correct / total))


# TODO delete when we finish implementing StreamVC module.
class AdditiveModule(nn.Module):
    def __init__(self):
        super(AdditiveModule, self).__init__()
        # Define the parameter 'a' as a learnable parameter
        self.a = nn.Parameter(torch.tensor(0.0))

    def forward(self, x1, x2):
        # Add the parameter 'a' to the input tensor 'x'
        return x1 + self.a


def get_reconstruction_loss(orig_audio: torch.Tensor, generated_audio: torch.Tensor) -> torch.Tensor:
    """
    Returns the reconstruction loss for the generated audio compared to the original audio.
    :param orig_audio: The real original audio of shape [batch_size, 1, samples_num].
    :param generated_audio: The generated audio of the same shape as `orig_audio`.
    :return: The reconstruction loss.
    """
    assert orig_audio.shape == generated_audio.shape
    loss = torch.tensor(0.).to(DEVICE)
    for s_exp in range(6, 12):
        s = 2 ** s_exp
        n_fft = 2 ** 11  # Should satisfy n_fft >= win_length && ((n_fft // 2) + 1) >= n_mels.
        window_size = s
        hop_length = int(s / 4)
        mel_spectrogram = MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            win_length=window_size,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=MEL_BINS
        )
        orig_audio_spec = mel_spectrogram(orig_audio)
        generated_audio_spec = mel_spectrogram(generated_audio)

        # TODO: Check that this is indeed the number of frames and the loss is computed correctly.
        n_frames = orig_audio_spec.shape[-1]
        reshaped_orig_audio_spec = orig_audio_spec.reshape(-1, n_frames)
        reshaped_generated_audio_spec = generated_audio_spec.reshape(-1, n_frames)

        alpha_s = torch.sqrt(torch.tensor(s).to(DEVICE) / 2).to(DEVICE)
        for t in range(n_frames):
            orig_t_frame = reshaped_orig_audio_spec[:, t]
            generated_t_frame = reshaped_generated_audio_spec[:, t]
            l1_loss = F.l1_loss(orig_t_frame, generated_t_frame)
            l2_log_loss = F.mse_loss(torch.log(orig_t_frame), torch.log(generated_t_frame))
            loss += l1_loss + alpha_s * l2_log_loss
    return loss


def train_streamvc(streamvc_model: StreamVC, args: argparse.Namespace) -> None:
    """
    Trains a StreamVC model.

    :param streamvc_model: The model to train.
    :param args: Hyperparameters for training.
    """
    # TODO consider passing the parameters one by one instead of passing args.
    streamvc_model.to(DEVICE)
    root = Path(args.save_path)
    load_root = Path(args.load_path) if args.load_path else None
    root.mkdir(parents=True, exist_ok=True)

    ####################################
    # Dump arguments and create logger #
    ####################################
    with open(root / "args.yml", "w") as f:
        yaml.dump(args, f)
    writer = SummaryWriter(str(root))

    #######################
    # Load PyTorch Models #
    #######################
    # TODO: When we finish implementing StreamVC, replace AdditiveModule with streamvc.
    netG = AdditiveModule()
    netG.to(DEVICE)
    netD = Discriminator(
        args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor
    ).to(DEVICE)
    # fft = Audio2Mel(n_mel_channels=args.n_mel_channels).cuda()

    #####################
    # Create optimizers #
    #####################
    optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if load_root and load_root.exists():
        netG.load_state_dict(torch.load(load_root / "netG.pt"))
        optG.load_state_dict(torch.load(load_root / "optG.pt"))
        netD.load_state_dict(torch.load(load_root / "netD.pt"))
        optD.load_state_dict(torch.load(load_root / "optD.pt"))

    ##########################
    # Dumping original audio #
    ##########################
    test_voc = []
    test_audio = []

    costs = []
    start = time.time()

    # enable cudnn autotuner to speed up training
    torch.backends.cudnn.benchmark = True

    steps = 0
    dataset = load_dataset(DATASET_PATH, "clean", split=TRAIN_SPLIT, streaming=True)
    for epoch in range(1, args.svc_epochs + 1):
        for batch in batch_generator(dataset, BATCH_SIZE):
            print(f"{steps=}")
            batch.to(DEVICE)
            x_pred_t = netG(batch, batch)
            x_pred_t = x_pred_t.unsqueeze(1)
            batch = batch.unsqueeze(1)
            x_pred_t = x_pred_t[..., SAMPLES_PER_FRAME * 2:]
            batch = batch[..., :x_pred_t.shape[-1]]

            #######################
            # Train Discriminator #
            #######################

            D_fake_det = netD(x_pred_t.detach())
            D_real = netD(batch)

            loss_D = torch.tensor(0.).to(DEVICE)
            for scale in D_fake_det:
                # TODO: Shouldn't it be min(0,...)? Relu is max...
                loss_D += F.relu(1 + scale[-1]).mean()

            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            netD.zero_grad()
            loss_D.backward()
            optD.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.to(DEVICE))

            # Compute adversarial loss.
            loss_G = torch.tensor(0.).to(DEVICE)
            for scale in D_fake:
                loss_G += -scale[-1].mean()

            # Compute feature loss.
            loss_feat = torch.tensor(0.).to(DEVICE)
            feat_weights = 4.0 / (args.n_layers_D + 1)
            D_weights = 1.0 / args.num_D
            wt = D_weights * feat_weights
            for i in range(args.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

            # Compute reconstruction loss.
            loss_rec = get_reconstruction_loss(batch, x_pred_t)

            netG.zero_grad()
            (args.lambda_adv * loss_G + args.lambda_feat * loss_feat + args.lambda_rec * loss_rec).backward()
            optG.step()

            ######################
            # Update tensorboard #
            ######################
            costs.append([loss_D.item(), loss_G.item(), loss_feat.item()])

            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            steps += 1

            # TODO save checkpoint.

            if steps % args.log_interval == 0:
                print(
                    "Epoch {} | Iters {} | ms/batch {:5.2f} | loss {}".format(
                        epoch,
                        steps,
                        1000 * (time.time() - start) / args.log_interval,
                        np.asarray(costs).mean(0),
                    )
                )
                costs = []
                start = time.time()


def main(args: argparse.Namespace, show_accuracy: bool = True) -> None:
    """Main function for training StreamVC model."""
    streamvc = StreamVC(SAMPLES_PER_FRAME).to(DEVICE)
    # TODO consider adding an option to load content encoder instead of training.
    # content_encoder = streamvc.content_encoder
    # hubert_model = torch.hub.load("bshall/hubert:main", "hubert_discrete", trust_repo=True) \
    #     .to(DEVICE).eval()
    # wrapped_content_encoder = train_content_encoder(content_encoder, hubert_model, args.ce_lr, args.ce_epochs)
    # if show_accuracy:
    #     compute_content_encoder_accuracy(wrapped_content_encoder, hubert_model)
    train_streamvc(streamvc, args)
    # TODO: Train `streamvc_model`.


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    # TODO: Consider making some of the arguments constants.

    parser = argparse.ArgumentParser(
        prog='StreamVC Training Script',
        description='Training script for StreamVC model, using the LibriTTS dataset. Consists of two training phases:'
                    '(1) Training the content encoder and (2) Training StreamVC which contains the trained content'
                    'encoder')

    parser.add_argument("--save_path", type=str, default="out")
    # parser.add_argument("--save_path", type=str, default="out", required=True)
    parser.add_argument("--load_path", default=None)

    # Encoder training arguments.
    parser.add_argument("--ce_lr", type=float, default=0.005,
                        help="Learning rate for content encoder training.")
    parser.add_argument("--ce_epochs", type=int, default=1,
                        help="Number of epochs for content encoder training.")

    # StreamVC training arguments.
    parser.add_argument("--svc_lr", type=float, default=0.005,
                        help="Learning rate for StreamVC training.")
    parser.add_argument("--svc_epochs", type=int, default=1,
                        help="Number of epochs for StreamVC training.")

    parser.add_argument("--n_mel_channels", type=int, default=80)
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=3)

    parser.add_argument("--ndf", type=int, default=16)
    parser.add_argument("--num_D", type=int, default=3)
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--downsamp_factor", type=int, default=4)
    parser.add_argument("--lambda_feat", type=float, default=100)
    parser.add_argument("--lambda_rec", type=float, default=1)
    parser.add_argument("--lambda_adv", type=float, default=1)
    parser.add_argument("--cond_disc", action="store_true")

    parser.add_argument("--data_path", default=None, type=Path)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=8192)

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--n_test_samples", type=int, default=8)

    args = parser.parse_args()

    main(args)
