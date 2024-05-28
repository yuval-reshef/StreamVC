import argparse

import safetensors as st
import safetensors.torch
import soundfile as sf
import torch
import torchaudio.functional as F

from streamvc import StreamVC

SAMPLE_RATE = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


@torch.no_grad()
def main(args):
    """Main function for StreamVC model inference."""
    model = StreamVC().to(device=DEVICE, dtype=DTYPE).eval()

    encoder_state_dict = st.torch.load_file(args.checkpoint, device=DEVICE)

    model.load_state_dict(encoder_state_dict)

    source_speech, orig_sr = sf.read(args.source_speech)
    source_speech = torch.from_numpy(
        source_speech).to(device=DEVICE, dtype=DTYPE)
    if orig_sr != SAMPLE_RATE:
        source_speech = F.resample(source_speech, orig_sr, SAMPLE_RATE)

    target_speech, orig_sr = sf.read(args.target_speech)
    target_speech = torch.from_numpy(
        target_speech).to(device=DEVICE, dtype=DTYPE)
    if orig_sr != SAMPLE_RATE:
        target_speech = F.resample(target_speech, orig_sr, SAMPLE_RATE)

    output = model(source_speech, target_speech)
    sf.write(args.output_path, output, SAMPLE_RATE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='StreamVC Inference Script',
        description='Inference script for StreamVC model, performs voice conversion on a single audio file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-c", "--checkpoint", type=str,
                        default="./model.safetensors",
                        help="A path the a pretrained StreamVC model checkpoint (safetensors).")
    parser.add_argument("-s", "--source-speech", type=str,
                        help="A path to a an audio file with the source speech input for the model.")
    parser.add_argument("-t", "--target-speech", type=str,
                        help="A path to a an audio file with the target speech input for the model.")
    parser.add_argument("-o", "--output-path", type=str,
                        default="./out.wav",
                        help="Output file path.")

    main(parser.parse_args())
