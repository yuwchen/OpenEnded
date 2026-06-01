import argparse
import gc
import os

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from model import SpeechAssessmentModel
from whisper_voice_quality import WhisperWrapper

SAMPLE_RATE = 16000

gc.collect()
torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckptdir", type=str, help="Path to pretrained checkpoint.")
    parser.add_argument("--wavpath", type=str, help="Path to wavfile.")

    args = parser.parse_args()

    wavpath = args.wavpath

    my_checkpoint_dir = args.ckptdir
    pretrain_dim = 256
    SAMPLE_RATE = 16000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-voice-quality"
    ).to(device)

    featureset = ["Accuracy", "Fluency", "Prosody"]

    assessment_model = SpeechAssessmentModel(
        pretrain_model, pretrain_dim, featureset
    ).to(device)

    assessment_model.eval()
    assessment_model.load_state_dict(
        torch.load(os.path.join(my_checkpoint_dir, "best"))
    )

    max_length = 15

    with torch.no_grad():

        wav, sr = torchaudio.load(wavpath)
        if sr != SAMPLE_RATE:
            transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            wav = transform(wav)

        window_size = max_length * SAMPLE_RATE
        hop_size = window_size // 2

        wav_length = wav.shape[1]
        if wav_length <= window_size:
            windows = [wav]
        else:
            windows = []
            start = 0
            while start + window_size <= wav_length:
                window = wav[:, start : start + window_size]
                windows.append(window)
                start += hop_size

            if start < wav_length:
                last_window = wav[:, -window_size:]
                windows.append(last_window)

        all_logits = []
        for window in windows:
            window = window.to(device)
            outputs = assessment_model(window)
            logits = torch.stack([outputs[f].cpu().squeeze() for f in featureset])
            all_logits.append(logits)

        logits = torch.stack(all_logits).mean(dim=0)
        result = {f: p.cpu().item() for f, p in zip(featureset, logits)}
        print(result)


if __name__ == "__main__":
    main()
