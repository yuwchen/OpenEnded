import argparse
import gc
import os

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from model import SpeechAssessmentModel
from whisper_voice_quality import WhisperWrapper
from scipy.stats import pearsonr

SAMPLE_RATE = 16000

gc.collect()
torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        default="openended_data/wav",
        type=str,
        help="Path of your DATA/ directory",
    )
    parser.add_argument("--ckptdir", type=str, help="Path to pretrained checkpoint.")

    args = parser.parse_args()

    my_checkpoint_dir = args.ckptdir
    datadir = args.datadir

    pretrain_dim = 256
    SAMPLE_RATE = 16000

    print("Loading checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: " + str(device))

    pretrain_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-voice-quality"
    ).to(device)

    output_dir = "Results"

    featureset = ["Accuracy", "Fluency", "Prosody"]

    assessment_model = SpeechAssessmentModel(
        pretrain_model, pretrain_dim, featureset
    ).to(device)

    assessment_model.eval()
    assessment_model.load_state_dict(
        torch.load(os.path.join(my_checkpoint_dir, "best"))
    )

    test_csv_path = "openended_data/csv/human_test.csv"
    outputpath = os.path.join(
        output_dir,
        os.path.basename(os.path.normpath(my_checkpoint_dir))
        + "-"
        + os.path.basename(test_csv_path),
    )

    df = pd.read_csv(test_csv_path)
    max_length = 15
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filepath = os.path.join(datadir, row["wavname"])

        with torch.no_grad():

            wav, sr = torchaudio.load(filepath)
            if sr != SAMPLE_RATE:
                transform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                wav = transform(wav)
                sr = SAMPLE_RATE

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

            # Average predictions across all windows
            logits = torch.stack(all_logits).mean(dim=0)

            row_dict = {}
            row_dict["wavname"] = os.path.basename(filepath)
            row_dict.update({f: p.cpu().item() for f, p in zip(featureset, logits)})

            rows.append(row_dict)
            del wav

            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(outputpath, index=False)


if __name__ == "__main__":
    main()
