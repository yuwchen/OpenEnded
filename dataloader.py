import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data.dataset import Dataset

SAMPLE_RATE = 16000


class MyDataset(Dataset):

    def __init__(self, rootdir, data_csv_path):

        data_df = pd.read_csv(data_csv_path)
        self.wavfiles = []
        self.labels = []
        data_df[data_df.columns[1:]] = data_df[data_df.columns[1:]].astype(float)

        for _, row in data_df.iterrows():
            self.wavfiles.append(os.path.join(rootdir, row["wavname"]))
            self.labels.append(row.iloc[1:].values.astype(np.float32))

    def __getitem__(self, idx):
        wavpath = self.wavfiles[idx]
        label = self.labels[idx]
        wav, sr = torchaudio.load(wavpath)

        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        return (
            wav,
            label,
            wavpath,
        )

    def __len__(self):
        return len(self.wavfiles)

    def collate_fn(self, batch):  ## zero padding

        batch = list(filter(lambda x: x is not None, batch))

        (
            wav,
            labels,
            wavfile,
        ) = zip(*batch)

        wavs = list(wav)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), "constant", 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)

        labels = torch.stack([torch.tensor(x) for x in list(labels)], dim=0)

        return (
            output_wavs,
            labels,
            wavfile,
        )
