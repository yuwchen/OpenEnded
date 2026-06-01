import argparse
import os
import random

import pandas as pd
import torch
import torch.nn as nn
from whisper_voice_quality import WhisperWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MyDataset

random.seed(1984)


def mha_pool(feat, length, q, attn):

    B, T, D = feat.shape

    # pool query: [1, B, D]
    query = q.flatten().unsqueeze(0).unsqueeze(1).expand(-1, B, -1)
    key = value = feat.transpose(0, 1)  # [T, B, D]

    mask = torch.arange(T, device=feat.device).unsqueeze(0) >= length.unsqueeze(
        1
    )  # [B, T]

    pooled, _ = attn(query, key, value, key_padding_mask=mask)  # [1, B, D]
    return pooled.squeeze(0)  # [B, D]


class SpeechAssessmentModel(nn.Module):

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        input_lengths = input_lengths // 160
        input_lengths = (input_lengths - 1) // 2 + 1
        return input_lengths

    def __init__(self, foundation_model, pretrain_dim, feature_set):

        super(SpeechAssessmentModel, self).__init__()

        num_heads = 4

        self.foundation_model = foundation_model
        self.backbone_model = self.foundation_model.backbone_model
        self.model_seq = self.foundation_model.model_seq
        self.output_layer = self.foundation_model.output_layer
        self.backbone_model.encoder.embed_positions = (
            self.backbone_model.encoder.embed_positions.from_pretrained(
                self.foundation_model.embed_positions[:750]
            )
        )

        for p in self.backbone_model.parameters():
            p.requires_grad = False

        for p in self.model_seq.parameters():
            p.requires_grad = False

        for p in self.output_layer.parameters():
            p.requires_grad = False

        self.feature_set = feature_set

        self.new_heads = nn.ModuleDict(
            {
                f: nn.Sequential(
                    nn.LayerNorm(pretrain_dim),
                    nn.Linear(pretrain_dim, pretrain_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                )
                for f in feature_set
            }
        )

        self.attn_pool_q = nn.ParameterDict(
            {
                f: nn.Parameter(torch.randn(num_heads, pretrain_dim // num_heads))
                for f in self.feature_set
            }
        )
        self.attn = nn.ModuleDict(
            {
                f: nn.MultiheadAttention(pretrain_dim, num_heads, batch_first=False)
                for f in self.feature_set
            }
        )

        self.output_layer = nn.ModuleDict(
            {f: nn.Linear(pretrain_dim, 1) for f in feature_set}
        )

    def forward(self, x, length=None):

        if length is not None:
            max_audio_len = 15 * 16000

            new_x = list()
            for idx in range(len(length)):
                new_x.append(x[idx].detach().cpu().numpy())

            features = self.foundation_model.feature_extractor(
                new_x,
                return_tensors="pt",
                sampling_rate=16000,
                max_length=max_audio_len,
            )
            features = features.input_features.cuda()
        else:
            max_audio_len = 15 * 16000
            features = self.foundation_model.feature_extractor(
                x[0].detach().cpu(),
                return_tensors="pt",
                sampling_rate=16000,
                max_length=max_audio_len,
            )
            features = features.input_features.cuda()

        if length is not None:
            length = self._get_feat_extract_output_lengths(length.detach().cpu())
        else:
            length = torch.tensor([len(x[0])])
            length = self._get_feat_extract_output_lengths(length)

        length = length.cuda()

        features = self.backbone_model.encoder(
            features, output_hidden_states=True
        ).hidden_states

        features = torch.stack(features, dim=0)[-1]

        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)

        features = {f: self.new_heads[f](features) for f in self.feature_set}

        outputs = {}
        for f, feat in features.items():
            pooled = mha_pool(feat, length, self.attn_pool_q[f], self.attn[f])
            outputs[f] = self.output_layer[f](pooled).squeeze(1)

        return outputs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        default="openended_data/wav",
        type=str,
        help="Path to root data directory",
    )
    parser.add_argument(
        "--finetune_from_checkpoint",
        type=str,
        required=False,
        help="Path to the checkpoint to finetune from",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=False,
        default="checkpoint/voxpa",
        help="Output directory for trained checkpoints",
    )
    parser.add_argument(
        "--csv_path_train",
        type=str,
        required=False,
        default="openended_data/csv/gemini2_train.csv",
    )
    parser.add_argument(
        "--csv_path_val",
        type=str,
        required=False,
        default="openended_data/csv/gemini2_dev.csv",
    )
    args = parser.parse_args()

    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint_dir = args.finetune_from_checkpoint
    csv_path_train = args.csv_path_train
    csv_path_val = args.csv_path_val

    if not os.path.exists(ckptdir):
        os.makedirs(os.path.join(ckptdir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: " + str(device))

    wavdir = os.path.join(datadir, "")

    pretrain_model = WhisperWrapper.from_pretrained(
        "tiantiaf/whisper-large-v3-voice-quality"
    ).to(device)
    pretrain_dim = 256

    trainset = MyDataset(wavdir, csv_path_train)

    trainloader = DataLoader(
        trainset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=trainset.collate_fn,
    )
    validset = MyDataset(wavdir, csv_path_val)
    validloader = DataLoader(
        validset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        collate_fn=validset.collate_fn,
    )
    df = pd.read_csv(csv_path_train)
    featureset = df.columns[1:]
    median_dict = df.iloc[:, 1:].median().to_dict()

    net = SpeechAssessmentModel(pretrain_model, pretrain_dim, featureset)
    net = net.to(device)

    if my_checkpoint_dir != None:
        net.load_state_dict(torch.load(os.path.join(my_checkpoint_dir, "best")))

    optimizer = torch.optim.SGD(net.parameters(), lr=0.00005, momentum=0.7)

    PREV_VAL_LOSS = 9999999999
    orig_patience = 2
    patience = orig_patience

    for epoch in range(1, 100):
        STEPS = 0
        net.train()
        running_loss = 0.0

        for i, data in enumerate(tqdm(trainloader), 0):

            (
                wav,
                labels,
                _,
            ) = data

            wav = wav.to(device)
            labels = labels.to(device)

            wav_input = wav.squeeze(1)
            optimizer.zero_grad()

            outputs = net(wav_input)
            loss = 0.0

            for i, f in enumerate(featureset):
                pred = outputs[f]
                target = labels[:, i]
                weights = 1.0 + torch.abs(target - median_dict[f]).pow(2)
                weights = weights / (weights.mean() + 1e-6)
                loss_f = (weights * (pred - target) ** 2).mean()

                loss += loss_f

            loss.backward()
            optimizer.step()
            STEPS += 1
            running_loss += loss.item()

        print("EPOCH: " + str(epoch))
        print("AVG EPOCH TRAIN LOSS: " + str(running_loss / STEPS))

        ## validation
        VALSTEPS = 0
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.memory_allocated()
            torch.cuda.synchronize()

        for i, data in enumerate(tqdm(validloader), 0):
            VALSTEPS += 1

            (
                wav,
                labels,
                _,
            ) = data

            wav = wav.to(device)
            labels = labels.to(device)

            wav_input = wav.squeeze(1)

            with torch.no_grad():

                outputs = net(wav_input)
                loss = 0.0

                for i, f in enumerate(featureset):
                    pred = outputs[f]
                    target = labels[:, i]
                    weights = 1.0 + torch.abs(target - median_dict[f]).pow(2)
                    weights = weights / (weights.mean() + 1e-6)
                    loss_f = (weights * (pred - target) ** 2).mean()

                    loss += loss_f

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / VALSTEPS
        print("EPOCH VAL LOSS: " + str(avg_val_loss))
        if avg_val_loss < PREV_VAL_LOSS:
            print("Loss has decreased")
            PREV_VAL_LOSS = avg_val_loss
            torch.save(net.state_dict(), os.path.join(ckptdir, "best"))
            patience = orig_patience
        else:
            patience -= 1
            if patience == 0:
                print(
                    "loss has not decreased for "
                    + str(orig_patience)
                    + " epochs; early stopping at epoch "
                    + str(epoch)
                )
                break

    print("Finished Training of Pronunciation Assessment Model")


if __name__ == "__main__":
    main()
