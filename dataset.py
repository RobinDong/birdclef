import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as torchdata

from pathlib import Path
from config import CFG
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, TimeStretch, PitchShift

class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 waveform_transforms = None,
                 period = 5,
                 validation = False):
        self.df = df[["filename", "primary_label"]]
        if not validation:
            nobird = pd.DataFrame({
                "filename": [f"{index}.ogg" for index in range(480)],
                "primary_label": ["nobird"] * 480
            })
            self.df = pd.concat([self.df, nobird]).sample(frac=1.0).reset_index(drop=True)
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation
        poss = 0.3
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=poss),
            AddGaussianSNR(min_snr_in_db=5.0, max_snr_in_db=40.0, p=poss),
            TimeStretch(min_rate=0.8, max_rate=1.2, p=poss),
            PitchShift(min_semitones=-5, max_semitones=5, p=0.3)
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        wav_name = wav_name[:-len("ogg")] + "npy"
        ebird_code = sample["primary_label"]

        try:
            sound = np.load(self.datadir / ebird_code / wav_name, mmap_mode="r")
        except:
            print(self.datadir / ebird_code / wav_name)

        len_sound = len(sound)
        effective_len = int(CFG.sample_rate * self.period)
        if len_sound < effective_len:
            new_sound = np.zeros(effective_len, dtype=sound.dtype)
            if self.validation:
                start = 0
            else:
                start = np.random.randint(effective_len - len_sound)
            new_sound[start: start + len_sound] = sound
            sound = new_sound.astype(np.float32)
        elif len_sound > effective_len:
            if self.validation:
                start = 0
            else:
                start = np.random.randint(len_sound - effective_len)
            sound = sound[start: start + effective_len].astype(np.float32)
        else:
            sound = sound.astype(np.float32)

        if not self.validation:
            sound = self.augment(samples=sound, sample_rate=CFG.sample_rate)

        sound = np.nan_to_num(sound)

        if self.waveform_transforms:
            sound = self.waveform_transforms(sound)

        sound = np.nan_to_num(sound)

        # labels = np.zeros(len(CFG.target_columns), dtype=float)
        # labels[CFG.target_columns.index(ebird_code)] = 1.0
        label = CFG.target_columns.index(ebird_code)

        return sound, label
