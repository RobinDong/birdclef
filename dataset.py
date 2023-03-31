import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as torchdata

from pathlib import Path
from config import CFG

class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 waveform_transforms = None,
                 period = 5,
                 validation = False):
        self.df = df
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]

        sound, sr = sf.read(self.datadir / ebird_code / wav_name)

        len_sound = len(sound)
        effective_len = int(sr * self.period)
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

        sound = np.nan_to_num(sound)

        if self.waveform_transforms:
            sound = self.waveform_transforms(sound)

        sound = np.nan_to_num(sound)

        # labels = np.zeros(len(CFG.target_columns), dtype=float)
        # labels[CFG.target_columns.index(ebird_code)] = 1.0
        label = CFG.target_columns.index(ebird_code)

        return sound, label
