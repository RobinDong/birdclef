import torch
import numpy as np
import pandas as pd
import torch.utils.data as torchdata

from collections import defaultdict
from pathlib import Path
from config import CFG


def get_transforms(phase: str):
    transforms = CFG.transforms
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None


class WaveformDataset(torchdata.Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 datadir: Path,
                 waveform_transforms=None,
                 period=5,
                 validation=False):
        self.df = df
        self.datadir = datadir
        self.waveform_transforms = waveform_transforms
        self.period = period
        self.validation = validation
        # remember filenames for every category
        cat_map = defaultdict(set)
        for _, row in self.df.iterrows():
            cat_map[row["primary_label"]].add(row["filename"])
        self.cat_map = {key: list(value) for key, value in cat_map.items()}
        self.categories = list(self.cat_map.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        ebird_code = sample["primary_label"]
        sound = self.get_sound(wav_name, ebird_code)

        # 50% possibility to get pair_sound in same category, and
        # 50% possibility to get pair_sound in another category
        if np.random.randint(0, 2):  # same category but different wav_name
            arr = self.cat_map[ebird_code]
            if len(arr) > 1:
                pair_wav_name = np.random.choice(arr)
                while pair_wav_name == wav_name:
                    pair_wav_name = np.random.choice(arr)
            else:
                pair_wav_name = wav_name
            pair_sound = self.get_sound(pair_wav_name, ebird_code)
            distance = 0
        else:  # another category
            pair_ebird_code = np.random.choice(self.categories)
            while pair_ebird_code == ebird_code:
                pair_ebird_code = np.random.choice(self.categories)
            arr = self.cat_map[pair_ebird_code]
            pair_wav_name = np.random.choice(arr)
            pair_sound = self.get_sound(pair_wav_name, pair_ebird_code)
            distance = 1
        #return sound, pair_sound, torch.from_numpy(np.array([distance], dtype=np.float32))
        return sound, pair_sound, distance

    def get_sound(self, wav_name: str, ebird_code: str):
        wav_name = wav_name[:-len("ogg")] + "npy"

        sound = np.load(self.datadir / ebird_code / wav_name, mmap_mode="r")

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

        sound = np.nan_to_num(sound)

        if self.waveform_transforms:
            sound = self.waveform_transforms(sound)

        sound = np.nan_to_num(sound)

        return sound


if __name__ == "__main__":
    df = pd.read_csv(CFG.train_csv)
    print(df)
    wf = WaveformDataset(
        df,
        CFG.train_datadir,
        waveform_transforms=None,
        period=CFG.period,
        validation=False
    )
    print(wf[1023])
