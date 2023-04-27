import glob
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch.utils.data as torchdata

from pathlib import Path
from config import CFG
from audiomentations import Compose, AddGaussianNoise, AddGaussianSNR, TimeStretch, PitchShift, Normalize, ClippingDistortion, AirAbsorption

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
        bg_list = glob.glob("../background/*.wav")
        self.noise_list = []
        for filename in bg_list:
            noise, nsr = sf.read(filename)
            noise = noise.astype(np.float32)
            self.noise_list.append(noise)
            self.noise_list.append(librosa.resample(noise, orig_sr=nsr, target_sr=16000))
        poss = 0.5
        self.augment = Compose([
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=poss),
            AddGaussianSNR(min_snr_in_db=5.0, max_snr_in_db=40.0, p=poss),
            #TimeStretch(min_rate=0.8, max_rate=1.2, p=poss),
            #PitchShift(min_semitones=-5, max_semitones=5, p=poss),
            #ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p=poss),
            #AirAbsorption(p=poss),
        ])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.loc[idx, :]
        wav_name = sample["filename"]
        wav_name = wav_name[:-len("ogg")] + "npy"
        ebird_code = sample["primary_label"]

        sound = np.load(self.datadir / ebird_code / wav_name, mmap_mode="r")

        len_sound = len(sound)
        effective_len = int(16000 * self.period)
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
            sound = self.augment(samples=sound, sample_rate=16000)
            # add background randomly
            '''length = len(self.noise_list)
            pick = np.random.randint(0, 2*length)
            if pick < length:
                noise = self.noise_list[pick]
                start = np.random.randint(0, sound.shape[0])
                length = min(sound.shape[0] - start, noise.shape[0])
                scale = np.max(noise) / (np.max(sound) + 1e-9) / 4
                new_noise = np.zeros(sound.shape[0])
                new_noise[start:] = noise[:length]
                sound = np.add(sound, new_noise/scale).astype(np.float32)'''

        sound = np.nan_to_num(sound)

        if self.waveform_transforms:
            sound = self.waveform_transforms(sound)

        sound = np.nan_to_num(sound)

        # labels = np.zeros(len(CFG.target_columns), dtype=float)
        # labels[CFG.target_columns.index(ebird_code)] = 1.0
        label = CFG.target_columns.index(ebird_code)

        return sound, label
