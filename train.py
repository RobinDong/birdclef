import torch
import numpy as np
import pandas as pd

from config import CFG
from dataset import WaveformDataset
from sklearn import model_selection

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


class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class Trainer:
    def run(self):
        splitter = getattr(model_selection, CFG.split)(**CFG.split_params)
        df = pd.read_csv(CFG.train_csv)
        for index, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df["primary_label"])):
            trn_df = df.loc[trn_idx, :].reset_index(drop=True)
            val_df = df.loc[val_idx, :].reset_index(drop=True)

            dataset = WaveformDataset(
                trn_df,
                CFG.train_datadir,
                img_size = CFG.img_size,
                waveform_transforms = get_transforms("train"),
                period = CFG.period,
                validation = False
            )
            '''loaders = {
                phase: torchdata.DataLoader(
                    WaveformDataset(
                        df_,
                        CFG.train_datadir,
                        img_size = CFG.img_size,
                        waveform_transforms = get_transforms(phase),
                        period = CFG.period,
                        validation = (phase == "valid")
                    ),
                    **CFG.loader_params[phase])
                for phase, df_ in zip(["train", "valid"], [trn_df, val_df])
            }'''
            print(dataset[100])


if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
