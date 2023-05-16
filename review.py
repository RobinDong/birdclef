import fire
import faiss
import tqdm
import pandas as pd

import torch

from collections import defaultdict
from config import CFG
from model import TimmSED
from dataset import WaveformDataset, get_transforms

def contrastive_loss(label, predict):
    margin = 1.0
    loss = (1 - label) * torch.square(predict) + label * torch.square(torch.clamp(margin - predict, min=0.0))
    return torch.mean(loss)

def review(ckpt_file="ckpt/bird_cls_80223.pth"):
    net = TimmSED("tf_mobilenetv3_large_100", num_classes=len(CFG.target_columns))
    net.load_state_dict(torch.load(ckpt_file))
    net.load_state_dict(torch.load(ckpt_file))
    net = net.cuda(device=torch.cuda.current_device()).eval()

    # traverse eval sounds
    val_df = pd.read_csv("val.csv")
    eval_ds = WaveformDataset(
        val_df,
        CFG.train_datadir,
        waveform_transforms = get_transforms("valid"),
        period = CFG.period,
        validation = True
    )
    train_df = pd.read_csv("train.csv")
    train_df = train_df.sort_values(["primary_label"]).groupby("primary_label").head(200)
    cluster = faiss.IndexFlatL2(1280)
    _map = {}
    _seq = 0
    for _, trow in tqdm.tqdm(train_df.iterrows(), total=len(train_df)):
        train_sound = eval_ds.get_sound(trow["filename"], trow["primary_label"])
        train_sound = torch.tensor(train_sound).cuda().unsqueeze(0)
        out = net.sub_forward(train_sound).cpu().detach().numpy()
        cluster.add(out)
        _map[_seq] = trow["primary_label"]
        _seq += 1

    print(train_df)
    correct = top5_correct = 0
    for dx, row in tqdm.tqdm(val_df.iterrows(), total=len(val_df)):
        wav_name, ebird_code = row["filename"], row["primary_label"]
        eval_sound = eval_ds.get_sound(wav_name, ebird_code)
        eval_sound = torch.tensor(eval_sound).cuda().unsqueeze(0)
        out = net.sub_forward(eval_sound).cpu().detach().numpy()
        scores, indices = cluster.search(out, 50)
        dists = []
        dist_map = defaultdict(float)
        for index, score in enumerate(scores[0]):
            name = _map[indices[0][index]]
            dist_map[name] += 1.0/score
        cands = sorted(list(dist_map.items()), key=lambda x: x[1], reverse=True)[:5]
        if cands[0][0] == ebird_code:
            correct += 1
        if ebird_code in set([name for name, score in cands]):
            top5_correct += 1
        #if dx % 100 == 0:
        #    print(f"{correct*100.0/len(val_df):02f}%, {top5_correct*100.0/len(val_df):02f}%")
    print(f"{correct*100.0/len(val_df):02f}%, {top5_correct*100.0/len(val_df):02f}%")

if __name__ == "__main__":
    fire.Fire(review)
