import time
import datetime
import argparse
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn as nn

from config import CFG
from dataset import WaveformDataset
from model import TimmSED, init_weights
from loss import BCEFocal2WayLoss
from sklearn import model_selection

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

config = {
    "num_classes": len(CFG.target_columns),
    "num_workers": 12,
    "save_folder": "ckpt/",
    "ckpt_name": "bird_cls",
    "temperature": 2.0,
}


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
        y_vol = y * 1 / (max_vol + 1e-9)
        return np.asfortranarray(y_vol)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_ckpt(net, iteration):
    torch.save(
        net.state_dict(),
        config["save_folder"]
        + config["ckpt_name"]
        + "_"
        + str(iteration)
        + ".pth",
    )


def evaluate(args, net, eval_loader):
    batch_iterator = iter(eval_loader)
    sum_correct = 0
    eval_samples = 0
    for iteration in range(len(eval_loader)):
        sounds, type_ids = next(batch_iterator)
        if torch.cuda.is_available():
            sounds = sounds.cuda()
            type_ids = type_ids.cuda()

        # forward
        out = net(sounds)["clipwise_output"]
        # accuracy
        _, predict = torch.max(out, 1)
        correct = (predict == type_ids)
        sum_correct += correct.sum().item()
        eval_samples += sounds.shape[0]
    return sum_correct / eval_samples


def warmup_learning_rate(optimizer, steps, warmup_steps):
    min_lr = args.lr / 100
    slope = (args.lr - min_lr) / warmup_steps

    lr = steps * slope + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def criterion(outputs, targets):
    return torch.sum(-targets * F.log_softmax(outputs, -1), -1).mean()

criterion = BCEFocal2WayLoss()

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b), lam * y_a + (1 - lam) * y_b


def train(args, train_loader, eval_loader):
    net = TimmSED("tf_efficientnetv2_s_in21k", num_classes=config["num_classes"])
    init_weights(net)
    print(net)
    net = net.cuda(device=torch.cuda.current_device())
    # net = torch.compile(net)

    if args.resume:
        print("Resuming training, loading {}...".format(args.resume))
        ckpt_file = (
            config["save_folder"]
            + config["ckpt_name"]
            + "_"
            + str(args.resume)
            + ".pth"
        )
        net.load_state_dict(torch.load(ckpt_file))

    if args.finetune:
        print("Finetuning......")
        # Freeze all layers
        for param in net.parameters():
            param.requires_grad = False
        # Unfreeze some layers
        for layer in [net.s4.b1, net.s3.b13, net.s3.b12]:
            for param in layer.parameters():
                param.requies_grad = True
        net.head.fc.weight.requires_grad = True
        optimizer = optim.AdamW(
            filter(lambda param: param.requires_grad, net.parameters()),
            lr=args.lr,
        )
    else:
        optimizer = optim.AdamW(
            net.parameters(),
            lr=args.lr,
        )

    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        factor=0.5,
        patience=1,
        verbose=True,
        threshold=5e-3,
        threshold_mode="abs",
    )
    '''T_0 = 5 * len(train_loader.dataset) // args.batch_size
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0,
        T_mult=2,
        verbose=True
    )'''

    batch_iterator = iter(train_loader)
    sum_accuracy = 0
    step = 0
    config["eval_period"] = len(train_loader.dataset) // args.batch_size
    config["verbose_period"] = config["eval_period"] // 5
    print("config:", config)

    train_start_time = time.time()
    for iteration in range(
        args.resume + 1,
        args.max_epoch * len(train_loader.dataset) // args.batch_size,
    ):
        t0 = time.time()
        try:
            sounds, type_ids = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(train_loader)
            sounds, type_ids = next(batch_iterator)
        except Exception as ex:
            print("Loading data exception:", ex)

        if torch.cuda.is_available():
            sounds = sounds.cuda()
            type_ids = type_ids.cuda()

        '''if torch.cuda.is_available():
            one_hot = torch.cuda.FloatTensor(
                type_ids.shape[0], config["num_classes"]
            )
        else:
            one_hot = torch.FloatTensor(
                type_ids.shape[0], config["num_classes"]
            )
        one_hot.fill_(0.5 / (config["num_classes"] - 1))
        one_hot.scatter_(1, type_ids.unsqueeze(1), 0.5)'''
        one_hot = F.one_hot(type_ids, num_classes=config["num_classes"])

        for index in range(2):  # Let's mixup two times
            if iteration % config["verbose_period"] == 0:
                out = net(sounds)
                loss = criterion(out, one_hot)
            else:
                # 'sounds' is input and 'one_hot' is target
                inputs, targets_a, targets_b, lam = mixup_data(sounds, one_hot)
                # forward
                out = net(inputs)
                loss, out_mixup = mixup_criterion(out, targets_a, targets_b, lam)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

        t1 = time.time()

        if iteration % config["verbose_period"] == 0:
            out = out["clipwise_output"]
            # accuracy
            _, predict = torch.max(out, 1)
            correct = (predict == type_ids)
            accuracy = correct.sum().item() / correct.size()[0]
            print(
                "iter: %d loss: %.4f | acc: %.4f | time: %.4f sec."
                % (iteration, loss.item(), accuracy, (t1 - t0)),
                flush=True,
            )
            sum_accuracy += accuracy
            step += 1

        warmup_steps = config["verbose_period"]
        if iteration < warmup_steps:
            warmup_learning_rate(optimizer, iteration, warmup_steps)

        if (
            iteration % config["eval_period"] == 0
            and iteration != 0
            and step != 0
        ):
            with torch.no_grad():
                accuracy = evaluate(args, net, eval_loader)
            hours = int(time.time() - train_start_time) // 3600
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            print(
                "[{}] [{}] Eval accuracy:{:4f} | Train accuracy:{:4f}".format(
                    now, hours, accuracy, sum_accuracy / step
                ),
                flush=True,
            )
            scheduler.step(accuracy)
            if get_lr(optimizer) < 1e-7:
                print("The learning rate is below 1e-7 now so let's go back to initial lr")
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.lr

            sum_accuracy = 0
            step = 0

        if iteration % config["eval_period"] == 0 and iteration != 0:
            # save checkpoint
            print("Saving state, iter:", iteration, flush=True)
            save_ckpt(net, iteration)

    # final checkpoint
    save_ckpt(net, iteration)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epoch",
        default=100,
        type=int,
        help="Maximum epoches for training",
    )
    parser.add_argument(
        "--dataset_root",
        default="/media/data2/song/V7.npy",
        type=str,
        help="Root path of data",
    )
    parser.add_argument(
        "--lr", default=4e-4, type=float, help="Initial learning rate"
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum value for optimizer",
    )
    parser.add_argument(
        "--resume",
        default=0,
        type=int,
        help="Checkpoint steps to resume training from",
    )
    parser.add_argument(
        "--finetune",
        default=False,
        type=bool,
        help="Finetune model by using all categories",
    )
    parser.add_argument(
        "--label_path",
        default="/media/data2/label/V7.npy",
        type=str,
        help="Root path of sounds",
    )

    args = parser.parse_args()

    t0 = time.time()
    df = pd.read_csv(CFG.train_csv)
    splitter = getattr(model_selection, CFG.split)(**CFG.split_params)
    for index, (trn_idx, val_idx) in enumerate(splitter.split(df, y=df["primary_label"])):
        print(f"Fold: {index}")
        trn_df = df.loc[trn_idx, :].reset_index(drop=True)
        val_df = df.loc[val_idx, :].reset_index(drop=True)

        train_loader = data.DataLoader(
            WaveformDataset(
                trn_df,
                CFG.train_datadir,
                waveform_transforms = get_transforms("train"),
                period = CFG.period,
                validation = False
            ),
            args.batch_size,
            num_workers=config["num_workers"],
            shuffle=True,
            pin_memory=True,
        )
        eval_loader = data.DataLoader(
            WaveformDataset(
                val_df,
                CFG.train_datadir,
                waveform_transforms = get_transforms("valid"),
                period = CFG.period,
                validation = True
            ),
            args.batch_size,
            num_workers=config["num_workers"],
            shuffle=False,
            pin_memory=True,
        )
        t1 = time.time()
        print("Load dataset with {} secs".format(t1 - t0))

        train(args, train_loader, eval_loader)
        break # just run once
