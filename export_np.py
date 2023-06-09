import os
import cv2
import sys
import time
import torch
import librosa
import argparse
import traceback
import numpy as np
import soundfile as sf

from config import CFG
from functools import partial
from multiprocessing import Pool, current_process
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


RETRY = 10
RETRY_WAIT = 10  # seconds

TOO_SMALL_SIZE = 4096
TOO_BIG_SIZE = 100 * 1048576


class SignalExtractor:
    def __init__(self):
        self.spectrogram_extractor = Spectrogram(n_fft=CFG.n_fft, hop_length=CFG.hop_length, win_length=CFG.n_fft, window="hann", center=True, pad_mode="reflect", freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=CFG.sample_rate, n_fft=CFG.n_fft, n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.factors = [2.0, 1.8, 1.6, 1.4, 1.2, 1.1]
        self.kernel_size = 15
        self.sn_threshold = 0.2

    def extract(self, input):
        x = torch.from_numpy(input)
        x = x[None, :].float()

        x = self.spectrogram_extractor(x)
        x = self.logmel_extractor(x)

        x = x.squeeze(0).squeeze(0)
        x = x.permute(1, 0).numpy()
        x = x - np.amin(x)

        for factor in self.factors:
            sound, sn_ratio = self._factor_extract(input, x, factor)
            if sn_ratio >= self.sn_threshold:
                break

        return sound, sn_ratio

    def _factor_extract(self, input, x, factor: float):
        rows, cols = x.shape
        row_median = np.median(x, axis=1)
        row_median_matrix = np.tile(row_median, (cols, 1)).T * factor
        col_median = np.median(x, axis=0)
        col_median_matrix = np.tile(col_median, (rows, 1)) * factor

        y = x > row_median_matrix
        z = x > col_median_matrix
        res = np.logical_and(y, z) + np.zeros(x.shape)

        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        img = cv2.dilate(res, kernel, iterations=1)

        indicator = np.sum(img, axis=0)
        chunk_size = input.shape[0] // indicator.shape[0]
        sounds = []
        for index, chunk in enumerate(indicator):
            if chunk > 0:
                sounds.append(input[index*chunk_size:(index+1)*chunk_size])
        if len(sounds) <= 0:
            return None, 0.0
        sound = np.concatenate(sounds)
        return sound, sound.shape[0]/input.shape[0]


def process(full_path, args):
    sub_dir = full_path.split("/")[-2]
    output_path = os.path.join(args.output_root, sub_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    for _ in range(RETRY):
        try:
            file_size = os.stat(full_path).st_size
        except Exception as ex:
            print("Stat file failed:", ex)
            traceback.print_exc(file=sys.stdout)
            time.sleep(RETRY_WAIT)
            continue
        break

    if file_size <= TOO_SMALL_SIZE or file_size > TOO_BIG_SIZE:
        return

    sound_name, _ = os.path.splitext(full_path.split("/")[-1])
    new_path = os.path.join(output_path, f"{sound_name}.npy")
    sound, sr = sf.read(full_path)
    sound = librosa.resample(sound, orig_sr=sr, target_sr=32000).astype(np.float16)
    sn_extractor = SignalExtractor()
    sound, sn_ratio = sn_extractor.extract(sound)
    if sound is None or sn_ratio < 0.1:
        print(f"{full_path} extract failed {sound}:{sn_ratio}")
        return
    np.save(new_path, sound)
    print(full_path, new_path)

    '''for _ in range(RETRY):
        result = subprocess.run(
            f"python3 -u export_single.py --input \"{full_path}\" --output \"{new_path}\"",
            shell=True)
        if result.returncode != 0:
            print("retry")
        else:
            break'''

    job_index = current_process().name.split("-")[-1]
    job_index = int(job_index) - 1
    with open(f"finish.{job_index}.lst", "a+") as fp:
        fp.write(full_path + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpus", default=12, type=int, help="Number of cpu cores"
    )
    parser.add_argument(
        "--output_root", default="../numpy", type=str, help="Root directory of output"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_root):
        os.mkdir(args.output_root)

    finish_list = set()
    for cpu in range(args.cpus):
        if os.path.exists(f"finish.{cpu}.lst"):
            with open(f"finish.{cpu}.lst") as fp:
                while True:
                    line = fp.readline()
                    if not line:
                        break
                    finish_list.add(line)

    need_to_do = []
    with open("task.lst", "r") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            if line in finish_list:
                continue
            need_to_do.append(line.strip())

    pool = Pool(processes=args.cpus)
    pool.map(
        partial(
            process,
            args=args,
        ),
        need_to_do
    )
