import os
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import random
import torch
import soundfile as sf
import librosa
import glob

class TSEDataset(Dataset):
    def __init__(self, base_dir, vae_dir, timbre_dir, tag=None, debug=False
        ):

        self.debug = debug
        self.base_dir = base_dir
        self.timbre_dir = timbre_dir
        self.vae_dir = vae_dir
        self.tag = tag
        self.files = glob.glob(os.path.join(vae_dir, "mix_dir", "*.pt"))
        
    def __len__(self):
        return len(self.files) if not self.debug else len(self.files) // 100


    def __getitem__(self, idx):
        file_id = self.files[idx].split('/')[-1].split('.pt')[0]

        # read mixture
        mixture_path = os.path.join(self.base_dir, "mix_dir", file_id+'.wav')
        mixture = torch.load(os.path.join(self.vae_dir, "mix_dir", file_id+'.pt'))
        source_path = os.path.join(self.base_dir, "s1", file_id+'.wav')
        source = torch.load(os.path.join(self.vae_dir, "s1", file_id+'.pt'))
        
        # read enrollment
        enroll_path = os.path.join(self.base_dir, "ref", file_id+'.wav')
        enroll = torch.tensor(np.load(os.path.join(self.timbre_dir, "s1", file_id+"_"+self.tag+".npz"))['arr_0'])
        
        return mixture, source, enroll, file_id, mixture_path, source_path, enroll_path


if __name__ == "__main__":
    dataset = TSEDataset(
        base_dir='/export/corpora7/HW/TSEDataMix/fsd-val/wav24000/val', 
        vae_dir='/export/corpora7/HW/TSEDataMix/fsd-val-vae/wav24000/val', 
        timbre_dir='/export/corpora7/HW/TSEDataMix/fsd-val-clap/wav24000/val', 
        tag="text1", 
        debug=True
    )
    mixture, source, enroll, mix_id, mixture_path, source_path, enroll_path = dataset[0]
    print(mixture.shape, source.shape, enroll.shape)
    print(mixture_path)
    print(source_path)
    print(enroll_path)