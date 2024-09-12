import os
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import glob
import numpy as np

TARGET_SAMPLE_RATE = 48000
BATCH_SIZE = 16
DEVICE = 'cuda'


class AudioDataset(Dataset):
    def __init__(self, data_dir, subset='train',
                 target_sample_rate=48000, target_duration=10, processor=None):
        self.audios = glob.glob(os.path.join(data_dir, "*.wav"))
        self.target_sample_rate = target_sample_rate
        self.target_duration = target_duration
        self.target_num_samples = self.target_sample_rate * self.target_duration
        self.processor = processor

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio_file_path = self.audios[idx]
        audio_sample, sample_rate = torchaudio.load(audio_file_path)

        # Convert to mono if necessary
        if audio_sample.shape[0] > 1:
            audio_sample = torch.mean(audio_sample, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            audio_sample = audio_sample
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            audio_sample = resampler(audio_sample)

        # Ensure audio_sample has the correct number of samples
        num_samples = audio_sample.shape[1]
        if num_samples > self.target_num_samples:
            # Truncate if too long
            audio_sample = audio_sample[:, :self.target_num_samples]
        elif num_samples < self.target_num_samples:
            # Pad if too short
            padding = self.target_num_samples - num_samples
            audio_sample = torch.nn.functional.pad(audio_sample, (0, padding))

        # Process the audio input separately
        audio_inputs = self.processor(
            audios=[audio_sample.squeeze().numpy()],
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True  # Pad audio to the required length, if necessary
        )
        inputs = {
            "input_features": audio_inputs["input_features"][0]  # Audio features
        }

        return inputs, self.audios[idx]


def process_batch(model, processor, batch, output_base_dir, input_base_dir, split_index):
    inputs, files = batch
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.get_audio_features(**inputs)

    for i in range(len(files)):
        savename = os.path.join(output_base_dir, files[i].split('/')[-1].replace('.wav', '_audio.npz'))
        np.savez_compressed(savename, outputs[i].cpu().numpy())
        


def process_audio_folder(input_base_dir, output_base_dir, split_index):
    os.makedirs(output_base_dir, exist_ok=True)
    model = ClapModel.from_pretrained('laion/larger_clap_general').to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained('laion/larger_clap_general')

    dataset = AudioDataset(input_base_dir, target_sample_rate=TARGET_SAMPLE_RATE, processor=processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)

    for batch in tqdm(dataloader):
        process_batch(model, processor, batch, output_base_dir, input_base_dir, split_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process audio files with CLAP model")
    parser.add_argument('--input_base_dir', type=str, required=True)
    parser.add_argument('--output_base_dir', type=str, required=True)
    parser.add_argument('--split', type=int, default=1, help='Index of the split to process')

    args = parser.parse_args()

    process_audio_folder(args.input_base_dir, args.output_base_dir, args.split)
