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
    def __init__(self, meta_dir, split_index, processor=None):
        df = pd.read_csv(meta_dir)
        self.filenames = list(df['output_filename'])
        self.texts = list(df['s1_text'])
        self.processor = processor
        self.split_index = split_index

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        if self.split_index == 1:
            text = self.texts[idx].replace('_', ' ').lower()
        elif self.split_index == 2:
            text = 'The sound of '+self.texts[idx].replace('_', ' ').lower()
        elif self.split_index == 3:
            text = 'An audio clip of '+self.texts[idx].replace('_', ' ').lower()

        text_inputs = self.processor(
            text=[text],
            max_length=10,  # Fixed length for text
            padding='max_length',  # Pad text to max length
            truncation=True,  # Truncate text if it's longer than max length
            return_tensors="pt"
        )

        # Combine the processed text and audio inputs
        inputs = {
            "input_ids": text_inputs["input_ids"][0],  # Text input IDs
            "attention_mask": text_inputs["attention_mask"][0],  # Attention mask for text
        }
        return inputs, filename



def process_batch(model, processor, batch, output_base_dir, input_base_dir, split_index):
    inputs, files = batch
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        # print(outputs.shape)

    for i in range(len(files)):
        if split_index == 1:
            savename = os.path.join(output_base_dir, files[i].replace('.wav', '_text1.npz'))
        elif split_index == 2:
            savename = os.path.join(output_base_dir, files[i].replace('.wav', '_text2.npz'))
        elif split_index == 3:
            savename = os.path.join(output_base_dir, files[i].replace('.wav', '_text3.npz'))
            
        np.savez_compressed(savename, outputs[i].cpu().numpy())
        


def process_audio_folder(input_base_dir, output_base_dir, split_index):
    os.makedirs(output_base_dir, exist_ok=True)
    model = ClapModel.from_pretrained('laion/larger_clap_general').to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained('laion/larger_clap_general')

    dataset = AudioDataset(input_base_dir, split_index, processor=processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)

    for batch in tqdm(dataloader):
        process_batch(model, processor, batch, output_base_dir, input_base_dir, split_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process audio files with CLAP model")
    parser.add_argument('--input_base_dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd_mix_test.csv')
    parser.add_argument('--output_base_dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-test-clap/wav24000/test/s1')
    parser.add_argument('--split', type=int, default=1, help='Index of the split to process (1, 2 or 3)')

    args = parser.parse_args()

    process_audio_folder(args.input_base_dir, args.output_base_dir, args.split)
