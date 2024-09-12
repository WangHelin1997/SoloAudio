import os
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
import torch
from tqdm import tqdm
from vae_modules.autoencoder_wrapper import Autoencoder
import glob
import argparse

def main(datapath, output_dir):
    audios = glob.glob(os.path.join(datapath, '*.wav'))
    device = 'cuda'
    os.makedirs(output_dir, exist_ok=True)
    max_sample = 10000000000
    autoencoder = Autoencoder('./pretrained_models/audio-vae.pt',
                              'stable_vae',
                              quantization_first=True)
    autoencoder.to(device)
    autoencoder.eval()

    with torch.no_grad():
        latents = []
        step = 0
        for i in tqdm(range(len(audios))):
            audio_id = audios[i].split('/')[-1].split('.wav')[0]
            audio_clip, sr = librosa.load(audios[i], sr=24000)
            desired_length = 10 * sr
            if len(audio_clip) < desired_length:
                padding = desired_length - len(audio_clip)
                audio_clip = np.pad(audio_clip, (0, padding), mode='constant')
            if np.abs(audio_clip).max() > 1:
                audio_clip /= np.abs(audio_clip).max()
            audio_clip = torch.tensor(audio_clip).unsqueeze(0).to(device)
            audio_clip = autoencoder(audio=audio_clip.unsqueeze(1))
            audio_clip = audio_clip.cpu()[0]
            torch.save(audio_clip, f'{output_dir}/{audio_id}.pt')
            latents.append(audio_clip)
            step += 1

            if step >= max_sample:
                break

    latents = torch.cat(latents, dim=-1)
    print('shift: ' + f'{-latents.mean()}')
    print('scale: ' + f'{1/latents.std()}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)

