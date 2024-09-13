import os
import pandas as pd
import soundfile as sf
from tango import Tango
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Generate audio with Tango")
    parser.add_argument('--audio_pre_prompt', type=int, default=1, help='Number of audio samples per prompt')
    parser.add_argument('--save_path', type=str, default='../tse_gen_output/', help='Path to save generated audio')

    args = parser.parse_args()

    df = pd.read_csv('tse_gen_meta_final.csv')
    df = df[df['remove'] == 0]
    #df = df.sample(10)

    tango = Tango("declare-lab/tango-af-ac-ft-ac")
    steps = 50
    guidance = 3
    os.makedirs(args.save_path, exist_ok=True)

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        audio_id = row['audio_id']
        prompt = row['caption']
        os.makedirs(f"{args.save_path}/{audio_id}", exist_ok=True)
        for j in range(args.audio_pre_prompt):
            audio = tango.generate(prompt, steps=steps, guidance=guidance,
                                   samples=1,
                                   disable_progress=True)
            sf.write(f"{args.save_path}/{audio_id}/{j}.wav", audio, samplerate=16000)
