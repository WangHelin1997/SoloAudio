import os
import numpy as np
import soundfile as sf
import pandas as pd
import argparse
from utils import read_scaled_wav, quantize, fix_length, create_mixes
import multiprocessing
import random

FILELIST_STUB = 'fsd_mix_{}.csv'

MIX_DIR = 'mix_dir'
S1_DIR = 's1'
REF_DIR = 'ref'
SPLITS = ['train']

def create_one(i_utt, output_name, fsdmix_df, SAMPLE_RATES, output_root, splt, FIXED_LEN):
    wav_dir = 'wav' + str(SAMPLE_RATES)
    output_path = os.path.join(output_root, wav_dir, splt)
    if not os.path.exists(os.path.join(output_path, MIX_DIR, output_name)):
        utt_row = fsdmix_df[fsdmix_df['output_filename'] == output_name]
        source_number = utt_row['source_number'].iloc[0]
        
        noise_path = utt_row['noise_path'].iloc[0]
        reference_path = utt_row['reference_path'].iloc[0]
        noise_snr = 10**(float(utt_row["noise_snr"].iloc[0]) / 20)

        s1_path = utt_row['s1_path'].iloc[0]
        s2_path = utt_row['s2_path'].iloc[0]
        s3_path = utt_row['s3_path'].iloc[0]
        s4_path = utt_row['s4_path'].iloc[0]

        s1_start = float(utt_row["s1_start"].iloc[0])
        s1_end = float(utt_row["s1_end"].iloc[0])
        s1_tag = float(utt_row["s1_tag"].iloc[0])
        s1_snr = 10**(float(utt_row["s1_snr"].iloc[0]) / 20)

        s2_start = float(utt_row["s2_start"].iloc[0])
        s2_end = float(utt_row["s2_end"].iloc[0])
        s2_tag = float(utt_row["s2_tag"].iloc[0])
        s2_snr = 10**(float(utt_row["s2_snr"].iloc[0]) / 20)

        s3_start = float(utt_row["s3_start"].iloc[0])
        s3_end = float(utt_row["s3_end"].iloc[0])
        s3_tag = float(utt_row["s3_tag"].iloc[0])
        s3_snr = 10**(float(utt_row["s3_snr"].iloc[0]) / 20)

        s4_start = float(utt_row["s4_start"].iloc[0])
        s4_end = float(utt_row["s4_end"].iloc[0])
        s4_tag = float(utt_row["s4_tag"].iloc[0])
        s4_snr = 10**(float(utt_row["s4_snr"].iloc[0]) / 20)


        snr_ratio = 0.9/(s1_snr + s2_snr + s3_snr + s4_snr + noise_snr) # avoid out-of-range
        noise = quantize(read_scaled_wav(noise_path, noise_snr*snr_ratio, 0, None, SAMPLE_RATES))
        ref = quantize(read_scaled_wav(reference_path, 1.0, 0, None, SAMPLE_RATES))
        s1 = quantize(read_scaled_wav(s1_path, s1_snr*snr_ratio, s1_start, s1_end, SAMPLE_RATES))
        s2 = quantize(read_scaled_wav(s2_path, s2_snr*snr_ratio, s2_start, s2_end, SAMPLE_RATES))
        s3 = quantize(read_scaled_wav(s3_path, s3_snr*snr_ratio, s3_start, s3_end, SAMPLE_RATES))
        s4 = quantize(read_scaled_wav(s4_path, s4_snr*snr_ratio, s4_start, s4_end, SAMPLE_RATES))
        noise, s1, s2, s3, s4 = fix_length(noise, s1, s2, s3, s4, s1_tag, s2_tag, s3_tag, s4_tag, FIXED_LEN, SAMPLE_RATES)
    
        mix = create_mixes(source_number, noise, s1, s2, s3, s4)

        # write audio
        sf.write(os.path.join(output_path, MIX_DIR, output_name), mix, SAMPLE_RATES, subtype='FLOAT')
        sf.write(os.path.join(output_path, S1_DIR, output_name), s1, SAMPLE_RATES, subtype='FLOAT')
        sf.write(os.path.join(output_path, REF_DIR, output_name), ref, SAMPLE_RATES, subtype='FLOAT')
    
        if (i_utt + 1) % 500 == 0:
            print('Completed {} of {} utterances'.format(i_utt + 1, len(fsdmix_df)))
                    
                
def create_wham(args, output_root):
    FIXED_LEN = args.fixed_len
    SAMPLE_RATES = args.sr

    for splt in SPLITS:

        fsdmix_path = FILELIST_STUB.format(splt)
        fsdmix_df = pd.read_csv(fsdmix_path)

        wav_dir = 'wav' + str(SAMPLE_RATES)
        output_path = os.path.join(output_root, wav_dir, splt)
        os.makedirs(os.path.join(output_path, MIX_DIR), exist_ok=True)
        os.makedirs(os.path.join(output_path, S1_DIR), exist_ok=True)
        os.makedirs(os.path.join(output_path, REF_DIR), exist_ok=True)

        utt_ids = fsdmix_df['output_filename']

        cmds = []
        for i_utt, output_name in enumerate(utt_ids):
            cmds.append((i_utt, output_name, fsdmix_df, SAMPLE_RATES, output_root, splt, FIXED_LEN))
        print('Totally {} utterances'.format(len(cmds)))
        random.shuffle(cmds) # For parallel CPU processing, which can run several scripts at the same time.
        with multiprocessing.Pool(processes=50) as pool:
            pool.starmap(create_one, cmds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='./fsd-train/',
                        help='Output directory for writing datasets.')
    parser.add_argument('--sr', type=int, default=24000,
                help='Sampling rate')
    parser.add_argument('--fixed-len', type=float, default=10,
            help='Fixed length of simulated audio')

    args = parser.parse_args()
    print('All arguments:', args)
    os.makedirs(args.output_dir, exist_ok=True)
    create_wham(args, args.output_dir)
