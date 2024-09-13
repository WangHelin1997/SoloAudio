import csv
import pandas as pd
from tqdm import tqdm
import random
import glob
import os

csvdata = [
    [
        "output_filename",       # filename to be saved
        "source_number",         # number of audio sources, 1~4 in this case
        "noise_path",               # filepath of the 1st source
        "noise_snr",                # the snr the 1st source in the simulated data (used for mixture), set random values in -6.0~0
        "reference_path",               # filepath of the 1st source
        "s1_path",               # filepath of the 1st source
        "s1_start",              # the start time of the 1st source (used to crop and extract segment)
        "s1_end",                # the end time of the 1st source (used to crop and extract segment), set the duration *(s1_end - s1_start) between 3~10 seems realistic
        "s1_tag",                # the start time of the 1st source in the simulated data (used for mixture)
        "s1_snr",                # the snr the 1st source in the simulated data (used for mixture), set random values in -3.0~3.0
        "s1_text",               # the text (or label) the 1st source
        "s2_path",
        "s2_start",
        "s2_end",
        "s2_tag",
        "s2_snr",
        "s2_text",
        "s3_path",
        "s3_start",
        "s3_end",
        "s3_tag",
        "s3_snr",
        "s3_text",
        "s4_path",
        "s4_start",
        "s4_end",
        "s4_tag",
        "s4_snr",
        "s4_text",
    ]
]

noisepath = './TAU-urban-acoustic-scenes-2019-development/audio'
noisefiles = glob.glob(os.path.join(noisepath, '*.wav'))
print(len(noisefiles))

audiopath = './FSDKaggle2018.audio_train'
df = pd.read_csv('./FSDKaggle2018.meta/train_post_competition.csv')
groups = df.groupby('label')
grouped_data = [group for _, group in groups]

sampled_data = []

for _ in range(3):
    for group in grouped_data:
        current_label = group['label'].iloc[0]
        other_data = df[df['label'] != current_label]
        this_data = df[df['label'] == current_label]
        for _, row in group.iterrows():
            sampled = other_data.sample(n=3)
            sampled_ = this_data.sample(n=1)
            sampled_data.append((row, sampled, sampled_))
        

sampled_data = sampled_data

for i in tqdm(range(len(sampled_data))):
    s1_duration = random.uniform(5.0, 10.0)
    s1_start = 0.0 # to avoid short audios
    s1_tag = random.uniform(0.0, 10.0 - s1_duration)
    
    s2_duration = random.uniform(5.0, 10.0)
    s2_start = 0.0
    s2_tag = random.uniform(0.0, 10.0 - s2_duration)
    
    s3_duration = random.uniform(5.0, 10.0)
    s3_start = 0.0
    s3_tag = random.uniform(0.0, 10.0 - s3_duration)
    
    s4_duration = random.uniform(5.0, 10.0)
    s4_start = 0.0
    s4_tag = random.uniform(0.0, 10.0 - s4_duration)

    csvdata.append(
        [
            f'{i+1}.wav',
            random.randint(1, 4),
            random.choice(noisefiles),
            random.uniform(-6.0, 3.0),
            os.path.join(audiopath, list(sampled_data[i][2]['fname'])[0]),
            os.path.join(audiopath, sampled_data[i][0]['fname']),
            s1_start,
            s1_start+s1_duration,
            s1_tag,
            random.uniform(-3.0, 3.0),
            sampled_data[i][0]['label'],
            os.path.join(audiopath, list(sampled_data[i][1]['fname'])[0]),
            s2_start,
            s2_start+s2_duration,
            s2_tag,
            random.uniform(-3.0, 3.0),
            list(sampled_data[i][1]['label'])[0],
            os.path.join(audiopath, list(sampled_data[i][1]['fname'])[1]),
            s3_start,
            s3_start+s3_duration,
            s3_tag,
            random.uniform(-3.0, 3.0),
            list(sampled_data[i][1]['label'])[1],
            os.path.join(audiopath, list(sampled_data[i][1]['fname'])[2]),
            s4_start,
            s4_start+s4_duration,
            s4_tag,
            random.uniform(-3.0, 3.0),
            list(sampled_data[i][1]['label'])[2],
        ]
    )


file_path = 'fsd_mix_train.csv'
# Writing to the CSV file
with open(file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csvdata)
print(f'Data has been written to {file_path}')