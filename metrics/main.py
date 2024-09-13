import os
import numpy as np
import pandas as pd
from extractor_wrapper import FeatureExtractor
from metric_funcs import calculate_frechet_distance, calculate_kld, calculate_clap_score, calculate_isc
import argparse
import glob
import sys
sys.path.append('../pretrained_models/visqol/visqol_lib_py')
import visqol_lib_py
import visqol_config_pb2
import similarity_result_pb2
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.transforms import Resample # Resampling
import numpy as np
import tempfile
import soundfile as sf

VISQOLMANAGER = visqol_lib_py.VisqolManager()
VISQOLMANAGER.Init(visqol_lib_py.FilePath( \
    '../pretrained_models/visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite'), \
    True, False, 60, True)


def visqol_speech_24k(ests, refs, sr=16000):
    if sr != 16000:
        resample = Resample(sr, 16000)
        ests = resample(ests)
        refs = resample(refs)
        sr = 16000
    ests = ests.view(-1, ests.shape[-1])
    refs = refs.view(-1, refs.shape[-1])
    outs = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for curinx in range(ests.shape[0]):
            sf.write("{}/est_{:07d}.wav".format(tmpdirname,curinx),ests[curinx].detach().cpu().numpy(),sr)
            sf.write("{}/ref_{:07d}.wav".format(tmpdirname,curinx),refs[curinx].detach().cpu().numpy(),sr)
            out = VISQOLMANAGER.Run( \
                visqol_lib_py.FilePath("{}/ref_{:07d}.wav".format(tmpdirname,curinx)), \
                visqol_lib_py.FilePath("{}/est_{:07d}.wav".format(tmpdirname,curinx)))
            outs.append(out.moslqo)
    return np.mean(outs)


def get_wav_paths(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                relative_path = os.path.join(root, file)
                wav_files.append(os.path.relpath(relative_path, start=directory))
    return wav_files



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute measures.")
    parser.add_argument(
        '--test_dir',
        required=True,
        help="Reference wave folder."
    )
    args = parser.parse_args()

    pred_wav_files = sorted(glob.glob(f"{args.test_dir}/*_pred.wav"))
    pred_wav_files = [item.split('/')[-1] for item in pred_wav_files]
    gt_wav_files = [item.replace("_pred.wav", "_ref.wav") for item in pred_wav_files]

    
    gt_df = pd.DataFrame({'audio_path': gt_wav_files})
    gt_df['caption'] = ''
    pred_df = pd.DataFrame({'audio_path': pred_wav_files})
    pred_df['caption'] = ''
    device = 'cuda'
    scores = {}

    # FAD by PANNs
    model = FeatureExtractor(sr=16000, backbone='cnn14', device=device,
                             feature_key='2048')
    audio_gen_features, _ = model.extract_features(pred_df, base_folder=args.test_dir)
    audio_real_features, _ = model.extract_features(gt_df, base_folder=args.test_dir)

    scores.update(calculate_frechet_distance(audio_real_features,
                                             audio_gen_features,
                                             model_name='cnn14'))

    model = FeatureExtractor(sr=16000, backbone='cnn14', device=device, 
                             feature_key='logits')

    # print(scores)

    # previous work use softmax for kl
    # however sigmoid might be more reasonable
    audio_gen_features, _ = model.extract_features(pred_df, base_folder=args.test_dir)
    audio_real_features, _ = model.extract_features(gt_df, base_folder=args.test_dir)
    scores.update(calculate_kld(audio_real_features,
                                audio_gen_features,
                                model_name='cnn14'))

    scores.update(calculate_isc(audio_gen_features,
                                rng_seed=2024,
                                samples_shuffle=True,
                                splits=10,))
    # print(scores)

    model = FeatureExtractor(sr=48000, backbone='clap', device=device, 
                             model_name='laion/larger_clap_general')
    audio_gen_features, _ = model.extract_features(pred_df, base_folder=args.test_dir)
    audio_real_features, _ = model.extract_features(gt_df, base_folder=args.test_dir)
    
    scores.update(calculate_clap_score(audio_gen_features, audio_real_features))

    # print(scores)
    
    input_files = glob.glob(f"{args.test_dir}/*_pred.wav")
    visqol = []
    for deg_wav in tqdm(input_files):
        ref_wav = deg_wav.replace("_pred.wav", "_ref.wav")
        deg_wav, fs = torchaudio.load(deg_wav)
        ref_wav, fs = torchaudio.load(ref_wav)
        cur_visqol = visqol_speech_24k(deg_wav, ref_wav, sr=fs)
        visqol.append(cur_visqol)
        
    scores.update({'visqol': np.mean(visqol)})
    print(args.test_dir)
    print(scores)
