import os
import numpy as np
import pandas as pd
from extractor_wrapper import FeatureExtractor
from metric_funcs import calculate_frechet_distance, calculate_kld, calculate_clap_score, calculate_isc


def get_wav_paths(directory):
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                relative_path = os.path.join(root, file)
                wav_files.append(os.path.relpath(relative_path, start=directory))
    return wav_files



if __name__ == '__main__':
    real_dir = 'ground truth path'  # 替换为你的 ground truth 目录路径
    gen_dir = 'pred path'           # 替换为你的 prediction 目录路径
    
    gt_wav_files = get_wav_paths(real_dir)
    pred_wav_files = get_wav_paths(gen_dir)
    
    gt_df = pd.DataFrame({'audio_path': gt_wav_files})
    gt_df['caption'] = ''
    pred_df = pd.DataFrame({'audio_path': pred_wav_files})
    pred_df['caption'] = ''
    device = 'cuda'
    scores = {}

    # FD KL by VGGish
    # model = FeatureExtractor(sr=16000, backbone='vggish', device=device,
    #                          use_pca=False, use_activation=False)
    # audio_gen_features, _ = model.extract_features(df, base_folder=gen_dir)
    # audio_real_features, _ = model.extract_features(df_ori, base_folder=real_dir)

    # scores.update(calculate_frechet_distance(audio_real_features,
    #                                          audio_gen_features,
    #                                          model_name='vggish'))
    # print(scores)

    # FAD by PANNs
    model = FeatureExtractor(sr=16000, backbone='cnn14', device=device,
                             feature_key='2048')
    audio_gen_features, _ = model.extract_features(df, base_folder=gen_dir)
    audio_real_features, _ = model.extract_features(df_ori, base_folder=real_dir)

    scores.update(calculate_frechet_distance(audio_real_features,
                                             audio_gen_features,
                                             model_name='cnn14'))

    model = FeatureExtractor(sr=16000, backbone='cnn14', device=device, 
                             feature_key='logits')

    print(scores)

    audio_gen_features, _ = model.extract_features(df, base_folder=gen_dir)
    audio_real_features, _ = model.extract_features(df_ori, base_folder=real_dir)

    # previous work use softmax for kl
    # however sigmoid might be more reasonable
    scores.update(calculate_kld(audio_real_features,
                                audio_gen_features,
                                model_name='cnn14'))

    scores.update(calculate_isc(audio_gen_features,
                                rng_seed=2024,
                                samples_shuffle=True,
                                splits=10,))
    print(scores)

    model = FeatureExtractor(sr=48000, backbone='clap', device=device, 
                             model_name='laion/larger_clap_general')
    audio_gen_features, _ = model.extract_features(df, base_folder=gen_dir)
    clap_score = {'clap_gen': np.mean(audio_gen_features)}
    scores.update(clap_score)
    # audio_real_features, _ = model.extract_features(df_ori, base_folder=real_dir)
    # clap_score = {'clap_real': np.mean(audio_real_features)}
    # scores.update(clap_score)

    print(scores)
    result = pd.DataFrame([scores])