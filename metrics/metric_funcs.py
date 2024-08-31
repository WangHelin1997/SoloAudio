import os
import pandas
import torch
import numpy as np
from scipy import linalg


# mean and sigma for FD scores
def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(gts, gens, eps=1e-6, model_name='default'):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    Adapted from: https://github.com/gudgud96/frechet-audio-distance/blob/main/frechet_audio_distance/fad.py

    Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- mu1: Embedding's mean statistics for generated samples.
    -- mu2: Embedding's mean statistics for reference samples.
    -- sigma1: Covariance matrix over embeddings for generated samples.
    -- sigma2: Covariance matrix over embeddings for reference samples.
    Returns:
    --  Fréchet Distance.
    """
    mu1, sigma1 = calculate_embd_statistics(gts)
    mu2, sigma2 = calculate_embd_statistics(gens)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fd_score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return {f'fd_score_{model_name}': fd_score}


def calculate_kld(gts, gens, eps=1e-6, model_name='default'):
    assert len(gts) == len(gens)
    gts = torch.tensor(gts)
    gens = torch.tensor(gens)
    # print(gts.shape)
    # print(gens.shape)
    # softmax could be wrong for audioset multi-label classification
    kl_softmax = torch.nn.functional.kl_div(
        (gts.softmax(dim=1) + eps).log(), gens.softmax(dim=1), reduction="sum") / len(gts)
    kl_sigmoid = torch.nn.functional.kl_div(
        (gts.sigmoid() + eps).log(), gens.sigmoid(), reduction="sum") / len(gts)

    return {f'kl_softmax_{model_name}': float(kl_softmax),
            f'kl_sigmoid_{model_name}': float(kl_sigmoid)}


def calculate_isc(features, rng_seed=2024, samples_shuffle=True, splits=10):
    # print("Computing Inception Score")
    features = torch.tensor(features)
    assert torch.is_tensor(features) and features.dim() == 2
    N, C = features.shape
    if samples_shuffle:
        rng = np.random.RandomState(rng_seed)
        features = features[rng.permutation(N), :]
    features = features.double()

    p = features.softmax(dim=1)
    log_p = features.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        p_chunk = p[(i * N // splits) : ((i + 1) * N // splits), :]  # 一部分的预测概率
        log_p_chunk = log_p[(i * N // splits) : ((i + 1) * N // splits), :]  # log
        q_chunk = p_chunk.mean(dim=0, keepdim=True)  # 概率的均值
        kl = p_chunk * (log_p_chunk - q_chunk.log())  #
        kl = kl.sum(dim=1).mean().exp().item()
        scores.append(kl)
    # print("scores",scores)
    return {
        "inception_score_mean": float(np.mean(scores)),
        "inception_score_std": float(np.std(scores)),
    }


def calculate_clap_score(audio_features, text_features, model_name='default'):
    score = 0.0
    count = 0
    for i in range(len(audio_features)):
        audio_embedding = audio_features[i]
        text_feature = text_features[i]
        cosine_sim = torch.nn.functional.cosine_similarity(torch.tensor(audio_embedding).unsqueeze(0),
                                                           torch.tensor(text_feature).unsqueeze(0),
                                                           dim=1, eps=1e-8)[0]
        score += cosine_sim
        count += 1

    score = score / count if count > 0 else 0
    return {f'clap_{model_name}': score.numpy()}
