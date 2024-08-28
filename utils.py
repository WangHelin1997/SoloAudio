# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import torch

# import matplotlib as mpl
# mpl.use('TkAgg')


def sisnr_loss(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor, estimate value
          s: reference signal, N x S tensor, True value
    Return:
          sisnr: N tensor
    """
    if x.shape != s.shape:
        if x.shape[-1] > s.shape[-1]:
            x = x[:, :s.shape[-1]]
        else:
            s = s[:, :x.shape[-1]]
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    loss = -20. * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))
    return torch.sum(loss) / x.shape[0]


def sisnri(x, s, m): # sisnr improvement
    """
    Arguments:
    x: separated signal, BS x S predicted sound
    s: reference signal, BS x S target sound
    m: mixture signal, BS x S mixture sound
    Return:
    sisnri: N tensor
    """
    sisnr = sisnr_loss(x, s)
    sisnr_ori = sisnr_loss(m, s)
    return sisnr_ori - sisnr #


def get_loss(est_wav, lab_wav, mix_wav, onset, offset, sr=16000):
    sisnrI_w = 0.0
    loss_sisnr_w = 0.0
    onset = onset.cpu().numpy()
    offset = offset.cpu().numpy()
    sample_num = onset.shape[0] # batch_size
    for i in range(sample_num):
        assert onset[i] < offset[i]
        est_wav_w = est_wav[i]
        lab_wav_w = lab_wav[i]
        mix_wav_w = mix_wav[i]

        max_wav = min(est_wav_w.shape[-1], lab_wav_w.shape[-1], mix_wav_w.shape[-1])
        est_wav_w = est_wav_w[:max_wav]
        lab_wav_w = lab_wav_w[:max_wav]
        mix_wav_w = mix_wav_w[:max_wav]

        onset_wav = round(sr * onset[i]) if round(sr * onset[i]) >= 0 else 0  # target sound begin sample)
        offset_wav = round(sr * offset[i]) if round(sr * offset[i]) < max_wav else max_wav  # end

        est_wav_w = est_wav_w[onset_wav:offset_wav]  # est_wav
        est_wav_w = est_wav_w[None, :]  # (1,N)
        lab_wav_w = lab_wav_w[onset_wav:offset_wav]  # lab_wav
        lab_wav_w = lab_wav_w[None, :]
        mix_wav_w = mix_wav_w[onset_wav:offset_wav] # mix wav
        mix_wav_w = mix_wav_w[None, :]

        loss_sisnr_w += sisnr_loss(est_wav_w, lab_wav_w)  # weighted sisnr
        sisnrI_w += sisnri(est_wav_w, lab_wav_w, mix_wav_w) # inmprovemnt

    loss_sisnr_w = loss_sisnr_w / sample_num
    sisnrI_w = sisnrI_w / sample_num
    loss_sisnr_all = sisnr_loss(est_wav, lab_wav)
    sisnrI_all = sisnri(est_wav, lab_wav, mix_wav)

    return loss_sisnr_w, sisnrI_w, loss_sisnr_all, sisnrI_all


def save_plot(tensor, savepath):
    tensor = tensor.squeeze().cpu()
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()


def save_audio(file_path, sampling_rate, audio):
    audio = np.clip(audio.cpu().squeeze().numpy(), -0.999, 0.999)
    wavfile.write(file_path, sampling_rate, (audio * 32767).astype("int16"))


def minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, vmin, vmax)
    tensor = 2 * (tensor - vmin) / (vmax - vmin) - 1
    return tensor


def reverse_minmax_norm_diff(tensor: torch.Tensor, vmax: float = 2.5, vmin: float = -12) -> torch.Tensor:
    tensor = torch.clip(tensor, -1.0, 1.0)
    tensor = (tensor + 1) / 2
    tensor = tensor * (vmax - vmin) + vmin
    return tensor