import torch
import os
from utils import save_audio, get_loss
from tqdm import tqdm
import shutil
import numpy as np

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@torch.no_grad()
def eval_ddim(unet, autoencoder, scheduler, eval_loader, args, device, epoch=0, 
              uncond_path=None,
              guidance_scale=False, guidance_rescale=0.0,
              ddim_steps=50, eta=1, 
              random_seed=2024,):
    # todo: might need to add cfg
    # add load uncond embedding here
    # if guidance_scale:
    #     uncond = torch.load
    
    if random_seed is not None:
        generator = torch.Generator(device=device).manual_seed(random_seed)
    else:
        generator = torch.Generator(device=device)
        generator.seed()
        
    unet.eval()

    for step, (mixture, target, timbre, mix_id, mixture_path, source_path, enroll_path) in enumerate(tqdm(eval_loader)):
        mixture = mixture.to(device)
        target = target.to(device)
        timbre = timbre.to(device)

        # init noise
        noise = torch.randn(mixture.shape, generator=generator, device=device)
        pred = noise

        for t in scheduler.timesteps:
            pred = scheduler.scale_model_input(pred, t)
            if guidance_scale:
                uncond = torch.tensor(np.load(uncond_path)['arr_0']).unsqueeze(0).to(device)
                pred_combined = torch.cat([pred, pred], dim=0)
                mixture_combined = torch.cat([mixture, mixture], dim=0)
                timbre_combined = torch.cat([timbre, uncond], dim=0)
                output_combined = unet(x=pred_combined, timesteps=t, mixture=mixture_combined, timbre=timbre_combined)
                output_pos, output_neg = torch.chunk(output_combined, 2, dim=0)
    
                model_output = output_neg + guidance_scale * (output_pos - output_neg)
                if guidance_rescale > 0.0:
                    # avoid overexposed
                    model_output = rescale_noise_cfg(model_output, output_pos,
                                                     guidance_rescale=guidance_rescale)
            else:
                model_output = unet(x=pred, timesteps=t, mixture=mixture, timbre=timbre)
            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred,
                                  eta=eta, generator=generator).prev_sample

        pred_wav = autoencoder(embedding=pred)

        os.makedirs(f'{args.log_dir}/audio/{epoch}/', exist_ok=True)

        for j in range(pred_wav.shape[0]):
            shutil.copyfile(mixture_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_mixture.wav')
            shutil.copyfile(source_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_source.wav')
            shutil.copyfile(enroll_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_enroll.wav')
            save_audio(f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}.wav', 24000, pred_wav[j].unsqueeze(0))
