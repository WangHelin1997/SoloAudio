import torch
import os
from utils import save_audio, get_loss
from tqdm import tqdm
import shutil

@torch.no_grad()
def eval_ddim(unet, autoencoder, scheduler, eval_loader, args, device, epoch=0, ddim_steps=50, eta=1):
    # noise generator for eval

    generator = torch.Generator(device=device).manual_seed(args.random_seed)
    scheduler.set_timesteps(ddim_steps)
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

            
@torch.no_grad()
def eval_unipc(unet, autoencoder, scheduler, eval_loader, args, device, epoch=0, unipc_steps=10):
    # noise generator for eval

    generator = torch.Generator(device=device).manual_seed(args.random_seed)
    scheduler.set_timesteps(unipc_steps)
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
            model_output = unet(x=pred, timesteps=t, mixture=mixture, timbre=timbre)
            pred = scheduler.step(model_output=model_output, timestep=t, sample=pred).prev_sample

        pred_wav = autoencoder(embedding=pred)

        os.makedirs(f'{args.log_dir}/audio/{epoch}/', exist_ok=True)

        for j in range(pred_wav.shape[0]):
            shutil.copyfile(mixture_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_mixture.wav')
            shutil.copyfile(source_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_source.wav')
            shutil.copyfile(enroll_path[j], f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}_enroll.wav')
            save_audio(f'{args.log_dir}/audio/{epoch}/pred_{mix_id[j]}.wav', 24000, pred_wav[j].unsqueeze(0))
