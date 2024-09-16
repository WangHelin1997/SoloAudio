import yaml
import random
import argparse
import os
import torch
import librosa
from tqdm import tqdm
from diffusers import DDIMScheduler
from transformers import AutoProcessor, ClapModel, ClapProcessor
from model.udit import UDiT
from utils import save_audio
import shutil
from vae_modules.autoencoder_wrapper import Autoencoder
import torchaudio
import glob
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--output_dir', type=str, default='./test-languageTSE/')
parser.add_argument('--test_dir', type=str, default='/YOUR_PATH_TO_TEST/')
parser.add_argument('--meta_dir', type=str, default='/YOUR_PATH_TO_TEST/fsd_mix_test.csv')
# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/audio-vae.pt')
parser.add_argument('--uncond_path', type=str, default='./pretrained_models/uncond.npz')
parser.add_argument('--guidance_scale', type=float, default=3.0) # 2.5 for audio, 3.0 for text
parser.add_argument('--guidance_rescale', type=float, default=0.0)

parser.add_argument("--num_infer_steps", type=int, default=50)
# model configs
parser.add_argument('--diffusion-config', type=str, default='./config/SoloAudio.yaml')
parser.add_argument('--diffusion_ckpt', type=str, default='./pretrained_models/soloaudio_v2.pt')


# log and random seed
parser.add_argument('--random-seed', type=int, default=2024)
args = parser.parse_args()

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)

args.v_prediction = args.diff_config["ddim"]["v_prediction"]


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
def sample_diffusion(args, unet, autoencoder, scheduler,
                     mixture, timbre, device, ddim_steps=50, eta=0, seed=2023,
                     uncond_path=None, guidance_scale=False, guidance_rescale=0.0,):
    unet.eval()
    scheduler.set_timesteps(ddim_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
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

    pred = autoencoder(embedding=pred).squeeze(1)

    return pred



if __name__ == '__main__':

    os.makedirs(args.output_dir, exist_ok=True)
    clapmodel = ClapModel.from_pretrained("laion/larger_clap_general").to(args.device)
    processor = AutoProcessor.from_pretrained('laion/larger_clap_general')
    
    autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(args.device)

    unet = UDiT(
        **args.diff_config['diffwrap']['UDiT']
    ).to(args.device)
    unet.load_state_dict(torch.load(args.diffusion_ckpt)['model'])

    total = sum([param.nelement() for param in unet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    

    if args.v_prediction:
        print('v prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    else:
        print('noise prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    
    # these steps reset dtype of noise_scheduler params
    latents = torch.randn((1, 128, 128),
                          device=args.device)
    noise = torch.randn(latents.shape).to(latents.device)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                              (noise.shape[0],),
                              device=latents.device).long()
    _ = noise_scheduler.add_noise(latents, noise, timesteps)
    
    
    mixtures = glob.glob(os.path.join(args.test_dir, "mix_dir", "*.wav"))
    print(len(mixtures))
    references = [item.replace("mix_dir", "s1") for item in mixtures]
    
    df = pd.read_csv(args.meta_dir)
    filenames = list(df['output_filename'])
    texts = list(df['s1_text'])
    meta_data = {}
    for i in range(len(filenames)):
        meta_data[filenames[i].split('/')[-1]] = texts[i].replace('_', ' ').lower()
        
    for mix, ref in tqdm(zip(mixtures, references)):
        with torch.no_grad():
            mixture, _ = librosa.load(mix, sr=24000)
            mixture = torch.tensor(mixture).unsqueeze(0).to(args.device)
            mixture = autoencoder(audio=mixture.unsqueeze(1))
            
            text = meta_data[mix.split('/')[-1]]

            text_inputs = processor(
                text=[text],
                max_length=10,  # Fixed length for text
                padding='max_length',  # Pad text to max length
                truncation=True,  # Truncate text if it's longer than max length
                return_tensors="pt"
            )
            inputs = {
                "input_ids": text_inputs["input_ids"][0].unsqueeze(0),  # Text input IDs
                "attention_mask": text_inputs["attention_mask"][0].unsqueeze(0),  # Attention mask for text
            }
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            timbre = clapmodel.get_text_features(**inputs)


        pred = sample_diffusion(args, unet, autoencoder, noise_scheduler, mixture, timbre, args.device, ddim_steps=args.num_infer_steps, eta=0, seed=args.random_seed, uncond_path=args.uncond_path, guidance_scale=args.guidance_scale, guidance_rescale=args.guidance_rescale)

        savename = mix.split('/')[-1].split('.wav')[0]
        shutil.copyfile(mix, f'{args.output_dir}/{savename}_mix.wav')
        shutil.copyfile(ref, f'{args.output_dir}/{savename}_ref.wav')
        save_audio(f'{args.output_dir}/{savename}_pred.wav', 24000, pred)