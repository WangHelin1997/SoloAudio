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

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--output_dir', type=str, default='./output/')
parser.add_argument('--mixture', type=str, default='./demo/1_mix.wav')
parser.add_argument('--enrollment', type=str, default='./demo/1_enrollment.wav')

# pre-trained model path
parser.add_argument('--autoencoder-path', type=str, default='./pretrained_models/audio-vae.pt')
parser.add_argument('--segment', type=int, default=3)
parser.add_argument('--vae_sr', type=int, default=50)

parser.add_argument("--num_infer_steps", type=int, default=50)
# model configs
parser.add_argument('--diffusion-config', type=str, default='./config/SoloAudio.yaml')
parser.add_argument('--diffusion-ckpt', type=str, default='./pretrained_models/soloaudio_v2.pt')

# log and random seed
parser.add_argument('--random-seed', type=int, default=2024)
args = parser.parse_args()

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)

args.v_prediction = args.diff_config["ddim"]["v_prediction"]

@torch.no_grad()
def sample_diffusion(args, unet, autoencoder, scheduler,
                     mixture, timbre, device, ddim_steps=50, eta=0, seed=2023):
    unet.eval()
    scheduler.set_timesteps(ddim_steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    # init noise
    noise = torch.randn(mixture.shape, generator=generator, device=device)
    pred = noise

    for t in tqdm(scheduler.timesteps):
        pred = scheduler.scale_model_input(pred, t)
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
        
    with torch.no_grad():
        mixture, _ = librosa.load(args.mixture, sr=24000)
        mixture = torch.tensor(mixture).unsqueeze(0).to(args.device)
        mixture = autoencoder(audio=mixture.unsqueeze(1))
        
        audio_sample, sample_rate = torchaudio.load(args.enrollment)
        if audio_sample.shape[0] > 1:
            audio_sample = torch.mean(audio_sample, dim=0, keepdim=True)
        if sample_rate != 48000:
            audio_sample = audio_sample
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)
            audio_sample = resampler(audio_sample)
        num_samples = audio_sample.shape[1]
        target_num_samples = 48000*10
        if num_samples > target_num_samples:
            audio_sample = audio_sample[:, :target_num_samples]
        elif num_samples < target_num_samples:
            padding = target_num_samples - num_samples
            audio_sample = torch.nn.functional.pad(audio_sample, (0, padding))
            
        audio_inputs = processor(
            audios=[audio_sample.squeeze().numpy()],
            sampling_rate=48000,
            return_tensors="pt",
            padding=True  # Pad audio to the required length, if necessary
        )
        inputs = {
            "input_features": audio_inputs["input_features"][0].unsqueeze(0)  # Audio features
        }
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        timbre = clapmodel.get_audio_features(**inputs)

    
    pred = sample_diffusion(args, unet, autoencoder, noise_scheduler, mixture, timbre, args.device, ddim_steps=args.num_infer_steps, eta=0, seed=args.random_seed)
    
    savename = args.mixture.split('/')[-1].split('.wav')[0]
    shutil.copyfile(args.mixture, f'{args.output_dir}/{savename}_mix.wav')
    shutil.copyfile(args.enrollment, f'{args.output_dir}/{savename}_enrollment.wav')
    save_audio(f'{args.output_dir}/{savename}_pred.wav', 24000, pred)

    print(f'the prediction is save to {savename}_pred.wav')
