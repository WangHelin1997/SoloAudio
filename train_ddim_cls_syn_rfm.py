import yaml
import random
import argparse
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import time
import copy

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
# from torch.cuda.amp import autocast, GradScaler

from accelerate import Accelerator
# trailing timestep support during inference
from solvers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

from model.udit import UDiT
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff
from inference_rfm import eval_ddim
from dataset import TSEDataset
from vae_modules.autoencoder_wrapper import Autoencoder

parser = argparse.ArgumentParser()

# data loading settings
parser.add_argument('--train-base-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-train-syn/wav24000/train_syn')
parser.add_argument('--train-vae-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-train-syn-vae/wav24000/train_syn')
parser.add_argument('--train-timbre-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-train-syn-clap/wav24000/train_syn')
parser.add_argument('--val-base-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-val-syn/wav24000/val_syn')
parser.add_argument('--val-vae-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-val-syn-vae/wav24000/val_syn')
parser.add_argument('--val-timbre-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-val-syn-clap/wav24000/val_syn')
parser.add_argument('--test-base-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-test-syn/wav24000/test_syn')
parser.add_argument('--test-vae-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-test-syn-vae/wav24000/test_syn')
parser.add_argument('--test-timbre-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-test-syn-clap/wav24000/test_syn')

parser.add_argument('--sample_rate', type=int, default=24000)
parser.add_argument('--debug', type=bool, default=False)

parser.add_argument("--num_infer_steps", type=int, default=50)

# training settings
parser.add_argument("--amp", type=str, default='fp16')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=16)
parser.add_argument('--num-threads', type=int, default=1)
parser.add_argument('--save-every', type=int, default=5)
parser.add_argument("--adam-epsilon", type=float, default=1e-08)

# model configs
parser.add_argument('--diffusion-config', type=str, default='config/DiffTSE_udit_rotary_v_b_rfm.yaml')
parser.add_argument('--autoencoder-path', type=str, default='/export/corpora7/HW/audio-vae/100k.pt')

# optimization
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2024)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--log-dir', type=str, default='logs_syn/')
parser.add_argument('--save-dir', type=str, default='ckpt_syn/')


args = parser.parse_args()

with open(args.diffusion_config, 'r') as fp:
    args.diff_config = yaml.safe_load(fp)


args.v_prediction = args.diff_config["ddim"]["v_prediction"]
args.log_dir = args.log_dir.replace('log', args.diff_config["system"] + '_log')
args.save_dir = args.save_dir.replace('ckpt', args.diff_config["system"] + '_ckpt')

if os.path.exists(args.log_dir + '/pic/gt') is False:
    os.makedirs(args.log_dir + '/pic/gt')

if os.path.exists(args.log_dir + '/audio/gt') is False:
    os.makedirs(args.log_dir + '/audio/gt')

if os.path.exists(args.save_dir) is False:
    os.makedirs(args.save_dir)


def get_sigmas(noise_scheduler, timesteps, n_dim=3, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


if __name__ == '__main__':
    # Fix the random seed
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Set device
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'

    train_set1 = TSEDataset(
        base_dir=args.train_base_dir, 
        vae_dir=args.train_vae_dir, 
        timbre_dir=args.train_timbre_dir,
        tag="audio", 
        debug=args.debug
    )
    train_set2 = TSEDataset(
        base_dir=args.train_base_dir, 
        vae_dir=args.train_vae_dir, 
        timbre_dir=args.train_timbre_dir,
        tag="text1", 
        debug=args.debug
    )
    train_set = ConcatDataset([train_set1, train_set2])

    train_loader = DataLoader(train_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, pin_memory=True, persistent_workers=True)

    # use this load for check generated audio samples
    eval_set = TSEDataset(
        base_dir=args.val_base_dir, 
        vae_dir=args.val_vae_dir, 
        timbre_dir=args.val_timbre_dir,
        tag="audio", 
        debug=args.debug
    )
    eval_loader = DataLoader(eval_set, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, pin_memory=True, persistent_workers=True)
    # use these two loaders for benchmarks

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp)

    unet = UDiT(
        **args.diff_config['diffwrap']['UDiT']
    ).to(accelerator.device)

    total = sum([param.nelement() for param in unet.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    
    autoencoder = Autoencoder(args.autoencoder_path, 'stable_vae', quantization_first=True)
    autoencoder.eval()
    autoencoder.to(accelerator.device)

    noise_scheduler = FlowMatchEulerDiscreteScheduler(**args.diff_config["ddim"]['diffusers'])
    noise_scheduler_eval = copy.deepcopy(noise_scheduler)

    optimizer = torch.optim.AdamW(unet.parameters(),
                                  lr=args.learning_rate,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay,
                                  eps=args.adam_epsilon,
                                  )
    loss_func = torch.nn.MSELoss()

    # scaler = GradScaler()
    # put to accelerator
    unet, autoencoder, optimizer, train_loader = accelerator.prepare(unet, autoencoder, optimizer, train_loader)

    global_step = 0
    losses = 0

    if accelerator.is_main_process:
        eval_ddim(unet, autoencoder, noise_scheduler, eval_loader, args, accelerator.device, epoch='test', ddim_steps=args.num_infer_steps, eta=0)
    accelerator.wait_for_everyone()

    for epoch in range(args.epochs):
        unet.train()
        for step, batch in enumerate(tqdm(train_loader)):
            # compress by vae
            mixture, target, timbre, _, _, _, _ = batch

            # adding noise
            noise = torch.randn_like(target)
            # flow matching velocity
            velocity = noise.clone() - target.clone()

            # Sample a random timestep for each image
            bsz = target.shape[0]

            # todo: sd3 said lognorm sampling is better than uniform
            # u = compute_density_for_timestep_sampling(
            #         weighting_scheme='logit_normal',
            #         batch_size=bsz,
            #         logit_mean=0.0,
            #         logit_std=1.0,
            #         mode_scale=None,
            #     )
            # indices = (u * noise_scheduler.config.num_train_timesteps).long()

            indices = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,))
            timesteps = noise_scheduler.timesteps[indices].to(device=target.device)

            # Add noise according to flow matching.
            sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=target.ndim, 
                                dtype=target.dtype)
            noisy_target = sigmas * noise + (1.0 - sigmas) * target

            # inference
            pred = unet(x=noisy_target, timesteps=timesteps, 
                        mixture=mixture, timbre=timbre)

            # it seems we dont need weighting
            # weighting = compute_loss_weighting_for_sd3(weighting_scheme='logit_normal', sigmas=sigmas)
            # weighting = torch.ones_like(sigmas)

            loss = loss_func(pred, velocity)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            losses += loss.item()

            if accelerator.is_main_process:
                if global_step % args.log_step == 0:
                    n = open(args.log_dir + 'ddim_cls_log.txt', mode='a')
                    n.write(time.asctime(time.localtime(time.time())))
                    n.write('\n')
                    n.write('Epoch: [{}][{}]    Batch: [{}][{}]    Loss: {:.6f}\n'.format(
                        epoch + 1, args.epochs, step+1, len(train_loader), losses / args.log_step))
                    n.close()
                    losses = 0.0

        if accelerator.is_main_process:
            eval_ddim(unet, autoencoder, noise_scheduler, eval_loader, args, accelerator.device, epoch=epoch+1, ddim_steps=args.num_infer_steps, eta=0)

            if (epoch + 1) % args.save_every == 0:
                accelerator.wait_for_everyone()
                unwrapped_unet = accelerator.unwrap_model(unet)
                accelerator.save({
                    "model": unwrapped_unet.state_dict(),
                }, args.save_dir+str(epoch)+'.pt')
        accelerator.wait_for_everyone()
