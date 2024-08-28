import yaml
import random
import argparse
import os
import librosa
import soundfile as sf
from tqdm import tqdm
import time

import torch
import torchaudio
from torch.utils.data import DataLoader, ConcatDataset
# from torch.cuda.amp import autocast, GradScaler

from accelerate import Accelerator
from diffusers import DDIMScheduler

from model.udit import UDiT
from utils import save_plot, save_audio, minmax_norm_diff, reverse_minmax_norm_diff
from inference import eval_ddim
from dataset import TSEDataset
from vae_modules.autoencoder_wrapper import Autoencoder

parser = argparse.ArgumentParser()

# data loading settings
parser.add_argument('--train-base-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-train/wav24000/train')
parser.add_argument('--train-vae-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-train-vae/wav24000/train')
parser.add_argument('--train-timbre-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-train-clap/wav24000/train')
parser.add_argument('--val-base-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-val/wav24000/val')
parser.add_argument('--val-vae-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-val-vae/wav24000/val')
parser.add_argument('--val-timbre-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-val-clap/wav24000/val')
parser.add_argument('--test-base-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-test/wav24000/test')
parser.add_argument('--test-vae-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-test-vae/wav24000/test')
parser.add_argument('--test-timbre-dir', type=str, default='/export/corpora7/HW/TSEDataMix/fsd-test-clap/wav24000/test')

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
parser.add_argument('--diffusion-config', type=str, default='config/DiffTSE_udit_rotary_v_b_1000.yaml')
parser.add_argument('--autoencoder-path', type=str, default='/export/corpora7/HW/audio-vae/100k.pt')

# optimization
parser.add_argument('--learning-rate', type=float, default=1e-4)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--weight-decay', type=float, default=1e-4)

# log and random seed
parser.add_argument('--random-seed', type=int, default=2024)
parser.add_argument('--log-step', type=int, default=50)
parser.add_argument('--log-dir', type=str, default='logs/')
parser.add_argument('--save-dir', type=str, default='ckpt/')


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
    train_set3 = TSEDataset(
        base_dir=args.train_base_dir, 
        vae_dir=args.train_vae_dir, 
        timbre_dir=args.train_timbre_dir,
        tag="text2", 
        debug=args.debug
    )
    train_set4 = TSEDataset(
        base_dir=args.train_base_dir, 
        vae_dir=args.train_vae_dir, 
        timbre_dir=args.train_timbre_dir,
        tag="text3", 
        debug=args.debug
    )
    train_set = ConcatDataset([train_set1, train_set2, train_set3, train_set4])

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

    if args.v_prediction:
        print('v prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])
    else:
        print('noise prediction')
        noise_scheduler = DDIMScheduler(**args.diff_config["ddim"]['diffusers'])

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
            noise = torch.randn(target.shape).to(accelerator.device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],),
                                      device=accelerator.device,).long()
            noisy_target = noise_scheduler.add_noise(target, noise, timesteps)
            # v prediction - model output
            velocity = noise_scheduler.get_velocity(target, noise, timesteps)

            # inference
            pred = unet(x=noisy_target, timesteps=timesteps, mixture=mixture, timbre=timbre)

            # backward
            if args.v_prediction:
                loss = loss_func(pred, velocity)
            else:
                loss = loss_func(pred, noise)

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
