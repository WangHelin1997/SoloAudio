import torch
import torch.nn as nn
from diffusers import UNet1DModel
import yaml


class DiffTSE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.unet = UNet1DModel(**self.config['unet'])
        self.unet.set_use_memory_efficient_attention_xformers(True)

    def forward(self, x, t, mixture, timbre=None): #b,C
        # print(x.shape,t.shape,mixture.shape,timbre.shape)
        # b, N, t
        timbre_all = timbre.unsqueeze(2)#b,C,1
        timbre_all = torch.cat(x.shape[2]*[timbre_all], 2)#b,C,t
        x = torch.cat([x, mixture, timbre_all], dim=1)
        noise = self.unet(sample=x, timestep=t)['sample']

        return noise


if __name__ == "__main__":
    with open('/export/corpora7/HW/DPMTSE-main/src/config/DiffTSE_cls_v_b_1000.yaml', 'r') as fp:
        config = yaml.safe_load(fp)
    device = 'cuda'

    model = DiffTSE(config['diffwrap']).to(device)

    x = torch.rand((1, 128, 128)).to(device)
    t = torch.randint(0, 1000, (1, )).long().to(device)
    mixture = torch.rand((1, 128, 128)).to(device)
    timbre = torch.rand((1, 512)).to(device)

    y = model(x, t, mixture, timbre)
    print(y.shape)
