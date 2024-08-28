import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms


class LogMelSpectrogram(torch.nn.Module):
    def __init__(self, sr=16000, frame_length=1024, hop_length=160, n_mel=64, f_min=0, f_max=8000,
                 mel_length=1024):
        super().__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.mel = transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=frame_length,
            win_length=frame_length,
            hop_length=hop_length,
            center=False,
            power=1.0,
            norm="slaney",
            n_mels=n_mel,
            mel_scale="slaney",
            f_min=f_min,
            f_max=f_max
        )
        self.target_length = mel_length

    @torch.no_grad()
    def forward(self, x):
        x = F.pad(x, ((self.frame_length - self.hop_length) // 2,
                      (self.frame_length - self.hop_length) // 2), "reflect")
        mel = self.mel(x)

        logmel = torch.zeros(mel.shape[0], mel.shape[1], self.target_length).to(mel.device)
        logmel[:, :, :mel.shape[2]] = mel

        logmel = torch.log(torch.clamp(logmel, min=1e-5))
        return logmel