import numpy as np
import torchaudio
import torch
import torch.nn as nn
from feature_extractors.panns import Cnn14
from tqdm import tqdm


"""
wrap different encoder in following format to make code consistent and neat
"""


class CNN14_wrapper():
    def __init__(self, sr, device, feature_key='2048'):
        # FD, KL
        print('CNN14 is used for FD and KL')

        key_usage = {'2048': '2048 feature for FD',
                     'logits': 'logits feature for KL'}
        print(key_usage[feature_key])

        self.sampling_rate = sr
        self.device = device
        self.feature_key = feature_key
        features_list = ["2048", "logits"]
        if self.sampling_rate == 16000:
            self.model = Cnn14(
                features_list=features_list,
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            ).to(device).eval()
        elif self.sampling_rate == 32000:
            self.model = Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            ).to(device).eval()
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )

    # load wav as mono channel
    def load_wav_resample(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                      new_freq=self.sampling_rate)(waveform)
            sample_rate = self.sampling_rate
        # Convert to mono by averaging the channels if the waveform has more than one channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    @torch.no_grad()
    def extract(self, audio_path, text=None):
        audio, sr = self.load_wav_resample(audio_path)
        audio = audio.to(self.device)
        a_feature = self.model(audio)
        t_feature = None
        return a_feature[self.feature_key], t_feature


class Vggish_wrapper():
    def __init__(self, sr, device, use_pca=False, use_activation=False):
        # only for FAD
        print('Vggish is used for FAD')
        assert sr == 16000
        self.sampling_rate = sr
        self.device = device
        model = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.model = model.to(device)
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()

    # load wav as mono channel
    def load_wav_resample(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                      new_freq=self.sampling_rate)(waveform)
            sample_rate = self.sampling_rate
        # Convert to mono by averaging the channels if the waveform has more than one channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    @torch.no_grad()
    def extract(self, audio_path, text=None):
        audio, sr = self.load_wav_resample(audio_path)
        assert sr == 16000
        audio = audio.numpy()[0]
        a_feature = self.model(audio, self.sampling_rate)
        t_feature = None
        return a_feature, t_feature


class LaionClap_wrapper():
    def __init__(self, sr, device, model_id=-1):
        import laion_clap
        assert sr == 48000
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt(model_id=model_id)
        self.model = model.to(device).eval()

    @torch.no_grad()
    def extract(self, audio_path, text):
        audio_path = [audio_path]
        a_feature = self.model.get_audio_embedding_from_filelist(x=audio_path,
                                                                 use_tensor=True)
        t_feature = self.model.get_text_embedding([text],
                                                  use_tensor=True)
        return a_feature, t_feature


class Clap_wrapper():
    def __init__(self, sr, device, model_name='laion/larger_clap_general'):
        from transformers import AutoProcessor, ClapModel
        assert sr == 48000
        self.sampling_rate = sr
        self.model = ClapModel.from_pretrained(model_name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    def load_wav_resample(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sampling_rate:
            # First resample to 16k
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            # Then resample to the target sample rate
            waveform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.sampling_rate)(waveform)
            sample_rate = self.sampling_rate
        # Convert to mono by averaging the channels if the waveform has more than one channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    @torch.no_grad()
    def extract(self, audio_path, text):
        audio, sr = self.load_wav_resample(audio_path)
        text_inputs = self.processor(
            text=[text],
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors="pt")

        audio_inputs = self.processor(
            audios=[audio.squeeze().numpy()],
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True)

        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "input_features": audio_inputs["input_features"]}
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        outputs = self.model(**inputs)
        cos = outputs.logits_per_audio/self.model.logit_scale_a.exp()
        a_feature = torch.diag(cos)

        return a_feature, None


class MSClap_wrapper():
    def __init__(self, sr, device):
        raise NotImplementedError


class FeatureExtractor():
    def __init__(self, sr, backbone, device='cuda', **kwargs):
        self.sr = sr
        self.device = device
        self.backbone = backbone
        if backbone == "cnn14":
            self.feature_model = CNN14_wrapper(self.sr, device, **kwargs)
        elif backbone == 'vggish':
            self.feature_model = Vggish_wrapper(sr, device, **kwargs)
        elif backbone == 'laionclap':
            self.feature_model = LaionClap_wrapper(sr, device, **kwargs)
        elif backbone == 'clap':
            self.feature_model = Clap_wrapper(sr, device, **kwargs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def extract_features(self, meta, base_folder=None):
        if base_folder is None:
            base_folder = ''
        audio_features = []
        text_features = []
        for i in tqdm(range(len(meta))):
            row = meta.iloc[i]
            audio_path = f"{base_folder}/{row['audio_path']}"
            caption = row['caption']
            # wrap all backbone model to a feature extraction class
            audio_feature, text_feature = self.feature_model.extract(audio_path, caption)
            audio_features.append(audio_feature.cpu().numpy())
            if text_feature is not None:
                # only used text feature for clap score
                text_features.append(text_feature.cpu().numpy())

        audio_features = np.concatenate(audio_features, axis=0)
        if len(text_features) != 0:
            text_features = np.concatenate(text_features, axis=0)

        return audio_features, text_features
