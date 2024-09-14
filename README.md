<p align="center">
  <img src="demo/soloaudio.webp" alt="SoloAudio" width="300" height="300" style="max-width: 100%;">
</p>


[![Paper](https://img.shields.io/badge/arXiv-24xx.xxxxx-brightgreen.svg?style=flat-square)](https://arxiv.org/)  [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/) [![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/westbrook/SoloAudio)  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/)  [![Demo page](https://img.shields.io/badge/Audio_Samples-blue?logo=Github&style=flat-square)](https://wanghelin1997.github.io/SoloAudio-Demo/)

Official Pytorch implementation of the paper: SoloAudio: Target Sound Extraction with Language-oriented Audio Diffusion Transformer.


## TODO
- [x] Release model weights
- [ ] Release data
- [ ] HuggingFace Spaces demo
- [ ] arxiv paper


## Environment setup
```bash
conda env create -f env.yml
conda activate soloaudio
```

## Pretrained Models

Download our pretrained models from [huggingface](https://huggingface.co/westbrook/SoloAudio).

After downloading the files, put them under this repo, like:
```
SoloAudio/
    -config/
    -demo/
    -pretrained_models/
    ....
```

<!-- ## Gradio
### Run in colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1g4-Oqd1Fu9WfDFb-nicfxqsWIPvsGb91?usp=sharing)

### Run locally
After environment setup install additional dependencies:
```bash
apt-get install -y espeak espeak-data libespeak1 libespeak-dev
apt-get install -y festival*
apt-get install -y build-essential
apt-get install -y flac libasound2-dev libsndfile1-dev vorbis-tools
apt-get install -y libxml2-dev libxslt-dev zlib1g-dev
pip install -r gradio_requirements.txt
```

Run gradio server from terminal:
```bash
python gradio_app.py
```
It is ready to use on [default url](http://127.0.0.1:7860).

### How to use it
1. (optionally) Select models
2. Load models
3. Transcribe
4. Align
5. Run -->

## Inference examples
For audio-oriented TSE, please run:

```bash
python test_audioTSE.py --output_dir './output-audioTSE/' --mixture './demo/1_mix.wav' --enrollment './demo/1_enrollment.wav'
```

For language-oriented TSE, please run:

```bash
python test_languageTSE.py --output_dir './output-languageTSE/' --mixture './demo/1_mix.wav' --enrollment 'Acoustic guitar'
```


## Data Preparation
To train a SoloAudio model, you need to prepare the following parts:
1. Prepare the FSD-Mix DataSet, please run:
```bash
cd data_preparating/
python create_filenames.py
python create_fsdmix.py
```

2. Prepare the TangoSyn DataSet, please run:
```bash
cd tango/
sh gen.sh
```

3. Prepare the TangoSyn-Mix DataSet like step 1.

4. Extract the VAE features, please run:

```bash
python extract_vae.py --data_dir "YOUR_DATA_DIR" --output_dir "YOUR_OUTPUT_DIR"
```

5. Extract the CLAP features, please run:

```bash
python extract_clap_audio.py --input_base_dir "YOUR_DATA_DIR" --output_base_dir "YOUR_OUTPUT_DIR"
python extract_clap_text.py --input_base_dir "YOUR_DATA_DIR" --output_base_dir "YOUR_OUTPUT_DIR" --split 1
python extract_clap_text.py --input_base_dir "YOUR_DATA_DIR" --output_base_dir "YOUR_OUTPUT_DIR" --split 2
python extract_clap_text.py --input_base_dir "YOUR_DATA_DIR" --output_base_dir "YOUR_OUTPUT_DIR" --split 3
```

## Training

Now, you are good to start training!

1. Train with a single GPU, please run:
```bash
python train.py
```

2. Train with multiple GPUs, please run:
```bash
accelerate launch train.py
```

## Training

Now, you are good to start training!

1. Train with a single GPU, please run:
```bash
python train.py
```

2. Train with multiple GPUs, please run:
```bash
accelerate launch train.py
```

## Test
To test a folder of audio files, please run:

```bash
python test_audioTSE.py --output_dir './test-audioTSE/' --test_dir '/YOUR_PATH_TO_TEST/'
```

OR

```bash
python test_languageTSE.py --output_dir './test-languageTSE/' --test_dir '/YOUR_PATH_TO_TEST/'
```

To calculate the metrics used in the paper, please run:
```bash
cd metircs/
python main.py
```

## License
The codebase is under [MIT LICENSE](./LICENSE). 



## Citations
```
@article{helin2024soloaudio,
  author    = {Wang, Helin and Hai, Jiarui and Lu, Yen-Ju and Thakkar, Karan and Elhilali, Mounya and Dehak, Najim},
  title     = {SoloAudio: Target Sound Extraction with Language-oriented Audio Diffusion Transformer},
  journal   = {arXiv},
  year      = {2024},
}

@INPROCEEDINGS{jiarui2024dpmtse,
  author={Hai, Jiarui and Wang, Helin and Yang, Dongchao and Thakkar, Karan and Dehak, Najim and Elhilali, Mounya},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={DPM-TSE: A Diffusion Probabilistic Model for Target Sound Extraction}, 
  year={2024},
  pages={1196-1200},
  }

```
