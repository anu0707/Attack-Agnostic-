# Speaker Aware Deepfake Detectors

The code in this repository is the official implementation of the paper "Beyond Attacks: Advancing Fake Speech Detection with Attack-Agnostic Method" accepted at *INTERSPEECH 2025*. The link to the paper will be made available soon.

This repository provides an attack-agnoistic framework by utlizing two modules for detecting speech deepfakes.

## Getting started
First, clone the repository locally

```bash
git clone https://github.com/shilpac131/AttackAgnostic.git
cd AttackAgnostic
```

The models are trained on the logical access (LA) train  partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The models are tested on the LA and DF partion of ASVspoof 2021 and also on dataset created by us *IndicTTS*.


### Training

## 1. To train the code first you have to trained the Attack-Invariant Encoder-Decoder (AIED) AIED submodule:
```bash
python main_w2v2_AASIST_AIED_L5_noMean_hugface.py
```

## 2. After this, the entire framework can be trained using:

```bash
python main_w2v2_AASIST_AIED_CSD_hugface.py --track=LA --lr=0.000001 --batch_size=14 --loss=WCE  
```







