import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import random
import numpy as np
import torch.nn as nn
from data_utils_AIED import (genSpoof_list,Dataset_ASVspoof2019_train_DA)
from tqdm import tqdm
from autoencoder_model import (Autoencoder)
from model_OnlyAASIST import Model
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)

'''to train w2v2-> AAISST-> AIED'''

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def get_loader(
        database_path: str,
        seed: int,
        algo:int) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train"""

    trn_database_path = database_path / "ASVspoof2019_LA_train/"

    trn_list_path = "./metadata/AIED_pairs.txt"
    d_label_trn, file_train1, file_train2 = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train1))

    train_set = Dataset_ASVspoof2019_train_DA(args,list_IDs1=file_train1, list_IDs2=file_train2,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path,algo=algo)
    gen = torch.Generator()
    gen.manual_seed(seed)

    print(f"train_set view 1:> {train_set[0]}")
    print(f"train_set view 2:> {train_set[3]}")
    num_samples = 10000
    subset_indices = random.sample(range(len(train_set)), num_samples)
    subset_train_set = torch.utils.data.Subset(train_set, subset_indices)
    trn_loader = DataLoader(subset_train_set, batch_size=512, num_workers = 8, shuffle=True)

    return trn_loader

def train_epoch(
    trn_loader: DataLoader,
    w2v2,
    aasist,
    model,
    optim,
    device: torch.device,):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0

    w2v2.eval()
    aasist.eval()
    model.train() 
    criterion = nn.MSELoss()
    for batch_x1,batch_x2,label in tqdm((trn_loader)):
        # print(f"batch_x1: {batch_x1}")
        batch_size = batch_x1.size(0) 
        num_total += batch_size
        batch_x1 = batch_x1.to(device)
        batch_x2 = batch_x2.to(device)

        # Perform a forward pass through the model with output_hidden_states=True
        with torch.no_grad():
            outputs_x1 = w2v2(batch_x1, output_hidden_states=True)  # This will return hidden states from all layers
            outputs_x2 = w2v2(batch_x2, output_hidden_states=True)
        # Access the hidden states (i.e., features from each layer)
        hidden_states_x1 = outputs_x1.hidden_states  # List of hidden states, one for each layer
        hidden_states_x2 = outputs_x2.hidden_states
        # The fifth layer corresponds to index 4 (since the list is 0-indexed)
        fifth_layer_embeddings_x1 = hidden_states_x1[4]  # Shape: [batch_size, seq_length, hidden_size]
        fifth_layer_embeddings_x2 = hidden_states_x2[4]
        # print(f"Shape of the fifth layer embeddings: {fifth_layer_embeddings_x1.shape}")

        x1 = aasist(fifth_layer_embeddings_x1)
        x2 = aasist(fifth_layer_embeddings_x2)
        # Forward pass
        batch_out = model(x1)
        # # Compute loss
        batch_loss = criterion(batch_out, x2)
        running_loss += batch_loss.item() * batch_size
        # # Backward pass and optimization
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    running_loss /= num_total
    return running_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIAE system")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument('--model_path', type=str,
                        default="./w2v2_AASIST/best_model.pth", help='Model checkpoint')
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')
    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    args = parser.parse_args()

    print("MAIN AIED layer 5 WITH NO MEAN with DA and ontogo w2v2-AASIST")
    # make experiment reproducible

    # Example training loop
    database_path = Path("./datasets/LA")

    # set device
    device = torch.device('cuda')
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture - VAE
    model = Autoencoder(input_dim=160, hidden_dim1=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

    # define dataloaders
    trn_loader = get_loader(
        database_path, args.seed,args.algo)
    
    print("dataloader done")
    '''training on dataloader with already xlsr-53'''

    ## 'facebook/wav2vec2-large-960h'
    model_name = 'facebook/wav2vec2-xls-r-300m'  # You can replace with 'wav2vec2-large-xlsr-53' if needed
    w2v2 = Wav2Vec2Model.from_pretrained(model_name).to(device)  # Load model
    configuration = w2v2.config
    print("model loaded")

    ## load aasist
    aasist = Model(args,device)
    aasist =aasist.to(device)
    if args.model_path:
        aasist.load_state_dict(torch.load(args.model_path,map_location=device))
        print('ASSIST model loaded : {}'.format(args.model_path))

    print("begin training")
    num_epochs = 100
    best_loss = 17

    # Define the directory where you want to save the models
    model_save_dir = './models/w2v2_AASIST_AIED'
    os.makedirs(model_save_dir, exist_ok=True)

    # for epoch in tqdm(range(num_epochs)):
    for epoch in range(num_epochs):
        print("Start training epoch {:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, w2v2, aasist, model, optimizer, device)
        print("loss", running_loss)
    
        if running_loss < best_loss:
            best_loss = running_loss
            model_path = os.path.join(model_save_dir, f"best.pth")
            torch.save(model.state_dict(), model_path)
            print(f"Model saved as {model_path}")