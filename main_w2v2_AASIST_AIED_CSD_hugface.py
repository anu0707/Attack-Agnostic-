import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
import numpy as np
import torch.nn as nn
from data_utils_CSD import genSpoof_list,Dataset_ASVspoof2021_eval,Dataset_ASVspoof2019_train_DA
from tqdm import tqdm
from autoencoder_model import (Autoencoder) #model2
from model_OnlyAASIST import Model #model1
from model_OnlyCSD import SimpleDNN #model3
import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Model
from tqdm import tqdm
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
'''to train w2v2-> AAISST-> AIED-> CSD+DNN| Interspeech code'''
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def combined_model(x, labels, domain):
        x = model1(x)
        x = model2(x)  # Frozen
        x = model3(x, labels, domain)
        return x

def produce_evaluation_file_2021(dataset, w2v2, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, num_workers=8, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    w2v2.eval()
    model1.eval()
    model2.eval() 
    model3.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x, utt_id,batch_domain in tqdm(data_loader):
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_domain = batch_domain.to(device)
        # zero = torch.tensor(0, device='cuda:0')
        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)  # This will return hidden states from all layers
        hidden_states_x = outputs_x.hidden_states  # List of hidden states, one for each layer
        fifth_layer_embeddings_x = hidden_states_x[4]  # Shape: [batch_size, seq_length, hidden_size]

        _,outputs,_,_,_ = combined_model(fifth_layer_embeddings_x,batch_domain,batch_domain)
        
        batch_score = (outputs[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(
    trn_loader: DataLoader,
    w2v2,
    model,
    optim,
    device: torch.device,):
    """Train the model for one epoch"""
    running_loss = 0
    classL = 0
    specificL = 0
    orthL = 0
    totalL = 0
    correct = 0
    total = 0
    num_total = 0.0

    w2v2.eval()
    model1.train()
    model2.eval() 
    model3.train()
    for batch_x, batch_y, batch_domain in (trn_loader):
        batch_size = batch_x.size(0) 
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_domain = batch_domain.to(device)

        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)  # This will return hidden states from all layers
        hidden_states_x = outputs_x.hidden_states  # List of hidden states, one for each layer
        fifth_layer_embeddings_x = hidden_states_x[4]  # Shape: [batch_size, seq_length, hidden_size]
        loss, predicted_one_hot,class_Loss, specific_Loss, orth_Loss = combined_model(fifth_layer_embeddings_x, batch_y, batch_domain)
        totalL += loss.item()
        classL += class_Loss.item()
        specificL += specific_Loss.item()
        orthL +=orth_Loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Compute training accuracy
        total += batch_y.size(0)
        predicted_int = torch.argmax(predicted_one_hot, dim=1)
        correct += (predicted_int == batch_y).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    print(f'Train Accuracy: {train_accuracy:.2%}')

    
    return train_loss

def evaluate_accuracy(dev_loader, w2v2, model, device):
    val_running_loss = 0.0
    num_total = 0.0
    total = 0
    correct = 0
    w2v2.eval()
    model1.eval()
    model2.eval() 
    model3.eval()
    for batch_x, batch_y, batch_domain in (dev_loader):
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_domain = batch_domain.to(device)

        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)  # This will return hidden states from all layers
        hidden_states_x = outputs_x.hidden_states  # List of hidden states, one for each layer
        fifth_layer_embeddings_x = hidden_states_x[4]  # Shape: [batch_size, seq_length, hidden_size]
        loss, predicted,_,_,_ = combined_model(fifth_layer_embeddings_x, batch_y, batch_domain)
        val_running_loss += loss.item()
        
    val_loss = val_running_loss / len(dev_loader)

    return val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main AIED-CSD (Attack agnoistic) system")
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument('--model_path', type=str,
                        default="./models/huggface_LA_freeze/w2v2_AASIST/best_model.pth", help='Model checkpoint')
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF','ITW'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default="./eval_output/w2v2_AASIST_AIED_CSD.txt",
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--resume', type=str, default='./w2v2_AASIST_AIED_CSD_DNN/best_model.pth', help='path to checkpoint to resume training from')
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

    print("w2v2-> AASIST-> AIED-> CSD")
    # make experiment reproducible

    # Example training loop
    database_path = Path("./datasets/LA/")

    # set device
    device = torch.device('cuda')
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    if args.eval:
        '''loading models'''
        model_name = 'facebook/wav2vec2-xls-r-300m'  # You can replace with 'wav2vec2-large-xlsr-53' if needed
        w2v2 = Wav2Vec2Model.from_pretrained(model_name).to(device)  # Load model
        configuration = w2v2.config
        print("w2v2 model loaded")

        ## load model 1 aasist
        model1 = Model(args,device)
        model1 =model1.to(device)

        # define model2 AIED
        model2 = Autoencoder(input_dim=160, hidden_dim1=128).to(device)

        # define CSD+DNN model 3
        model3 = SimpleDNN(input_size=128, hidden_size=80, num_classes=2, domain_size=7, K=4).to(device)

        # Load the checkpoint
        checkpoint = torch.load("./w2v2_AASIST_AIED_CSD_DNN/best_model.pth")

        # Restore model and optimizer states
        model1.load_state_dict(checkpoint['model1_state_dict'])
        model2.load_state_dict(checkpoint['model2_state_dict'])
        model3.load_state_dict(checkpoint['model3_state_dict'])
        print("all models loaded")


        print(f"doing evaluation of ASVspoof 2021 {args.track}")
        if args.track == "LA":
            file_eval = genSpoof_list(dir_meta = "./datasets/ASVspoof2021/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt",
    is_train=False,is_eval=False)
            print('no. of eval trials',len(file_eval))
            eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir ="./datasets/ASVspoof2021/ASVspoof2021_LA_eval/")
            produce_evaluation_file_2021(eval_set, w2v2, combined_model, device, args.eval_output)

        elif args.track == "DF":
            file_eval = genSpoof_list(dir_meta = "./datasets/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt",
    is_train=False,is_eval=False)
            print('no. of eval trials',len(file_eval))
            print(f"writing in {args.eval_output} file")
            eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir ="./datasets/ASVspoof2021_DF_eval/")
            produce_evaluation_file_2021(eval_set, w2v2, combined_model, device, args.eval_output)

        else:
            print("enter proper track: either LA/DF")
        sys.exit(0)

    '''dataloader '''     
    # define train dataloader
    d_label_trn,file_train,domain_train= genSpoof_list( dir_meta = "./datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train_DA(args,list_IDs = file_train,labels = d_label_trn,domains = domain_train,is_set = "train",algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    print(f"train_set view:> {train_set[7802]}")
    del train_set,d_label_trn

    # define development dataloader
    d_label_dev,file_dev,domain_dev = genSpoof_list(dir_meta = "./datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",is_train=True,is_eval=False)
    print('no. of validation trials',len(file_dev))
    dev_set = Dataset_ASVspoof2019_train_DA(args,list_IDs = file_dev,labels = d_label_dev,domains = domain_dev,is_set = "dev",algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev
    
    print("dataloader done")
    '''training on dataloader with already xlsr-53'''

    ## 'facebook/wav2vec2-large-960h'
    model_name = 'facebook/wav2vec2-xls-r-300m'  # You can replace with 'wav2vec2-large-xlsr-53' if needed
    w2v2 = Wav2Vec2Model.from_pretrained(model_name).to(device)  # Load model
    configuration = w2v2.config
    print("w2v2 model loaded")

    ## load model 1 aasist
    model1 = Model(args,device)
    model1 =model1.to(device)
    if args.model_path:
        model1.load_state_dict(torch.load(args.model_path,map_location=device))
        print('[MODEL 1] ASSIST model loaded : {}'.format(args.model_path))

    # define model2 AIED
    model2 = Autoencoder(input_dim=160, hidden_dim1=128).to(device)
    model2.load_state_dict(torch.load("./models/w2v2_AASIST_AIED/best.pth",map_location=device))
    for param in model2.parameters():
        param.requires_grad = False
    print("[MODEL 2] AIED model loaded: ./models/w2v2_AASIST_AIED/best.pth")

    # define CSD+DNN model 3
    model3 = SimpleDNN(input_size=128, hidden_size=80, num_classes=2, domain_size=7, K=4).to(device)

    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model3.parameters()),lr=args.lr,weight_decay=args.weight_decay)


    print("begin training")
    num_epochs = 200
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model1.load_state_dict(checkpoint['model1_state_dict'])
        model3.load_state_dict(checkpoint['model3_state_dict'])
        # model2 (AIED) is frozen/pretrained separately — no need to overwrite it,
        # but loading it too keeps everything consistent with what was actually running
        if 'model2_state_dict' in checkpoint:
            model2.load_state_dict(checkpoint['model2_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint.get('epoch', -1) + 1
        print(f"Resumed at epoch {start_epoch}")
    else:
        print("No valid resume checkpoint found — starting from scratch.")
    # Define the directory where you want to save the models
    model_save_path = './w2v2_AASIST_AIED_CSD_DNN'
    os.makedirs(model_save_path, exist_ok=True)
    best_model_path = os.path.join(model_save_path, 'best_model.pth')  # Path for the best model
    best_val_loss = 5  # Initialize to infinity

    for epoch in range(100, num_epochs):
        
        running_loss = train_epoch(train_loader,w2v2,combined_model,optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, w2v2,combined_model, device)
        print('\nEpoch: {} - Train loss: {} - Val_loss:{} '.format(epoch,
                                                   running_loss,val_loss))

        save_path = os.path.join(model_save_path, 'epoch_{}.pth'.format(epoch))
        # Save the state_dicts
        torch.save({
            'model1_state_dict': model1.state_dict(),
            'model2_state_dict': model2.state_dict(),
            'model3_state_dict': model3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,  # Save the epoch number
        }, save_path)

        # Check if the current validation loss is the best we've seen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model
            torch.save({
            'model1_state_dict': model1.state_dict(),
            'model2_state_dict': model2.state_dict(),
            'model3_state_dict': model3.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, best_model_path)