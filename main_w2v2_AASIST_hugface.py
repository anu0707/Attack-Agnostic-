import argparse
import sys
import warnings
import os
import random
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_final import genSpoof_list,Dataset_ASVspoof2021_eval,Dataset_ASVspoof2019_train_DA
from model_OnlyAASIST import Model
from core_scripts.startup_config import set_random_seed
import multiprocessing as mp
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Config
import glob
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


''' for w2v2 (frozen DA) - > AASIST {BASELINE} | in interspeech code'''
def produce_evaluation_file_2021(dataset, w2v2, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=14, num_workers = 10, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    
    for batch_x,utt_id in tqdm(data_loader):
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)
        # Access the hidden states (i.e., features from each layer)
        hidden_states_x = outputs_x.hidden_states  # List of hidden states, one for each layer
        # The fifth layer corresponds to index 4 (since the list is 0-indexed)
        fifth_layer_embeddings_x = hidden_states_x[4]  # Shape: [batch_size, seq_length, hidden_size]
        batch_out = model(fifth_layer_embeddings_x)
        
        batch_score = (batch_out[:, 1]  
                       ).data.cpu().numpy().ravel() 
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))



def produce_evaluation_file(
    eval_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:

    """Perform evaluation on 2019 data and save the score to a file"""
    model.eval()

    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []

    for batch_x, utt_id in eval_loader:
        
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        
        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)
        # Access the hidden states (i.e., features from each layer)
        hidden_states_x = outputs_x.hidden_states  # List of hidden states, one for each layer
        # The fifth layer corresponds to index 4 (since the list is 0-indexed)
        fifth_layer_embeddings_x = hidden_states_x[4]  # Shape: [batch_size, seq_length, hidden_size]
        batch_out = model(hidden_states_x)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        
        # add outputs

        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    # assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            # assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    total = 0
    correct = 0
    model.eval()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in (dev_loader):
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)
        # Access the hidden states (i.e., features from each layer)
        hidden_states_x = outputs_x.hidden_states  # List of hidden states, one for each layer
        # The fifth layer corresponds to index 4 (since the list is 0-indexed)
        fifth_layer_embeddings_x = hidden_states_x[4]  # Shape: [batch_size, seq_length, hidden_size]
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(fifth_layer_embeddings_x)
        
        batch_loss = criterion(batch_out, batch_y)
        val_loss += (batch_loss.item() * batch_size)

        ##New
    # val_accuracy = correct / total
        predicted_int = torch.argmax(batch_out, dim=1)
        total += batch_y.size(0)
        correct += (predicted_int == batch_y).sum().item()
    
    # val_accuracy = correct / total
        
    val_loss /= num_total
   
    return val_loss

def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in (train_loader):
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        with torch.no_grad():
            outputs_x = w2v2(batch_x, output_hidden_states=True)
        # Access the hidden states (i.e., features from each layer)
        hidden_states_x = outputs_x.hidden_states  # List of hidden states, one for each layer
        # The fifth layer corresponds to index 4 (since the list is 0-indexed)
        fifth_layer_embeddings_x = hidden_states_x[4]  # Shape: [batch_size, seq_length, hidden_size]

        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        batch_out = model(fifth_layer_embeddings_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss

if __name__ == '__main__':

    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='./datasets/LA/', help='Change this to user\'s full directory address of LA database (ASVspoof2019- for training & development (used as validation), ASVspoof2021 for evaluation scores). We assume that all three ASVspoof 2019 LA train, LA dev and ASVspoof2021 LA eval data folders are in the same database_path directory.')
    '''
    % database_path/
    %   |- LA
    %      |- ASVspoof2021_LA_eval/flac
    %      |- ASVspoof2019_LA_train/flac
    %      |- ASVspoof2019_LA_dev/flac

 
 
    '''

    parser.add_argument('--protocols_path', type=str, default='./datasets/LA/ASVspoof2019_LA_cm_protocols', help='Change with path to user\'s LA database protocols directory address')
    '''
    % protocols_path/
    %   |- ASVspoof_LA_cm_protocols
    %      |- ASVspoof2021.LA.cm.eval.trl.txt
    %      |- ASVspoof2019.LA.cm.dev.trl.txt 
    %      |- ASVspoof2019.LA.cm.train.trn.txt
  
    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default="./w2v2_AASIST/best_model.pth", help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF','ITW'], help='LA/PA/DF/ITW')
    parser.add_argument('--eval_output', type=str, default="./eval_output/w2v2_froz_DA_aasist_LA.txt",
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    ## NEW
    parser.add_argument('--partition', action='store_true', default="LA",
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation ======================================================================#

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
    

    # if not os.path.exists('models'):
    #     os.mkdir('models')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    track = args.track

    assert track in ['LA', 'PA','DF','ITW'], 'Invalid track given'

    #database
    prefix      = 'ASVspoof_{}'.format(track)
    prefix_2019 = 'ASVspoof2019.{}'.format(track)
    prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    '''main model w2v2 - AASIST '''
    model = Model(args,device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    ''' load saved model if any '''
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    model_name = 'facebook/wav2vec2-xls-r-300m'  # You can replace with 'wav2vec2-large-xlsr-53' if needed
    w2v2 = Wav2Vec2Model.from_pretrained(model_name).to(device)  # Load model
    configuration = w2v2.config
    print("w2v2 huggingface model loaded")
    #evaluation on 2021 LA
    if args.eval:
        print(f"doing evaluation of ASVspoof 2021 {args.track}")
        if args.track == "LA":
            file_eval = genSpoof_list(dir_meta = "./datasets/ASVspoof2021_LA_eval/ASVspoof2021.LA.cm.eval.trl.txt",
    is_train=False,is_eval=True)
            print('no. of eval trials',len(file_eval))
            eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir ="/home/pm_students/shilpa/datasets/ASVspoof2021_LA_eval/")
            produce_evaluation_file_2021(eval_set, w2v2, model, device, args.eval_output)

        if args.track == "DF":
            file_eval = genSpoof_list(dir_meta = "./datasets/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt",
    is_train=False,is_eval=True)
            print('no. of eval trials',len(file_eval))
            print(f"writing in {args.eval_output} file")
            eval_set=Dataset_ASVspoof2021_eval(list_IDs = file_eval,base_dir ="./datasets/ASVspoof2021_DF_eval/")
            produce_evaluation_file_2021(eval_set, w2v2, model, device, args.eval_output)
        
        
        else:
            print("enter proper track: either LA or DF")
        sys.exit(0)

    '''dataloader '''     
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta = "./datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_ASVspoof2019_train_DA(args,list_IDs = file_train,labels = d_label_trn,is_set = "train",algo=args.algo)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    print(f"train_set view:> {train_set[0]}")
    del train_set,d_label_trn


    # define validation dataloader
    d_label_dev,file_dev = genSpoof_list(dir_meta = "./datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_ASVspoof2019_train_DA(args,list_IDs = file_dev,labels = d_label_dev,is_set = "dev",algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    
    '''begin training'''

    print("begin training")
    model_save_path = './models/huggface_LA_freeze/w2v2_AASIST'
    os.makedirs(model_save_path, exist_ok=True)
    
    # Training and validation 
    num_epochs = args.num_epochs
    best_val_loss = 1  # Initialize to infinity
    best_model_path = os.path.join(model_save_path, 'best_model.pth')
    patience_counter = 0

    for epoch in range(num_epochs):      
        running_loss = train_epoch(train_loader, model, args.lr,optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, model, device)
        print('\nEpoch: {} - Train loss: {} - Val_loss:{} '.format(epoch,
                                                   running_loss,val_loss))
        # saving the model
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
        # Check if the current validation loss is the best we've seen
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)

        else:
            patience_counter += 1

        if patience_counter >= 7:
            print("Early stopping triggered, best model is epoch: ", epoch)
            break
