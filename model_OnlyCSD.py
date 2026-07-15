import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

'''CSD code'''

class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, domain_size, K):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120,hidden_size)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size) 
     
        # CSD layer parameters
        self.sms = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K+1, hidden_size, num_classes], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.sm_biases = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K+1, num_classes], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.embs = torch.nn.Parameter(torch.normal(mean=0., std=1e-4, size=[domain_size, K], dtype=torch.float, device='cuda:0'), requires_grad=True)
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda:0'), requires_grad=True)

    def csd(self, x, labels, domains, num_classes, num_domains, K):
        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        logits_common = torch.matmul(x, w_c) + b_c    
        domains = torch.nn.functional.one_hot(domains, num_domains)
        domains = domains.to(dtype=torch.float)
        c_wts = torch.matmul(domains, self.embs)  
 
        # B x K
        batch_size = x.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), dtype=torch.float, device='cuda:0') * self.cs_wt, c_wts), 1)
        c_wts = torch.tanh(c_wts)
        w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d

        # criterion = nn.CrossEntropyLoss()
        weight = torch.FloatTensor([0.1, 0.9]).to('cuda:0')
        criterion = nn.CrossEntropyLoss(weight=weight)

        specific_loss = criterion(logits_specialized, labels)
        class_loss = criterion(logits_common, labels)
        sms = self.sms
        # create a stack of identity matrices on the GPU with dimensions (K+1) x (K+1)
        diag_tensor = torch.stack([torch.eye(K+1) for _ in range(num_classes)], dim=0).to('cuda:0')
        cps = torch.stack([torch.matmul(sms[:, :, _], torch.transpose(sms[:, :, _], 0, 1)) for _ in range(num_classes)], dim=0).to('cuda:0')
        orth_loss = torch.mean((cps - diag_tensor)**2).to('cuda:0')  ##########
        loss = class_loss + specific_loss + orth_loss
        
        return loss, logits_common,class_loss, specific_loss, orth_loss

    def forward(self, x, labels, domain):
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.layer_norm(x)

        # Using the CSD layer
        # Assuming you have labels and num_domains defined elsewhere
        #labels = torch.zeros_like(domain)  # Replace this with your actual labels tensor
        num_domains = 7 # Replace this with the actual number of domains in your data
        loss, logits_common, class_loss, specific_loss, orth_loss= self.csd(x, labels, domain, num_classes=2, num_domains=num_domains, K=4)
        return loss, logits_common,class_loss, specific_loss, orth_loss
