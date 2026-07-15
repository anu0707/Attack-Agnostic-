import torch
import torch.nn as nn

'''AASIST - AIED'''

class Autoencoder(nn.Module):
    def __init__(self, input_dim=160, hidden_dim1=128):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim1, input_dim),
            nn.ReLU()
        )

    def forward(self, x1):
        # Encode x1 to a 128-dimensional vector
        encoded = self.encoder(x1)
        # Decode to produce x2
        x2 = self.decoder(encoded)
        return encoded
        # return encoded