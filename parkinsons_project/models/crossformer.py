import torch
import torch.nn as nn
from einops import rearrange

class CrossFormer(nn.Module):
    def __init__(self, seq_len, num_features, num_classes, d_model=64, n_heads=4, d_ff=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(seq_len * num_features * d_model, num_classes)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = rearrange(x, 'b t d -> b t d 1')
        x = self.embedding(x)
        x = rearrange(x, 'b t d m -> b (t d) m')
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        return self.fc(x)