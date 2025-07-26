# In file: models/itransformer.py
import torch
import torch.nn as nn

class iTransformer(nn.Module):
    def __init__(self, seq_len, num_features, num_classes, d_model=64, n_heads=4, d_ff=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(num_features * d_model, num_classes)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        return self.fc(x)