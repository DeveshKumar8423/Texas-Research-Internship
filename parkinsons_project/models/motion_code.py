import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, d_embed, commitment_cost):
        super().__init__()
        self.d_embed = d_embed
        self.n_codes = n_codes
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.n_codes, self.d_embed)
        self.embedding.weight.data.uniform_(-1 / self.n_codes, 1 / self.n_codes)

    def forward(self, z):
        b, c, t = z.shape
        # --- THIS IS THE CORRECTED LINE ---
        z_flattened = z.reshape(-1, self.d_embed)
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_codes).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.commitment_cost * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        return z_q, loss

class MotionCode(nn.Module):
    def __init__(self, input_dim, num_classes, n_codes=128, d_model=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.encoder_rnn = nn.GRU(input_dim, d_model, n_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.encoder_fc = nn.Linear(d_model * 2, d_model)
        self.quantizer = VectorQuantizer(n_codes, d_model, commitment_cost=0.25)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        z, _ = self.encoder_rnn(x)
        z = torch.tanh(self.encoder_fc(z))
        z_permuted = z.permute(0, 2, 1)
        z_q, vq_loss = self.quantizer(z_permuted)
        z_q = z_q.permute(0, 2, 1)
        z_pooled = torch.mean(z_q, dim=1)
        logits = self.classifier(z_pooled)
        return logits