import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=d_conv, padding=d_conv - 1, groups=d_model)
        
        # --- FIX 1: Correct the projection dimension for C ---
        self.x_proj = nn.Linear(d_model, d_state + d_state + d_state)
        self.dt_proj = nn.Linear(d_state, d_model)
        
        self.A = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        _, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :L]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        # --- FIX 2: Correct the split sizes ---
        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.d_state, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        
        y = self.ssm(x, dt, B, C)
        
        y = y * F.silu(z)
        return self.out_proj(y)

    # --- FIX 3: Correct the state-space model (SSM) calculation ---
    def ssm(self, u, delta, B, C):
        A = -torch.exp(self.A.float()) # (D, N)
        
        delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        delta_B_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)

        h = torch.zeros(u.size(0), self.d_model, self.d_state, device=u.device)
        ys = []
        for i in range(u.size(1)):
            h = delta_A[:, i] * h + delta_B_u[:, i]
            y = torch.einsum('bdn,bln->bld', h, C[:, i].unsqueeze(1))
            ys.append(y.squeeze(1))
        
        return torch.stack(ys, dim=1)

class Mamba(nn.Module):
    def __init__(self, seq_len, num_features, num_classes, d_model=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.fc(x)