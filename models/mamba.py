import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional

class SelectiveSSM(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        self.dt_rank = dt_rank or math.ceil(self.d_model / 16)
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )
        
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seqlen, dim = hidden_states.shape
        
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seqlen]
        x = rearrange(x, "b d l -> b l d")
        
        x = F.silu(x)
        
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        dt = self.dt_proj.weight @ dt.transpose(-1, -2)
        dt = rearrange(dt, "d (b l) -> b l d", l=seqlen)
        dt = dt + self.dt_proj.bias
        dt = F.softplus(dt)
        
        A = -torch.exp(self.A_log.float())
        
        y = self.selective_scan(
            x, dt, A, B, C, self.D.float()
        )
        
        y = y * F.silu(z)
        
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        batch, seqlen, dim = u.shape
        n = A.shape[1]
        
        deltaA = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        deltaB_u = torch.einsum("bld,bln,bld->bldn", delta, B, u)
        
        x = torch.zeros((batch, dim, n), device=u.device, dtype=u.dtype)
        ys = []
        
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = torch.einsum("bdn,bn->bd", x, C[:, i])
            ys.append(y)
        
        y = torch.stack(ys, dim=1)
        y = y + u * D
        
        return y


class MambaBlock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mixer = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        x = self.dropout(x)
        x = residual + x
        return x


class Mamba(nn.Module):
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_layers: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embedding = nn.Linear(n_features, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        
        x = x.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        
        x = x.mean(dim=1)
        
        return self.head(x).squeeze(-1)
