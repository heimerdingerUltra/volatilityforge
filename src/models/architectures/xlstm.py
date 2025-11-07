import torch
import torch.nn as nn
import torch.nn.functional as F


class sLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i = nn.Linear(input_size, hidden_size, bias=True)
        self.W_f = nn.Linear(input_size, hidden_size, bias=True)
        self.W_z = nn.Linear(input_size, hidden_size, bias=True)
        self.W_o = nn.Linear(input_size, hidden_size, bias=True)
        
        self.R_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.R_o = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.group_norm = nn.GroupNorm(1, hidden_size)
        
    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        h_prev, c_prev, n_prev, m_prev = state
        
        i = torch.sigmoid(self.W_i(x) + self.R_i(h_prev))
        f = torch.sigmoid(self.W_f(x) + self.R_f(h_prev))
        z = torch.tanh(self.W_z(x) + self.R_z(h_prev))
        o = torch.sigmoid(self.W_o(x) + self.R_o(h_prev))
        
        m = torch.max(f * m_prev, i)
        c = f * c_prev + i * z
        n = f * n_prev + i
        
        c_stabilized = c / n
        
        h = o * torch.tanh(self.group_norm(c_stabilized))
        
        return h, (h, c, n, m)


class mLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(input_size, hidden_size, bias=False)
        self.W_v = nn.Linear(input_size, hidden_size, bias=False)
        
        self.W_i = nn.Linear(input_size, hidden_size, bias=True)
        self.W_f = nn.Linear(input_size, hidden_size, bias=True)
        self.W_o = nn.Linear(input_size, hidden_size, bias=True)
        
        self.group_norm = nn.GroupNorm(1, hidden_size)
        
    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        C_prev, n_prev, m_prev = state
        batch_size = x.shape[0]
        
        q = self.W_q(x)  # (batch, hidden)
        k = self.W_k(x)  # (batch, hidden)
        v = self.W_v(x)  # (batch, hidden)
        
        i = F.softplus(self.W_i(x))  # (batch, hidden)
        f = torch.sigmoid(self.W_f(x))  # (batch, hidden)
        o = torch.sigmoid(self.W_o(x))  # (batch, hidden)
        
        # Update C: (batch, hidden, hidden)
        # kv outer product: (batch, hidden, 1) @ (batch, 1, hidden) = (batch, hidden, hidden)
        k_expanded = k.unsqueeze(2)  # (batch, hidden, 1)
        v_expanded = v.unsqueeze(1)  # (batch, 1, hidden)
        kv_outer = k_expanded @ v_expanded  # (batch, hidden, hidden)
        
        # Expand f and i for broadcasting with C
        f_expanded = f.unsqueeze(2)  # (batch, hidden, 1)
        i_expanded = i.unsqueeze(2)  # (batch, hidden, 1)
        
        # Update C: element-wise operations
        C = f_expanded * C_prev + i_expanded * kv_outer  # (batch, hidden, hidden)
        
        # Update n and m
        n = f * n_prev + i * k  # (batch, hidden)
        m = torch.max(f * m_prev, i)  # (batch, hidden)
        
        # Compute h_tilde: (C @ q) normalized
        # C: (batch, hidden, hidden), q: (batch, hidden, 1)
        q_expanded = q.unsqueeze(2)  # (batch, hidden, 1)
        h_tilde_raw = torch.bmm(C, q_expanded).squeeze(2)  # (batch, hidden)
        
        # Normalization denominator
        # n: (batch, hidden), q: (batch, hidden)
        norm_denom = torch.sum(n * q, dim=1, keepdim=True).clamp(min=1.0)  # (batch, 1)
        h_tilde = h_tilde_raw / norm_denom  # (batch, hidden)
        
        # Apply output gate and normalization
        h = o * self.group_norm(h_tilde)  # (batch, hidden)
        
        return h, (C, n, m)

class xLSTMBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, use_mlstm: bool = True, dropout: float = 0.1):
        super().__init__()
        
        self.use_mlstm = use_mlstm
        self.hidden_size = hidden_size
        
        if use_mlstm:
            self.lstm = mLSTMCell(input_size, hidden_size)
        else:
            self.lstm = sLSTMCell(input_size, hidden_size)
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def init_state(self, batch_size: int, device: torch.device) -> tuple:
        if self.use_mlstm:
            C = torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device)
            n = torch.zeros(batch_size, self.hidden_size, device=device)
            m = torch.zeros(batch_size, self.hidden_size, device=device)
            return (C, n, m)
        else:
            h = torch.zeros(batch_size, self.hidden_size, device=device)
            c = torch.zeros(batch_size, self.hidden_size, device=device)
            n = torch.zeros(batch_size, self.hidden_size, device=device)
            m = torch.zeros(batch_size, self.hidden_size, device=device)
            return (h, c, n, m)
    
    def forward(self, x: torch.Tensor, state: tuple) -> tuple:
        C_prev, n_prev, m_prev = state
        
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        i = F.softplus(self.W_i(x))
        f = torch.sigmoid(self.W_f(x))
        o = torch.sigmoid(self.W_o(x))
        
        k_expanded = k.unsqueeze(-1)
        v_expanded = v.unsqueeze(-2)
        
        f_expanded = f.unsqueeze(-1).unsqueeze(-1)
        i_expanded = i.unsqueeze(-1).unsqueeze(-1)
        
        kv_outer = k_expanded @ v_expanded
        
        f_exp = f.unsqueeze(-1)
        i_exp = i.unsqueeze(-1)
        
        C = f_exp * C_prev + i_exp * kv_outer
        
        n = f * n_prev + i * k
        m = torch.max(f * m_prev, i)
        
        h_tilde = (C @ q.unsqueeze(-1)).squeeze(-1) / torch.max(
            (n.unsqueeze(-1) * q.unsqueeze(-2)).sum(-1), 
            torch.ones_like(q)
        )
        
        h = o * self.group_norm(h_tilde)
        
        return h, (C, n, m)


class xLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int = 256,
        n_layers: int = 4,
        use_mlstm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.embed = nn.Linear(n_features, hidden_size)
        
        self.blocks = nn.ModuleList([
            xLSTMBlock(hidden_size, hidden_size, use_mlstm, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through xLSTM model.
        
        Args:
            x: Input tensor of shape (batch, n_features)
            
        Returns:
            Output predictions of shape (batch,)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Embed input
        x = self.embed(x)
        
        # Initialize states for all blocks
        states = [block.init_state(batch_size, device) for block in self.blocks]
        
        # Process through all blocks
        for i, block in enumerate(self.blocks):
            x, states[i] = block(x, states[i])
        
        # Final normalization
        x = self.norm(x)
        
        # Head for prediction
        return self.head(x).squeeze(-1)





