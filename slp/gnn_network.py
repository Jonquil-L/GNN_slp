import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionBlock(nn.Module):
    """带残差连接和 FFN 的多头图注意力层"""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        x: (B, N, D)
        adj: (B, N, N) 邻接掩码，1=有边，0=无边
        """
        B, N, D = x.shape
        H, d = self.num_heads, self.head_dim

        res = x
        x = self.norm1(x)

        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)  # (B, H, N, d)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
        attn = attn.masked_fill(adj.unsqueeze(1) == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, 0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        x = res + out

        x = x + self.ffn(self.norm2(x))
        return x


class SLPPolicyValueNet(nn.Module):
    """
    GNN 编码器 + 两步指针解码器 + 价值头。
    动作空间从 O(n^2) 分解为两步 O(n)：先选 u，再条件选 v。
    """

    def __init__(self, input_dim, hidden_dim=128, num_gnn_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.gnn_layers = nn.ModuleList([
            GraphAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 第一步指针：选 u
        self.u_query = nn.Linear(hidden_dim, hidden_dim)
        self.u_key = nn.Linear(hidden_dim, hidden_dim)

        # 第二步指针：条件于 u 选 v
        self.v_query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v_key = nn.Linear(hidden_dim, hidden_dim)

        # 价值评估头（预测剩余门数，非负输出）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # AlphaZero 价值头：预测 gates-remaining（非负）
        self.az_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # 非负：预测剩余门数
        )

    def encode(self, features, adj):
        h = self.input_proj(features)
        for layer in self.gnn_layers:
            h = layer(h, adj)
        return self.final_norm(h)

    def _global_pool(self, h, valid_mask):
        mask = valid_mask.unsqueeze(-1)
        return (h * mask).sum(1) / (mask.sum(1) + 1e-8)

    def get_u_logits(self, h, valid_mask):
        g = self._global_pool(h, valid_mask)
        q = self.u_query(g).unsqueeze(1)
        k = self.u_key(h)
        logits = (q * k).sum(-1) / math.sqrt(self.hidden_dim)
        logits = logits.masked_fill(valid_mask == 0, float('-inf'))
        return logits

    def get_v_logits(self, h, u_idx, v_mask, valid_mask):
        g = self._global_pool(h, valid_mask)
        u_emb = h[torch.arange(h.size(0), device=h.device), u_idx]
        q_input = torch.cat([g, u_emb], dim=-1)
        q = self.v_query(q_input).unsqueeze(1)
        k = self.v_key(h)
        logits = (q * k).sum(-1) / math.sqrt(self.hidden_dim)
        logits = logits.masked_fill(v_mask == 0, float('-inf'))
        return logits

    def get_value(self, h, valid_mask):
        g = self._global_pool(h, valid_mask)
        return self.value_head(g)

    def get_az_value(self, h, valid_mask):
        """AlphaZero 价值：预测从当前状态到完成还需多少门"""
        g = self._global_pool(h, valid_mask)
        return self.az_value_head(g)

    def forward(self, features, adj, valid_mask, u_idx=None, v_mask=None):
        """
        完整前向传播。
        若提供 u_idx 和 v_mask，同时返回 v_logits。
        """
        h = self.encode(features, adj)
        u_logits = self.get_u_logits(h, valid_mask)
        value = self.get_value(h, valid_mask)
        v_logits = None
        if u_idx is not None and v_mask is not None:
            v_logits = self.get_v_logits(h, u_idx, v_mask, valid_mask)
        return u_logits, v_logits, value
