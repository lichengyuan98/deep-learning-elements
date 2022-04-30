import torch
import torch.nn as nn
from model_elements.PreNorm import PreNorm
from model_elements.Attention import Attention


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """
    这个Transformer架构可以生成多个blocks的layer
    """
    
    def __init__(self, dim, depth, heads, dim_head, mlp_ratio, dropout=0.):
        """
        :param dim: 一个词向量维度
        :param depth: 多少个Transformer块
        :param heads: 多头自注意力的头数
        :param dim_head: 单头自注意力变换维度
        :param mlp_ratio: Transformer MLP模块中间层神经元缩减倍数
        :param dropout: dropout比例
        """
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim,
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        norm=nn.LayerNorm),
                PreNorm(dim,
                        FeedForward(dim, int(mlp_ratio * dim), dropout=dropout),
                        norm=nn.LayerNorm)
            ]))
    
    def forward(self, x):
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x


# %%
if __name__ == '__main__':
    # %%
    x = torch.randn([1, 100, 64])  # [B, N, D]
    tblock = Transformer(dim=64, depth=8, heads=8, dim_head=128, mlp_ratio=0.3)
    output = tblock(x)
    print(x.shape)  # [1, 100, 64]
