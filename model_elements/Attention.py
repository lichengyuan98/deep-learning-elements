import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        :param dim: 词向量维度
        :param heads: 多头自注意力头数
        :param dim_head: 单头自注意力变换的维度
        :param dropout: dropout比例
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        # 先变成3个dim_head，再掰断成q, k, v
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # 从多头自注意力值的concatenate维度变回输入词向量维度
        # nn.Identity() 不区分参数的占位符标识运算符
        # if 某个操作 else Identity()
        # 在增减网络过程中，可以使得整个网络层数据不变，便于迁移权重数据
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        """
        :param x: [batch, length, dim]
        :return:
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        # b=batch; n=sequence length; h=heads; d=dim_head
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # 维度同输入没有发生变化，仍为[batch, length, dim]


if __name__ == '__main__':
    x = torch.rand([1, 64, 128])
    attn = Attention(dim=128, heads=8, dim_head=64, dropout=0.1)
    output = attn(x)
    print(x.shape)
