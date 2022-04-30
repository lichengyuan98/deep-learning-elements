# %%
import torch
import torch.nn as nn


class PreNorm(nn.Module):
    """
    self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
    self.attn = PreNorm(inp, self.attn, nn.LayerNorm)
    PreNorm目的是在应用一个nn.Module前进行一次归一化操作，提高训练稳定性。
    一般nlp中使用的是nn.LayerNorm
    在CV中使用的是nn.BatchNorm
    在vit中使用的是nn.LayerNorm
    """
    
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


if __name__ == '__main__':
    # %% 当输入是个图像数据时
    x = torch.randn([1, 3, 16, 16])  # [B, C, H, W]
    conv2d = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3),
                       stride=(1, 1), padding=(1, 1))
    conv2d_prenorm = PreNorm(3, conv2d, nn.BatchNorm2d)  # 3是输入通道数，也是输入BatchNorm2d的维度
    output = conv2d_prenorm(x)
    print(output.shape)  # [1, 5, 16, 16]

    # %% 当输入是个词向量时
    # B是Batch大小，N是词向量长度，D是词向量的嵌入维度
    from model_elements.Attention import Attention

    x = torch.randn([1, 64, 128])  # [B, N, D]
    attn = Attention(dim=128, heads=8, dim_head=64, dropout=0.1)
    attn_prenorm = PreNorm(128, attn, nn.LayerNorm)  # 128是词向量嵌入维度，也是LayerNorm的影响维度
    output = attn_prenorm(x)
    print(output.shape)  # [1, 64, 128]
