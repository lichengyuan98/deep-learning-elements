# %%
import torch
import torch.nn as nn
from einops import rearrange


class CompileSeq(nn.Module):
    def __init__(self, dim, latent_dim, layers=2, reverse=False):
        """
        使用深层LSTM进行序列的编码，隐变量维度和输入维度相同
        :param dim: 词向量嵌入维度
        :param latent_dim: 词向量序列编码维度
        :param reverse: 若序列的第一个词向量最重要，则需要翻转
        """
        super(CompileSeq, self).__init__()
        self.reverse = reverse
        
        self.fc = nn.Sequential(nn.Linear(layers * dim, latent_dim * 2),
                                nn.GELU(),
                                nn.Dropout(0.),
                                nn.Linear(latent_dim * 2, latent_dim),
                                nn.ELU(),
                                nn.Dropout(0.))
        self.h0 = nn.Parameter(torch.randn([layers, 1, dim]))
        self.c0 = nn.Parameter(torch.randn([layers, 1, dim]))
        
        self.rnn = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=layers, batch_first=True)
    
    def forward(self, x):
        """
        :param x: [B, N, D]
        :return: [B, latent_dim]
        """
        B, N, D = x.shape
        
        if self.reverse:
            x = x.flip(dims=(1,))
        
        h0 = self.h0.repeat(1, B, 1)  # [num_layers * num_directions, batch, hidden_size]
        c0 = self.c0.repeat(1, B, 1)
        _, (_, ct) = self.rnn(x, (h0, c0))
        ct = rearrange(ct, "l b d -> b (l d)")
        out = self.fc(ct)
        return out


# %%
if __name__ == '__main__':
    # %% 测试对词向量序列进行编码，最后一个词向量最重要
    x = torch.randn([5, 64, 256])  # [B, N, D]
    compileseq = CompileSeq(dim=256, latent_dim=128)
    out = compileseq(x)
    print(out.shape)  # [5, 128]
