import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

from torch import Tensor


def positional_encoding(length, dim) -> Tensor:
    """
    这个位置编码可以支持词向量嵌入维度为奇的情况
    :param length: 词向量的长度
    :param dim: 每个词向量的嵌入维度
    :return:
    """
    pe = torch.zeros(length, dim)
    position = torch.arange(0, length).unsqueeze(1)
    
    if dim % 2 == 0:
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
    else:
        div_term = torch.exp((torch.arange(0, dim + 1, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term[:-1])
    
    return pe


# %%
if __name__ == '__main__':
    pe = positional_encoding(length=50, dim=5).numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pe)
    plt.show()
