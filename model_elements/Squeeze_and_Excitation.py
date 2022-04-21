# %%
import torch
import torch.nn as nn
from torch import Tensor


class SE(nn.Module):
    """
    用于自动学习卷积后每个通道的显著性权重，并且对通道分别加权
    """
    
    def __init__(self, in_channel, mlp_dim):
        """
        :param in_channel: 输入通道数
        :param mlp_dim: 全连接中间层神经元个数，一般小于输入通道数
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channel, mlp_dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_dim, in_channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: [b, c, h, w]
        :return: [b, c, h, w]
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


if __name__ == '__main__':
    # %%
    x = torch.randn([1, 3, 64, 64])
    se = SE(in_channel=3, mlp_dim=5)
    output = se(x)
    print(output.shape)
