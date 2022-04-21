import torch
import torch.nn as nn

from model_elements.Squeeze_and_Excitation import SE
from model_elements.PreNorm import PreNorm


class MBConv(nn.Module):
    def __init__(self, inp, oup, downsample=False, expansion=4):
        """
        MBConv由一个包含分组卷积、SE模块的前馈通道和短路残差连接组合而成。网络经过了通道数量先变胖再变瘦的过程
        :param inp: 输入通道数
        :param oup: 输出通道数
        :param downsample: bool，是否降采样
        :param expansion: 通道数变胖的倍数
        """
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_channels = int(inp * expansion)
        
        # 残差连接需要保持维度一致
        if inp != oup:
            self.trans = nn.Conv2d(inp, oup, 1, bias=False)
        else:
            self.trans = nn.Identity()
        
        # 若开启降采样时的残差连接
        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)  # kernel size, stride, padding这步进行降采样
            self.proj = nn.Conv2d(inp, oup, 1, bias=False)
        
        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride,
                          1, groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # 同时在第一个卷积层进行降采样
                nn.Conv2d(inp, hidden_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1,
                          groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
                SE(hidden_channels, inp),
                # pw-linear
                nn.Conv2d(hidden_channels, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
    
    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return self.trans(x) + self.conv(x)


# %%
if __name__ == '__main__':
    # %% 测试没有降采样的MBConv
    x = torch.randn([1, 3, 64, 64])
    mbconv = MBConv(inp=3, oup=10, downsample=False)
    output = mbconv(x)
    print(output.shape)
    
    # %% 测试降采样的MBConv
    x = torch.randn([1, 3, 64, 64])
    mbconv = MBConv(inp=3, oup=10, downsample=True)
    output = mbconv(x)
    print(output.shape)  # [1, 3, 32, 32]
