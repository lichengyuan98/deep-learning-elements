# %%
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 残差函数
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        
        # 向前短路连接
        # 短路连接后的叠加需要维度相同，若维度发生了变化，通过1X1卷积核增加通道数
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """
    若ResNet网络结构过深（50层以上），则需要通过Bottle Neck结构，通过在ResNet中不断增加、减少通道数，方便训练
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 残差函数，通道数从in_channels变换至out_channels * BottleNeck.expansion
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )
        
        # 短路连接，需要保证维度一致
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


if __name__ == '__main__':
    # %% 用图像数据测试basicblock
    x = torch.randn([1, 3, 64, 64])
    basic_resblock = BasicBlock(in_channels=3, out_channels=10, stride=1)
    output = basic_resblock(x)
    print(output.shape)
    
    # %% 用图像数据测试bottleneck
    x = torch.randn([1, 3, 64, 64])
    bottleneck_resblock = BottleNeck(in_channels=3, out_channels=10, stride=1)
    output = bottleneck_resblock(x)
    print(output.shape)  # 注意这里扩大了四倍的目标通道数，目的是在ResNet中第2个残差layer开始使用stride为2的卷积
