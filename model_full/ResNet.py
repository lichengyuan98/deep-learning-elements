#%%
import torch
import torch.nn as nn
from model_structure.Residual_block import BasicBlock, BottleNeck


class ResNet(nn.Module):
    
    def __init__(self, block, num_block, num_classes=100):
        """
        :param block: 选择什么类型的残差结构，basic还是bottle neck
        :param num_block: 网络有多少个残差块，每个残差块由多个残差结构组成
        :param num_classes: 最后用于输出分类时有多少个目标类别
        """
        super().__init__()
        
        self.in_channels = 64
        
        # 第一个卷积将通道从3增至self.in_channels，图像大小没发生变化
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        # 第二个残差layer将包含num_block[0]个残差块，stride为1，输出通道数为64
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        用于构建残差layer，每个layer由多个残差block组成
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        
        return output

if __name__ == '__main__':
    # %% 用图像数据测试resnet18
    # 使用的是basicblock残差块，2~5残差layer中的block数量分别是[2,2,2,2]
    x = torch.randn([1, 3, 64, 64]) # [B, C, H, W]
    res18 = ResNet(BasicBlock, [2, 2, 2, 2])
    output = res18(x)
    print(output.shape)
    
    # %% 用图像数据测试resnet152
    x = torch.randn([1, 3, 64, 64])
    res152 = ResNet(BottleNeck, [3, 8, 36, 3])
    output = res152(x)
    print(output.shape)