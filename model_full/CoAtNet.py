import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

from model_structure.MBConv import MBConv
from model_elements.PreNorm import PreNorm


def conv_3x3_bn(inp, oup, image_size, downsample=False):
    """
    CoAtNet的第一层卷积层
    :param inp: input channels
    :param oup: out channels
    :param image_size: 不需要输入，只是为了在_make_layers中保证输入格式统一
    :param downsample: 是否进行一次降采样
    :return:
    """
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class FeedForward(nn.Module):
    """
    此处的全连接神经网络输入输出维度不发生变化
    """
    
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


class Attention(nn.Module):
    """
    相较于ViT中的多头自注意力机制，这里使用的自适应注意力矩阵
    ViT中的每个词向量代表着patch embedding
    此处的词向量代表着每个像素在通道上的展平向量
    """
    
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        """
        :param inp: 输入的图片通道数
        :param oup: 输出的图片通道数
        :param image_size: 图像尺寸
        :param heads: 多少个头
        :param dim_head: 每个头词向量嵌入维度的变换值
        :param dropout: dropout比例
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)
        
        self.ih, self.iw = image_size
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))
        
        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        
        # 图像坐标两两相对位置，因此relative_coords形状是[2, image_pixels, image_pixels]
        relative_coords = coords[:, :, None] - coords[:, None, :]
        
        # 这两步把负值都归零了
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)
        
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        # x: [B, N, D]
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # qkv: ([B, N, inner_dim], ..., ...)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # q(k,v): [B, heads, N, dim_head]
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # dots: [B, heads, N, N]
        
        # Use "gather" for more efficiency on GPUs
        # relative_bias_table: [(2*self.ih-1) * (2*self.iw-1), heads]
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        
        # 其中relative_bias就是自注意力矩阵上添加的偏置
        # 一个batch共用一个偏置
        dots = dots + relative_bias
        
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    """
    这是利用相对位置自注意力机制的Transformer，和ViT中的Transformer有一些区别
    且ViT中利用的Transformer可以直接生成深度Transformer block的layer
    此处的Transformer只能生成单一block
    """
    
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        
        self.downsample = downsample
        
        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.ih, self.iw = tuple(map(lambda x: int(x / 2), image_size))
        else:
            self.ih, self.iw = image_size
        
        self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.attn = Attention(inp, oup, (self.ih, self.iw), heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)
        
        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )
        
        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )
    
    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = self.proj(x) + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=1000,
                 block_types=['C', 'C', 'T', 'T']):
        """
        :param image_size: 输入图片尺寸
        :param in_channels: 输入图片通道数
        :param num_blocks: 每一个layer包含的block数量
        :param channels: 每一个layer输出的通道数
        :param num_classes: 最终输出的分类标签数
        :param block_types: 后四层layer的block类型，"C"代表MBConv，"T"代表相对自注意力的Transformer
        """
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}
        
        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih, iw))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 2, iw // 2))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 4, iw // 4))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 8, iw // 8))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 16, iw // 16))
        
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)
    
    def forward(self, x):
        x = self.s0(x)  # layer conv 3X3 is done
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        
        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x
    
    def _make_layer(self, block, inp, oup, depth, image_size):
        """
        将同一种类型的block重复多次，形成某个block的layer
        :param block: block类型，conv_3x3_bn、MBConv、Transformer
        :param inp: 输入的通道数
        :param oup: 输出的通道数
        :param depth: 该layer的block个数
        :param image_size: 图像尺寸
        :return: nn.Sequential
        """
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
                image_size = tuple(map(lambda x: x // 2, image_size))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


# %%
if __name__ == '__main__':
    # %% 用图像数据测试本文核心的相对自注意力层
    x = torch.randn([2, 64, 3])  # [B, N, D]，其中B为batch_size，N为一张图片的像素个数，D为图片通道数
    attn = Attention(inp=3, oup=5,  # inp为输入的图片通道数，oup为输出的图片通道数
                     image_size=(8, 8))  # 图片的尺寸
    output = attn(x)
    print(output.shape)  # [2, 64, 5]
    
    # %% 用图像数据测试利用先谷底自注意力层的Transformer
    x = torch.randn([2, 3, 8, 8])  # [B, C, H, W]，图片数据为正常的形状
    transformer = Transformer(inp=3, oup=5,
                              image_size=(8, 8))
    output = transformer(x)
    print(output.shape)  # [2, 5, 8, 8]
    
    # %% 用图像数据测试利用先谷底自注意力层的Transformer_downsample
    x = torch.randn([2, 3, 8, 8])  # [B, C, H, W]，图片数据为正常的形状
    transformer = Transformer(inp=3, oup=5,
                              image_size=(8, 8), downsample=True)
    output = transformer(x)
    print(output.shape)  # [2, 5, 4, 4]
    
    # %% 测试完整的网络CoAtNet网络结构
    x = torch.randn([1, 3, 64, 64])  # [B, C, H, W] 为正常的图像形状
    num_blocks = [2, 2, 3, 5, 2]  # 每一层的网络结构数量
    channels = [64, 96, 192, 384, 768]  # 每一层输入和输出的通道数
    block_types = ['C', 'T', 'T', 'T']  # 网络层的block类型 'C' for MBConv, 'T' for Transformer
    net = CoAtNet(image_size=(64, 64), in_channels=3,
                  num_blocks=num_blocks, channels=channels,
                  block_types=block_types, num_classes=1000)
    output = net(x)
    print(output.shape)  # [1, 1000]
