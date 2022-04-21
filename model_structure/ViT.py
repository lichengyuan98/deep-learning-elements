"""
ViT
1. 文章推荐：AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
2. Transformer优势：并行计算、全局视野、灵活的堆叠能力
3. 做的任务：ViT和ResNet在图像分类问题上能力相当
4. 意义：CV中使用Transformer结构的可能
"""
# %%
import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn

from model_elements.Attention import Attention
from model_elements.PreNorm import PreNorm


def pair(t: int):
    """
    输出为元组，输入可以为元组或整数
    :param t: int
    :return: tuple
    """
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(PreNorm):
    """
    重写Prenorm仅用Layernorm归一化
    """
    
    def __init__(self, dim, fn):
        super().__init__(dim, fn, nn.LayerNorm)


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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        :param dim: 一个词向量维度
        :param depth: 多少个自注意力层
        :param heads: 多头自注意力的头数
        :param dim_head: 单头自注意力变换维度
        :param mlp_dim: Transformer Encoder模块最终输出的最后一维维度
        :param dropout: dropout比例
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        """
        :param image_size: (h, w) or h, 图像尺寸
        :param patch_size: (h, w) or h, 每个patch的尺寸
        :param num_classes: 输出分类的类别个数
        :param dim: 一个patch嵌入空间维度
        :param depth: 多少个Transformer的encoder模块
        :param heads: 多头自注意力头数
        :param mlp_dim: 全连接层隐藏层神经元个数
        :param pool: 信息收缩的方式
        :param channels: 输入的通道数
        :param dim_head: 每个头所变换的维度
        :param dropout: Transformer中的dropout比例
        :param emb_dropout: 加入位置编码后的dropout比例
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width  # 一个patch三个通道进行展平就是patch的维度
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )  # 将一个图像转换为h*w个patch的embedding

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 位置编码是通过学习得到的
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 支出来的类别编码也是通过学习得到的
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        self.pool = pool
        self.to_latent = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        
        x = self.to_latent(x)
        return self.mlp_head(x)


if __name__ == '__main__':
    # %%
    v = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    img = torch.randn(1, 3, 256, 256)
    preds = v(img)
    print(preds.shape)
