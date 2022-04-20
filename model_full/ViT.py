"""
ViT
1. 文章推荐：AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
2. Transformer优势：并行计算、全局视野、灵活的堆叠能力
3. 做的任务：ViT和ResNet在图像分类问题上能力相当
4. 意义：CV中使用Transformer结构的可能
5. 论文结构：
"""
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class PreNorm(nn.Module):
    """
    self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)
    self.attn = PreNorm(inp, self.attn, nn.LayerNorm)
    PreNorm目的是在应用一个nn.Module前进行一次归一化操作，提高训练稳定性。
    一般nlp中使用的是nn.LayerNorm
    在CV中使用的是nn.BatchNorm
    在vit中使用的是nn.LayerNorm
    """
    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # 从多头自注意力值的concatenate维度变回输入词向量维度
        # nn.Identity() 不区分参数的占位符标识运算符
        #
        # if 某个操作 else Identity()
        # 在增减网络过程中，可以使得整个网络层数据不变，便于迁移权重数据
    
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
        return self.to_out(out)


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
        :param mlp_dim:
        :param pool:
        :param channels:
        :param dim_head:
        :param dropout:
        :param emb_dropout:
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
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
    
    preds = v(img)  # (1, 1000)
