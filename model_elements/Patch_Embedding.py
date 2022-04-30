import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class PatchEmbed(nn.Module):
    """
    目的是把一个多通道的图像转化为词向量序列
    """
    
    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans=3,
                 embed_dim=768):
        """
        :param img_size: 图像的大小[H, W]
        :param patch_size: 每个小patch的长宽[h, w]， 长宽分别要能够被图像长宽整除
        :param in_chans: 输入的通道数
        :param embed_dim: 嵌入词向量的维度
        """
        super().__init__()
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_dim = patch_size[0] * patch_size[1] * in_chans
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=self.grid_size[0], p2=self.grid_size[1]),
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x):
        """
        :param x: Tensor, [b, c, h, w]
        :return: [b, n, d]
        """
        return self.to_patch_embedding(x)


if __name__ == '__main__':
    x = torch.randn([1, 3, 64, 64])
    patch_embed = PatchEmbed(img_size=[64, 64],
                             patch_size=[8, 8],
                             in_chans=3,
                             embed_dim=100)
    output = patch_embed(x)
    print(output.shape)  # [1, 64, 100]
