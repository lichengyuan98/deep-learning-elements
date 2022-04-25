from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

from model_elements.Transformer import Transformer


class MaskedAutoencoderViT(nn.Module):
    """
    MaskedAE本质上是将一个词向量序列遮住一部分，剩下的部分通过若干个Transformer进行编码学习到隐变量
    然后通过若干次Transformer对隐变量进行编码，恢复至原来的词向量序列维度
    因此在mask过程中直接将Patch Embedding部分摘除[B, N, D] -> [B, N*(1-mask_ratio), D]
    """
    
    def __init__(self, length,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=0.8, norm_layer=nn.LayerNorm):
        """
        :param length: 词向量序列长度
        :param embed_dim: 词向量嵌入维度
        :param depth: encoder中Transformer块的个数
        :param num_heads: encoder中多头自注意力的头数
        :param decoder_embed_dim: decoder中输入的词向量嵌入维度
        :param decoder_depth: decoder中transformer块个数
        :param decoder_num_heads: decoder中多头自注意力头数
        :param mlp_ratio: Transformer MLP层中间层神经元的比例
        :param norm_layer: 对词向量进行norm的方法，一般为nn.Layernorm
        """
        super().__init__()
        
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 词向量序列中增加一个类别token
        self.pos_embed = nn.Parameter(torch.zeros(1, length + 1, embed_dim),
                                      requires_grad=False)  # 需要给加入类别token的词向量序列配备绑定的位置编码
        
        self.blocks = Transformer(dim=embed_dim,
                                  depth=depth,
                                  heads=num_heads,
                                  dim_head=embed_dim * 2,
                                  mlp_ratio=mlp_ratio)  # 随机遮掩之后将剩余可见的词向量序列投入Transformer中进行编码
        
        self.norm = norm_layer(embed_dim)
        
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)  # 进入Decoder前首先把词向量嵌入维度扩大至理想值
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  #
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, length + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding
        
        self.decoder_blocks = Transformer(dim=decoder_embed_dim,
                                          depth=decoder_depth,
                                          heads=decoder_num_heads,
                                          dim_head=decoder_embed_dim * 2,
                                          mlp_ratio=mlp_ratio)
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim,
                                      bias=True)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        首先将x在dim=1（词向量序列长度维度）上进行打乱，然后选取mask_ratio的比例进行遮掩
        最后输出的x_mask[N, L*mask_ratio, D]是打乱并遮掩的词向量序列batch
        mask是形状为[N, L]的位置掩码，restore[N, L]则为每一个序列打乱的index
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # 保留下来的词向量长度
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        # add pos embed w/o cls token
        # [B, N, D]
        x = x + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio
        # [B, N*(1-mask_ratio), D]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)  # # 这里latent是对遮掩后的词向量编码后的隐变量
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p0*p1*c]
        return pred


# %%
if __name__ == '__main__':
    x = torch.randn([1, 100, 32])
    mae = MaskedAutoencoderViT(length=100,
                               embed_dim=32)
    output = mae(x, mask_ratio=0.8)
    print(output.shape)  # [1, 100, 32]
