import torch
import torch.nn as nn

from model_elements.Transformer import Transformer
from model_elements.Compile_Seq import CompileSeq
from einops.layers.torch import Rearrange


class PaddedAE(nn.Module):
    """
    基于MAE开发的网络框架，形状`[B, N, D]`的张量在序列维度按照比例随机进行归零遮掩，
    在Encoder经过多层Transformer后形状为`[B, N, D]`，然后利用LSTM在`N`维度上滑动进行编码获得序列的隐变量，此时张量的形状变成`[B, latent_dim]`，
    随后通过线性层和维度变化将张量形状还原至`[B, N, D]`，最后通过多层Transformer编码后输出
    """
    
    def __init__(self, length,
                 embed_dim=128, latent_dim=32, depth=4, num_heads=4,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
                 mlp_ratio=0.8, norm_layer=nn.LayerNorm):
        """
        :param length: 词向量序列长度
        :param embed_dim: 词向量嵌入维度
        :param latent_dim: 隐变量维度
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
        self.encoder_to_latent = CompileSeq(dim=embed_dim, latent_dim=latent_dim, reverse=True)
        
        # MAE decoder specifics
        self.latent_to_decoder = nn.Sequential(nn.Linear(latent_dim, embed_dim * length // 2),
                                               nn.GELU(),
                                               nn.Dropout(0.),
                                               nn.Linear(embed_dim * length // 2, embed_dim * length),
                                               nn.ELU(),
                                               nn.Dropout(0.),
                                               Rearrange("B (N D) -> B N D", N=length))
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)  # 进入Decoder前首先把词向量嵌入维度扩大至理想值
        self.cls_token_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)  # 进入Decoder前首先把cls token嵌入维度扩大至理想值
        
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
        mask是形状为[N, L]的位置掩码
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))  # 保留下来的词向量长度
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).bool()
        
        # pad x with mask
        x_padded = torch.masked_fill(x, mask.unsqueeze(-1).repeat(1, 1, D), 0)
        
        return x_padded, mask  # x: [B, N, D]
    
    def forward_encoder(self, x, mask_ratio):
        # add pos embed w/o cls token
        # x: [B, N, D]
        x = x + self.pos_embed[:, 1:, :]
        
        x, mask = self.random_masking(x, mask_ratio)  # x: [B, N, D]
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # [1, 1, D]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # x: [B, N+1, D]
        
        # apply Transformer blocks
        x = self.blocks(x)  # x: [B, N+1, D]
        x = self.norm(x)  # x: [B, N+1, D]
        
        return x, mask
    
    def forward_decoder(self, x, cls_token):
        # embed tokens
        x = self.decoder_embed(x)
        cls_token = self.cls_token_embed(cls_token)
        
        # append mask tokens to sequence
        x = torch.cat([cls_token, x], dim=1)  # append cls token 这个cls token是由encoder部分传递过来的，即从encoder部分的信息直接到达decoder
        
        # add pos embed 这个pos embed在encoder和decoder中不同的
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
        x, mask = self.forward_encoder(imgs, mask_ratio)
        latent = self.encoder_to_latent(x)
        reconstruct = self.latent_to_decoder(latent)
        pred = self.forward_decoder(reconstruct, x[:, :1, :])
        return pred, latent


# %%
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn([4, 100, 32]).to(device)
    pae = PaddedAE(length=100,
                   embed_dim=32,
                   latent_dim=16).to(device)
    output, latent = pae(x, mask_ratio=0.8)
    print(output.shape)  # [4, 100, 32]
    print(latent.shape)  # [4, 16]
