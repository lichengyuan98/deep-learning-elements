# %%
import torch
from torch import nn


class VanillaVAE(nn.Module):
    
    def __init__(self, in_features, latent_dim):
        """
        这只是一个超级简易的VAE框架，Encoder和Decoder可以根据任务需要进行添加不同的网络结构
        :param in_features: 输入特征数量
        :param latent_dim: 多少个正态分布的隐变量
        """
        super(VanillaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 定义encoder到隐变量分布参数的映射
        
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_var = nn.Linear(in_features, latent_dim)
        
        # 定义decoder
        self.decoder = nn.Linear(latent_dim, in_features)
    
    def reparameterize(self, mu, logvar):
        """
        为了保证网络通路的连贯性，对隐变量进行Reparameterization
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.reparameterize(mu, log_var)
        reconstruct = self.decoder(z)
        return reconstruct


# %%
if __name__ == '__main__':
    # %%
    x = torch.randn([1, 64])
    vae = VanillaVAE(in_features=64, latent_dim=10)
    output = vae(x)
    print(output.shape)
