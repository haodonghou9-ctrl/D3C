import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.distributions as dist
from torch.nn import Parameter


from .autoencoder import Encoder, Decoder

# from utils import mask_image


def mask_image(img, patch_size=4, mask_ratio=0.5, return_img=True):
    """mask image like MAE.

    Args:
        img (Tensor): (B, C, H, W) images.
        patch_size (int, optional): masked patch size. Defaults to 4.
        mask_ratio (float, optional): mask ratio. Defaults to 0.5.
        return_img (bool, optional): Return masked image if ture, whether return return visable image.
    Returns:
        img (Tensor): (B, C, H, W) masked images.
    """
    b, c, h, w = img.shape
    patch_h = patch_w = patch_size
    num_patches = (h // patch_h) * (w // patch_w)

    patches = img.view(
        b, c,
        h // patch_h, patch_h,
        w // patch_w, patch_w
    ).permute(0, 2, 4, 3, 5, 1).reshape(b, num_patches, -1)

    num_masked = int(mask_ratio * num_patches)
    shuffle_indices = torch.rand(b, num_patches).argsort()
    mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]
    batch_ind = torch.arange(b).unsqueeze(-1)
    if return_img:
        # masked
        patches[batch_ind, mask_ind] = 0
        x_masked = patches.view(
            b, h // patch_h, w // patch_w,
            patch_h, patch_w, c
        ).permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)

        return x_masked
    else:
        return patches[batch_ind, unmask_ind]


class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters=10, hidden=10, cluster_centers=None, alpha=1.0):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.hidden = hidden
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters,
                self.hidden,
                dtype=torch.float
            ).cuda()
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, x):
        norm_squared = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        t_dist = (numerator.t() / torch.sum(numerator, 1)).t()  # soft assignment using t-distribution
        return t_dist


class ConsistencyAE(nn.Module):

    def __init__(self,
                 basic_hidden_dim=16,
                 c_dim=64,
                 continous=True,
                 in_channel=3,
                 num_res_blocks=3,
                 ch_mult=[1, 2, 4, 8],
                 block_size=8,
                 latent_ch=10,
                 temperature=1.0,
                 kld_weight=0.00025,
                 views=2,
                 alpha=1.0,
                 categorical_dim=10) -> None:
        """
        """
        super().__init__()

        self.c_dim = c_dim
        self.continous = continous
        self.in_channel = in_channel
        self.ch_mult = ch_mult
        self.block_size = block_size
        self.basic_hidden_dim = basic_hidden_dim
        self.num_res_blocks = num_res_blocks
        self.latent_ch = latent_ch
        self.anneal_rate = 0.00003
        self.min_temp = 0.5
        self.temp = temperature
        self.views = 1
        self.kld_weight = kld_weight
        self.categorical_dim = categorical_dim
        self.alpha = alpha

        self._encoder = Encoder(hidden_dim=self.basic_hidden_dim,
                                in_channels=self.in_channel,
                                z_channels=self.latent_ch,
                                ch_mult=self.ch_mult,
                                num_res_blocks=self.num_res_blocks,
                                resolution=1,
                                use_attn=False,
                                attn_resolutions=None,
                                double_z=False)

        self.decoders = nn.ModuleList([Decoder(hidden_dim=self.basic_hidden_dim,
                                               out_channels=self.in_channel,
                                               in_channels=self.latent_ch,
                                               z_channels=self.latent_ch,
                                               ch_mult=self.ch_mult,
                                               num_res_blocks=self.num_res_blocks,
                                               resolution=1,
                                               use_attn=False,
                                               attn_resolutions=None,
                                               double_z=False) for _ in range(self.views)])

        if self.continous:

            self.fc_z = nn.Linear(512 * self.views, self.c_dim * 2)
            self.to_decoder_input = nn.Linear(self.c_dim, self.latent_ch * self.block_size ** 2)
        else:
            # discrete code.
            self.fc_z = nn.Linear(self.latent_ch * self.block_size ** 2, self.c_dim * self.categorical_dim)
            self.to_decoder_input = nn.Linear(self.c_dim * self.categorical_dim, self.latent_ch * self.block_size ** 2)

    def forward(self, Xs):

        if self.continous:
            mu, logvar = self.encode(Xs)
            z = self.cont_reparameterize(mu, logvar)
        else:
            beta = self.encode(Xs)
            z = self.disc_reparameterize(beta)

        recons = self.decode(z)
        return recons, z

    def cont_reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def disc_reparameterize(self, z, eps=1e-7):
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        # if self.training:
        # Sample from Gumbel
        u = torch.rand_like(z)
        g = - torch.log(- torch.log(u + eps) + eps)
        # Gumbel-Softmax sample
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.c_dim * self.categorical_dim)
        return s

    def encode(self, Xs):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param x: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        latents = []
        for x in Xs:

            latent = self._encoder(x)

            latent = torch.flatten(latent, start_dim=1)
            latents.append(latent)
        latent = torch.cat(latents, dim=-1)




        z = self.fc_z(latent)
        if self.continous:
            mu, logvar = torch.split(z, self.c_dim, dim=1)
            return mu, logvar
        else:
            return z.view(-1, self.c_dim, self.categorical_dim)

    def decode(self, z):
        z = self.to_decoder_input(z)
        z = z.view(-1, self.latent_ch, self.block_size, self.block_size)
        return [dec(z) for dec in self.decoders]

    def get_loss(self, Xs, epoch, mask_ratio, mask_patch_size):
        # Masked cross-view distribution modeling.
        Xs_masked = [mask_image(x, mask_patch_size, mask_ratio=mask_ratio) for x in Xs]

        if self.continous:
            mu, logvar = self.encode(Xs_masked)
            kld_loss = self.con_loss(mu, logvar)

            z = self.cont_reparameterize(mu, logvar)
        else:
            beta = self.encode(Xs_masked)
            kld_loss = self.disc_loss(beta, epoch)

            z = self.disc_reparameterize(beta)

        recons = self.decode(z)
        recon_loss = 0.
        return_details = {}
        for v, (x, recon) in enumerate(zip(Xs, recons)):
            sub_loss = F.mse_loss(x, recon, reduction='sum')
            return_details[f'v{v + 1}-loss'] = sub_loss.item()
            recon_loss += sub_loss

        loss = self.alpha * recon_loss + kld_loss


        return_details['total_loss'] = loss.item()
        return_details['recon_loss'] = recon_loss.item()
        return_details['kld_loss'] = kld_loss.item()
        return_details['temparature'] = self.temp

        return loss, return_details

    def con_loss(self, mu, log_var):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return self.kld_weight * kld_loss

    def disc_loss(self, Q, epoch) -> dict:
        """"
        Computes the discreate-VAE loss function.
        """

        B, N, K = Q.shape
        Q = Q.view(B * N, K)
        q = dist.Categorical(logits=Q)
        p = dist.Categorical(
            probs=torch.full((B * N, K), 1.0 / K).to(Q.device))  # uniform bunch of K-class categorical distributions
        kl = dist.kl.kl_divergence(q, p).view(B, N)  # kl is of shape [B*N]

        if epoch % 5 == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * epoch),
                                   self.min_temp)

        return torch.mean(torch.sum(kl, dim=1))

    def consistency_features(self, Xs):
        if self.continous:
            mu, logvar = self.encode(Xs)
            z = self.cont_reparameterize(mu, logvar)
        return z

