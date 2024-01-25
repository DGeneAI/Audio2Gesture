# region Import.
import os

import torch

import torch.nn.functional as F


import numpy as np
import torch.nn as nn

from typing import List
# from vqvae_modules import VectorQuantizerEMA
# endregion


__all__ = ["Conv1d", "Transformer","vqvae1d"]

# class vqvae1d_encoder(nn.Module):
#     def __init__(self,encoder_config,embedding_dim) -> None:
#         super().__init__()
#         num_layers = len(encoder_config)
#         modules = []
#         for i, c in enumerate(encoder_config):
#             modules.append(nn.Conv1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
            
#             if i < (num_layers-1):
#                 modules.append(nn.BatchNorm1d(c[1]))
#                 modules.append(nn.LeakyReLU(1.0, inplace=True))
#         self.encoder = nn.Sequential(*modules)
#         self.pre_vq_conv = nn.Conv1d(encoder_config[-1][1], embedding_dim, 1, 1)
#         self.lxm_embed = nn.Linear(encoder_config[-1][1],embedding_dim, bias=True)
#     def forward(self,x):
#         # for i in range(len(self.encoder)):
#         #     x = self.encoder[i](x)
#         z = self.encoder(x)
#         # z = self.pre_vq_conv(z)
#         # z: [b,192,1]
#         z = z.permute(0, 2, 1).contiguous()
#         z = self.lxm_embed(z)
#         z = z.permute(0, 2, 1).contiguous()
#         return z
    
# class vqvae1d_decoder(nn.Module):
#     def __init__(self,decoder_config,embedding_dim) -> None:
#         super().__init__()
#         num_layers = len(decoder_config)
#         modules = []
#         for i, c in enumerate(decoder_config):
#             modules.append(nn.ConvTranspose1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
           
#             if i < (num_layers-1):
#                 modules.append(nn.BatchNorm1d(c[1]))
#                 modules.append(nn.LeakyReLU(1.0, inplace=True))
#         self.decoder = nn.Sequential(*modules)
#         self.aft_vq_conv = nn.Conv1d(embedding_dim, decoder_config[0][0], 1, 1)
#         self.decoder_embed = nn.Linear(embedding_dim, decoder_config[0][0], bias=True)
        
        
#     def forward(self,e):
#         # e = self.aft_vq_conv(e)
#         e = e.permute(0, 2, 1).contiguous()
#         e = self.decoder_embed(e)
#         e = e.permute(0, 2, 1).contiguous()  # 5,192,1
#         re_con = self.decoder(e)
#         return re_con
    
# class vqvae1d(nn.Module):
    
#     def __init__(self, 
#                  encoder_config : List[List[int]], 
#                  decoder_config : List[List[int]],
#                  vq_config: dict) -> None:
#         super().__init__()
#         # embedding_dim,num_embeddings = 192,50
#         self.embedding_dim = vq_config["embedding_dim"]
#         self.num_embeddings = vq_config["num_embeddings"]
#         commitment_cost=0.25
#         vq_cost = 0
#         decay=0.99

        
#         self.encoder = vqvae1d_encoder(encoder_config,self.embedding_dim)
#         self.vq_layer = VectorQuantizerEMA(self.embedding_dim,self.num_embeddings, commitment_cost, vq_cost,decay)
#         self.decoder = vqvae1d_decoder(decoder_config,self.embedding_dim)
        
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.)
        
    
#     def forward(self, x : torch.Tensor):
#         """
#         x : (batch_size, dim_feat, time).
#         """
        
#         z = self.encoder(x)
#         e,  loss_commit, loss_vq = self.vq_layer(z)
#         gt_recon= self.decoder(e)
#         return  loss_commit, loss_vq, gt_recon



class ConvNormRelu(nn.Module):
    '''
    (B,C_in,H,W) -> (B, C_out, H, W)
    there exist some kernel size that makes the result is not H/s
    #TODO: there might some problems with residual
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 leaky=False,
                 sample='none',
                 p=0,
                 groups=1,
                 residual=False,
                 norm='bn'):
        '''
        conv-bn-relu
        '''
        super(ConvNormRelu, self).__init__()
        self.residual = residual
        self.norm_type = norm
        padding = 1

        if sample == 'none':
            kernel_size = 3
            stride = 1
        elif sample == 'one':
            padding = 0
            kernel_size = stride = 1
        elif sample in ['down_resample',"up_resample"] :  # for input T is 12, out 10, to uniform the enc/dec
            kernel_size = 3
            stride = 1
            padding = 0
            sample = sample[:-9]
        else:
            kernel_size = 4
            stride = 2
            padding = 0
        if self.residual:
            
            if sample == 'down':
                self.residual_layer = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            elif sample == 'up':
                self.residual_layer = nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            else:
                if in_channels == out_channels:
                    self.residual_layer = nn.Identity()
                else:
                    self.residual_layer = nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding
                        )
                    )

        in_channels = in_channels * groups
        out_channels = out_channels * groups
        if sample == 'up':
            self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride, padding=padding,
                                           groups=groups)
        else:
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding,
                                  groups=groups)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=p)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x, **kwargs):
        out = self.norm(self.dropout(self.conv(x)))
        if self.residual:
            residual = self.residual_layer(x)
            out += residual
        return self.relu(out)


class Res_CNR_Stack(nn.Module):
    def __init__(self,
                 channels,
                 layers,
                 sample='none',
                 leaky=False,
                 casual=False,
                 ):
        super(Res_CNR_Stack, self).__init__()

        if casual:
            kernal_size = 1
            padding = 0
            conv = CasualConv
        else:
            kernal_size = 3
            padding = 1
            conv = ConvNormRelu

        if sample == 'one':
            kernal_size = 1
            padding = 0

        self._layers = nn.ModuleList()
        for i in range(layers):
            self._layers.append(conv(channels, channels, leaky=leaky, sample=sample))
        self.conv = nn.Conv1d(channels, channels, kernal_size, 1, padding)
        self.norm = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x, pre_state=None):
        # cur_state = []
        h = x
        for i in range(self._layers.__len__()):
            # cur_state.append(h[..., -1:])
            h = self._layers[i](h, pre_state=pre_state[i] if pre_state is not None else None)
        h = self.norm(self.conv(h))
        return self.relu(h + x)


class ExponentialMovingAverage(nn.Module):
    """Maintains an exponential moving average for a value.

      This module keeps track of a hidden exponential moving average that is
      initialized as a vector of zeros which is then normalized to give the average.
      This gives us a moving average which isn't biased towards either zero or the
      initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

      Initially:
          hidden_0 = 0
      Then iteratively:
          hidden_i = hidden_{i-1} - (hidden_{i-1} - value) * (1 - decay)
          average_i = hidden_i / (1 - decay^i)
    """

    def __init__(self, init_value, decay):
        super().__init__()

        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros_like(init_value))

    def forward(self, value):
        self.counter += 1
        self.hidden.sub_((self.hidden - value) * (1 - self.decay))
        average = self.hidden / (1 - self.decay ** self.counter)
        return average


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. Use EMA to update embeddings.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
        decay (float): decay for the moving averages.
        epsilon (float): small float constant to avoid numerical instability.
    """

    def __init__(self, embedding_dim, num_embeddings, commitment_cost,vq_cost,decay,
                 epsilon=1e-5, ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.vq_cost = vq_cost
        self.epsilon = epsilon
        # self.infer = infer
        # initialize embeddings as buffers
        embeddings = torch.empty(self.num_embeddings, self.embedding_dim)
        nn.init.xavier_uniform_(embeddings)
        self.register_buffer("embeddings", embeddings)
        self.ema_dw = ExponentialMovingAverage(self.embeddings, decay)

        # also maintain ema_cluster_size， which record the size of each embedding
        self.ema_cluster_size = ExponentialMovingAverage(torch.zeros((self.num_embeddings,)), decay)

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        flat_x = x.reshape(-1, self.embedding_dim)

        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x)  # [B, W, C]

        # if not self.training:
        #     quantized = quantized.permute(0, 2, 1).contiguous()
        #     return quantized, encoding_indices.view(quantized.shape[0], quantized.shape[2])

        # update embeddings with EMA
        with torch.no_grad():
            encodings = F.one_hot(encoding_indices, self.num_embeddings).float()
            updated_ema_cluster_size = self.ema_cluster_size(torch.sum(encodings, dim=0))
            n = torch.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
            dw = torch.matmul(encodings.t(), flat_x)  # sum encoding vectors of each cluster
            updated_ema_dw = self.ema_dw(dw)
            normalised_updated_ema_w = (
                    updated_ema_dw / updated_ema_cluster_size.reshape(-1, 1))
            self.embeddings.data = normalised_updated_ema_w

        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        e_vq_loss = F.mse_loss(x.detach(), quantized)
        loss_vq = self.vq_cost*e_vq_loss
        loss_commit = self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 2, 1).contiguous()
        
        # if not self.training and self.infer:
        #     return quantized, encoding_indices 
        return quantized, loss_commit,loss_vq ,encoding_indices 

    def get_code_indices(self, flat_x):
        # compute L2 distance
        distances = (
                torch.sum(flat_x ** 2, dim=1, keepdim=True) +
                torch.sum(self.embeddings ** 2, dim=1) -
                2. * torch.matmul(flat_x, self.embeddings.t())
        )  # [N, M]
        encoding_indices = torch.argmin(distances, dim=1)  # [N,]
        return encoding_indices

    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return F.embedding(encoding_indices, self.embeddings)
    
class Encoder(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens,uniform_length=12):
        super(Encoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self.uniform_length  = uniform_length
        if self.uniform_length == 10:
            self.project = nn.Conv1d(in_dim, self._num_hiddens // 4, 1, 1)
            # self._enc_0 = Res_CNR_Stack(self._num_hiddens // 8, self._num_residual_layers, leaky=True)
            # self._down_0 = ConvNormRelu(self._num_hiddens // 8, self._num_hiddens // 4, leaky=True, residual=True,
            #                             sample='down_resample')
        elif self.uniform_length == 12:
            self.project = nn.Conv1d(in_dim, self._num_hiddens // 8, 1, 1)
            self._enc_0 = Res_CNR_Stack(self._num_hiddens // 8, self._num_residual_layers, leaky=True)
            self._down_0 = ConvNormRelu(self._num_hiddens // 8, self._num_hiddens // 4, leaky=True, residual=True,
                                        sample='down_resample')
        else:
            raise ValueError
            
        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)
        self._down_1 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, residual=True, sample='down')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._down_2 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens, leaky=True, residual=True, sample='down')
        
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)

        self.pre_vq_conv = nn.Conv1d(self._num_hiddens, embedding_dim, 1, 1)

    def forward(self, x):
        h = self.project(x)
        if self.uniform_length == 10:
            pass
        elif self.uniform_length == 12:
            h = self._enc_0(h)
            h = self._down_0(h)
        else:
            raise ValueError
        h = self._enc_1(h)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        h = self._enc_3(h)
        h = self.pre_vq_conv(h)
        return h


class Frame_Enc(nn.Module):
    def __init__(self, in_dim, num_hiddens):
        super(Frame_Enc, self).__init__()
        self.in_dim = in_dim
        self.num_hiddens = num_hiddens

        # self.enc = transformer_Enc(in_dim, num_hiddens, 2, 8, 256, 256, 256, 256, 0, dropout=0.1, n_position=4)
        self.proj = nn.Conv1d(in_dim, num_hiddens, 1, 1)
        self.enc = Res_CNR_Stack(num_hiddens, 2, leaky=True)
        self.proj_1 = nn.Conv1d(256*4, num_hiddens, 1, 1)
        self.proj_2 = nn.Conv1d(256*4, num_hiddens*2, 1, 1)

    def forward(self, x):
        # x = self.enc(x, None)[0].reshape(x.shape[0], -1, 1)
        x = self.enc(self.proj(x)).reshape(x.shape[0], -1, 1)
        second_last = self.proj_2(x)
        last = self.proj_1(x)
        return second_last, last



class Decoder(nn.Module):
    def __init__(self, out_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, ae=False, uniform_length = 12):
        super(Decoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens
        self.uniform_length =  uniform_length
        self.aft_vq_conv = nn.Conv1d(embedding_dim, self._num_hiddens, 1, 1)

        self._dec_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._up_2 = ConvNormRelu(self._num_hiddens, self._num_hiddens // 2, leaky=True, residual=True, sample='up')
        self._dec_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._up_3 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens // 4, leaky=True, residual=True,sample='up')
        self._dec_3 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)

        if ae:
            self.frame_enc = Frame_Enc(out_dim, self._num_hiddens // 4)
            self.gru_sl = nn.GRU(self._num_hiddens // 2, self._num_hiddens // 2, 1, batch_first=True)
            self.gru_l = nn.GRU(self._num_hiddens // 4, self._num_hiddens // 4, 1, batch_first=True)
        if self.uniform_length == 10:
            self.project = nn.Conv1d(self._num_hiddens // 4, out_dim, 1, 1)
        elif self.uniform_length == 12:
            self._up_4 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 8, leaky=True, residual=True, sample='up_resample')
            self._dec_4 = Res_CNR_Stack(self._num_hiddens // 8, self._num_residual_layers, leaky=True)
            self.project = nn.Conv1d(self._num_hiddens // 8, out_dim, 1, 1)
        else:
            raise ValueError
    def forward(self, h, last_frame=None):

        h = self.aft_vq_conv(h)
        h = self._dec_1(h)
        h = self._up_2(h)
        h = self._dec_2(h)
        h = self._up_3(h)
        h = self._dec_3(h)
        if self.uniform_length == 10:
            pass
        elif self.uniform_length == 12:
            h = self._up_4(h)
            h = self._dec_4(h)
        else:
            raise ValueError
        recon = self.project(h)
        return recon, None


class Pre_VQ(nn.Module):
    def __init__(self, num_hiddens, embedding_dim, num_chunks):
        super(Pre_VQ, self).__init__()
        self.conv = nn.Conv1d(num_hiddens, num_hiddens, 1, 1, 0, groups=num_chunks)
        self.bn = nn.GroupNorm(num_chunks, num_hiddens)
        self.relu = nn.ReLU()
        self.proj = nn.Conv1d(num_hiddens, embedding_dim, 1, 1, 0, groups=num_chunks)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.proj(x)
        return x


class vqvae1d(nn.Module):
    """VQ-VAE"""

    def __init__(self, encoder_config, decoder_config,vq_config, in_dim=45, 
                 
                #  embedding_dim=64, num_embeddings=1024,
                #  num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=512,
                #  commitment_cost=0.25,vq_cost=1, 
                 decay=0.99, share=False, infer=False):
        super().__init__()
        self.in_dim = vq_config["in_dim"]
        embedding_dim= vq_config['embedding_dim']
        num_embeddings= vq_config['num_embeddings']
        num_hiddens= vq_config['num_hiddens']
        num_residual_layers = vq_config['num_residual_layers']
        num_residual_hiddens = vq_config['num_residual_hiddens']
        commitment_cost = vq_config['commitment_cost']
        vq_cost = vq_config['vq_cost']
        self.infer = infer
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.share_code_vq = share
        self.uniform_length = 10
        self.encoder = Encoder(self.in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens,uniform_length=self.uniform_length)
        self.vq_layer = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, vq_cost,decay)
        self.decoder = Decoder(self.in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens,uniform_length=self.uniform_length)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward(self, gt_poses, id=None, pre_state=None):
        z = self.encoder(gt_poses)
        # if not self.training and self.infer:
        #     e, _ , _, encoding_indices  = self.vq_layer(z)
        #     # x_recon, cur_state = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)
        #     return e, encoding_indices 

        e, loss_commit, loss_vq,  encoding_indices  = self.vq_layer(z)
        gt_recon, cur_state = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)

        return loss_commit, loss_vq, gt_recon, e, encoding_indices, z
    def encode(self, gt_poses, id=None):
        z = self.encoder(gt_poses.transpose(1, 2))
        e, latents = self.vq_layer(z)
        return e, latents

    def decode(self, b, w, e=None, latents=None, pre_state=None):
        if e is not None:
            x = self.decodder(e, pre_state.transpose(1, 2) if pre_state is not None else None)
        else:
            e = self.vq_layer.quantize(latents)
            e = e.view(b, w, -1).permute(0, 2, 1).contiguous()
            x = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)
        return x


class Conv1d(nn.Module):
    def __init__(self, 
                 encoder_config : List[List[int]], 
                 decoder_config : List[List[int]]) -> None:
        super().__init__()
        
        num_layers = len(encoder_config)
        
        modules = []
        for i, c in enumerate(encoder_config):
            modules.append(nn.Conv1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
            
            if i < (num_layers-1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))
        
        self.encoder = nn.Sequential(*modules)
        
        
        num_layers = len(decoder_config)
        
        modules = []
        for i, c in enumerate(decoder_config):
            modules.append(nn.ConvTranspose1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
            
            if i < (num_layers-1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, x : torch.Tensor):
        """
        x : (batch_size, dim_feat, time).
        """
        
        latent_code = self.encoder(x)
        x_recon = self.decoder(latent_code)
        return latent_code, x_recon 


# --------------------------------------------------------
# Reference: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
# --------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100) -> None:
        super().__init__()

        assert d_model % 2 == 0

        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe.data[0, :, 0::2].copy_(torch.sin(position * div_term))
        self.pe.data[0, :, 1::2].copy_(torch.cos(position * div_term))

    def forward(self, x) -> torch.Tensor:
        """
        x: [N, L, D]
        """
        x = x + self.pe[:, :x.shape[1], :]

        return self.dropout(x)


# --------------------------------------------------------
# Reference: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, mo_dim, lxm_dim,
                 embed_dim=512, depth=6, num_heads=8,
                 decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
                 mlp_ratio=4, activation='gelu', norm_layer=nn.LayerNorm, dropout=0.1) -> None:
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.encoder_embed = nn.Linear(mo_dim, embed_dim, bias=True)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding(embed_dim, dropout)  # hack: max len of position is 100

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim,
                                                                        nhead=num_heads,
                                                                        dim_feedforward=int(mlp_ratio*embed_dim),
                                                                        dropout=dropout,
                                                                        activation=activation,
                                                                        batch_first=True),
                                             num_layers=depth)

        self.norm = norm_layer(embed_dim)
        self.lxm_embed = nn.Linear(embed_dim, lxm_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(lxm_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = PositionalEncoding(decoder_embed_dim, dropout)  # hack: max len of position is 100

        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=decoder_embed_dim,
                                                                        nhead=decoder_num_heads,
                                                                        dim_feedforward=int(mlp_ratio*decoder_embed_dim),
                                                                        dropout=dropout,
                                                                        activation=activation,
                                                                        batch_first=True),
                                             num_layers=decoder_depth)

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, mo_dim, bias=True)
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize nn.Parameter
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed.pe, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed.pe, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward_encoder(self, x):
        """
        x: [N, D, L]
        """
        x = torch.einsum('NDL->NLD', x)

        # embed motion sequence
        x = self.encoder_embed(x)

        # append cls token
        cls_token = self.cls_token.repeat(x.shape[0], x.shape[1], 1)
        x = torch.cat([cls_token, x], dim=1)

        # add pos embed
        x = self.pos_embed(x)

        # apply Transformer blocks
        x = self.encoder(x)
        x = self.norm(x)
        x = self.lxm_embed(x)

        lxm = torch.einsum('NLD->NDL', x[:, :1, :])

        return lxm

    def forward_decoder(self, lxm, mo_len):
        '''
        lxm: [N, D, L]
        '''
        lxm = torch.einsum('NDL->NLD', lxm)

        # embed lexeme
        x = self.decoder_embed(lxm)

        # append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], mo_len, 1)
        x = torch.cat([x, mask_tokens], dim=1)

        # add pos embed
        x = self.decoder_pos_embed(x)

        # add Transformer blocks
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove lxm token
        x = x[:, 1:, :]

        x = torch.einsum('NLD->NDL', x)

        return x

    def forward(self, x):
        """
        x: [N, D, L]
        """
        _, _, L = x.shape

        lxm = self.forward_encoder(x)  # x[5,45,10] -> [5,96,1]
        x_hat = self.forward_decoder(lxm, L)

        return lxm, x_hat

# region Test.

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # region Conv1d
    
    encoder_config = [
        [45, 64, 3, 1, 1],
        [64, 128, 4, 2, 1],
        [128, 156, 3, 1, 1],
        [156, 192, 4, 2, 1]
    ]
    decoder_config = [
        [192, 156, 4, 2, 1],
        [156, 128, 3, 1, 1],
        [128, 64, 4, 2, 1],
        [64, 45, 3, 1, 1]
    ]
    vq_config= {
            "in_dim": 45,
            "embedding_dim": 512,
            "num_embeddings": 512,

            "num_hiddens": 1024,  # minize 512

            "num_residual_layers":2,
            "num_residual_hiddens":512,

            "commitment_cost":0.02,
            "vq_cost":0.0,
    }
    #
    
    # encoder_config = [
    #     [45, 64, 3, 1, 1],
    #     [64, 128, 4, 2, 1],
    #     [128, 156, 3, 1, 1],
    #     [156, 192, 4, 2, 1]
    # ]
    # decoder_config = [
    #     [192, 156, 3, 1, 1],
    #     [156, 128, 4, 2, 1],
    #     [128, 64, 3, 1, 1],
    #     [64, 45, 4, 2, 1]
    # ]
    # conv_1d = vqvae1d(encoder_config, decoder_config,vq_config).to(device)
    #
    # x = torch.randn((10, 45, 10)).to(device)
    # loss_commit, loss_vq, gt_recon = conv_1d(x)
    #
    # print(motif.shape, x_hat.shape)
    
    # endregion

    # region Transformer

    model = Transformer(45, 96).to(device)
    #
    x = torch.randn((5, 45, 10)).to(device)  # [N, D, L]
    
    lexeme, x_hat = model(x)
    #
    print(lexeme.shape)
    print(x_hat.shape)

    # endregion
    
    # region network statistics

    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    #
    # print(get_parameter_number(model))

    # endregion

# endregion