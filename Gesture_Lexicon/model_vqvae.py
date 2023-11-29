# region Import.

import torch

import numpy as np
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List
from vqvae_modules import VectorQuantizerEMA, ConvNormRelu, Res_CNR_Stack
# endregion


__all__ = ["Conv1d", "Transformer","vqvae1d"]

class vqvae1d_encoder(nn.Module):
    def __init__(self,encoder_config,embedding_dim) -> None:
        super().__init__()
        num_layers = len(encoder_config)
        modules = []
        for i, c in enumerate(encoder_config):
            modules.append(nn.Conv1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
            
            if i < (num_layers-1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))
        self.encoder = nn.Sequential(*modules)
        self.pre_vq_conv = nn.Conv1d(encoder_config[-1][1], embedding_dim, 1, 1)
        self.lxm_embed = nn.Linear(encoder_config[-1][1],embedding_dim, bias=True)
    def forward(self,x):
        z = self.encoder(x)
        # z = self.pre_vq_conv(z)
        # z: [b,192,1]
        z = z.permute(0, 2, 1).contiguous()
        z = self.lxm_embed(z)
        z = z.permute(0, 2, 1).contiguous()
        return z
    
class vqvae1d_decoder(nn.Module):
    def __init__(self,decoder_config,embedding_dim) -> None:
        super().__init__()
        num_layers = len(decoder_config)
        modules = []
        for i, c in enumerate(decoder_config):
            modules.append(nn.ConvTranspose1d(in_channels=c[0], out_channels=c[1], kernel_size=c[2], stride=c[3], padding=c[4]))
           
            if i < (num_layers-1):
                modules.append(nn.BatchNorm1d(c[1]))
                modules.append(nn.LeakyReLU(1.0, inplace=True))
        self.decoder = nn.Sequential(*modules)
        self.aft_vq_conv = nn.Conv1d(embedding_dim, decoder_config[0][0], 1, 1)
        self.decoder_embed = nn.Linear(embedding_dim, decoder_config[0][0], bias=True)
        
        
    def forward(self,e):
        # e = self.aft_vq_conv(e)
        e = e.permute(0, 2, 1).contiguous()
        e = self.decoder_embed(e)
        e = e.permute(0, 2, 1).contiguous()  # 5,192,1
        re_con = self.decoder(e)
        return re_con
    
class vqvae1d(nn.Module):
    
    def __init__(self, 
                 encoder_config : List[List[int]], 
                 decoder_config : List[List[int]]) -> None:
        super().__init__()
        embedding_dim,num_embeddings = 192,50
        self.embeddiQng_dim = embedding_dim
        self.num_embeddings = num_embeddings
        commitment_cost=1.0
        decay=0.99

        
        self.encoder = vqvae1d_encoder(encoder_config,embedding_dim)
        self.vq_layer = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder = vqvae1d_decoder(decoder_config,embedding_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)
        
    
    def forward(self, x : torch.Tensor):
        """
        x : (batch_size, dim_feat, time).
        """
        
        z = self.encoder(x)
        e, e_loss = self.vq_layer(z)
        gt_recon= self.decoder(e)
        return  e_loss, gt_recon





class Encoder(nn.Module):
    def __init__(self, in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.project = ConvNormRelu(in_dim, self._num_hiddens // 4, leaky=True)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)
        self._down_1 = ConvNormRelu(self._num_hiddens // 4, self._num_hiddens // 2, leaky=True, residual=True,
                                    sample='down')
        self._enc_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._down_2 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens, leaky=True, residual=True, sample='down')
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)

        self.pre_vq_conv = nn.Conv1d(self._num_hiddens, embedding_dim, 1, 1)

    def forward(self, x):
        h = self.project(x)
        h = self._enc_1(h)
        h = self._down_1(h)
        h = self._enc_2(h)
        h = self._down_2(h)
        h = self._enc_3(h)
        h = self.pre_vq_conv(h)
        return h
    
class Decoder(nn.Module):
    def __init__(self, out_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, ae=False):
        super(Decoder, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.aft_vq_conv = nn.Conv1d(embedding_dim, self._num_hiddens, 1, 1)

        self._dec_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True)
        self._up_2 = ConvNormRelu(self._num_hiddens, self._num_hiddens // 2, leaky=True, residual=True, sample='up')
        self._dec_2 = Res_CNR_Stack(self._num_hiddens // 2, self._num_residual_layers, leaky=True)
        self._up_3 = ConvNormRelu(self._num_hiddens // 2, self._num_hiddens // 4, leaky=True, residual=True,
                                  sample='up')
        self._dec_3 = Res_CNR_Stack(self._num_hiddens // 4, self._num_residual_layers, leaky=True)
        self.project = nn.Conv1d(self._num_hiddens // 4, out_dim, 1, 1)

    def forward(self, h, last_frame=None):

        h = self.aft_vq_conv(h)
        h = self._dec_1(h)
        h = self._up_2(h)
        h = self._dec_2(h)
        h = self._up_3(h)
        h = self._dec_3(h)

        recon = self.project(h)
        return recon, None
    
class VQVAE(nn.Module):
    """VQ-VAE"""

    def __init__(self, in_dim, embedding_dim, num_embeddings,
                 num_hiddens, num_residual_layers, num_residual_hiddens,
                 commitment_cost=0.25, decay=0.99, share=False):
        super().__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.share_code_vq = share

        self.encoder = Encoder(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)
        self.vq_layer = VectorQuantizerEMA(embedding_dim, num_embeddings, commitment_cost, decay)
        self.decoder = Decoder(in_dim, embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens)

    def forward(self, gt_poses, id=None, pre_state=None):
        z = self.encoder(gt_poses.transpose(1, 2))
        if not self.training:
            e, _ = self.vq_layer(z)
            x_recon, cur_state = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)
            return e, x_recon

        e, e_q_loss = self.vq_layer(z)
        gt_recon, cur_state = self.decoder(e, pre_state.transpose(1, 2) if pre_state is not None else None)

        return e_q_loss, gt_recon.transpose(1, 2)



if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # region Conv1d
    
    encoder_config = [
        [42, 64, 5, 1, 0],
        [64, 128, 4, 2, 0],
        [128, 156, 4, 1, 0],
        [156, 192, 4, 1, 0]
    ]
    decoder_config = [
        [192, 156, 4, 1, 0],
        [156, 128, 4, 1, 0],
        [128, 64, 4, 2, 0],
        [64, 42, 5, 1, 0]
    ]
    #
    conv_1d = vqvae1d(encoder_config, decoder_config).to(device)
    #
    x = torch.randn((5, 42, 20)).to(device)
    motif, x_hat = conv_1d(x)
    #
    print(motif.shape, x_hat.shape)
    
    # endregion

    # region Transformer

    # model = Transformer(48, 96).to(device)
    #
    # x = torch.randn((5, 48, 10)).to(device)  # [N, D, L]
    #
    # lexeme, x_hat = model(x)
    #
    # print(lexeme.shape)
    # print(x_hat.shape)

    # endregion
    
    # region network statistics

    # def get_parameter_number(model):
    #     total_num = sum(p.numel() for p in model.parameters())
    #     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     return {'Total': total_num, 'Trainable': trainable_num}
    #
    # print(get_parameter_number(model))

    # endregion
