# region Import.

from network2 import *
from einops import rearrange
# endregion


__all__ = ["infer_train", "initialize_net"]


def infer_train(batch, device, net, uniform_len, num_blocks, name_net):
    if name_net == "RNN":
        aud = batch["audio"].to(device)  # [N, L, D]
        wav = batch["audio_wav"].to(device) # [N, 10*12*800]
        wav = rearrange(wav, 'n (block uni_len hop) -> n (block uni_len) hop', block=10, hop=800) # [N, 10, 12*800]
        mo = batch["motion"].to(device)  # [N, L, D]
        lxm = batch["lexeme"].to(device)  # [N, B, D]
        vid_indices = batch["vid_indices"].to(device) 
        mo_gt = mo[:, uniform_len: (num_blocks-1)*uniform_len, :]
        # lxm_gt = lxm[:, 1: (num_blocks-1), :]

        x_audio = aud.permute((0, 2, 1)).contiguous()
        x_motion = mo.permute((0, 2, 1)).contiguous()
        x_lexeme = lxm.permute((0, 2, 1)).contiguous()
        x_wav = wav.contiguous()
        ret = net(x_audio, x_motion, x_lexeme, x_wav,vid_indices)
        ret['mo_hat'] = ret['mo_hat'].permute((0, 2, 1)).contiguous()
        ret['z_mu'] = ret['z_mu'].permute((0, 2, 1)).contiguous()
        ret['z_logvar'] = ret['z_logvar'].permute((0, 2, 1)).contiguous()
        ret['mo_gt'] = mo_gt
        return ret

    else:
        raise NotImplementedError


def initialize_net(config, config_data_preprocessing):
    if config['network']['name'] == "RNN":
        net = MotionGenerator_RNN(**config['network']['hparams'])

    else:
        raise NotImplementedError

    return net