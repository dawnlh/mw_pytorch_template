from .builder import LOSSES
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import SSIM, MS_SSIM  # pip install pytorch-msssim


# ===========================
# weighted_loss
# ===========================

@LOSSES.register_module 
class WeightedLoss(nn.Module):
    """
    weighted multi-loss
    loss_conf_dict: {loss_type1: weight|[weight,{kwargs_dict_for_init}], ...}
        eg: loss_conf_dict = {'CharbonnierLoss':0.5, 'EdgeLoss':0.5}
        eg: loss_conf_dict = {'CharbonnierLoss':[0.5, {'eps':1e-3}], 'EdgeLoss':0.5}
    """

    def __init__(self, loss_conf_dict):
        super(WeightedLoss, self).__init__()
        self.loss_conf_dict = loss_conf_dict

        # instantiate classes
        self.losses = []
        for k, v in loss_conf_dict.items():
            if isinstance(v, (float, int)):
                assert v >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES.get(k)(), 'weight': v})
            elif isinstance(v, list) and len(v) == 2:
                assert v[0] >= 0, f"loss'weight {k}:{v} should be positive"
                self.losses.append({'cls': LOSSES.get(k)(**v[1]), 'weight': v[0]})
            else:
                raise ValueError(
                    f"the Key({k})'s Value {v} in Dict(loss_conf_dict) should be scalar(weight) | list[weight, args] ")

    def forward(self, output, target):
        loss_v = 0
        for loss in self.losses:
            loss_v += loss['cls'](output, target)*loss['weight']

        return loss_v

# ===========================
# basic_loss
# ===========================


@LOSSES.register_module 
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        diff = output.to('cuda:0') - target.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss


@LOSSES.register_module 
class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


@LOSSES.register_module 
class SSIMLoss(SSIM):
    """Structural Similarity Index Measure Loss
    Directly use the SSIM class provided by pytorch_msssim
    """
    pass


@LOSSES.register_module 
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, output, target):
        loss = self.loss(self.laplacian_kernel(output.to('cuda:0')),
                         self.laplacian_kernel(target.to('cuda:0')))
        return loss


@LOSSES.register_module 
class FFTLoss(nn.Module):
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, output, target):
        diff = torch.fft.fft2(output.to('cuda:0')) - \
            torch.fft.fft2(target.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss


@LOSSES.register_module 
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self, output, *args):
        batch_size = output.size()[0]
        h_x = output.size()[2]
        w_x = output.size()[3]
        count_h = self._tensor_size(output[:, :, 1:, :])
        count_w = self._tensor_size(output[:, :, :, 1:])
        h_tv = torch.pow(
            (output[:, :, 1:, :]-output[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow(
            (output[:, :, :, 1:]-output[:, :, :, :w_x-1]), 2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size
