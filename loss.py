import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import json

## ================================================================================
def main():
    sn = SqueezeNet()
    return

## ================================================================================
def huber(pred, target):
    return F.huber_loss(pred, target, delta=3)

# ---------------------------------------------------------------------------------
def mse(pred, target):
    loss = F.mse_loss(pred, target)
    return loss

# ---------------------------------------------------------------------------------
def wt_mse(pred, target):
    wt = torch.abs(target)
    return F.mse_loss(pred, target, weight=wt)

# ---------------------------------------------------------------------------------
def quantile(pred, target, alpha=0.95):
    diff = pred - target
    return torch.mean(torch.where(diff > 0, (1 - alpha) * diff, alpha * torch.abs(diff)))

# ---------------------------------------------------------------------------------
def neg_relu(pred):
    return torch.mean(F.relu(-1 * pred))  # relu set neg values to 0 --> switch

# ---------------------------------------------------------------------------------
def mae(pred, target):
    return F.l1_loss(pred, target)

# ---------------------------------------------------------------------------------
def wt_mae(pred, target):
    wt = torch.where(target <= MIN, MIN, target)
    wt = torch.where(wt >= MAX, MAX, wt)
    l1 = torch.mean(wt * torch.abs(pred - target))
    return l1

# ---------------------------------------------------------------------------------
def pcc(pred, target):
    t = torch.exp(target) - 1
    mt = torch.mean(t, dim=(2, 3), keepdim=True)
    ts = t - mt
    p = torch.exp(pred) - 1
    mp = torch.mean(p, dim=(2, 3), keepdim=True)
    ps = p - mp
    eps = 0
    pcc_loss = torch.sum(ps * ts) / torch.sqrt(torch.sum(ps ** 2) * torch.sum(ts ** 2) + eps)
    return 1 - pcc_loss

# ---------------------------------------------------------------------------------
def fft(pred, target):
    p = pred - torch.mean(pred, dim=(2, 3), keepdim=True)
    t = target - torch.mean(target, dim=(2, 3), keepdim=True)
    fft_pred = torch.abs(torch.fft.fft2(p, norm='ortho'))
    fft_target = torch.abs(torch.fft.fft2(t, norm='ortho'))
    fft_wt = torch.log(1 + F.l1_loss(fft_pred, fft_target, reduction='none'))
    fft_loss = F.l1_loss(fft_pred, fft_target, weight=fft_wt)
    return fft_loss

# ---------------------------------------------------------------------------------
def cross_entropy(pred, target):
    target = target.to(torch.long)
    loss = F.cross_entropy(pred, target)#, weight=wt)
    return loss

# ---------------------------------------------------------------------------------
def dice(pred, target):
    a = pred.contiguous().view(-1)
    b = target.contiguous().view(-1)
    return 1 - (2 * (a * b).sum() + 1e-5) / (a.sum() + b.sum() + 1e-5)
    
# ---------------------------------------------------------------------------------
def comp_loss_fn(pred, target):
    mse_loss = mse(pred, target)
    #mae_loss = mae(pred, target)
    fft_loss = fft(pred, target)
    #return mse_loss + 0.25 * fft_loss
    return mse_loss + 0.35 * fft_loss

# ---------------------------------------------------------------------------------
def unstd(pred, mn, std):
    return pred * std + mn 

# ---------------------------------------------------------------------------------
def linex(pred, target, a=-0.5):
    diff = pred - target
    return torch.mean(torch.exp(a * diff) - a * diff - 1)

# ---------------------------------------------------------------------------------
class SqueezeNet(nn.Module):
    """SqueezeNet/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """
    def __init__(self, model_name, conv_index: str = 'mid'):
        super(SqueezeNet, self).__init__()
        weights = torchvision.models.SqueezeNet1_0_Weights.DEFAULT
        features = torchvision.models.squeezenet1_0(weights=weights).features
        self.transforms = weights.transforms()
        modules = [m for m in features]
        # 0: Conv, 1: ReLU, 2: MaxPool, 3-5: "Fire", 6: MaxPool, 7-10: "Fire", 11: MaxPool, 12: Fire
        if conv_index == 'start':
            self.sn = nn.Sequential(*modules[:3])
        elif conv_index == 'mid':
            self.sn = nn.Sequential(*modules[:8])
        elif conv_index == 'late':
            self.sn = nn.Sequential(*modules[:11])
        elif conv_index == 'full':
            self.sn = nn.Sequential(*modules)

        with open(f'./models/{model_name}/norm_vars.json', 'r') as f:
            stats = json.load(f)
            self.p_mn = stats['precipitationmn']
            self.p_std = stats['precipitationstd']

        #vgg_mean = (0.485, 0.456, 0.406)
        #vgg_std = (0.229, 0.224, 0.225)
        #self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.sn.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SqueezeNet/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """
        def _forward(x):
            #x = self.sub_mean(x)
            x = self.sn(x)
            return x

        mse_loss = mse(pred, target)
        unstd_pred = unstd(pred.clone(), self.p_mn, self.p_std)
        neg_pen_loss = neg_relu(unstd_pred)  # consider shifting this by one..? or use SELU or something.?
        quantile_loss = quantile(pred, target, alpha=0.75)
        q2_loss = quantile(pred, target, alpha=0.95)
        #fft = fft(pred, target)
        mae_loss = mae(pred, target)
        #linex_loss = linex(pred, target, a=-0.1)
        #pcc = pcc(pred, target)

        _pred = torch.cat([pred] * 3, dim=1)
        _target = torch.cat([target] * 3, dim=1)
        #_pred = self.transforms(_pred)
        #_target = self.transforms(_target)
        _pred = _forward(_pred)  # should this also be with torch.no_grad() when validating.?

        with torch.no_grad():
            _target = _forward(_target.detach())

        perceptual_loss = F.mse_loss(_pred, _target)  # formerly mse!

        #return 1e-3 * perceptual_loss + mse_loss + neg_pen_loss
        #print(perceptual_loss.item(), mae_loss.item(), linex_loss.item(), neg_pen_loss.item())
        return 1e-3 * perceptual_loss + 5 * quantile_loss + 2 * q2_loss + mse_loss + neg_pen_loss

## ================================================================================
if __name__ == '__main__':
    main()
