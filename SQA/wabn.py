from copy import deepcopy

import torch
import torch.nn as nn

class WABN2d(nn.Module):
    """Weighted Batch Normalization"""

    def __init__(self, BN, w = 0.5, n = 8, alpha = 0.9):
        super(WABN2d, self).__init__()
        self.w = w
        self.n = n
        self.alpha = alpha

        self.eps = deepcopy(BN.eps)
        self.affine = deepcopy(BN.affine)
        self.running_mean = deepcopy(BN.running_mean)
        self.running_var = deepcopy(BN.running_var)
        self.weight = deepcopy(BN.weight)
        self.bias = deepcopy(BN.bias)
        self.training = True

    def forward(self, x):
        with torch.no_grad():
            if self.training:
                if self.n > 0:
                    x_t = x[0]
                    x_aug = x[1:]

                    mu = self.w * x_t.mean([1,2]) + (1 - self.w) * x_aug.mean([0,2,3])
                    var_t = ((x_t - mu[:, None, None]) ** 2).sum([1,2]) / (x_t.size(1) * x_t.size(2) - 1)
                    var_aug = ((x_aug - mu[:, None, None]) ** 2).sum([0,2,3]) / (x_aug.size(0) * x_aug.size(2) * x_aug.size(3) - 1)
                    var = (self.w * var_t + (1 - self.w) * var_aug)
                else:
                    mu = x.mean()
                    var = x.var()

                self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * mu
                self.running_var = self.alpha * self.running_var + (1 - self.alpha) * var

            mu = self.running_mean
            var = self.running_var

        x = (x - mu[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x

    def __repr__(self):
        rep = '{name}(omega={w}, alpha={alpha}, num_features={num_features}, eps={eps},' \
              ' affine={affine}, requires_grad={requires_grad_})'
        return rep.format(name=self.__class__.__name__, **self.__dict__)