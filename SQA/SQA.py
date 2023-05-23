from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.transforms as T

from SQA.wabn import WABN2d
import random

class SQA(nn.Module):
    """
    SQA adapts a model to the unseen domain by a single query.
    The model adapts to the query by updating on every forward.
    """
    def __init__(self, model, optimizer, lambd = 0.0051, aug_n = 8, steps = 1, episodic=True, task='c10c'):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.lambd = lambd
        self.aug_n = aug_n
        self.task = task
        assert steps > 0, "SQA requires >= 1 step(s) for forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            out = forward_and_adapt(x, self.model, self.optimizer, self.lambd, self.aug_n, self.task)

        return out

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")

        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer, lambd, aug_n, task):
    """
    Forward and adapt model on batch of data.
    Measure correlation matrix and update params.
    """
    raw = x
    x = augSamples(x, aug_n)
    # waBN Forward
    model.train()
    out = model(x)
    # Affine Calibration
    if task == 'ReID':
    # # 2D -> 1D if needed, can be changed to other pooling methods or a projector
    # # directly faltten would significantly increase GPU memory cost and a much bigger loss 
    # out = torch.flatten(out, start_dim = 1)
        z_t = out[0].repeat(aug_n, 1)
        z_aug = out[1:]

        c = z_t.T @ z_aug / aug_n
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # disable moving average recal for eval
    model.eval()
    with torch.no_grad():
        out = model(raw)

    return out


def collect_params(model):
    """
    Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, WABN2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, w = 0.5, alpha = 0.9):
    """Configure model for use with SQA."""
    # train mode, because SQA optimizes the model by contrastive learning
    model.train()
    # disable grad, then (re-)enable only what SQA updates
    model.requires_grad_(False)
    # configure norm for SQA updates: enable grad + force batch statisics
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            newBN = WABN2d(m, w, alpha)
            set_module(model, nm, newBN)
    #check model
    check_model(model)

    return model


def check_model(model):
    """Check model for compatability with SQA."""
    is_training = model.training
    assert is_training, "SQA needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "SQA needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "SQA should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, WABN2d) for m in model.modules()])
    assert has_bn, "SQA needs normalization for its optimization"


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def augSamples(x, aug_n):
    trans = [T.ColorJitter(0.2, 0.1, 0.5),
            T.RandomHorizontalFlip(1),
            T.RandomVerticalFlip(1),
            T.GaussianBlur(3),
            T.RandomErasing(1),
            T.RandomRotation(15),
            T.Grayscale(3),
            T.RandomPerspective(p=1)
            ]

    trans = random.sample(trans, aug_n)

    raw = x[0,:,:,:]
    out = []
    out.append(raw)

    for i in range(len(trans)):
        out.append(trans[i](raw))

    out = torch.stack(out, 0)

    return out