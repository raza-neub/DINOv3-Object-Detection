"""
MuSGD: hybrid Muon + SGD optimizer for detection head training.
Adapted from Ultralytics implementation (AGPL-3.0 License).

MuSGD applies an orthogonalized Muon update (via Newton-Schulz iterations)
plus a standard SGD momentum update to 2D+ weight tensors (conv kernels,
linear weights).  1-D parameters (biases, GroupNorm scales/shifts, the
per-level learnable scales in FCOSHead) use plain SGD momentum.

Usage in training:
    from src.optimizer import build_musgd_optimizer
    optimizer = build_musgd_optimizer(
        head_params=model_head.parameters(),
        lr=1e-4, momentum=0.95, weight_decay=1e-4,
        muon_weight=0.5, sgd_weight=0.5,
    )
"""

from __future__ import annotations

import torch
from torch import optim


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalization
# ---------------------------------------------------------------------------

def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Approximate orthogonalization of G via 5 Newton-Schulz steps.

    Returns a matrix whose singular values are in ~[0.5, 1.5] (empirically
    works well as an optimizer update direction).  Computation is done in
    bfloat16 for speed; result is cast back to the original dtype.
    """
    assert G.ndim == 2
    orig_dtype = G.dtype
    X = G.bfloat16()
    X = X / (X.norm() + eps)           # ensure top singular value <= 1
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for a, b, c in [
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
        (3.4445, -4.7750, 2.0315),
    ]:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(orig_dtype)


def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    """Compute the orthogonalized Muon update for one parameter.

    For 4-D conv filters the tensor is reshaped to (out_ch, *) before
    orthogonalization.  The update is scaled by sqrt(max(1, rows/cols))
    to keep the effective step size comparable across layer shapes.
    """
    momentum.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum.clone()

    if update.ndim > 2:                         # conv filters: 3-D (Conv1d) or 4-D (Conv2d)
        update = update.view(update.size(0), -1)

    update = zeropower_via_newtonschulz5(update)
    # Scale by sqrt(max(1, rows/cols)) so updates are ~unit-norm regardless of shape
    rows, cols = update.size(0), update.size(1)
    update = update * max(1.0, rows / cols) ** 0.5
    return update


# ---------------------------------------------------------------------------
# MuSGD optimizer
# ---------------------------------------------------------------------------

class MuSGD(optim.Optimizer):
    """Hybrid Muon + SGD optimizer.

    Parameter groups with ``use_muon=True`` receive both an orthogonalized
    Muon component (scaled by ``self.muon``) and a classical SGD momentum
    component (scaled by ``self.sgd``).  Parameter groups with
    ``use_muon=False`` receive only SGD with momentum.

    The Muon orthogonalization works best for 2-D+ tensors (conv kernels,
    linear weights).  1-D tensors (biases, norms) should go into a
    ``use_muon=False`` group.

    Args:
        params: parameter groups or raw iterable of parameters.
        lr (float): learning rate.
        momentum (float): momentum coefficient (beta for Muon, classic for SGD).
        weight_decay (float): L2 weight decay applied in the SGD component.
        nesterov (bool): use Nesterov momentum in both components.
        use_muon (bool): default for whether a group uses Muon.
        muon (float): scaling factor for the Muon component lr.
        sgd (float): scaling factor for the SGD component lr.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            use_muon=use_muon,
        )
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]

            if group["use_muon"]:
                # --- Muon + SGD hybrid ---
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if not state:
                        state["muon_buf"] = torch.zeros_like(p)
                        state["sgd_buf"] = torch.zeros_like(p)

                    # Muon component (orthogonalized)
                    if p.ndim >= 2:
                        mu_update = muon_update(grad, state["muon_buf"],
                                                beta=momentum, nesterov=nesterov)
                        p.add_(mu_update.reshape(p.shape), alpha=-(lr * self.muon))
                    else:
                        # Fallback for 1-D params in a muon group: plain SGD
                        state["muon_buf"].mul_(momentum).add_(grad)
                        p.add_(state["muon_buf"], alpha=-(lr * self.muon))

                    # SGD component (with weight decay)
                    g = grad
                    if wd != 0.0:
                        g = grad.add(p, alpha=wd)
                    state["sgd_buf"].mul_(momentum).add_(g)
                    sgd_upd = (g.add(state["sgd_buf"], alpha=momentum)
                               if nesterov else state["sgd_buf"])
                    p.add_(sgd_upd, alpha=-(lr * self.sgd))

            else:
                # --- Pure SGD with momentum ---
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if wd != 0.0:
                        grad = grad.add(p, alpha=wd)
                    state = self.state[p]
                    if not state:
                        state["momentum_buf"] = torch.zeros_like(p)
                    state["momentum_buf"].mul_(momentum).add_(grad)
                    upd = (grad.add(state["momentum_buf"], alpha=momentum)
                           if nesterov else state["momentum_buf"])
                    p.add_(upd, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_musgd_optimizer(
    head_params,
    lr: float = 1e-4,
    momentum: float = 0.95,
    weight_decay: float = 1e-4,
    muon_weight: float = 0.5,
    sgd_weight: float = 0.5,
    nesterov: bool = True,
) -> MuSGD:
    """Build a MuSGD optimizer with two parameter groups.

    - 2-D+ tensors  (conv kernels, linear weights): Muon + SGD hybrid.
    - 1-D tensors   (biases, GroupNorm params, FCOSHead per-level scales):
      plain SGD with momentum (no Muon; weight decay also disabled for these).

    Args:
        head_params: iterable of parameters (e.g. model_head.parameters()).
        lr: learning rate shared by both groups.
        momentum: momentum coefficient (default 0.95, works well for Muon).
        weight_decay: L2 penalty applied only in the 2-D+ group.
        muon_weight: fractional weight for the Muon component (0–1).
        sgd_weight: fractional weight for the SGD component (0–1).
        nesterov: use Nesterov momentum.
    Returns:
        Configured MuSGD instance.
    """
    muon_params, sgd_params = [], []
    for p in head_params:
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            sgd_params.append(p)

    param_groups = [
        {
            "params": muon_params,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "use_muon": True,
        },
        {
            "params": sgd_params,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": 0.0,        # never decay biases / norm params
            "nesterov": nesterov,
            "use_muon": False,
        },
    ]
    return MuSGD(param_groups, muon=muon_weight, sgd=sgd_weight)
