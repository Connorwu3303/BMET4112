"""
Aim 3 shared utilities: device selection, seeding, masked pooling, metrics.

None of this depends on the architecture, so changing the network can't quietly
change how it is trained, scored or seeded.

Masked pooling is what lets Aim 3 keep the Aim 1 correction. A CNN ends with a
pooling step that collapses time into a fixed length summary. Plain global
average pooling divides by the full window length, so a night with two hours of
unmeasurable signal gets its summary diluted by two hours of zeros. After
centring, zero means 95 percent, so this pulls every summary towards normal in
proportion to how much of the night was unmeasurable, and it does it most in the
poor quality recordings where artefact and disease go together. Masked pooling
divides by the analysable length instead, so unmeasured time contributes
nothing. It's the same idea as Aim 2 using valid time as the denominator of
every rate measure, just inside the network.

Device: Apple MPS when available. MPS doesn't implement every op, so CPU
fallback is switched on with an environment variable before torch loads any
kernels. MPS also has no float64 support (and float32 reductions over long
sequences are faster anyway), so everything is float32.
"""

import os
import random
import sys

import numpy as np

# Has to be set before the first MPS kernel runs. Ops missing from MPS then fall
# back to the CPU instead of raising, so one unimplemented op can't kill a run
# hours in.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "aim2"))
from ml_common import (  # noqa: E402
    evaluate_predictions, bootstrap_auc_ci, delong_test,
)

RESULTS_DIR = "/Volumes/ThesisHD/Data/processed/aim3/results"


def get_device(prefer="mps"):
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    """
    Seed every RNG that can affect a run.

    Bitwise determinism isn't possible on MPS, so runs only reproduce to within
    small numerical differences from the backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class MaskedAvgPool(nn.Module):
    """
    Global average over time using only analysable samples.

    x: (B, C, L), mask: (B, L) with 1 where analysable. Returns (B, C).

    The clamp covers a window with nothing analysable in it (only possible if
    it's all padding), which then returns zeros instead of dividing by zero.
    """

    def forward(self, x, mask):
        m = mask.unsqueeze(1)
        return (x * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)


class MaskedMaxPool(nn.Module):
    """
    Global max over time using only analysable samples.

    Unmeasured positions are set to a large negative value so they never win the
    max. Max pooling is useful because the clinically interesting part of an
    oximetry trace is episodic: a bad night is bad because of its worst
    stretches, and an average over eight hours can hide a severe half hour
    completely.
    """

    def forward(self, x, mask):
        m = mask.unsqueeze(1)
        return x.masked_fill(m == 0, float("-1e4")).max(dim=2).values


class MaskedStatsPool(nn.Module):
    """
    Concatenates masked mean, masked SD and masked max.

    These give the head the overall level, the variability and the extreme,
    which line up with what the oximetry literature looks at in an abnormal
    night: low baseline saturation, unstable saturation and deep individual
    desaturations. Triples the head's input width for almost no cost.
    """

    def __init__(self):
        super().__init__()
        self.avg = MaskedAvgPool()
        self.max = MaskedMaxPool()

    def forward(self, x, mask):
        m = mask.unsqueeze(1)
        n = m.sum(dim=2).clamp(min=1.0)
        mean = (x * m).sum(dim=2) / n
        var = ((x - mean.unsqueeze(2)) ** 2 * m).sum(dim=2) / n
        return torch.cat([mean, var.clamp(min=1e-8).sqrt(),
                          self.max(x, mask)], dim=1)


def downsample_mask(mask, factor):
    """
    Reduce a mask in step with a stride or pooling layer in the network.

    A block counts as analysable if any of its samples were. That matches the
    convolution: a kernel over a partly valid block still gives an output based
    on real samples, so marking the whole block invalid would throw away
    measured data.
    """
    if factor == 1:
        return mask
    b, length = mask.shape
    trimmed = (length // factor) * factor
    if trimmed == 0:
        return mask[:, :1]
    return mask[:, :trimmed].reshape(b, -1, factor).amax(dim=2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe_model(model, name=""):
    total = count_parameters(model)
    lines = [f"{name} parameters: {total:,}"]
    for child, module in model.named_children():
        n = count_parameters(module)
        if n:
            lines.append(f"    {child:<24s} {n:>10,}")
    return "\n".join(lines)


__all__ = [
    "get_device", "set_seed", "MaskedAvgPool", "MaskedMaxPool",
    "MaskedStatsPool", "downsample_mask", "count_parameters",
    "describe_model", "evaluate_predictions", "bootstrap_auc_ci",
    "delong_test", "RESULTS_DIR",
]
