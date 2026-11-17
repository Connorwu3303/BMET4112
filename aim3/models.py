"""
Aim 3 step 6: the networks, plus the registry of every pre-declared arm.

Architecture A is the pre-declared primary. Picking an architecture per outer
fold was ruled out because the inner criterion's standard error (about 0.012
AUC) is bigger than the likely gap between candidates, so the alternatives run
as fixed secondary arms on the same folds and are compared by paired DeLong.

A is the convolutional branch of OxiNet (Levy et al. 2023), shrunk about 3.5
times for a much weaker label: a strided residual front end, then dilated
residual blocks for a receptive field of about 70 minutes without the depth or
long kernels that overfit small time series datasets (Ismail Fawaz et al. 2020).

The primary reads saturation only (data.py says why there's no mask or REM
channel). Staging comes in only through pooling, which takes a masked mean and
top decile mean over NREM and, separately, over REM. That matches Aim 2, which
computes its statistics separately over the two.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Prevalence of the primary outcome. The head bias starts at its log odds so the
# untrained network already predicts the base rate and does not waste the first
# epochs getting from zero to the right intercept.
BASE_RATE = 0.19
STEP_VALID_FRACTION = 0.5    # a step counts for a stage at half coverage
MIN_REM_STEPS = 10           # below this, the REM vector is copied from NREM


# --- pooling ---

def masked_mean_top(x, m, k_fraction=0.10):
    """
    Masked mean and masked mean of the top k fraction, over one stage's steps.

    Used instead of mean and max. The top decile mean picks up a night's worst
    stretches, which an eight hour mean can bury, but unlike the max it doesn't
    grow with night length, so the network can't read length as risk.
    """
    mm = m.unsqueeze(1)
    counts = mm.sum(dim=2).clamp(min=1.0)
    mean = (x * mm).sum(dim=2) / counts

    n_valid = m.sum(dim=1)
    k = torch.ceil(n_valid * k_fraction).clamp(min=1.0)
    # invalid steps get a large finite negative, not -inf (which tended to give
    # silent NaN on MPS), and the top k are taken by position
    ordered, _ = torch.sort(x.masked_fill(mm == 0, -1e4), dim=2,
                            descending=True)
    pos = torch.arange(x.shape[2], device=x.device).view(1, 1, -1)
    take = (pos < k.view(-1, 1, 1)).to(x.dtype)
    top = (ordered * take).sum(dim=2) / k.view(-1, 1)
    return torch.cat([mean, top], dim=1)


class StageStratifiedPool(nn.Module):
    """
    Pool NREM steps and REM steps separately, then concatenate.

    Output width is 4 times the channel count (mean and top decile for each
    stage). About 2.8 percent of the cohort have under five minutes of
    analysable REM, so a night with fewer than MIN_REM_STEPS valid REM steps
    copies its REM vector from NREM. That avoids a REM mean over one or two
    noisy steps without adding a "no REM" indicator, which would be a sleep
    architecture variable.
    """

    def __init__(self, k_fraction=0.10):
        super().__init__()
        self.k_fraction = float(k_fraction)

    def forward(self, z, m_nrem, m_rem):
        nrem = masked_mean_top(z, m_nrem, self.k_fraction)
        rem = masked_mean_top(z, m_rem, self.k_fraction)
        enough = (m_rem.sum(dim=1) >= MIN_REM_STEPS).unsqueeze(1).to(z.dtype)
        rem = enough * rem + (1.0 - enough) * nrem
        return torch.cat([nrem, rem], dim=1)


def pool_mask(mask, stride, min_fraction=STEP_VALID_FRACTION):
    """
    Reduce a per sample mask to a per step mask that lines up with the network.

    A step counts when at least min_fraction of its samples were valid sleep of
    that stage. Average pooling plus a threshold is the right reduction here:
    max would mark a step valid on one measured sample, and min would throw a
    step away for one missing sample.
    """
    if stride <= 1:
        return mask
    if mask.dim() == 2:
        mask = mask.unsqueeze(1)
    n = (mask.shape[-1] // stride) * stride
    if n == 0:
        return (mask[..., :1] > 0).float().squeeze(1)
    pooled = F.avg_pool1d(mask[..., :n], kernel_size=stride, stride=stride)
    return (pooled >= min_fraction).float().squeeze(1)


# --- blocks ---

class ResBlock(nn.Module):
    """
    Two convolutions with a skip connection.

    Conv, BatchNorm, GELU, Dropout, Conv, BatchNorm, then add the skip and
    activate. Dropout1d drops whole channels. On a smooth, heavily
    autocorrelated signal like saturation, dropping single samples does almost
    nothing because the neighbour has nearly the same value, while dropping a
    channel removes a whole learned feature and actually regularises.
    """

    def __init__(self, c_in, c_out, kernel, dilation=1, dropout=0.1):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, padding=pad,
                               dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, padding=pad,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.skip = None
        if c_in != c_out:
            self.skip = nn.Sequential(nn.Conv1d(c_in, c_out, 1, bias=False),
                                      nn.BatchNorm1d(c_out))
        self.act = nn.GELU()

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        z = self.drop(self.act(self.bn1(self.conv1(x))))
        return self.act(self.bn2(self.conv2(z)) + identity)


# --- architecture A ---

class SpO2Net(nn.Module):
    """
    Architecture A: strided residual front end, then dilated residual blocks.

    Five front end stages with max pooling by 2 give a stride of 32, so one
    output step is 32 seconds. Five dilated blocks (dilations 1 to 16) widen
    the receptive field to about 70 minutes, roughly one NREM to REM cycle.
    BatchNorm, not GroupNorm, so in eval mode a night's normalisation can't
    depend on its length or padding. The feature map is re-zeroed outside the
    mask after every stage, and the padding in data.py stops the rest of the
    leakage across the mask edge.
    """

    def __init__(self, n_channels=1, widths=(16, 32, 48, 64, 64),
                 kernels=(7, 7, 5, 5, 5), dilated_width=64, dilated_kernel=3,
                 dilations=(1, 2, 4, 8, 16), dropout=0.1, head_dropout=0.5,
                 stem_width=16, stem_kernel=7, k_fraction=0.10):
        super().__init__()
        assert len(widths) == len(kernels), "widths and kernels must align"
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, stem_width, stem_kernel,
                      padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(stem_width), nn.GELU())

        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = stem_width
        for w, k in zip(widths, kernels):
            self.blocks.append(ResBlock(c, w, k, dropout=dropout))
            self.pools.append(nn.MaxPool1d(2))
            c = w
        self.total_stride = 2 ** len(widths)

        self.dilated = nn.ModuleList([
            ResBlock(c if i == 0 else dilated_width, dilated_width,
                     dilated_kernel, dilation=d, dropout=dropout)
            for i, d in enumerate(dilations)])

        self.pool = StageStratifiedPool(k_fraction)
        self.head_drop = nn.Dropout(head_dropout)
        self.embed_dim = dilated_width * 4
        self.head = nn.Linear(self.embed_dim, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.fill_(float(np.log(BASE_RATE / (1 - BASE_RATE))))

    def features(self, x, m_any=None):
        """
        Feature map before pooling, with unmeasured positions re-zeroed.

        Exposed for Grad-CAM in Aim 4, which needs the last dilated block's
        activations on the 32 second grid.
        """
        z = self.stem(x)
        stride = 1
        if m_any is not None:
            z = z * _mask_at(m_any, stride, z.shape[-1]).unsqueeze(1)
        for block, pool in zip(self.blocks, self.pools):
            z = block(z)
            if m_any is not None:
                z = z * _mask_at(m_any, stride, z.shape[-1]).unsqueeze(1)
            z = pool(z)
            stride *= 2
        for block in self.dilated:
            z = block(z)
            if m_any is not None:
                z = z * _mask_at(m_any, stride, z.shape[-1]).unsqueeze(1)
        return z

    def embed(self, x, m_nrem, m_rem):
        m_any = ((m_nrem + m_rem) > 0).to(x.dtype)
        z = self.features(x, m_any)
        sn = _match_length(pool_mask(m_nrem, self.total_stride), z.shape[-1])
        sr = _match_length(pool_mask(m_rem, self.total_stride), z.shape[-1])
        return self.pool(z, sn, sr)

    def forward(self, x, m_nrem, m_rem):
        return self.head(self.head_drop(self.embed(x, m_nrem, m_rem)))


class SpO2NetMIL(nn.Module):
    """
    Alternative B: additive multiple instance model over five minute windows.

    A shared encoder scores each non overlapping window, one convolution adds
    about fifteen minutes of context, and the night logit is the bias plus the
    mean NREM and mean REM window contributions. Because the logit is a mean it
    can't read night length, and the per window contributions are exact
    explanations that Aim 4 can use directly. It tests whether window level
    desaturation shape beats A's long range context.
    """

    def __init__(self, n_channels=1, window=300, widths=(16, 32, 48, 48),
                 kernel=5, context_width=48, dropout=0.5,
                 min_valid_fraction=0.5):
        super().__init__()
        self.window = int(window)
        self.min_valid_fraction = float(min_valid_fraction)
        layers, c = [], n_channels
        for w in widths:
            layers += [nn.Conv1d(c, w, kernel, padding=kernel // 2, bias=False),
                       nn.BatchNorm1d(w), nn.GELU(), nn.MaxPool1d(2)]
            c = w
        self.encoder = nn.Sequential(*layers)
        # Mean and max over each window's own time axis. Max is fine here
        # (unlike in global pooling) because every window is the same length,
        # so its expected max does not change with the night.
        self.context = nn.Conv1d(c * 2, context_width, 3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.head_nrem = nn.Linear(context_width, 1)
        self.head_rem = nn.Linear(context_width, 1)
        for h in (self.head_nrem, self.head_rem):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)
        self.bias = nn.Parameter(torch.tensor(
            float(np.log(BASE_RATE / (1 - BASE_RATE)))))
        self.embed_dim = context_width

    def window_contributions(self, x, m_nrem, m_rem):
        """
        Per window contributions and per window stage validity.

        Kept separate from forward so Aim 4 can read them directly. They sum to
        the night logit minus the bias, so they're exact explanations.
        """
        b, ch, length = x.shape
        n_win = length // self.window
        if n_win == 0:
            pad = self.window - length
            x = F.pad(x, (0, pad))
            m_nrem = F.pad(m_nrem, (0, pad))
            m_rem = F.pad(m_rem, (0, pad))
            n_win, length = 1, self.window
        keep = n_win * self.window
        x, m_nrem, m_rem = x[:, :, :keep], m_nrem[:, :keep], m_rem[:, :keep]

        xw = x.reshape(b, ch, n_win, self.window).permute(0, 2, 1, 3) \
              .reshape(b * n_win, ch, self.window)
        h = self.encoder(xw)
        summary = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        summary = summary.reshape(b, n_win, -1).transpose(1, 2)
        z = self.drop(self.act(self.context(summary))).transpose(1, 2)

        c_n = self.head_nrem(z).squeeze(-1)
        c_r = self.head_rem(z).squeeze(-1)
        frac_n = m_nrem.reshape(b, n_win, self.window).mean(dim=2)
        frac_r = m_rem.reshape(b, n_win, self.window).mean(dim=2)
        # a window belongs to whichever stage covers most of its valid sleep
        # samples, and only counts if half its samples are valid
        v_n = ((frac_n >= self.min_valid_fraction) &
               (frac_n >= frac_r)).to(x.dtype)
        v_r = ((frac_r >= self.min_valid_fraction) &
               (frac_r > frac_n)).to(x.dtype)
        return c_n, c_r, v_n, v_r

    def forward(self, x, m_nrem, m_rem):
        c_n, c_r, v_n, v_r = self.window_contributions(x, m_nrem, m_rem)
        n_n = v_n.sum(dim=1)
        n_r = v_r.sum(dim=1)
        term_n = (c_n * v_n).sum(dim=1) / n_n.clamp(min=1.0)
        term_r = (c_r * v_r).sum(dim=1) / n_r.clamp(min=1.0)
        # with fewer than two REM windows the REM term is evaluated on the NREM
        # windows instead (same fallback idea as A's REM rule)
        enough = (n_r >= 2).to(x.dtype)
        term_r = enough * term_r + (1.0 - enough) * \
            ((c_r * v_n).sum(dim=1) / n_n.clamp(min=1.0))
        return (self.bias + term_n + term_r).unsqueeze(-1)


class OffsetWrapper(nn.Module):
    """
    Alternative C: logit = frozen clinical offset + network output.

    The offset is the linear predictor of a cross fitted, confounders only
    logistic regression, fixed per participant. The head weights start at zero,
    so training starts from the clinical model (plus the network's constant
    bias) and only rewards the network where the confounders leave residual
    error, which targets incremental information.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, m_nrem, m_rem, offset):
        return self.net(x, m_nrem, m_rem) + offset.view(-1, 1)


class EMA:
    """
    Exponential moving average of the weights, which is what gets evaluated.

    With a weak label the validation metric jumps around between epochs, so a
    checkpoint from a lucky epoch is partly luck. Averaging over the trajectory
    smooths that and reduces the optimism in picking an epoch. Costs one extra
    copy of the model in memory, under 1 MB here.
    """

    def __init__(self, model, decay=0.99):
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1.0 - d)
            else:
                s.copy_(m)

    def module(self):
        return self.shadow


def _mask_at(mask, stride, target_len):
    """Per sample mask reduced to a stride and forced to a given length."""
    m = pool_mask(mask, stride) if stride > 1 else mask
    return _match_length(m, target_len)


def _match_length(mask, target_len):
    """
    Force a reduced mask to the feature map's length.

    Integer division at each pooling layer can leave them a step or two apart,
    and trimming or edge padding that much is fine. A bigger gap means the
    stride bookkeeping is wrong, so it raises instead of quietly misaligning the
    mask with the features.
    """
    have = mask.shape[-1]
    if have == target_len:
        return mask
    if abs(have - target_len) > 4:
        raise RuntimeError(
            f"mask length {have} and feature length {target_len} differ by more "
            f"than 4 steps; stride bookkeeping is wrong")
    if have > target_len:
        return mask[:, :target_len]
    pad = target_len - have
    return torch.cat([mask, mask[:, -1:].repeat(1, pad)], dim=1)


# --- pre-declared arms ---
# Fixed before the first CVD labelled run. Every arm ever scored on an outer
# test fold is listed here, so "nothing was chosen after seeing test
# performance" can actually be checked.

ARMS = {
    # primary: one channel, saturation only, stage stratified pooling
    "A": {"cls": "SpO2Net", "input": "primary", "channels": 1, "params": {}},
    # fixed secondary: additive window model, same single channel input
    "B": {"cls": "SpO2NetMIL", "input": "primary", "channels": 1, "params": {}},
    # required negative control: sleep structure and data quality only, no
    # saturation. Bounds how much of any deep result is just fragmentation.
    "NC": {"cls": "SpO2NetMIL", "input": "nc", "channels": 2, "params": {}},
    # deployability sensitivity: whole recording, validity mask only, no
    # technician hypnogram, so it could run on a wearable
    "D": {"cls": "SpO2Net", "input": "d", "channels": 1, "params": {}},
    # architecture matched negative control: A itself with the structure only
    # input. NC is the smaller window model, so only this arm isolates the
    # missing saturation as the difference from A.
    "NCA": {"cls": "SpO2Net", "input": "nc", "channels": 2, "params": {}},
}

_CLASSES = {"SpO2Net": SpO2Net, "SpO2NetMIL": SpO2NetMIL}


def build(arm, overrides=None, strict=False):
    """
    Build one pre-declared arm by name.

    A and the window model don't share a parameter set, so overrides are
    filtered to the ones the arm's class accepts and one config works for every
    arm. strict=True raises instead of filtering.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; available: {sorted(ARMS)}")
    spec = ARMS[arm]
    cls = _CLASSES[spec["cls"]]
    params = dict(spec["params"])
    if overrides:
        import inspect
        accepted = set(inspect.signature(cls.__init__).parameters)
        unknown = [k for k in overrides if k not in accepted]
        if unknown and strict:
            raise TypeError(
                f"{cls.__name__} does not accept {unknown}; arm {arm!r} has a "
                f"different parameter set from the primary arm")
        params.update({k: v for k, v in overrides.items() if k in accepted})
    return cls(n_channels=spec["channels"], **params)


def arm_input(arm):
    """Which input representation from data.py this arm uses."""
    return ARMS[arm]["input"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
