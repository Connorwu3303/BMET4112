"""
Aim 3 step 4: turns the packed signal store into batches for training. It also
builds the label free structure set S and the confounder transforms.

The primary input is one channel, saturation. The sleep and validity masks only
decide which time steps get pooled and are never network inputs, because a mask
or REM channel would show the network sleep fragmentation that the 34 OBM
features don't have. Arm NC gets the mask and REM channels and no saturation.

Anything outside valid sleep is filled with the participant's own median valid
sleep saturation, snapped to the oximeter's quantisation ladder. It's a within
night, label free value, so nothing leaks from other participants or the
outcome. Splicing the valid stretches together instead would turn a baseline
shift across a gap into a fake desaturation. Every night also gets at least
2,400 samples of fill at each end, wider than the network's one sided receptive
field, so the logit doesn't depend on how much padding the batching adds.

Training crops use five fixed lengths from 4.7 to 7.3 h (mean 6 h).
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DATA_ROOT = "/Volumes/ThesisHD/Data/processed"
STORE_DIR = os.path.join(DATA_ROOT, "aim3", "signal_store")
SCALE = 100          # store holds percent x 100

# fixed affine map, nothing fitted, so nothing can leak through the signal path
CENTRE = 95.0
SPREAD = 5.0
CLIP_LO, CLIP_HI = 50.0, 100.0

# padding rule (see module docstring): wider than the one sided receptive field
# (about 2,130 samples) and a common multiple of stride 32 and window 300
PAD_EACH_SIDE = 2400
SHAPE_MULTIPLE = 2400

# five fixed crop lengths, all multiples of SHAPE_MULTIPLE so a cropped night
# plus its padding is still a valid shape. They fit inside the sleep period
# (median 7.23 h) for 45.8 to 97.9 percent of nights, longest to shortest.
CROP_BUCKETS = (16800, 19200, 21600, 24000, 26400)

# half the oximeter's 0.78 percent quantisation step, in scaled units
DEQUANT_JITTER = 0.39 / SPREAD
QUANTUM = 100.0 / 128.0      # SaO2 ladder step, about 0.78 percent

STAGE_UNKNOWN, STAGE_WAKE, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_REM = range(6)
SLEEP_STAGES = (STAGE_N1, STAGE_N2, STAGE_N3, STAGE_REM)
NREM_STAGES = (STAGE_N1, STAGE_N2, STAGE_N3)


class SignalStore:
    """
    The packed store held in memory, cropped to the sleep period on demand.

    Make one per process and share it across datasets.
    """

    def __init__(self, store_dir=STORE_DIR, cache=True):
        self.index = pd.read_csv(os.path.join(store_dir, "index.csv"))
        self.signal = np.load(os.path.join(store_dir, "signal_q.npy"))
        self.valid = np.load(os.path.join(store_dir, "valid.npy"))
        self.stage = np.load(os.path.join(store_dir, "stage.npy"))
        self._row = {int(p): (int(o), int(n)) for p, o, n in zip(
            self.index["pid"], self.index["offset"], self.index["length"])}
        self._cache = {} if cache else None

    def __contains__(self, pid):
        return int(pid) in self._row

    def raw(self, pid):
        """Return (percent, valid_bool, stage) views for the full recording."""
        off, n = self._row[int(pid)]
        sl = slice(off, off + n)
        return (self.signal[sl].astype(np.float32) / SCALE,
                self.valid[sl].astype(bool), self.stage[sl])

    def sleep_period(self, pid):
        """
        Return one participant's sleep period as a dict.

        x        float32 saturation in percent, with everything outside valid
                 sleep already replaced by the fill value
        m_sleep  bool, valid and scored asleep
        m_nrem   bool, valid and N1, N2 or N3
        m_rem    bool, valid and REM
        fill     median valid sleep saturation, snapped to the oximeter's
                 quantisation ladder
        """
        pid = int(pid)
        if self._cache is not None and pid in self._cache:
            return self._cache[pid]

        x, v, st = self.raw(pid)
        asleep = np.isin(st, SLEEP_STAGES)
        idx = np.flatnonzero(asleep)
        if idx.size == 0:
            out = {"x": np.full(1, CENTRE, np.float32),
                   "m_sleep": np.zeros(1, bool), "m_nrem": np.zeros(1, bool),
                   "m_rem": np.zeros(1, bool), "fill": CENTRE}
        else:
            a, b = int(idx[0]), int(idx[-1]) + 1
            xs, vs, sts = x[a:b].copy(), v[a:b], st[a:b]
            m_sleep = vs & asleep[a:b]
            if m_sleep.any():
                # Snapped to the quantisation ladder so the fill is a value the
                # oximeter could actually report. An unsnapped median would sit
                # between two rungs and stand out from real signal.
                fill = float(np.round(np.median(xs[m_sleep]) / QUANTUM) * QUANTUM)
            else:
                fill = CENTRE
            xs[~m_sleep] = fill
            out = {"x": xs, "m_sleep": m_sleep,
                   "m_nrem": vs & np.isin(sts, NREM_STAGES),
                   "m_rem": vs & (sts == STAGE_REM), "fill": fill}

        if self._cache is not None:
            self._cache[pid] = out
        return out

    def recording(self, pid):
        """
        Whole recording with only the validity mask, no hypnogram.

        Used by arm D, the deployable sensitivity analysis. A wearable only gives
        a saturation trace, so a model that needs technician scored sleep stages
        can't be deployed as is.
        """
        x, v, _ = self.raw(pid)
        xs = x.copy()
        fill = float(np.round(np.median(xs[v]) / QUANTUM) * QUANTUM) \
            if v.any() else CENTRE
        xs[~v] = fill
        return {"x": xs, "m_sleep": v.copy(), "m_nrem": v.copy(),
                "m_rem": np.zeros_like(v), "fill": fill}


def scale_signal(percent):
    """Fixed affine map. Nothing is fitted, so nothing leaks."""
    return (np.clip(percent, CLIP_LO, CLIP_HI) - CENTRE) / SPREAD


def pad_to_shape(n):
    """Total padded length for a night of n samples."""
    total = n + 2 * PAD_EACH_SIDE
    return int(np.ceil(total / SHAPE_MULTIPLE) * SHAPE_MULTIPLE)


class SpO2Dataset(Dataset):
    """
    One item is one participant's night, padded and channelled for one arm.

    Arms:
      "primary"  saturation only; staging enters only through the pooling masks
      "nc"       sleep mask and REM indicator, no saturation (negative control)
      "d"        saturation over the whole recording, no hypnogram

    Returns (x, m_nrem, m_rem, y, pid). The masks are what the network pools
    over; for arm D they're the validity mask and an empty mask.
    """

    def __init__(self, store, pids, y, arm="primary", train=False, seed=42,
                 jitter=True, time_mask=True, crop_buckets=CROP_BUCKETS):
        if arm not in ("primary", "nc", "d"):
            raise ValueError(f"unknown arm {arm!r}")
        self.store = store
        self.pids = np.asarray(pids)
        self.y = np.asarray(y, dtype=np.float32)
        self.arm = arm
        self.train = bool(train)
        self.seed = int(seed)
        self.jitter = bool(jitter)
        self.time_mask = bool(time_mask)
        self.crop_buckets = tuple(crop_buckets)
        self.epoch = 0
        self.lengths = np.array([self._night(int(p))["x"].shape[0]
                                 for p in self.pids])

    def _night(self, pid):
        return self.store.recording(pid) if self.arm == "d" \
            else self.store.sleep_period(pid)

    def set_epoch(self, epoch):
        """
        Reseed augmentation each epoch.

        Without this the crop would be the same window every epoch, so it would
        just be a fixed subsample. The seed comes from (seed, epoch, index) so
        runs stay reproducible.
        """
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.pids)

    def _rng(self, i):
        return np.random.default_rng(
            (self.seed * 1_000_003 + self.epoch * 10_007 + i) % (2 ** 31 - 1))

    def __getitem__(self, i):
        pid = int(self.pids[i])
        night = self._night(pid)
        x = night["x"].copy()
        m_sleep = night["m_sleep"].copy()
        m_nrem = night["m_nrem"].copy()
        m_rem = night["m_rem"].copy()
        fill = night["fill"]
        n = len(x)

        if self.train:
            rng = self._rng(i)
            want = int(rng.choice(self.crop_buckets))
            if n > want:
                s0 = int(rng.integers(0, n - want + 1))
                sl = slice(s0, s0 + want)
                x, m_sleep = x[sl], m_sleep[sl]
                m_nrem, m_rem = m_nrem[sl], m_rem[sl]
                n = want
            if self.time_mask and rng.random() < 0.5:
                # Blank 1 to 3 blocks of 1 to 10 minutes: set to the fill value
                # with the pooling masks cleared. That's exactly how wake and
                # artefact already look, so the network learns fill stretches
                # carry no risk information, and it weakens any link between
                # the number of fill edges and the outcome.
                for _ in range(int(rng.integers(1, 4))):
                    span = int(rng.integers(60, 601))
                    if span >= n:
                        continue
                    s0 = int(rng.integers(0, n - span))
                    sl = slice(s0, s0 + span)
                    x[sl] = fill
                    m_sleep[sl] = False
                    m_nrem[sl] = False
                    m_rem[sl] = False

        xs = scale_signal(x)
        if self.train and self.jitter:
            # Dequantisation jitter of half a quantisation step, applied to the
            # fill too. If only measured samples were jittered, the fill would
            # be the only perfectly flat stretch in the night and the network
            # could find gaps just by looking for it.
            xs = xs + rng.uniform(-DEQUANT_JITTER, DEQUANT_JITTER, size=len(xs))

        # Pad with fill on both sides, wider than the receptive field, then out
        # to a fixed shape. The pooling masks are zero over all the padding, so
        # it doesn't feed into anything pooled.
        target = pad_to_shape(n)
        right = target - n - PAD_EACH_SIDE
        fill_scaled = float(scale_signal(np.float32(fill)))
        xs = np.concatenate([
            np.full(PAD_EACH_SIDE, fill_scaled, np.float32),
            xs.astype(np.float32),
            np.full(right, fill_scaled, np.float32)])

        def pad_mask(m):
            return np.concatenate([np.zeros(PAD_EACH_SIDE, bool), m,
                                   np.zeros(right, bool)])

        m_sleep, m_nrem, m_rem = (pad_mask(m_sleep), pad_mask(m_nrem),
                                  pad_mask(m_rem))

        if self.arm == "nc":
            # negative control only sees structure: where sleep was scored and
            # analysable, and where it was REM. No saturation.
            chan = np.stack([m_sleep.astype(np.float32),
                             m_rem.astype(np.float32)])
        else:
            chan = xs[None, :]

        return (torch.from_numpy(np.ascontiguousarray(chan)),
                torch.from_numpy(m_nrem.astype(np.float32)),
                torch.from_numpy(m_rem.astype(np.float32)),
                torch.tensor(self.y[i]), torch.tensor(float(pid)))


def collate_bucket(batch):
    """
    Pad a batch to the largest shape in it, which is already a fixed bucket.

    Every item is already padded to a multiple of 2,400 with at least 2,400
    samples of fill at each end, so padding it further can't change its logit:
    the extra is beyond the receptive field of any valid sample. A unit test
    checks this.
    """
    target = max(b[0].shape[-1] for b in batch)
    xs, mn, mr, ys, pids = [], [], [], [], []
    for chan, m_nrem, m_rem, y, pid in batch:
        pad = target - chan.shape[-1]
        if pad > 0:
            edge = chan[:, -1:].repeat(1, pad)
            chan = torch.cat([chan, edge], dim=-1)
            m_nrem = torch.cat([m_nrem, torch.zeros(pad)])
            m_rem = torch.cat([m_rem, torch.zeros(pad)])
        xs.append(chan)
        mn.append(m_nrem)
        mr.append(m_rem)
        ys.append(y)
        pids.append(pid)
    return (torch.stack(xs), torch.stack(mn), torch.stack(mr),
            torch.stack(ys), torch.stack(pids))


class LengthSortedSampler(torch.utils.data.Sampler):
    """
    Yields evaluation indices sorted by night length.

    Batching nights of similar length keeps padding small, so full night
    inference is affordable. The catch is that predictions come back
    out of dataset order, so every evaluation path carries the participant IDs
    with the predictions and realigns on them instead of trusting position.
    """

    def __init__(self, lengths):
        self.order = np.argsort(np.asarray(lengths), kind="stable")

    def __iter__(self):
        return iter(self.order.tolist())

    def __len__(self):
        return len(self.order)


def make_loader(store, pids, y, arm="primary", train=False, batch_size=32,
                seed=42, num_workers=0, jitter=True, time_mask=True):
    """
    Build one loader. Training shuffles, evaluation sorts by length.

    num_workers stays at 0. A batch takes 11 ms to build in the main process,
    while two workers took 956 ms per batch because each one inherits the
    0.7 GB in-memory store. The pipeline is GPU bound anyway.
    """
    ds = SpO2Dataset(store, pids, y, arm=arm, train=train, seed=seed,
                     jitter=jitter, time_mask=time_mask)
    if train:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers,
                            collate_fn=collate_bucket, drop_last=False)
    else:
        loader = DataLoader(ds, batch_size=min(batch_size, 16),
                            sampler=LengthSortedSampler(ds.lengths),
                            num_workers=num_workers,
                            collate_fn=collate_bucket)
    return loader, ds


# --- structure set S ---

def structure_features(store, pid):
    """
    Label free summaries of sleep structure and data quality for one night.

    The primary network never sees these. They feed the nuisance model, the
    correlations with the deep score, and the comparator that checks whether a
    deep increment survives them.
    """
    x, v, st = store.raw(int(pid))
    asleep = np.isin(st, SLEEP_STAGES)
    idx = np.flatnonzero(asleep)
    if idx.size == 0:
        return {k: np.nan for k in (
            "s_period_h", "s_valid_sleep_h", "s_rem_h", "s_rem_frac",
            "s_valid_frac", "s_waso_min", "s_wake_bouts", "s_rem_latency_min",
            "s_invalid_runs")}
    a, b = int(idx[0]), int(idx[-1]) + 1
    period = b - a
    sl = slice(a, b)
    m_sleep = v[sl] & asleep[sl]
    wake_in = (st[sl] == STAGE_WAKE)
    # a wake bout is a maximal run of wake epochs inside the sleep period
    runs = np.diff(np.concatenate([[0], wake_in.astype(np.int8), [0]]))
    n_bouts = int((runs == 1).sum())
    rem = (st[sl] == STAGE_REM)
    rem_idx = np.flatnonzero(rem)
    invalid = ~v[sl]
    inv_runs = np.diff(np.concatenate([[0], invalid.astype(np.int8), [0]]))
    starts = np.flatnonzero(inv_runs == 1)
    ends = np.flatnonzero(inv_runs == -1)
    long_runs = int(np.sum((ends - starts) > 30))
    return {
        "s_period_h": period / 3600.0,
        "s_valid_sleep_h": float(m_sleep.sum()) / 3600.0,
        "s_rem_h": float((v[sl] & rem).sum()) / 3600.0,
        "s_rem_frac": float((v[sl] & rem).sum()) / max(float(m_sleep.sum()), 1),
        "s_valid_frac": float(m_sleep.sum()) / max(period, 1),
        "s_waso_min": float(wake_in.sum()) / 60.0,
        "s_wake_bouts": n_bouts,
        "s_rem_latency_min": (float(rem_idx[0]) / 60.0 if rem_idx.size
                              else np.nan),
        "s_invalid_runs": long_runs,
    }


STRUCTURE_COLUMNS = ("s_period_h", "s_valid_sleep_h", "s_rem_h", "s_rem_frac",
                     "s_valid_frac", "s_waso_min", "s_wake_bouts",
                     "s_rem_latency_min", "s_invalid_runs")


# --- confounder handling ---
# These sit here because the confounders never go into a network. The stacking
# stage and the epoch selection criterion use them, with the same steps as Aim 2
# (SimpleImputer(strategy="median") then StandardScaler) fitted on training rows
# only. Rebuilding the Aim 2 comparators with them reproduces the published out
# of fold predictions to 0.00e+00.

def fit_confounder_transform(train_conf):
    """
    Fit median imputation and standardisation on the training rows only.

    Returned as plain vectors instead of a scikit-learn object, so applying it
    is obviously fixed arithmetic and the fitted values can be written into the
    run record and checked.
    """
    med = np.nanmedian(train_conf, axis=0)
    filled = np.where(np.isnan(train_conf), med, train_conf)
    mu = filled.mean(axis=0)
    sd = filled.std(axis=0)
    # A constant column would divide by zero. Using 1 leaves it at zero after
    # centring, which is right for a feature with no information in this fold,
    # and matches what StandardScaler does with a near zero variance column.
    sd = np.where(sd < 1e-12, 1.0, sd)
    return {"median": med, "mean": mu, "sd": sd}


def apply_confounder_transform(conf, tf):
    """Apply a transform fitted by fit_confounder_transform."""
    filled = np.where(np.isnan(conf), tf["median"], conf)
    return (filled - tf["mean"]) / tf["sd"]
