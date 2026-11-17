"""
Oximetry biomarker (OBM) feature library.

My implementation of the five digital oximetry biomarker families from Levy
et al. (2021), npj Digital Medicine 4:1, plus the standard SHHS overnight
oximetry measures. The families are general statistics, complexity,
periodicity, desaturations and hypoxic burden. Written from scratch instead of
calling the pobm package, so each definition can be checked directly and the
validity mask reaches every measure.

Input is the Aim 1 cleaned signal plus its validity mask. Invalid samples
(probe off, motion artifact) are left unimputed. Rates use valid time as the
denominator, and measures that need a continuous signal (desaturations,
entropy, spectra) run on contiguous valid segments and are pooled afterwards,
so nothing straddles a gap. For participant 200020, running on a gap-filled
signal turned one probe disconnection into 40 percentage points of fake T90.

Measures can be computed over the whole "recording" or only "sleep", "rem" or
"nrem" time, so they line up with the published SHHS variables.

Usage:
    from obm_library import compute_all_obm, build_stage_array

    stages = build_stage_array(events_df, n_samples, fs)
    features = compute_all_obm(signal, valid_mask, fs, stages)
"""

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from scipy.spatial import cKDTree

# --- config: biomarker parameters ---

# SpO2 thresholds (%) for the cumulative time and area measures. T90 is the
# standard clinical one; the deeper ones grade severity and tend to carry
# more independent CVD signal.
CT_THRESHOLDS = (95, 90, 88, 85, 80, 75, 70)

# desaturation depths to detect. 3% and 4% are the two AASM scoring rules,
# so ODI3 and ODI4 line up with published indices
DESAT_THRESHOLDS = (2.0, 3.0, 4.0)

# an event has to fall and recover within this window, otherwise it's just
# slow baseline drift
MAX_DESAT_DURATION_SEC = 120.0

# anything shorter isn't a resolvable respiratory desaturation at 1 Hz
MIN_DESAT_DURATION_SEC = 3.0

# an event counts as resolved once saturation climbs back this fraction of
# the fall. Needing a full return to baseline would lose events during
# progressive desaturation.
RECOVERY_FRACTION = 0.5

# segments shorter than this are too short for entropy or spectral
# estimates, so those families skip them (they still count for the
# distribution and threshold measures)
MIN_SEGMENT_SEC = 300.0

# ApEn / SampEn embedding. m = 2 with a tolerance of 0.2 SD is the usual
# choice in the physiological time series literature.
ENTROPY_M = 2
ENTROPY_R_FACTOR = 0.2

# CTM radius in % saturation units, used on the second-order difference plot
CTM_RADIUS = 0.25

# Skew and kurtosis are ratios of central moments, so on a nearly constant
# input the denominator collapses and the result is mostly floating point
# cancellation. This does happen here, e.g. someone with only a handful of
# desaturations that all have the same depth. Below this SD the shape stats
# are set to NaN, since a junk number in the feature matrix is worse than a
# missing one.
MIN_SD_FOR_SHAPE_STATS = 1e-8

# DFA box sizes, in samples
DFA_WINDOWS = (4, 8, 16, 32, 64, 128, 256)

# Welch window. 600 s at 1 Hz resolves the very low frequencies where
# periodic breathing sits.
PSD_WINDOW_SEC = 600.0

# band for repetitive apnoea / hypopnoea cycling. Obstructive events recur
# roughly every 25 to 100 s, which is 0.010 to 0.040 Hz.
PSD_OSA_BAND = (0.010, 0.040)

# delta index window. The delta index is the mean absolute difference
# between successive window means (short-term variability); 12 s is the
# value from the original description.
DELTA_INDEX_WINDOW_SEC = 12.0

# stage labels used by build_stage_array
STAGE_UNKNOWN = 0
STAGE_WAKE = 1
STAGE_N1 = 2
STAGE_N2 = 3
STAGE_N3 = 4
STAGE_REM = 5
NREM_STAGES = (STAGE_N1, STAGE_N2, STAGE_N3)
SLEEP_STAGES = (STAGE_N1, STAGE_N2, STAGE_N3, STAGE_REM)


# --- sleep staging ---

def build_stage_array(events_df, n_samples, fs):
    """
    Expand the scored sleep stage events into a per-sample stage label array.

    NSRR stage events are contiguous blocks aligned to 30 s epochs, given as
    a start time and a duration, so they can be painted straight onto the
    sample index.
    Samples outside any stage event stay STAGE_UNKNOWN (happens at the
    start/end of some recordings).

    Returns an int8 array of length n_samples.
    """
    stages = np.full(n_samples, STAGE_UNKNOWN, dtype=np.int8)
    if events_df is None or events_df.empty:
        return stages
    if "event_concept" not in events_df.columns:
        return stages

    concept_to_stage = {
        "wake": STAGE_WAKE,
        "stage 1 sleep": STAGE_N1,
        "stage 2 sleep": STAGE_N2,
        "stage 3 sleep": STAGE_N3,
        # SHHS scores N3 and N4 separately (older R&K rules). AASM merges
        # them into N3, so they're pooled here.
        "stage 4 sleep": STAGE_N3,
        "rem sleep": STAGE_REM,
    }

    for _, row in events_df.iterrows():
        key = str(row["event_concept"]).strip().lower()
        stage = concept_to_stage.get(key)
        if stage is None:
            continue
        start_idx = max(int(float(row["start_sec"]) * fs), 0)
        end_idx = min(
            int((float(row["start_sec"]) + float(row["duration_sec"])) * fs),
            n_samples,
        )
        if end_idx > start_idx:
            stages[start_idx:end_idx] = stage

    return stages


def window_mask(valid_mask, stages, window):
    """
    Boolean mask for the samples in an analysis window.

    Only valid samples can be included, so no window ever picks up an
    unmeasured sample. The stage filter goes on top of that.
    """
    if window == "recording":
        return valid_mask.copy()
    if window == "sleep":
        return valid_mask & np.isin(stages, SLEEP_STAGES)
    if window == "nrem":
        return valid_mask & np.isin(stages, NREM_STAGES)
    if window == "rem":
        return valid_mask & (stages == STAGE_REM)
    raise ValueError(f"unknown analysis window: {window}")


def contiguous_segments(mask, fs, min_seconds=0.0):
    """
    (start, stop) index pairs for each contiguous run of True in mask.

    All the contiguity-dependent biomarkers run per segment, so none of them
    can read across a period the oximeter wasn't measuring.
    """
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    min_samples = int(round(min_seconds * fs))
    return [
        (int(a), int(b))
        for a, b in zip(starts, stops)
        if (b - a) >= max(min_samples, 1)
    ]


# --- Family 1: general statistics ---

def safe_skew(values):
    """Skewness, or NaN if the input is too close to constant."""
    values = np.asarray(values, dtype=float)
    if values.size < 3 or np.std(values) < MIN_SD_FOR_SHAPE_STATS:
        return np.nan
    return float(sp_stats.skew(values))


def safe_kurtosis(values):
    """Excess kurtosis, or NaN if the input is too close to constant."""
    values = np.asarray(values, dtype=float)
    if values.size < 4 or np.std(values) < MIN_SD_FOR_SHAPE_STATS:
        return np.nan
    return float(sp_stats.kurtosis(values))


def general_statistics(x, fs):
    """
    Distribution of the saturation trace.

    x is the pooled vector of valid in-window samples. Everything here is
    order independent. Zero crossings and delta index need segments, so the
    caller computes those separately.
    """
    out = {}
    if x.size == 0:
        return out

    out["mean"] = float(np.mean(x))
    out["median"] = float(np.median(x))
    out["std"] = float(np.std(x))
    out["var"] = float(np.var(x))
    out["min"] = float(np.min(x))          # nadir saturation
    out["max"] = float(np.max(x))
    out["range"] = out["max"] - out["min"]

    for q in (1, 5, 10, 25, 75, 90, 95, 99):
        out[f"p{q}"] = float(np.percentile(x, q))

    # IQR, less sensitive to the deep tail than the SD
    out["iqr"] = out["p75"] - out["p25"]

    # shape: a healthy trace is tight and roughly symmetric, heavy negative
    # skew and high kurtosis mean repeated deep dips
    out["skew"] = safe_skew(x)
    out["kurtosis"] = safe_kurtosis(x)

    # CV is dimensionless, so it compares across people with different
    # baseline saturation
    out["cv"] = out["std"] / out["mean"] if out["mean"] else np.nan
    return out


def zero_crossings(segments_data):
    """
    Rate at which the trace crosses its own mean, per hour.

    Each segment is compared against its own mean and the counts are pooled,
    so a baseline shift between segments can't create crossings. A high rate
    means saturation keeps oscillating around its usual value, which is what
    repetitive apnoea looks like on oximetry.
    """
    total_crossings = 0
    total_samples = 0
    for seg, fs in segments_data:
        if seg.size < 2:
            continue
        centred = seg - np.mean(seg)
        # count sign changes, dropping exact zeros so a plateau sitting on
        # the mean isn't counted over and over
        signs = np.sign(centred)
        signs = signs[signs != 0]
        if signs.size < 2:
            continue
        total_crossings += int(np.sum(np.diff(signs) != 0))
        total_samples += seg.size
    if total_samples == 0:
        return np.nan
    seconds = total_samples / segments_data[0][1]
    return float(total_crossings / (seconds / 3600.0))


def delta_index(segments_data):
    """
    Mean absolute difference between the means of successive short windows.

    Measures short-term instability. It doesn't depend on the absolute
    saturation level, so it separates an unstable night from one that is
    just low.
    """
    diffs = []
    for seg, fs in segments_data:
        win = int(round(DELTA_INDEX_WINDOW_SEC * fs))
        if win < 1 or seg.size < 2 * win:
            continue
        n_win = seg.size // win
        # reshape into complete windows and average each one
        means = seg[: n_win * win].reshape(n_win, win).mean(axis=1)
        if means.size >= 2:
            diffs.append(np.abs(np.diff(means)))
    if not diffs:
        return np.nan
    return float(np.mean(np.concatenate(diffs)))


# --- Family 2: complexity ---

def _entropy_counts(x, m, r):
    """
    Count embedded-vector pairs within Chebyshev distance r, for dimensions
    m and m + 1.

    The naive ApEn / SampEn is O(N^2), far too slow for a full night at 1 Hz.
    A KD-tree with the infinity norm gets the pair counting down to roughly
    O(N log N), which makes it affordable across the whole cohort.

    Returns (count_m, count_m1, n_vectors).
    """
    n = x.size - m
    if n < 2:
        return None
    # embeddings for m and m + 1 over the same index range so the two counts
    # are comparable
    emb_m = np.lib.stride_tricks.sliding_window_view(x, m)[:n]
    emb_m1 = np.lib.stride_tricks.sliding_window_view(x, m + 1)[:n]

    tree_m = cKDTree(emb_m)
    tree_m1 = cKDTree(emb_m1)
    # count_neighbors with p = inf gives the Chebyshev-ball pair counts
    count_m = tree_m.count_neighbors(tree_m, r, p=np.inf)
    count_m1 = tree_m1.count_neighbors(tree_m1, r, p=np.inf)
    return float(count_m), float(count_m1), n


def approximate_entropy(x, m=ENTROPY_M, r_factor=ENTROPY_R_FACTOR):
    """
    Approximate entropy: negative log of the conditional probability that two
    sequences close in m dimensions stay close in m + 1 dimensions.

    Lower = more regular, more predictable trace. Self-matches are included,
    which is the difference from sample entropy.
    """
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    if x.size < m + 2 or sd == 0:
        return np.nan
    r = r_factor * sd
    counts = _entropy_counts(x, m, r)
    if counts is None:
        return np.nan
    count_m, count_m1, n = counts
    # both counts include self-matches here
    phi_m = count_m / (n * n)
    phi_m1 = count_m1 / (n * n)
    if phi_m <= 0 or phi_m1 <= 0:
        return np.nan
    return float(np.log(phi_m) - np.log(phi_m1))


def sample_entropy(x, m=ENTROPY_M, r_factor=ENTROPY_R_FACTOR):
    """
    Sample entropy: same idea without the self-matches, which removes the bias
    that makes ApEn depend on record length.

    Usually the more trustworthy of the two when people have different amounts
    of analysable data, which is the case here.
    """
    x = np.asarray(x, dtype=float)
    sd = np.std(x)
    if x.size < m + 2 or sd == 0:
        return np.nan
    r = r_factor * sd
    counts = _entropy_counts(x, m, r)
    if counts is None:
        return np.nan
    count_m, count_m1, n = counts
    # count_neighbors counts each ordered pair including the N self-matches,
    # so subtracting n takes them out of both dimensions
    a = count_m - n
    b = count_m1 - n
    if a <= 0 or b <= 0:
        return np.nan
    return float(-np.log(b / a))


def _lz76_substring_count(binary):
    """
    Kaspar and Schuster substring count for a binary sequence.

    Written as a plain loop over an int8 array with no numpy calls so numba
    can compile it. It's the hottest function in the library: interpreted it
    takes about 5 s per call on a full night and it runs twice per
    participant, which on its own would push a cohort run over two hours.
    Compiled it's roughly three orders of magnitude faster.
    """
    n = binary.size
    i = 0
    k = 1
    l = 1
    k_max = 1
    complexity = 1
    while True:
        if binary[i + k - 1] == binary[l + k - 1]:
            k += 1
            if l + k > n:
                complexity += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                complexity += 1
                l += k_max
                if l + 1 > n:
                    break
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return complexity


# compile the hot loop if numba is installed. The pure Python version stays
# as the fallback so numba isn't a hard dependency (it just runs slower).
try:
    from numba import njit as _njit

    _lz76_substring_count = _njit(cache=True, nogil=True)(_lz76_substring_count)
    LZ_ACCELERATED = True
except Exception:
    LZ_ACCELERATED = False


def lempel_ziv_complexity(x):
    """
    Normalised Lempel-Ziv complexity of the median-binarised trace.

    The signal is turned into a binary string (above / below its median) and
    the distinct substrings needed to build it are counted with the Kaspar
    and Schuster method. Dividing by n / log2(n) makes records of different
    length comparable, which matters since analysable time varies between
    participants.

    Close to 1 means an incompressible, disordered trace.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 10:
        return np.nan
    binary = np.ascontiguousarray((x > np.median(x)).astype(np.int8))
    # a constant trace has no complexity to measure
    if binary.min() == binary.max():
        return 0.0

    complexity = _lz76_substring_count(binary)
    normaliser = n / np.log2(n)
    return float(complexity / normaliser)


def central_tendency_measure(x, radius=CTM_RADIUS):
    """
    Fraction of points in the second-order difference plot that fall inside a
    circle of the given radius.

    Plotting (x[i+2] - x[i+1]) against (x[i+1] - x[i]) bunches up near the
    origin for a smooth trace and spreads out for a variable one, so a high
    CTM means stable saturation.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return np.nan
    d1 = np.diff(x[:-1])
    d2 = np.diff(x[1:])
    inside = np.sum(np.hypot(d1, d2) < radius)
    return float(inside / d1.size)


def detrended_fluctuation_alpha(x, windows=DFA_WINDOWS):
    """
    DFA scaling exponent.

    The cumulative deviation profile is cut into boxes, each box is linearly
    detrended, and the RMS fluctuation is regressed on box size on log-log
    axes. The slope describes long-range correlation: near 0.5 is
    uncorrelated noise, near 1.0 is 1/f behaviour (typical of healthy
    physiological control) and above 1.0 means non-stationary drift.
    """
    x = np.asarray(x, dtype=float)
    usable = [w for w in windows if x.size >= 4 * w]
    if len(usable) < 3:
        return np.nan

    profile = np.cumsum(x - np.mean(x))
    fluctuations = []
    for w in usable:
        n_boxes = profile.size // w
        if n_boxes < 2:
            fluctuations.append(np.nan)
            continue
        boxes = profile[: n_boxes * w].reshape(n_boxes, w)
        t = np.arange(w)
        # least squares line per box, vectorised across boxes
        t_mean = t.mean()
        t_var = np.sum((t - t_mean) ** 2)
        if t_var == 0:
            fluctuations.append(np.nan)
            continue
        box_means = boxes.mean(axis=1, keepdims=True)
        slopes = np.sum((t - t_mean) * (boxes - box_means), axis=1) / t_var
        trend = box_means + slopes[:, None] * (t - t_mean)
        residual = boxes - trend
        fluctuations.append(np.sqrt(np.mean(residual ** 2)))

    fluctuations = np.asarray(fluctuations, dtype=float)
    ok = np.isfinite(fluctuations) & (fluctuations > 0)
    if ok.sum() < 3:
        return np.nan
    sizes = np.asarray(usable, dtype=float)[ok]
    alpha = np.polyfit(np.log(sizes), np.log(fluctuations[ok]), 1)[0]
    return float(alpha)


# --- Family 3: periodicity ---

def periodicity_features(segments_data):
    """
    Spectral description of saturation cycling, pooled across segments.

    Repetitive obstructive events give a roughly periodic sawtooth in
    saturation with a cycle length of about 25 to 100 s. The power in that
    band, and especially its share of total power, works as a spectral
    version of the apnoea-hypopnoea index without any respiratory channels.

    Segments are weighted by length when pooling, so a long stretch of the
    night counts for more than a short one.
    """
    out = {}
    total_powers, band_powers, ratios, peak_freqs, peak_powers, weights = (
        [], [], [], [], [], []
    )

    for seg, fs in segments_data:
        nperseg = int(min(round(PSD_WINDOW_SEC * fs), seg.size))
        if nperseg < 64:
            continue
        # linear detrend takes out the person's baseline saturation, so the
        # spectrum only describes the fluctuation
        freqs, psd = sp_signal.welch(
            seg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend="linear"
        )
        if freqs.size < 3:
            continue

        df = freqs[1] - freqs[0]
        total = float(np.sum(psd) * df)
        band = (freqs >= PSD_OSA_BAND[0]) & (freqs <= PSD_OSA_BAND[1])
        band_power = float(np.sum(psd[band]) * df) if band.any() else np.nan

        total_powers.append(total)
        band_powers.append(band_power)
        ratios.append(band_power / total if total > 0 else np.nan)
        if band.any() and np.isfinite(psd[band]).any():
            local = psd[band]
            peak_freqs.append(float(freqs[band][np.argmax(local)]))
            peak_powers.append(float(np.max(local)))
        weights.append(seg.size)

    if not weights:
        return {
            "psd_total": np.nan,
            "psd_band": np.nan,
            "psd_ratio": np.nan,
            "psd_peak_freq": np.nan,
            "psd_peak_power": np.nan,
            "psd_peak_period_sec": np.nan,
        }

    w = np.asarray(weights, dtype=float)

    def wmean(values):
        values = np.asarray(values, dtype=float)
        if values.size != w.size:
            # peak stats can be missing for some segments
            ok = np.isfinite(values)
            return float(np.mean(values[ok])) if ok.any() else np.nan
        ok = np.isfinite(values)
        if not ok.any():
            return np.nan
        return float(np.sum(values[ok] * w[ok]) / np.sum(w[ok]))

    out["psd_total"] = wmean(total_powers)
    out["psd_band"] = wmean(band_powers)
    out["psd_ratio"] = wmean(ratios)
    out["psd_peak_freq"] = wmean(peak_freqs)
    out["psd_peak_power"] = wmean(peak_powers)
    out["psd_peak_period_sec"] = (
        1.0 / out["psd_peak_freq"]
        if out["psd_peak_freq"] and np.isfinite(out["psd_peak_freq"])
        and out["psd_peak_freq"] > 0
        else np.nan
    )
    return out


# --- Family 4: desaturations ---

def detect_desaturations(seg, fs, threshold):
    """
    Detect desaturation events in one contiguous valid segment (AASM rules).

    From each local maximum (the baseline), take the deepest local minimum
    within MAX_DESAT_DURATION_SEC and keep it if the fall is at least
    `threshold` percent. The event ends once saturation recovers
    RECOVERY_FRACTION of the fall, and scanning resumes from there so events
    never overlap.

    Returns one dict per event: depth, duration, area and the two slopes.
    """
    events = []
    n = seg.size
    if n < 3:
        return events

    max_len = int(round(MAX_DESAT_DURATION_SEC * fs))
    min_len = int(round(MIN_DESAT_DURATION_SEC * fs))

    # indices where the trace changes direction, plus both ends. These are
    # the candidate baselines and nadirs. Working on extrema instead of raw
    # samples means 1-bit quantisation jitter doesn't break the search.
    d = np.diff(seg)
    nonzero = d[d != 0]
    if nonzero.size < 2:
        return events
    direction = np.sign(d)
    # carry the last non-zero direction forward over flat stretches so a
    # plateau doesn't look like a turning point
    nz_idx = np.flatnonzero(direction != 0)
    filled = np.zeros_like(direction)
    filled[nz_idx] = direction[nz_idx]
    for i in range(1, filled.size):
        if filled[i] == 0:
            filled[i] = filled[i - 1]
    turns = np.flatnonzero(np.diff(filled) != 0) + 1
    extrema = np.unique(np.concatenate(([0], turns, [n - 1])))

    i = 0
    while i < extrema.size - 1:
        base_idx = extrema[i]
        base_val = seg[base_idx]

        # deepest minimum reachable within the max event duration
        window_end = base_idx + max_len
        candidates = extrema[(extrema > base_idx) & (extrema <= window_end)]
        if candidates.size == 0:
            i += 1
            continue

        nadir_pos = candidates[np.argmin(seg[candidates])]
        depth = base_val - seg[nadir_pos]

        if depth < threshold:
            i += 1
            continue

        # recovery = climbing back at least part way to baseline
        recovery_level = seg[nadir_pos] + RECOVERY_FRACTION * depth
        after = np.flatnonzero(seg[nadir_pos:] >= recovery_level)
        if after.size:
            end_idx = min(nadir_pos + int(after[0]), n - 1)
        else:
            # no recovery in this segment. Cap at the max duration so one
            # unresolved dip doesn't swallow the rest of the night.
            end_idx = min(nadir_pos + max_len, n - 1)

        duration_samples = end_idx - base_idx
        if duration_samples < min_len:
            i += 1
            continue

        span = seg[base_idx : end_idx + 1]
        # area between the pre-event baseline and the trace, in percent-seconds.
        # It's what the hypoxic burden measures integrate, and it combines
        # depth and duration in one number.
        area = float(np.sum(np.clip(base_val - span, 0, None)) / fs)

        fall_sec = max((nadir_pos - base_idx) / fs, 1.0 / fs)
        rise_sec = max((end_idx - nadir_pos) / fs, 1.0 / fs)

        events.append(
            {
                "start_idx": int(base_idx),
                "nadir_idx": int(nadir_pos),
                "end_idx": int(end_idx),
                "baseline": float(base_val),
                "nadir": float(seg[nadir_pos]),
                "depth": float(depth),
                "duration_sec": float(duration_samples / fs),
                "area": area,
                "slope_desat": float(depth / fall_sec),
                "slope_resat": float((seg[end_idx] - seg[nadir_pos]) / rise_sec),
            }
        )

        # continue from the recovery point so events don't overlap
        next_extrema = extrema[extrema >= end_idx]
        if next_extrema.size == 0:
            break
        i = int(np.searchsorted(extrema, next_extrema[0]))
        if extrema[i] <= base_idx:
            i += 1

    return events


def _distribution_stats(values, prefix, out):
    """
    Add mean / std / median / min / max / skew / kurtosis for one event-level
    quantity to out. The desaturation family needs the same seven summaries
    for depth, duration, area and the two slopes, hence the helper.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        for suffix in ("mean", "std", "median", "min", "max", "skew", "kurtosis"):
            out[f"{prefix}_{suffix}"] = np.nan
        return
    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values)) if values.size > 1 else 0.0
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_skew"] = safe_skew(values)
    out[f"{prefix}_kurtosis"] = safe_kurtosis(values)


def desaturation_features(segments_data, valid_hours, thresholds=DESAT_THRESHOLDS):
    """
    Event-level desaturation biomarkers at each detection threshold.

    The main output is the oxygen desaturation index (ODI), events per hour
    of analysable time. Dividing by valid hours instead of total recording
    hours stops ODI being diluted for people whose probe came off partway
    through the night.
    """
    out = {}
    for threshold in thresholds:
        tag = f"odi{int(threshold)}" if float(threshold).is_integer() else \
            f"odi{threshold}".replace(".", "p")

        all_events = []
        for seg, fs in segments_data:
            all_events.extend(detect_desaturations(seg, fs, threshold))

        n_events = len(all_events)
        out[tag] = float(n_events / valid_hours) if valid_hours > 0 else np.nan
        out[f"{tag}_n_events"] = n_events

        prefix = f"desat{int(threshold)}" if float(threshold).is_integer() else \
            f"desat{threshold}".replace(".", "p")

        if n_events == 0:
            for quantity in ("depth", "duration", "area", "slope_desat",
                             "slope_resat", "nadir"):
                _distribution_stats([], f"{prefix}_{quantity}", out)
            out[f"{prefix}_total_area"] = 0.0
            out[f"{prefix}_area_per_hour"] = 0.0
            continue

        _distribution_stats([e["depth"] for e in all_events],
                            f"{prefix}_depth", out)
        _distribution_stats([e["duration_sec"] for e in all_events],
                            f"{prefix}_duration", out)
        _distribution_stats([e["area"] for e in all_events],
                            f"{prefix}_area", out)
        _distribution_stats([e["slope_desat"] for e in all_events],
                            f"{prefix}_slope_desat", out)
        _distribution_stats([e["slope_resat"] for e in all_events],
                            f"{prefix}_slope_resat", out)
        _distribution_stats([e["nadir"] for e in all_events],
                            f"{prefix}_nadir", out)

        total_area = float(np.sum([e["area"] for e in all_events]))
        out[f"{prefix}_total_area"] = total_area
        # total area per analysable hour is a rate-like burden, comparable to
        # the Azarbarzin hypoxic burden
        out[f"{prefix}_area_per_hour"] = (
            total_area / valid_hours if valid_hours > 0 else np.nan
        )

    return out


# --- Family 5: hypoxic burden ---

def hypoxic_burden(x, fs, valid_hours, thresholds=CT_THRESHOLDS):
    """
    Cumulative time and area below fixed saturation thresholds.

    For each threshold T:
      ct{T}_pct   % of analysable time below T
      ct{T}_min   same thing in minutes (how clinicians usually see it)
      ca{T}       area between T and the trace in percent-minutes, so a deep
                  dip counts for more than a shallow one of the same length

    ct90_pct is T90, the most reported overnight oximetry biomarker and the
    direct match for the SHHS variable pctlt90.
    """
    out = {}
    if x.size == 0:
        for t in thresholds:
            out[f"ct{t}_pct"] = np.nan
            out[f"ct{t}_min"] = np.nan
            out[f"ca{t}"] = np.nan
        out["pod"] = np.nan
        out["ihb"] = np.nan
        return out

    total_minutes = x.size / (fs * 60.0)

    for t in thresholds:
        below = x < t
        frac = float(np.mean(below))
        out[f"ct{t}_pct"] = frac * 100.0
        out[f"ct{t}_min"] = frac * total_minutes
        # integral of the shortfall below T, in percent-minutes
        out[f"ca{t}"] = float(np.sum(np.clip(t - x, 0, None)) / (fs * 60.0))

    # POD: % of time desaturated relative to the person's own median. Unlike
    # the fixed thresholds it adapts to baseline, so it still picks up an
    # unstable night in someone with a high baseline saturation.
    baseline = float(np.median(x))
    out["pod"] = float(np.mean(x < baseline - 3.0) * 100.0)

    # integrated hypoxic burden against the person's own baseline, in
    # percent-minutes per analysable hour
    integrated = float(np.sum(np.clip(baseline - x, 0, None)) / (fs * 60.0))
    out["ihb"] = integrated / valid_hours if valid_hours > 0 else np.nan

    # area below 100%. The most conservative burden measure, since any
    # deficit at all counts.
    out["aod100"] = float(np.sum(np.clip(100.0 - x, 0, None)) / (fs * 60.0))
    out["aod100_per_hour"] = (
        out["aod100"] / valid_hours if valid_hours > 0 else np.nan
    )
    return out


# --- top-level driver ---

def compute_window_features(signal, mask, fs, valid_hours, heavy=True):
    """
    All five biomarker families for one analysis window.

    heavy=False skips complexity and periodicity, the two expensive families.
    It's used for REM and NREM, where the segments are fragmented and often
    short, so entropy and spectral estimates would be unstable anyway and
    not worth the run time.
    """
    features = {}
    x = signal[mask]

    features["valid_hours"] = valid_hours
    features["n_valid_samples"] = int(x.size)
    if x.size == 0:
        return features

    # Family 1
    features.update(general_statistics(x, fs))

    # Family 5, order independent so the pooled vector is fine
    features.update(hypoxic_burden(x, fs, valid_hours))

    # the contiguity-dependent families work on segments
    segments = contiguous_segments(mask, fs, min_seconds=0.0)
    segments_data = [(signal[a:b], fs) for a, b in segments]
    segments_data = [(s, f) for s, f in segments_data if np.isfinite(s).all()]

    features["n_segments"] = len(segments_data)
    if segments_data:
        features["longest_segment_min"] = max(
            s.size for s, _ in segments_data
        ) / (fs * 60.0)
    else:
        features["longest_segment_min"] = np.nan

    if segments_data:
        features["zero_crossings_per_hour"] = zero_crossings(segments_data)
        features["delta_index"] = delta_index(segments_data)
        # Family 4
        features.update(desaturation_features(segments_data, valid_hours))

    if heavy:
        long_segments = [
            (s, f)
            for s, f in segments_data
            if s.size >= int(round(MIN_SEGMENT_SEC * f))
        ]
        # Family 3
        features.update(periodicity_features(long_segments))

        # Family 2, on the single longest qualifying segment. Entropy and DFA
        # are defined for one continuous series, and gluing segments together
        # would add fake jumps that inflate every complexity estimate.
        if long_segments:
            longest = max(long_segments, key=lambda sf: sf[0].size)[0]
            features["apen"] = approximate_entropy(longest)
            features["sampen"] = sample_entropy(longest)
            features["lz_complexity"] = lempel_ziv_complexity(longest)
            features["ctm"] = central_tendency_measure(longest)
            features["dfa_alpha"] = detrended_fluctuation_alpha(longest)
            features["complexity_segment_min"] = longest.size / (fs * 60.0)
        else:
            for key in ("apen", "sampen", "lz_complexity", "ctm", "dfa_alpha",
                        "complexity_segment_min"):
                features[key] = np.nan

    return features


def compute_all_obm(signal, valid_mask, fs, stages=None,
                    windows=("recording", "sleep", "nrem", "rem")):
    """
    Compute the full biomarker set for every requested analysis window.

    Returns a flat dict with keys prefixed by window, e.g. "sleep_ct90_pct"
    is T90 over valid sleep time. Flat so it drops straight into a DataFrame
    row.

    signal      float array, NaN wherever not measured
    valid_mask  bool array, True where the sample is analysable
    fs          sampling rate in Hz (1.0 for SHHS SpO2)
    stages      per-sample stage labels from build_stage_array, or None to
                only compute the whole-recording window
    """
    signal = np.asarray(signal, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    if signal.shape != valid_mask.shape:
        raise ValueError(
            f"signal shape {signal.shape} does not match mask {valid_mask.shape}"
        )
    # the whole library relies on this, so check it up front
    if not np.isfinite(signal[valid_mask]).all():
        raise ValueError("validity mask marks non-finite samples as valid")

    if stages is None:
        stages = np.full(signal.size, STAGE_UNKNOWN, dtype=np.int8)
        windows = ("recording",)

    out = {}
    for window in windows:
        mask = window_mask(valid_mask, stages, window)
        valid_hours = float(mask.sum()) / (fs * 3600.0)
        # REM and NREM skip the expensive families (see
        # compute_window_features)
        heavy = window in ("recording", "sleep")
        window_features = compute_window_features(
            signal, mask, fs, valid_hours, heavy=heavy
        )
        for key, value in window_features.items():
            out[f"{window}_{key}"] = value

    # a few cross-window measures clinicians report directly. The REM vs NREM
    # nadir pair matches the SHHS variables losao2r and losao2nr.
    rem_nadir = out.get("rem_min", np.nan)
    nrem_nadir = out.get("nrem_min", np.nan)
    out["rem_minus_nrem_nadir"] = (
        rem_nadir - nrem_nadir
        if np.isfinite(rem_nadir) and np.isfinite(nrem_nadir)
        else np.nan
    )
    rem_mean = out.get("rem_mean", np.nan)
    nrem_mean = out.get("nrem_mean", np.nan)
    out["rem_minus_nrem_mean"] = (
        rem_mean - nrem_mean
        if np.isfinite(rem_mean) and np.isfinite(nrem_mean)
        else np.nan
    )
    # REM relaxes the upper airway, so OSA that's much worse in REM shows up
    # as a large REM / NREM ODI ratio
    rem_odi = out.get("rem_odi3", np.nan)
    nrem_odi = out.get("nrem_odi3", np.nan)
    out["rem_nrem_odi3_ratio"] = (
        rem_odi / nrem_odi
        if np.isfinite(rem_odi) and np.isfinite(nrem_odi) and nrem_odi > 0
        else np.nan
    )
    return out
