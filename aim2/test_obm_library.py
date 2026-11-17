"""
Correctness tests for the OBM library.

Every biomarker is checked on a signal where the answer is known by
construction. A wrong feature doesn't crash anything, it just gives a
plausible-looking number that quietly makes every model trained on it worse,
so testing matters more than usual here.

Tests are grouped by what could realistically go wrong:

  * threshold measures using the wrong denominator (total recording time
    instead of analysable time), the same kind of bug as in Aim 1
  * the desaturation detector miscounting, double counting, or counting
    events that straddle a gap
  * complexity and periodicity measures that don't actually tell ordered
    and disordered signals apart
  * stage masks leaking wake samples into sleep windows

Run:
    python3 aim2/test_obm_library.py
"""

import sys
import time

import numpy as np
import pandas as pd

from obm_library import (
    STAGE_N2,
    STAGE_REM,
    STAGE_WAKE,
    approximate_entropy,
    build_stage_array,
    central_tendency_measure,
    compute_all_obm,
    contiguous_segments,
    delta_index,
    desaturation_features,
    detect_desaturations,
    detrended_fluctuation_alpha,
    general_statistics,
    hypoxic_burden,
    lempel_ziv_complexity,
    periodicity_features,
    sample_entropy,
    window_mask,
    zero_crossings,
)

FS = 1.0
PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def approx(a, b, tol=1e-6):
    return np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol


print("\n[1] Segment decomposition")
mask = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=bool)
segs = contiguous_segments(mask, FS)
check("contiguous_segments finds all runs", segs == [(0, 2), (4, 7), (8, 9)], str(segs))

segs_min = contiguous_segments(mask, FS, min_seconds=3)
check("min_seconds filter drops short runs", segs_min == [(4, 7)], str(segs_min))

check("empty mask gives no segments",
      contiguous_segments(np.zeros(10, dtype=bool), FS) == [])


print("\n[2] Hypoxic burden - the denominator must be analysable time")
# 1000 s of signal: 200 s at 85%, 800 s at 95%. True T90 = 20%.
x = np.concatenate([np.full(200, 85.0), np.full(800, 95.0)])
hb = hypoxic_burden(x, FS, valid_hours=1000 / 3600.0)
check("ct90_pct = 20% exactly", approx(hb["ct90_pct"], 20.0), f"got {hb['ct90_pct']}")
check("ct95_pct = 20% (95 is strict less-than)",
      approx(hb["ct95_pct"], 20.0), f"got {hb['ct95_pct']}")
check("ct80_pct = 0%", approx(hb["ct80_pct"], 0.0), f"got {hb['ct80_pct']}")
# area below 90 = 200 s * 5% = 1000 percent-seconds = 16.667 percent-minutes
check("ca90 area correct", approx(hb["ca90"], 1000 / 60.0, 1e-6),
      f"got {hb['ca90']}")
check("ct90_min = 3.333 min", approx(hb["ct90_min"], 200 / 60.0, 1e-6),
      f"got {hb['ct90_min']}")

# regression test for the Aim 1 bug: same physiology but half the night
# wasn't measured, so T90 over analysable time must not change
signal_full = np.concatenate([x, np.full(1000, np.nan)])
mask_full = np.concatenate([np.ones(1000, dtype=bool), np.zeros(1000, dtype=bool)])
hb_gap = hypoxic_burden(signal_full[mask_full], FS,
                        valid_hours=mask_full.sum() / 3600.0)
check("T90 unchanged when half the record is unmeasured",
      approx(hb_gap["ct90_pct"], 20.0), f"got {hb_gap['ct90_pct']}")
# and the old failure mode: filling the gap halves it
filled = np.concatenate([x, np.full(1000, 95.0)])
check("gap-filling WOULD have corrupted it (bug reproduced)",
      approx(float(np.mean(filled < 90) * 100), 10.0),
      "forward fill halves T90 as expected")


print("\n[3] General statistics")
gs = general_statistics(x, FS)
check("min is the nadir", approx(gs["min"], 85.0))
check("max correct", approx(gs["max"], 95.0))
check("median correct", approx(gs["median"], 95.0))
check("mean correct", approx(gs["mean"], (200 * 85 + 800 * 95) / 1000.0))
check("range correct", approx(gs["range"], 10.0))
check("p1 sits in the low tail", gs["p1"] == 85.0, f"got {gs['p1']}")


print("\n[4] Desaturation detector")
def make_desats(n_events, depth, baseline=96.0, period=60, fall=15, dwell=10):
    """Build a trace with exactly n_events identical triangular dips."""
    trace = np.full(n_events * period, baseline)
    for k in range(n_events):
        s = k * period + 5
        trace[s : s + fall] = np.linspace(baseline, baseline - depth, fall)
        trace[s + fall : s + fall + dwell] = baseline - depth
        trace[s + fall + dwell : s + fall + dwell + fall] = np.linspace(
            baseline - depth, baseline, fall
        )
    return trace

trace = make_desats(10, depth=5.0)
ev4 = detect_desaturations(trace, FS, threshold=4.0)
check("detects all 10 dips at 4% threshold", len(ev4) == 10, f"got {len(ev4)}")

ev6 = detect_desaturations(trace, FS, threshold=6.0)
check("detects none when threshold exceeds depth", len(ev6) == 0, f"got {len(ev6)}")

depths = [e["depth"] for e in ev4]
check("measured depth matches the 5% construction",
      all(approx(d, 5.0, 0.3) for d in depths), f"got {depths[:3]}")

check("events do not overlap",
      all(ev4[i]["end_idx"] <= ev4[i + 1]["start_idx"] for i in range(len(ev4) - 1)),
      "overlapping events would double count ODI")

check("nadir lies between start and end",
      all(e["start_idx"] < e["nadir_idx"] <= e["end_idx"] for e in ev4))

check("desaturation slope is positive",
      all(e["slope_desat"] > 0 for e in ev4))

check("event area is positive",
      all(e["area"] > 0 for e in ev4))

# a flat trace should give no events
check("flat trace yields no desaturations",
      detect_desaturations(np.full(600, 96.0), FS, 3.0) == [])

# a slow monotonic drift isn't a desaturation
drift = np.linspace(96.0, 86.0, 3600)
ev_drift = detect_desaturations(drift, FS, 4.0)
check("slow overnight drift is not counted as many events",
      len(ev_drift) <= 2, f"got {len(ev_drift)} events from a pure 10% drift")


print("\n[5] ODI uses analysable time, and events never cross a gap")
# 10 dips in exactly 600 s of valid data = 60 events/hour
trace10 = make_desats(10, depth=5.0)
odi = desaturation_features([(trace10, FS)], valid_hours=trace10.size / 3600.0)
check("ODI4 = 60 events/hour for 10 dips in 600 s",
      approx(odi["odi4"], 60.0, 0.1), f"got {odi['odi4']}")
check("ODI4 event count recorded", odi["odi4_n_events"] == 10)

# same trace split into two segments with a gap. The dips are the same, so
# the count can't change and no phantom event should appear at the seam
# where the trace jumps.
seg_a, seg_b = trace10[:300], trace10[300:]
odi_split = desaturation_features(
    [(seg_a, FS), (seg_b, FS)], valid_hours=trace10.size / 3600.0
)
check("splitting into segments does not invent events at the seam",
      odi_split["odi4_n_events"] <= 10, f"got {odi_split['odi4_n_events']}")

# The actual seam problem: a high stretch of night, an unmeasured gap, then a
# lower stretch. Joining across the gap would read the baseline shift as one
# huge desaturation that never happened.
#
# Both stretches get a small ripple like real oximetry has. A perfectly flat
# step would be thrown out by the detector's degenerate-input guard (one
# direction change isn't an event), which would hide the problem instead of
# showing it.
t_ripple = np.arange(100)
high = 96.0 + 0.4 * np.sin(2 * np.pi * t_ripple / 25.0)
low = 79.0 + 0.4 * np.sin(2 * np.pi * t_ripple / 25.0)
seamed = desaturation_features([(high, FS), (low, FS)], valid_hours=200 / 3600.0)
concat = desaturation_features([(np.concatenate([high, low]), FS)],
                               valid_hours=200 / 3600.0)
check("per-segment detection ignores the cross-gap level jump",
      seamed["odi4_n_events"] == 0, f"got {seamed['odi4_n_events']}")
check("concatenating WOULD have invented a desaturation (hazard confirmed)",
      concat["odi4_n_events"] >= 1,
      f"got {concat['odi4_n_events']}, showing why segments matter")


print("\n[6] Complexity measures must discriminate order from disorder")
rng = np.random.default_rng(0)
noise = rng.normal(95, 2, 3000)
sine = 95 + 2 * np.sin(2 * np.pi * np.arange(3000) / 60.0)

check("SampEn higher for noise than for a sine",
      sample_entropy(noise) > sample_entropy(sine),
      f"noise {sample_entropy(noise):.3f} vs sine {sample_entropy(sine):.3f}")
check("ApEn higher for noise than for a sine",
      approximate_entropy(noise) > approximate_entropy(sine),
      f"noise {approximate_entropy(noise):.3f} vs sine {approximate_entropy(sine):.3f}")
check("LZ complexity higher for noise than for a sine",
      lempel_ziv_complexity(noise) > lempel_ziv_complexity(sine),
      f"noise {lempel_ziv_complexity(noise):.3f} vs sine {lempel_ziv_complexity(sine):.3f}")
check("CTM higher (more stable) for a smooth sine than for noise",
      central_tendency_measure(sine) > central_tendency_measure(noise),
      f"sine {central_tendency_measure(sine):.3f} vs noise {central_tendency_measure(noise):.3f}")
check("LZ of a constant trace is 0",
      lempel_ziv_complexity(np.full(1000, 95.0)) == 0.0)
check("SampEn of a constant trace is NaN not a number",
      not np.isfinite(sample_entropy(np.full(1000, 95.0))))

# DFA: white noise should sit near 0.5, a random walk near 1.5
alpha_noise = detrended_fluctuation_alpha(rng.normal(0, 1, 4000))
alpha_walk = detrended_fluctuation_alpha(np.cumsum(rng.normal(0, 1, 4000)))
check("DFA alpha near 0.5 for white noise",
      0.35 < alpha_noise < 0.68, f"got {alpha_noise:.3f}")
check("DFA alpha near 1.5 for a random walk",
      1.2 < alpha_walk < 1.8, f"got {alpha_walk:.3f}")
check("DFA separates the two regimes", alpha_walk > alpha_noise + 0.5)


print("\n[7] Periodicity must find a known cycle")
# a 40 s cycle is 0.025 Hz, inside the 0.010 to 0.040 Hz apnoea band
cycle40 = 95 + 2 * np.sin(2 * np.pi * np.arange(6000) / 40.0)
pf = periodicity_features([(cycle40, FS)])
check("recovers the 40 s cycle period",
      approx(pf["psd_peak_period_sec"], 40.0, 4.0),
      f"got {pf['psd_peak_period_sec']:.2f} s")
check("in-band power dominates for an in-band cycle",
      pf["psd_ratio"] > 0.8, f"got ratio {pf['psd_ratio']:.3f}")

# a 5 s cycle is 0.2 Hz, well outside the band, so the ratio should collapse
cycle5 = 95 + 2 * np.sin(2 * np.pi * np.arange(6000) / 5.0)
pf_out = periodicity_features([(cycle5, FS)])
check("out-of-band cycle gives a low band ratio",
      pf_out["psd_ratio"] < 0.2, f"got {pf_out['psd_ratio']:.4f}")


print("\n[8] Variability measures")
check("zero crossings higher for an oscillating trace than a flat one",
      zero_crossings([(sine, FS)]) > zero_crossings([(np.full(3000, 95.0), FS)])
      or not np.isfinite(zero_crossings([(np.full(3000, 95.0), FS)])),
      "a flat trace has no crossings to count")
check("delta index larger for noise than for a flat trace",
      delta_index([(noise, FS)]) > delta_index([(np.full(3000, 95.0), FS)]),
      f"noise {delta_index([(noise, FS)]):.4f}")


print("\n[9] Sleep staging and window masks")
events = pd.DataFrame(
    [
        {"event_concept": "Wake", "start_sec": 0.0, "duration_sec": 300.0},
        {"event_concept": "Stage 2 sleep", "start_sec": 300.0, "duration_sec": 600.0},
        {"event_concept": "REM sleep", "start_sec": 900.0, "duration_sec": 300.0},
        {"event_concept": "Arousal", "start_sec": 500.0, "duration_sec": 10.0},
    ]
)
stages = build_stage_array(events, 1200, FS)
check("wake epoch labelled wake", bool(np.all(stages[:300] == STAGE_WAKE)))
check("N2 epoch labelled N2", bool(np.all(stages[300:900] == STAGE_N2)))
check("REM epoch labelled REM", bool(np.all(stages[900:1200] == STAGE_REM)))
check("non-stage events ignored by the stage array",
      bool(np.all(stages[500:510] == STAGE_N2)),
      "an arousal must not overwrite the sleep stage")

all_valid = np.ones(1200, dtype=bool)
check("sleep window excludes wake",
      int(window_mask(all_valid, stages, "sleep").sum()) == 900)
check("rem window selects only REM",
      int(window_mask(all_valid, stages, "rem").sum()) == 300)
check("nrem window selects only NREM",
      int(window_mask(all_valid, stages, "nrem").sum()) == 600)
check("recording window takes everything valid",
      int(window_mask(all_valid, stages, "recording").sum()) == 1200)

# a window mask should never include an invalid sample
partial = all_valid.copy()
partial[300:600] = False
check("window mask never includes invalid samples",
      int(window_mask(partial, stages, "sleep").sum()) == 600,
      "invalid N2 samples must be dropped from the sleep window")


print("\n[10] End-to-end driver")
# 4 hours: wake, then N2 with dips, then REM with deeper dips, plus an
# unmeasured hole in the middle
n = 4 * 3600
sig = np.full(n, 96.0)
sig[3600:7200] = np.tile(make_desats(1, depth=5.0, period=60), 60)[:3600]
sig[7200:10800] = np.tile(make_desats(1, depth=8.0, period=60), 60)[:3600]
valid = np.ones(n, dtype=bool)
valid[5000:6000] = False          # 1000 s unmeasured
sig[~valid] = np.nan

ev = pd.DataFrame(
    [
        {"event_concept": "Wake", "start_sec": 0.0, "duration_sec": 3600.0},
        {"event_concept": "Stage 2 sleep", "start_sec": 3600.0, "duration_sec": 3600.0},
        {"event_concept": "REM sleep", "start_sec": 7200.0, "duration_sec": 3600.0},
        {"event_concept": "Wake", "start_sec": 10800.0, "duration_sec": 3600.0},
    ]
)
st = build_stage_array(ev, n, FS)

t0 = time.time()
feats = compute_all_obm(sig, valid, FS, st)
elapsed = time.time() - t0

check("driver returns a flat dict", isinstance(feats, dict) and len(feats) > 100,
      f"got {len(feats)} features")
check("all four windows present",
      all(any(k.startswith(w + "_") for k in feats)
          for w in ("recording", "sleep", "nrem", "rem")))
check("valid hours excludes the unmeasured hole",
      approx(feats["recording_valid_hours"], (n - 1000) / 3600.0, 1e-6),
      f"got {feats['recording_valid_hours']:.4f}")
check("sleep window smaller than recording window",
      feats["sleep_valid_hours"] < feats["recording_valid_hours"])
check("REM nadir deeper than NREM nadir (by construction)",
      feats["rem_min"] < feats["nrem_min"],
      f"rem {feats['rem_min']:.2f} vs nrem {feats['nrem_min']:.2f}")
check("REM minus NREM nadir is negative as constructed",
      feats["rem_minus_nrem_nadir"] < 0)
# Both stages get one dip per 60 s and only the depth differs (5% in N2, 8%
# in REM), so the event rates should match while the depths separate.
# Testing for a higher REM ODI would be testing for something that isn't in
# the signal.
check("REM and NREM ODI3 rates match, as constructed",
      abs(feats["rem_odi3"] - feats["nrem_odi3"]) < 5.0,
      f"rem {feats['rem_odi3']:.1f} vs nrem {feats['nrem_odi3']:.1f}")
check("REM desaturation depth exceeds NREM depth, as constructed",
      feats["rem_desat3_depth_mean"] > feats["nrem_desat3_depth_mean"] + 1.0,
      f"rem {feats['rem_desat3_depth_mean']:.2f} vs "
      f"nrem {feats['nrem_desat3_depth_mean']:.2f}")
check("segment count reflects the punched hole",
      feats["recording_n_segments"] == 2,
      f"got {feats['recording_n_segments']}")
check("no feature is infinite",
      not any(np.isinf(v) for v in feats.values() if isinstance(v, float)))

# check the invariant guard actually fires
bad_mask = np.ones(n, dtype=bool)
try:
    compute_all_obm(sig, bad_mask, FS, st)
    check("driver rejects a mask claiming NaN samples are valid", False,
          "no exception raised")
except ValueError:
    check("driver rejects a mask claiming NaN samples are valid", True)

try:
    compute_all_obm(sig, valid[:100], FS, st)
    check("driver rejects a shape mismatch", False, "no exception raised")
except ValueError:
    check("driver rejects a shape mismatch", True)

print(f"\n  timing: {elapsed:.2f} s for a 4 h record "
      f"({len(feats)} features)")
print(f"  projected for 2591 participants, single core: "
      f"{elapsed * 2591 / 60:.1f} min")


print("\n" + "=" * 70)
print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
print("=" * 70)
if FAILED:
    for name, detail in FAILED:
        print(f"  FAILED: {name}\n          {detail}")
    sys.exit(1)
print("All OBM library correctness tests passed.")
sys.exit(0)
