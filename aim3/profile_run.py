"""
Aim 3 stage 0: works out where the training time goes before the long runs.

The original compute plan assumed 20 seconds per training epoch, measured on
synthetic tensors of one fixed length. The first real run took 63.7 seconds per
epoch on 1,096 nights, and the per epoch time drifted from 52 to 75 seconds as
the machine warmed up. Scaled to the real inner fits, which have twice as many
nights, a 14 hour plan becomes something closer to 30 hours for the primary arm
alone.

This times the likely causes one at a time:

  1  GPU or data pipeline? The loader is timed on its own and the GPU is timed
     on pre-made tensors, so the two costs can be told apart.
  2  Does a bigger batch help? Longer sequences under-use the GPU at batch 32,
     and fewer, larger kernel launches often win.
  3  Do loader worker processes help? The store is in shared memory, so workers
     are cheap if the cost is in the per item numpy work.

Validation is timed separately from training, because validation runs on full
nights and training on crops, and the original estimate lumped them together.

Usage:
    python3 profile_run.py --n-train 512 --arm A
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as D            # noqa: E402
import models as M          # noqa: E402
import trainer as T         # noqa: E402
from folds import get_verified_folds  # noqa: E402


def timed_epoch(net, loader, opt, dev, train=True, max_batches=None):
    """One pass over the loader. Returns (seconds, nights, batches)."""
    net.train() if train else net.eval()
    t0, n, nb = time.time(), 0, 0
    for xb, mn, mr, yb, pb in loader:
        xb = xb.to(dev, non_blocking=True)
        mn = mn.to(dev, non_blocking=True)
        mr = mr.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        if train:
            opt.zero_grad(set_to_none=True)
            out = net(xb, mn, mr).squeeze(-1)
            T.compute_loss(out, yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        else:
            with torch.no_grad():
                net(xb, mn, mr)
        n += len(yb)
        nb += 1
        if max_batches and nb >= max_batches:
            break
    if dev.type == "mps":
        torch.mps.synchronize()
    return time.time() - t0, n, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=512)
    ap.add_argument("--arm", default="A")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {dev}, torch {torch.__version__}")
    data = get_verified_folds(quiet=True)
    store = D.SignalStore()
    y, pid = data["y"], data["pid"]
    sub = np.arange(args.n_train)

    print(f"\nprofiling arm {args.arm} on {len(sub)} nights")

    # --- 1. loader only ---
    print("\n1. DATA LOADER ALONE (no GPU work at all)")
    for workers in (0, 2, 4):
        loader, ds = D.make_loader(store, pid[sub], y[sub], arm="primary",
                                   train=True, batch_size=32, seed=42,
                                   num_workers=workers)
        ds.set_epoch(0)
        t0, nb = time.time(), 0
        for batch in loader:
            nb += 1
        dt = time.time() - t0
        print(f"   workers {workers}: {dt:.1f}s for {nb} batches "
              f"({1000 * dt / max(nb, 1):.0f} ms per batch)")

    # --- 2. GPU only, by sequence length and batch size ---
    print("\n2. GPU ALONE (one pre-made batch, repeated)")
    net = M.build(args.arm).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for L in (21600 + 4800, 26400 + 4800, 36000):
        for bs in (16, 32, 64):
            xb = torch.randn(bs, 1, L, device=dev)
            mn = torch.zeros(bs, L, device=dev)
            mn[:, 2400:L - 2400] = 1
            mr = torch.zeros(bs, L, device=dev)
            mr[:, L // 2:L - 2400] = 1
            yb = torch.randint(0, 2, (bs,), device=dev).float()
            # one warmup step so compilation time isn't counted
            opt.zero_grad(set_to_none=True)
            T.compute_loss(net(xb, mn, mr).squeeze(-1), yb).backward()
            opt.step()
            if dev.type == "mps":
                torch.mps.synchronize()
            t0 = time.time()
            steps = 3
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                T.compute_loss(net(xb, mn, mr).squeeze(-1), yb).backward()
                opt.step()
            if dev.type == "mps":
                torch.mps.synchronize()
            per = (time.time() - t0) / steps
            print(f"   L {L:6d} batch {bs:3d}: {per:.3f}s per step, "
                  f"{1000 * per / bs:.1f} ms per night")
            del xb, mn, mr, yb
            if dev.type == "mps":
                torch.mps.empty_cache()

    # --- 3. full training epoch ---
    print("\n3. FULL TRAINING EPOCH, by batch size and workers")
    for bs, workers in [(32, 0), (32, 2), (64, 0), (64, 2)]:
        net = M.build(args.arm).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        loader, ds = D.make_loader(store, pid[sub], y[sub], arm="primary",
                                   train=True, batch_size=bs, seed=42,
                                   num_workers=workers)
        ds.set_epoch(0)
        dt, n, nb = timed_epoch(net, loader, opt, dev, train=True)
        print(f"   batch {bs:3d} workers {workers}: {dt:.1f}s for {n} nights "
              f"({1000 * dt / n:.0f} ms per night, {nb} batches)")
        del net, opt
        if dev.type == "mps":
            torch.mps.empty_cache()

    # --- 4. validation epoch ---
    print("\n4. VALIDATION EPOCH (full nights, no crops)")
    for bs in (16, 32):
        net = M.build(args.arm).to(dev)
        loader, _ = D.make_loader(store, pid[sub], y[sub], arm="primary",
                                  train=False, batch_size=bs, seed=42)
        dt, n, nb = timed_epoch(net, loader, None, dev, train=False)
        print(f"   batch {bs:3d}: {dt:.1f}s for {n} nights "
              f"({1000 * dt / n:.0f} ms per night)")
        del net
        if dev.type == "mps":
            torch.mps.empty_cache()

    # --- 5. cost per arm ---
    print("\n5. ARM COST COMPARISON (one training epoch each, batch 32)")
    for arm in ["A", "B", "NC"]:
        net = M.build(arm).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
        loader, ds = D.make_loader(store, pid[sub], y[sub],
                                   arm=M.arm_input(arm), train=True,
                                   batch_size=32, seed=42)
        ds.set_epoch(0)
        dt, n, nb = timed_epoch(net, loader, opt, dev, train=True)
        print(f"   {arm:3s} ({M.count_parameters(net):>7,} params): {dt:.1f}s "
              f"for {n} nights ({1000 * dt / n:.0f} ms per night)")
        del net, opt
        if dev.type == "mps":
            torch.mps.empty_cache()

    print("\n6. IMPLIED COST OF THE REAL PLAN")
    print("   A real inner fit trains on 2,321 nights and validates on 1,161.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
