"""
Aim 3 step 5: trains one inner model and saves every epoch for selection later.

The trainer doesn't pick an epoch. It trains to a fixed cap and saves the
validation predictions and a checkpoint every epoch, and a separate script
chooses one epoch for all three inner models from their pooled predictions.
Picking each model's best epoch on its own validation fold would tune the same
rows the stacker is later fitted on. Two epochs are chosen from the same runs,
because the best standalone AUC can just be the best age proxy: E*_marg (best
pooled AUC) for the head to head contrast, and E*_inc (best deviance reduction
over a cross fitted clinical predictor) for the incremental contrasts.

Fixed in advance: plain binary cross entropy (at a 19 percent event rate, class
weighting, focal loss or resampling would add no discrimination and would hurt
calibration), head bias started at the prevalence, AdamW at 1e-3 with weight
decay 1e-2, one warmup epoch then cosine decay, gradient clipping at 1.0, and an
exponential moving average of the weights, which is what gets evaluated and
saved.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models import EMA


def compute_loss(out, target, loss_kind="bce"):
    """
    The only loss used anywhere in Aim 3.

    Kept in its own function so the unit tests can check numerically what it is.
    """
    if loss_kind == "bce":
        return F.binary_cross_entropy_with_logits(out, target)
    if loss_kind == "huber":
        return F.huber_loss(out, target)
    raise ValueError(f"unknown loss_kind {loss_kind!r}")


# Introspection helpers, checked by test_aim3.py. They test behaviour
# numerically and scan the syntax tree, not the text, since comments can
# mention terms like "pos_weight" or "patience" without using them.

TRAINING_FUNCTIONS = ("compute_loss", "make_optimiser", "make_scheduler",
                      "predict", "train_unit")


def _code_identifiers(only=TRAINING_FUNCTIONS):
    """
    Identifiers used in actual code inside the training functions.

    Limited to those functions on purpose. Scanning the whole module would find
    "pos_weight" inside uses_plain_bce, which uses it to build the
    counter-example it checks against.
    """
    import ast
    tree = ast.parse(open(__file__).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in only:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
                elif isinstance(sub, ast.Attribute):
                    names.add(sub.attr)
                elif isinstance(sub, ast.keyword) and sub.arg:
                    names.add(sub.arg)
                elif isinstance(sub, ast.arg):
                    names.add(sub.arg)
    return names


def uses_plain_bce():
    """
    True when the classification loss is exactly unweighted BCE with logits.

    Computes it on an imbalanced batch and compares against the unweighted
    reference, and against a class weighted version it must not equal. If any
    weighting crept in, the first comparison would fail and the second would
    pass.
    """
    torch.manual_seed(0)
    out = torch.randn(64)
    target = torch.zeros(64)
    target[:8] = 1.0                       # 12.5 percent positive
    got = float(compute_loss(out, target, "bce"))
    plain = float(F.binary_cross_entropy_with_logits(out, target))
    weighted = float(F.binary_cross_entropy_with_logits(
        out, target, pos_weight=torch.tensor(7.0)))
    identifiers = _code_identifiers()
    banned = {"pos_weight", "weight", "FocalLoss", "focal_gamma", "SMOTE",
              "SMOTETomek", "WeightedRandomSampler", "class_weight"}
    return (abs(got - plain) < 1e-9 and abs(got - weighted) > 1e-6
            and not (identifiers & banned))


def selects_epoch_after_training():
    """
    True when the fit itself picks no epoch and restores no checkpoint.

    train_unit should return one validation record per epoch and nothing naming
    a best epoch, so the choice can only happen in a later script. This checks
    its syntax tree for the identifiers a selecting loop would need.
    """
    import inspect
    src = inspect.getsource(train_unit)
    import ast
    names = set()
    for node in ast.walk(ast.parse(src.lstrip())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    banned = {"best_auc", "best_epoch", "best_state", "patience", "stalled",
              "early_stop"}
    return not (names & banned)


def make_optimiser(model, lr=1e-3, weight_decay=1e-2):
    """
    AdamW with no weight decay on normalisation scales or biases.

    Decaying a batch norm scale or a bias shrinks something that has no reason
    to be small, and it measurably hurts on small datasets. Splitting the
    parameters into two groups only takes four lines.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith("bias") else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(opt, epochs, steps_per_epoch, warmup_epochs=1):
    """
    One warmup epoch, then cosine decay to roughly 1 percent of the peak.

    Warmup is there because the first steps of a randomly initialised CNN on
    sequences this long give large gradients that can push the weights somewhere
    the optimiser then takes many epochs to recover from.
    """
    warmup = max(1, warmup_epochs * steps_per_epoch)
    total = max(warmup + 1, epochs * steps_per_epoch)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + float(np.cos(np.pi * min(t, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_at)


# A logit bigger than this means nothing: sigmoid(50) is 1 to machine precision,
# so no calibrated risk model should ever go past it. Only used to detect
# backend faults, never to clip real values.
SANE_LOGIT = 50.0


def _forward(model, xb, mn, mr, pb, offsets):
    if offsets is None:
        return model(xb, mn, mr)
    idx = pb.numpy().astype(np.int64)
    off = torch.tensor([offsets[int(p)] for p in idx],
                       dtype=torch.float32, device=xb.device)
    return model(xb, mn, mr, off)


@torch.no_grad()
def predict(model, loader, device, offsets=None, report=None):
    """
    Score every participant in a loader, returning (logits, labels, pids).

    Evaluation loaders are length sorted, so callers realign on the IDs, not
    position. The Apple GPU backend sometimes returns a garbage value (up to
    1.2e35) for one element of a batch while the weights are fine. That barely
    moves the AUC but would wreck the standardisation, the deviance criterion
    and any regression using the score, so a batch with a non-finite or
    implausible logit is rerun up to twice on the GPU and then on the CPU.
    Repairs are counted in report.
    """
    model.eval()
    logits, ys, pids = [], [], []
    n_repaired, n_batches = 0, 0
    for xb, mn, mr, yb, pb in loader:
        n_batches += 1
        xb_d = xb.to(device, non_blocking=True)
        mn_d = mn.to(device, non_blocking=True)
        mr_d = mr.to(device, non_blocking=True)
        out = _forward(model, xb_d, mn_d, mr_d, pb, offsets)
        vals = out.squeeze(-1).float().cpu().numpy()

        if not np.all(np.isfinite(vals)) or np.any(np.abs(vals) > SANE_LOGIT):
            repaired = False
            for _ in range(2):
                out = _forward(model, xb_d, mn_d, mr_d, pb, offsets)
                retry = out.squeeze(-1).float().cpu().numpy()
                if np.all(np.isfinite(retry)) and \
                        np.all(np.abs(retry) <= SANE_LOGIT):
                    vals, repaired = retry, True
                    break
            if not repaired:
                # the CPU path is deterministic and has never given these
                # values, so it wins when the GPU disagrees
                cpu_model = model.to("cpu")
                out = _forward(cpu_model, xb.to("cpu"), mn.to("cpu"),
                               mr.to("cpu"), pb, offsets)
                vals = out.squeeze(-1).float().cpu().numpy()
                model.to(device)
                if not np.all(np.isfinite(vals)) or \
                        np.any(np.abs(vals) > SANE_LOGIT):
                    raise RuntimeError(
                        f"a batch produced implausible logits on both the GPU "
                        f"and the CPU (range {vals.min():.4g} to "
                        f"{vals.max():.4g}); this is a real model fault, not a "
                        f"backend fault")
            n_repaired += 1

        logits.append(vals)
        ys.append(yb.numpy())
        pids.append(pb.numpy().astype(np.int64))

    if report is not None:
        report["batches"] = report.get("batches", 0) + n_batches
        report["repaired"] = report.get("repaired", 0) + n_repaired
    return (np.concatenate(logits), np.concatenate(ys), np.concatenate(pids))


def train_unit(model, train_loader, train_ds, val_loader, device, epochs=16,
               lr=1e-3, weight_decay=1e-2, grad_clip=1.0, ema_decay=0.99,
               offsets=None, out_dir=None, tag="", verbose=True,
               target=None, loss_kind="bce"):
    """
    Train one inner model to a fixed epoch cap, recording every epoch.

    Nothing is selected here. The return value has the per epoch validation
    logits, and with out_dir set there's a checkpoint per epoch on disk.
    target and loss_kind are only for the label free gate tasks (regressing the
    oxygen desaturation index with a Huber loss), which run this loop without
    touching the CVD label.
    """
    model = model.to(device)
    opt = make_optimiser(model, lr, weight_decay)
    sched = make_scheduler(opt, epochs, max(1, len(train_loader)))
    ema = EMA(model, ema_decay)

    history, val_records = [], []
    predict_report = {}
    n_skipped = 0
    t0 = time.time()

    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        model.train()
        running, n = 0.0, 0
        for xb, mn, mr, yb, pb in train_loader:
            xb = xb.to(device, non_blocking=True)
            mn = mn.to(device, non_blocking=True)
            mr = mr.to(device, non_blocking=True)
            if target is None:
                tgt = yb.to(device, non_blocking=True)
            else:
                idx = pb.numpy().astype(np.int64)
                tgt = torch.tensor([target[int(p)] for p in idx],
                                   dtype=torch.float32, device=xb.device)

            opt.zero_grad(set_to_none=True)
            if offsets is None:
                out = model(xb, mn, mr).squeeze(-1)
            else:
                idx = pb.numpy().astype(np.int64)
                off = torch.tensor([offsets[int(p)] for p in idx],
                                   dtype=torch.float32, device=xb.device)
                out = model(xb, mn, mr, off).squeeze(-1)

            # A bad batch shouldn't get a gradient step. The backend fault
            # described in predict() is transient, so just skip the batch and
            # carry on instead of aborting a fourteen minute unit. Skips are
            # counted and recorded.
            bad = (~torch.isfinite(out)).any() or \
                (out.abs() > SANE_LOGIT).any()
            if bool(bad):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue

            loss = compute_loss(out, tgt, loss_kind)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            ema.update(model)
            running += float(loss.detach().cpu()) * len(tgt)
            n += len(tgt)

        # the EMA weights are what gets evaluated and saved (never the raw
        # weights), so the recorded trajectory is the smoothed one
        shadow = ema.module().to(device)
        v_logit, v_y, v_pid = predict(shadow, val_loader, device, offsets,
                                      report=predict_report)
        rec = {"epoch": epoch, "logit": v_logit, "y": v_y, "pid": v_pid}
        val_records.append(rec)

        train_loss = running / max(n, 1)
        if loss_kind == "bce" and len(np.unique(v_y)) > 1:
            from sklearn.metrics import roc_auc_score
            v_metric = float(roc_auc_score(v_y, v_logit))
            metric_name = "val AUC"
        else:
            v_metric = float(np.corrcoef(v_logit, v_y)[0, 1]) \
                if len(v_y) > 2 else float("nan")
            metric_name = "val r"

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_metric": v_metric,
                        "lr": opt.param_groups[0]["lr"],
                        "seconds": time.time() - t0})

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            torch.save({k: v.detach().cpu() for k, v
                        in shadow.state_dict().items()},
                       os.path.join(out_dir, f"{tag}ema_epoch{epoch:02d}.pt"))

        if verbose:
            print(f"      epoch {epoch:2d}  train {train_loss:.4f}  "
                  f"{metric_name} {v_metric:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{history[-1]['seconds']:.0f}s", flush=True)

    return {"history": history, "val_records": val_records,
            "ema": ema.module(), "model": model,
            "seconds": time.time() - t0,
            # how often the backend misbehaved, so the run can report it (see
            # predict() for what these mean)
            "batches_skipped_in_training": n_skipped,
            "validation_batches": predict_report.get("batches", 0),
            "validation_batches_repaired": predict_report.get("repaired", 0)}


def inner_splits(train_idx, y, n_splits=3, seed=42):
    """
    Split an outer training fold the same way the Aim 2 grid search did.

    train_logistic_regression passes StratifiedKFold(3, shuffle=True,
    random_state=42) on the outer training rows to GridSearchCV, so using the
    same call puts the deep arm's cross fitted predictions on the same partition
    the stacker's own inner CV uses.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for tr, va in skf.split(np.zeros((len(train_idx), 1)), y[train_idx]):
        out.append((train_idx[tr], train_idx[va]))
    return out
