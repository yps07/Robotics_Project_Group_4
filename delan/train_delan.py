"""Train a Deep Lagrangian Network on the official manipulator inverse-dynamics data.

Examples
--------
    python train_delan.py                          # 100 % data, seed 42
    python train_delan.py --fraction 0.1
    python train_delan.py --all                    # 10/30/50/100 %, seed 42

Data
----
    train : nested subset identical to black_box_mlp (common.subset_indices)
    val   : early stopping (best-validation weights are kept)
    test / ood are never touched here.

Outputs
-------
    checkpoints/delan_f{fraction}_s{seed}.pt   best-on-val weights + config + history
    runs/<run name>/                           TensorBoard logs and history JSON
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from common import FRACTIONS, LOG_DIR, SEEDS, ckpt_path, load_split, subset_indices, tau_std
from delan import DeLaN, save_checkpoint

# Selected by tune_delan.py on the validation split (see results/tuning_results.csv).
DEFAULTS = dict(width=128, depth=3, activation="tanh", lr=1e-3, weight_decay=1e-5,
                batch_size=256, epochs=300, patience=40)


def weighted_mse(pred, target, scale):
    """MSE on std-normalised torques: identical to the MLP's training/validation loss."""
    return (((pred - target) / scale) ** 2).mean()


@torch.no_grad()
def predict(model, data, batch=4096):
    out = [model(data["q"][i:i + batch], data["qdot"][i:i + batch], data["qddot"][i:i + batch])
           for i in range(0, len(data["q"]), batch)]
    return torch.cat(out)


def evaluate(model, data, scale):
    pred = predict(model, data)
    return weighted_mse(pred, data["tau"], scale).item(), pred


def train(fraction=1.0, seed=42, width=DEFAULTS["width"], depth=DEFAULTS["depth"],
          activation=DEFAULTS["activation"], lr=DEFAULTS["lr"], weight_decay=DEFAULTS["weight_decay"],
          batch_size=DEFAULTS["batch_size"], epochs=DEFAULTS["epochs"], patience=DEFAULTS["patience"],
          learn_damping=True, out=None, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = out or ckpt_path(fraction, seed)
    run = os.path.splitext(os.path.basename(out))[0]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    train_full, val = load_split("train"), load_split("val")
    idx = torch.from_numpy(subset_indices(len(train_full["q"]), fraction))
    tr = {k: v[idx] for k, v in train_full.items()}
    scale = tau_std()
    n = len(idx)

    model = DeLaN(width=width, depth=depth, activation=activation, learn_damping=learn_damping)
    # The 4 damping scalars get a 10x step size (no weight decay). With the network rate they
    # move ~1e-3 per step; the 10 % subset has only 2400 steps in 300 epochs, and without this
    # its root damping stalled at 1.73 (true 2.0) and test RMSE was 0.19 N*m (see README).
    net_params = [p for n_, p in model.named_parameters() if n_ != "d_raw"]
    opt = torch.optim.Adam([{"params": net_params, "weight_decay": weight_decay},
                            {"params": [model.d_raw], "lr": 10 * lr, "weight_decay": 0.0}], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(LOG_DIR, run))
    except Exception:
        pass

    gen = torch.Generator().manual_seed(seed)
    best_val, best_state, best_epoch, history = float("inf"), None, -1, []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        running = 0.0
        for i in range(0, n, batch_size):
            b = perm[i:i + batch_size]
            loss = weighted_mse(model(tr["q"][b], tr["qdot"][b], tr["qddot"][b]), tr["tau"][b], scale)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            running += loss.item() * len(b)
        sched.step()

        model.eval()
        train_loss, (val_loss, _) = running / n, evaluate(model, val, scale)
        damping = model.damping.detach().tolist()
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss, "damping": damping})
        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            for j, d in enumerate(damping):
                writer.add_scalar(f"damping/joint{j}", d, epoch)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"[{run}] ep {epoch:4d}  train {train_loss:.3e}  val {val_loss:.3e}  "
                  f"best {best_val:.3e}@{best_epoch}  D={np.round(damping, 3)}  {time.time() - t0:.0f}s",
                  flush=True)
        if epoch - best_epoch >= patience:
            if verbose:
                print(f"[{run}] early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    hp = dict(fraction=fraction, seed=seed, n_train=n, lr=lr, weight_decay=weight_decay,
              batch_size=batch_size, epochs=epochs, patience=patience, epochs_run=epoch + 1)
    extra = {"history": history, "best_val": best_val, "best_epoch": best_epoch, "hparams": hp,
             "train_time_s": time.time() - t0}
    save_checkpoint(out, model, extra)
    with open(os.path.join(LOG_DIR, f"{run}_history.json"), "w") as f:
        json.dump({"hparams": hp, "config": model.config, "history": history}, f)
    if writer:
        writer.close()
    if verbose:
        print(f"[{run}] saved {out}  n_train={n}  best val {best_val:.4e} (epoch {best_epoch})", flush=True)
    return model, best_val


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all", action="store_true", help="every fraction in FRACTIONS (seed 42)")
    p.add_argument("--width", type=int, default=DEFAULTS["width"])
    p.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    p.add_argument("--activation", default=DEFAULTS["activation"], choices=["softplus", "tanh", "silu"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--no-damping", action="store_true", help="ablation: no dissipative term (D = 0)")
    p.add_argument("--out", default=None, help="checkpoint path (default checkpoints/delan_f*_s*.pt)")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)

    kw = dict(width=a.width, depth=a.depth, activation=a.activation, lr=a.lr,
              weight_decay=a.weight_decay, batch_size=a.batch_size, epochs=a.epochs,
              patience=a.patience, learn_damping=not a.no_damping)
    if a.all:
        for f in FRACTIONS:
            for s in SEEDS:
                train(fraction=f, seed=s, **kw)
    else:
        train(fraction=a.fraction, seed=a.seed, out=a.out, **kw)


if __name__ == "__main__":
    main()
