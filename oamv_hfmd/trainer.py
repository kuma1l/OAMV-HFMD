"""
Training loop for OAMV-HFMD.

Productionized from ``multi-view-hybrid/main.py`` (hyperparameters / optimizer
/ scheduler) and ``multi-view-hybrid/engine/trainer.py`` (train/eval epoch
structure). Adds the convergence-classification pipeline from
``spike_hotels50k_v2/convergence.py``.

PLAN.md §5, §6.0.
"""
from __future__ import annotations

import json
import time
import dataclasses
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import einops
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .convergence import classify
from .utils import get_logger


@dataclass
class TrainConfig:
    """Trainer hyperparameters. Defaults match multi-view-hybrid/main.py (v1.2)."""
    # Optimizer (SGD with momentum, NOT AdamW)
    optimizer:        str   = "sgd"
    lr:               float = 0.01      # OneCycleLR max_lr
    weight_decay:     float = 5e-4
    momentum:         float = 0.9

    # Scheduler (OneCycleLR, NOT cosine-with-warmup)
    scheduler:        str   = "onecycle"
    div_factor:       float = 10.0
    final_div_factor: float = 1000.0
    pct_start_epochs: int   = 5         # 5 / num_epochs of warmup
    anneal_strategy:  str   = "cos"

    # Training loop
    epochs:           int   = 50
    batch_size:       int   = 64        # train batch
    val_batch_size:   int   = 128
    grad_clip:        float = 80.0
    amp:              bool  = True      # torch.cuda.amp.autocast
    seed:             int   = 42
    n_views:          int   = 4
    num_workers:      int   = 8

    # CE loss (no label smoothing upstream)
    label_smoothing:  float = 0.0
    ce_ignore_index:  int   = -1

    # MD loss (Hinton-scaled: temp^2 * lambda already inside the loss fn)
    lambda_md:        float = 0.1
    md_temp:          float = 4.0

    # Overlap-weighted distillation (our method only)
    tau_overlap:      float = 4.0
    overlap_sign:     float = 1.0   # +1 up-weight similar; -1 up-weight complementary

    # Convergence pipeline (PLAN.md §5.5)
    overfit_gap_pp_threshold: float = 30.0  # best-ckpt guard


def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int = 1) -> int:
    pred = logits.topk(k, dim=1).indices
    return (pred == targets.unsqueeze(1)).any(dim=1).sum().item()


class Trainer:
    """Generic trainer for both MV-HFMD baseline and OAMV-HFMD variants.

    Mirrors ``TrainerEngine`` from upstream: per-batch CE on single + mv branches,
    optional MD loss (already Hinton-scaled), grad clip, OneCycleLR per-batch step.
    Best checkpoint saved by validation mv_collection top-1 (single top-1 for n=1).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        md_loss_fn: Optional[Callable] = None,
        device: str = "cuda",
        out_dir: str | Path = "./run",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        self.md_loss_fn = md_loss_fn
        self.device = device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(f"trainer.{self.out_dir.name}", self.out_dir / "train.log")

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.ce_ignore_index,
            label_smoothing=config.label_smoothing,
        )

        # Optimizer: SGD per upstream main.py.
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )

        # Scheduler: OneCycleLR per-batch step.
        steps_per_epoch = max(1, len(train_loader))
        pct_start = max(1e-3, min(0.99, config.pct_start_epochs / max(1, config.epochs)))
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.lr,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            div_factor=config.div_factor,
            final_div_factor=config.final_div_factor,
            pct_start=pct_start,
            anneal_strategy=config.anneal_strategy,
        )

        # Note: upstream does not use GradScaler — autocast only. Mirror that.
        self.use_amp = bool(config.amp) and torch.cuda.is_available()

        # Convergence-pipeline bookkeeping
        self._initial_loss: float | None = None
        self._running_losses: list[float] = []
        self._val_top1_trajectory: list[float] = []
        self._train_top1_trajectory: list[float] = []
        self._best_val_top1: float = 0.0
        self._best_epoch: int = 0
        self._best_state: dict | None = None

        # Persist config alongside training artifacts.
        with open(self.out_dir / "trainer_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

    # ------------------------------------------------------------ training step

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_batches = 0
        view_correct = {"single": 0, "mv_collection": 0}
        view_total = {"single": 0, "mv_collection": 0}
        first_loss_in_epoch: float | None = None

        for idx, batch in enumerate(self.train_loader):
            images, targets, _ = batch
            B, N = images.shape[0], images.shape[1]
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(images)
                total_loss = torch.zeros((), device=self.device)

                for view_type in ("single", "mv_collection"):
                    if view_type not in output:
                        continue
                    if view_type == "mv_collection":
                        t = targets[:, 0].flatten()
                    else:
                        t = targets.flatten()
                    logits = output[view_type]["logits"]
                    total_loss = total_loss + self.criterion(logits, t)

                    # accuracy bookkeeping (under autocast is fine — argmax is dtype-stable)
                    mask = t != self.cfg.ce_ignore_index
                    if mask.any():
                        pred = logits.argmax(dim=1)
                        view_correct[view_type] += (pred[mask] == t[mask]).sum().item()
                        view_total[view_type] += int(mask.sum().item())

                if self.md_loss_fn is not None and "mv_collection" in output:
                    z_mv = output["mv_collection"]["logits"]
                    z_single_BNK = einops.rearrange(
                        output["single"]["logits"], "(b n) k -> b n k", b=B, n=N
                    )
                    if "similarity" in output:
                        # overlap_md_loss signature
                        md = self.md_loss_fn(
                            z_mv, z_single_BNK, output["similarity"],
                            tau_overlap=self.cfg.tau_overlap,
                            tau_kl=self.cfg.md_temp,
                            lambda_hyperparam=self.cfg.lambda_md,
                            overlap_sign=self.cfg.overlap_sign,
                        )
                    else:
                        md = self.md_loss_fn(
                            z_mv, z_single_BNK,
                            tau=self.cfg.md_temp,
                            lambda_hyperparam=self.cfg.lambda_md,
                        )
                    total_loss = total_loss + md

            if not torch.isfinite(total_loss):
                self.logger.warning(
                    f"epoch {epoch} batch {idx}: non-finite loss = {total_loss.item()}, skipping"
                )
                continue

            total_loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            try:
                self.scheduler.step()
            except ValueError:
                # OneCycleLR raises after total_steps is exceeded — match upstream
                pass

            loss_val = float(total_loss.detach().item())
            loss_sum += loss_val
            n_batches += 1
            if first_loss_in_epoch is None:
                first_loss_in_epoch = loss_val
            if self._initial_loss is None:
                self._initial_loss = loss_val

            if idx % 50 == 0:
                cur_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"epoch {epoch} batch {idx}/{len(self.train_loader)} "
                    f"loss={loss_val:.4f} lr={cur_lr:.5f}"
                )

        elapsed = time.time() - t0
        avg_loss = loss_sum / max(1, n_batches)
        self._running_losses.append(avg_loss)

        report: dict = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": self.optimizer.param_groups[0]["lr"],
            "elapsed_sec": elapsed,
        }
        for vt in ("single", "mv_collection"):
            if view_total[vt] > 0:
                acc = view_correct[vt] / view_total[vt]
                report[f"train_top1_{vt}"] = acc
                if vt == "mv_collection":
                    self._train_top1_trajectory.append(acc)
        if "mv_collection" not in view_correct or view_total["mv_collection"] == 0:
            # n=1 case — track single
            if view_total["single"] > 0:
                self._train_top1_trajectory.append(view_correct["single"] / view_total["single"])

        self.logger.info(
            f"epoch {epoch} DONE train_loss={avg_loss:.4f} "
            f"top1_mv={report.get('train_top1_mv_collection', float('nan')):.4f} "
            f"top1_single={report.get('train_top1_single', float('nan')):.4f} "
            f"elapsed={elapsed:.1f}s"
        )
        return report

    # --------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """Quick evaluator used inside training (top-1 / top-5 for both branches).

        For exhaustive C(M, N) test-time eval with single-view dedup, use
        :class:`oamv_hfmd.eval.Evaluator`.
        """
        self.model.eval()
        view_correct = {"single": {1: 0, 5: 0}, "mv_collection": {1: 0, 5: 0}}
        view_total = {"single": 0, "mv_collection": 0}

        for batch in loader:
            images, targets, _ = batch
            B, N = images.shape[0], images.shape[1]
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(images)

            for view_type in ("single", "mv_collection"):
                if view_type not in output:
                    continue
                if view_type == "mv_collection":
                    t = targets[:, 0].flatten()
                else:
                    t = targets.flatten()
                logits = output[view_type]["logits"].float()
                mask = t != self.cfg.ce_ignore_index
                if not mask.any():
                    continue
                logits_m = logits[mask]
                t_m = t[mask]
                view_correct[view_type][1] += _topk_correct(logits_m, t_m, 1)
                view_correct[view_type][5] += _topk_correct(logits_m, t_m, min(5, logits_m.shape[1]))
                view_total[view_type] += int(mask.sum().item())

        out: dict = {}
        for vt in ("single", "mv_collection"):
            if view_total[vt] > 0:
                out[f"{vt}_top1"] = view_correct[vt][1] / view_total[vt]
                out[f"{vt}_top5"] = view_correct[vt][5] / view_total[vt]
        # Headline numbers expected by the caller:
        if "mv_collection_top1" in out:
            out["top1"] = out["mv_collection_top1"]
            out["top5"] = out["mv_collection_top5"]
        else:
            out["top1"] = out.get("single_top1", 0.0)
            out["top5"] = out.get("single_top5", 0.0)
        return out

    # --------------------------------------------------------------- checkpoint

    def _save_checkpoint(self, filename: str, epoch: int, val_top1: float) -> None:
        """Save model weights + light metadata. Matches the upstream
        ``engine/engine.py:save_models`` pattern (model state under "model" key)."""
        payload = {
            "model": self.model.state_dict(),
            "epoch": int(epoch),
            "val_top1": float(val_top1),
            "config": dataclasses.asdict(self.cfg),
        }
        path = self.out_dir / filename
        torch.save(payload, path)
        self.logger.info(f"saved checkpoint -> {path}")

    # ----------------------------------------------------------------- fit loop

    def fit(self) -> dict:
        """Run full training. Returns convergence report. PLAN.md §5.5."""
        self.model.to(self.device)

        # eval-at-epoch-0 to establish random-init baseline (matches upstream).
        self.logger.info("epoch 0: random-init baseline eval")
        baseline = self.evaluate(self.val_loader)
        self._val_top1_trajectory.append(baseline["top1"])
        self.logger.info(f"epoch 0 val: top1={baseline['top1']:.4f} top5={baseline['top5']:.4f}")
        self._best_val_top1 = baseline["top1"]
        self._best_epoch = 0
        self._best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self._save_checkpoint("checkpoint_best.pt", epoch=0, val_top1=baseline["top1"])

        t_start = time.time()
        for epoch in range(1, self.cfg.epochs + 1):
            train_report = self.train_one_epoch(epoch)
            # Val every 5 epochs + final epoch (saves ~80% of val compute).
            if epoch % 5 == 0 or epoch == self.cfg.epochs:
                val_report = self.evaluate(self.val_loader)
                self._val_top1_trajectory.append(val_report["top1"])
                self.logger.info(
                    f"epoch {epoch} val: top1={val_report['top1']:.4f} top5={val_report['top5']:.4f}"
                )

                # Best-checkpoint with overfit-gap guard.
                train_top1 = (
                    train_report.get("train_top1_mv_collection")
                    or train_report.get("train_top1_single", 0.0)
                )
                gap_pp = 100.0 * (train_top1 - val_report["top1"])
                if val_report["top1"] > self._best_val_top1:
                    if gap_pp > self.cfg.overfit_gap_pp_threshold:
                        self.logger.warning(
                            f"epoch {epoch}: train-val gap = {gap_pp:.1f} pp exceeds "
                            f"{self.cfg.overfit_gap_pp_threshold:.1f} pp threshold, still saving"
                        )
                    self._best_val_top1 = val_report["top1"]
                    self._best_epoch = epoch
                    self._best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    self._save_checkpoint("checkpoint_best.pt", epoch=epoch, val_top1=val_report["top1"])

        elapsed = time.time() - t_start

        # Always persist the final-epoch weights as "last" before swapping back to best.
        final_val_top1 = self._val_top1_trajectory[-1] if self._val_top1_trajectory else 0.0
        self._save_checkpoint("checkpoint_last.pt", epoch=self.cfg.epochs, val_top1=final_val_top1)

        # Restore the best weights before returning so caller can run final eval.
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        # ----- convergence classification
        init_loss = self._initial_loss if self._initial_loss is not None else 0.0
        final_loss = self._running_losses[-1] if self._running_losses else 0.0
        loss_reduction_frac = (
            (init_loss - final_loss) / init_loss if init_loss > 0 else 0.0
        )
        final_train_val_gap_pp = 0.0
        if self._train_top1_trajectory and self._val_top1_trajectory:
            final_train_val_gap_pp = 100.0 * (
                self._train_top1_trajectory[-1] - self._val_top1_trajectory[-1]
            )
        # Val cadence is every 5 epochs (Trainer.fit), so the last 3 val
        # points span ~10 training epochs — the field name reflects that.
        val_delta_last3_pp = 0.0
        if len(self._val_top1_trajectory) >= 4:
            recent = self._val_top1_trajectory[-3:]
            val_delta_last3_pp = 100.0 * (max(recent) - min(recent))

        # Honest effective_epochs: total samples seen / training-set size.
        # (Was previously hardcoded to cfg.epochs, making the "≥5 effective
        #  epochs" gate inert.)
        train_set_size = max(1, len(self.train_loader.dataset))
        steps_per_epoch = max(1, len(self.train_loader))
        effective_epochs = (
            self.cfg.epochs * steps_per_epoch * self.cfg.batch_size
            / train_set_size
        )

        conv_report = {
            "status": "UNDERTRAINED",
            "diagnosis": "",
            "epochs_completed": self.cfg.epochs,
            "effective_epochs": float(effective_epochs),
            "initial_loss": init_loss,
            "final_loss": final_loss,
            "loss_reduction_frac": loss_reduction_frac,
            "val_top1_trajectory": self._val_top1_trajectory,
            "val_top1_delta_last_3_val_points_pp": val_delta_last3_pp,
            "final_train_val_gap_pp": final_train_val_gap_pp,
            "max_train_val_gap_pp": final_train_val_gap_pp,
            "best_qualifying_ckpt_epoch": self._best_epoch,
            "best_val_top1": self._best_val_top1,
            "elapsed_train_seconds": elapsed,
        }
        status, diagnosis = classify(conv_report)
        conv_report["status"] = status
        conv_report["diagnosis"] = diagnosis

        with open(self.out_dir / "convergence_report.json", "w") as f:
            json.dump(conv_report, f, indent=2)
        self.logger.info(f"training complete: status={status} ({diagnosis})")
        return conv_report
