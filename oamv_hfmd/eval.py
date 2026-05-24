"""
Exhaustive evaluation and CCCE guard for OAMV-HFMD.

``Evaluator`` is productionized from ``multi-view-hybrid/engine/evaluator.py``.
``ccce_guard`` is the post-training perturbation check carried over from
``spike_hotels50k_v2/eval.py`` and adapted to MultiImageHybrid's dict output.

Combo enumeration lives in ``HotelsDataset.__init__(train=False)`` — the
Evaluator just iterates the DataLoader once.

PLAN.md §3.4, §17.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _top_k_acc(predictions: torch.Tensor, gt_labels: torch.Tensor, k: int) -> float:
    """predictions: (M, K) sorted indices descending. gt_labels: (M, 1)."""
    if predictions.shape[1] < k:
        k = predictions.shape[1]
    accuracy_per_sample = torch.any(predictions[:, :k] == gt_labels, dim=1).float()
    return float(torch.mean(accuracy_per_sample).item())


class Evaluator:
    """Exhaustive evaluator producing nested ``{single, mv_collection}`` metrics.

    Single-view paths are deduplicated (upstream behavior) so each test image
    contributes once to single-view top-k, regardless of how many C(M, N) combos
    contain it.
    """

    def __init__(self, model: nn.Module, n: int = 4, device: str = "cuda"):
        self.model = model
        self.n = n
        self.extract_device = device
        self.eval_device = device
        self.num_classes = model.num_classes

    @torch.no_grad()
    def extract(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        self.model.to(self.extract_device)

        num_collections = len(dataloader.dataset)
        num_total_images = num_collections * self.n

        results: dict = {
            "single": {
                "logits": np.zeros((num_total_images, self.num_classes), dtype=np.float32),
                "classes": np.zeros(num_total_images, dtype=np.int64),
                "paths": [],
            },
            "mv_collection": {
                "logits": np.zeros((num_collections, self.num_classes), dtype=np.float32),
                "classes": np.zeros(num_collections, dtype=np.int64),
                "paths": [],
            },
        }
        if self.n == 1:
            del results["mv_collection"]

        s = 0
        for batch in dataloader:
            images, targets, paths = batch
            B = images.shape[0]
            images = images.to(self.extract_device, non_blocking=True)
            output = self.model(images)
            e = s + B

            for view_type in ("single", "mv_collection"):
                if view_type not in output or view_type not in results:
                    continue
                logits = _to_numpy(output[view_type]["logits"].float())
                if view_type == "single":
                    multiplier = self.n
                    t = targets.view(B * self.n)
                    # paths: list of len N, each a tuple/list of len B
                    p_list = np.array(paths).T.flatten().tolist()
                else:
                    multiplier = 1
                    t = targets[:, 0]
                    # group per-collection: list of B sub-lists of len N
                    p_list = []
                    for bi in range(B):
                        p_list.append([paths[ni][bi] for ni in range(self.n)])

                results[view_type]["logits"][s * multiplier: e * multiplier] = logits
                results[view_type]["classes"][s * multiplier: e * multiplier] = _to_numpy(t)
                results[view_type]["paths"].extend(p_list)

            s = e

        # ---- single-view dedup
        if "single" in results:
            paths_arr = np.array(results["single"]["paths"])
            seen: set = set()
            keep: list[int] = []
            for i, p in enumerate(paths_arr):
                if p not in seen:
                    seen.add(p)
                    keep.append(i)
            for key in ("logits", "classes"):
                results["single"][key] = results["single"][key][keep]
            results["single"]["paths"] = paths_arr[keep]

        # Convert logits to torch tensors for the metrics pass.
        for view_type in results:
            results[view_type]["logits"] = torch.from_numpy(results[view_type]["logits"]).squeeze()
            results[view_type]["classes"] = results[view_type]["classes"].squeeze()
        return results

    def get_metrics(self, logits: torch.Tensor, gt_labels: torch.Tensor) -> dict:
        try:
            predictions = torch.argsort(logits, dim=1, descending=True)
        except torch.cuda.OutOfMemoryError:
            logits = logits.cpu()
            gt_labels = gt_labels.cpu()
            predictions = torch.argsort(logits, dim=1, descending=True)

        return {
            "top1_acc":   _top_k_acc(predictions, gt_labels, 1),
            "top2_acc":   _top_k_acc(predictions, gt_labels, 2),
            "top5_acc":   _top_k_acc(predictions, gt_labels, 5),
            "top10_acc":  _top_k_acc(predictions, gt_labels, 10),
            "top100_acc": _top_k_acc(predictions, gt_labels, 100),
        }

    def evaluate(self, dataloader: DataLoader) -> dict:
        results = self.extract(dataloader)
        out: dict = {}
        for view_type in results:
            gt = torch.from_numpy(np.expand_dims(results[view_type]["classes"], -1)).to(self.eval_device)
            logits = results[view_type]["logits"].to(self.eval_device)
            out[view_type] = self.get_metrics(logits, gt)
        return out


@torch.no_grad()
def ccce_guard(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cuda",
    n: int = 4,
    n_hotels: int = 256,
    chunk_size: int = 8,
) -> dict:
    """CCCE perturbation guard. PLAN.md §17, HANDOFF.md §1.3.

    Forwards a fixed batch of up to ``n_hotels`` test collections, then for
    each input slot k ∈ [0, n) re-forwards with slot k zeroed and measures the
    top-1 drop vs reference. Returns ``{ref_top1, slot_k_drop_pp..., pass}``.
    ``pass`` is True iff every ``slot_k_drop_pp >= 0.30``.

    Batched in chunks of ``chunk_size`` collections to fit 8 GB VRAM.
    """
    model.eval()
    model.to(device)

    collected_imgs: list[torch.Tensor] = []
    collected_labels: list[torch.Tensor] = []
    for batch in test_loader:
        images, targets, _ = batch
        if images.shape[1] != n:
            continue
        collected_imgs.append(images)
        collected_labels.append(targets[:, 0])
        if sum(b.shape[0] for b in collected_imgs) >= n_hotels:
            break
    if not collected_imgs:
        return {"pass": False, "error": "no eligible batches"}

    imgs = torch.cat(collected_imgs, dim=0)[:n_hotels]   # (H, N, 3, 224, 224)
    labels = torch.cat(collected_labels, dim=0)[:n_hotels].to(device)
    H = imgs.shape[0]

    def _forward_chunked(x: torch.Tensor) -> torch.Tensor:
        preds = []
        for i in range(0, x.shape[0], chunk_size):
            chunk = x[i:i + chunk_size].to(device, non_blocking=True)
            out = model(chunk)
            logits = out["mv_collection"]["logits"] if "mv_collection" in out else out["single"]["logits"]
            preds.append(logits.argmax(-1))
        return torch.cat(preds, dim=0)

    ref_pred = _forward_chunked(imgs)
    ref_top1 = float((ref_pred == labels).float().mean().item())

    result: dict = {"ref_top1": round(ref_top1, 6), "n_eligible_hotels": int(H)}
    drops: list[float] = []
    for k in range(n):
        perturbed = imgs.clone()
        perturbed[:, k] = 0.0
        p_pred = _forward_chunked(perturbed)
        p_top1 = float((p_pred == labels).float().mean().item())
        drop_pp = (ref_top1 - p_top1) * 100.0
        result[f"slot_{k}_drop_pp"] = round(drop_pp, 4)
        drops.append(drop_pp)

    min_drop = min(drops)
    result["min_slot_drop_pp"] = round(min_drop, 4)
    result["pass"] = bool(min_drop >= 0.30)
    return result
