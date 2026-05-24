"""
Data loading for OAMV-HFMD.

``HotelsDataset`` is productionized from
``multi-view-hybrid/datasets/hotels8k.py``. Loads splits from
``mvhfmd_data/{train,val,test}.npy`` and per-hotel image directories
``mvhfmd_data/<hotel_id>/<filename>``.

Key behaviors (preserved from upstream — see PLAN.md §5.2):

- Training is image-anchored: ``len(dataset) == n_images``, each sample anchors
  on one image and pulls n-1 others from the same hotel.
- Evaluation is exhaustive: ``len(dataset) == sum_hotels C(M_h, N)``.

PLAN.md §3, §5.2, §5.3.
"""
from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Standard ImageNet stats — matches MV-HFMD's transform
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform():
    """Resize(256) -> RandomCrop(224) -> HFlip -> ToTensor -> Normalize. PLAN.md §5.3."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform():
    """Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize. Deterministic. PLAN.md §5.3."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _extract_hotel_id(im_path: str) -> str:
    # path layout: <data_dir>/<hotel_id>/<image>.<ext>  — normalize separators first
    p = im_path.replace("\\", "/")
    return p.split("/")[-2]


def _generate_target2indices(targets: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for i, t in enumerate(targets):
        out.setdefault(t, []).append(i)
    return out


def _safe_open(path: str) -> Image.Image:
    """Open an image; fall back to neutral 224x224 gray on failure (matches
    upstream robustness — corrupt files in Hotels-50K are common enough that
    crashing the loader is the wrong default)."""
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(128, 128, 128))


class HotelsDataset(Dataset):
    """Productionized port of upstream ``HotelsDataset``.

    Args:
        data_dir: dataset root. Expects ``<data_dir>/{split}.npy`` and per-hotel
            image directories beneath it.
        split: one of ``"train"``, ``"val"``, ``"test"``.
        n: views per collection.
        train: if True, image-anchored sampling. If False, exhaustive
            ``C(M, N)`` enumeration via :meth:`get_all_collection_combos`.
        classes: optional list of class names (hotel IDs) — pass the training
            split's ``classes`` to val/test loaders so class indices align.

    Exposes (matching upstream): ``classes``, ``class_to_idx``, ``num_classes``,
    ``samples``, ``image_paths``, ``targets``, ``targets2indices``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        n: int = 2,
        train: bool = False,
        classes: list[str] | None = None,
    ):
        super().__init__()
        data_dir = str(data_dir)
        self.data_dir = data_dir
        self.split = split
        self.n = n
        self.train = train

        npy_path = os.path.join(data_dir, f"{split}.npy")
        self.all_paths = np.load(npy_path).tolist()
        for i in range(len(self.all_paths)):
            self.all_paths[i] = f"{data_dir}/" + self.all_paths[i]
        self.all_paths = np.array(self.all_paths)

        self.transform = build_train_transform() if split == "train" else build_eval_transform()

        if classes is None:
            self.classes, self.class_to_idx = self._find_classes()
        else:
            self.classes = list(classes)
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.num_classes = len(self.class_to_idx)

        self.samples: list = self._make_dataset()
        self.image_paths: list[str] = [s[0] for s in self.samples]
        self.targets: list[int] = [int(s[1]) for s in self.samples]
        self.targets2indices = _generate_target2indices(self.targets)

        if not self.train:
            self.samples = self.get_all_collection_combos()
            self.image_paths = [s[0] for s in self.samples]
            self.targets = [int(s[1]) for s in self.samples]

    # ------------------------------------------------------------------ helpers

    def _find_classes(self) -> tuple[list[str], dict[str, int]]:
        classes = set()
        for path in self.all_paths:
            classes.add(_extract_hotel_id(path))
        classes_list = sorted(classes)
        return classes_list, {c: i for i, c in enumerate(classes_list)}

    def _make_dataset(self) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []
        for path in self.all_paths:
            hotel_id = _extract_hotel_id(path)
            if hotel_id in self.class_to_idx:
                out.append((path, self.class_to_idx[hotel_id]))
        return out

    def get_all_collection_combos(self) -> list[list]:
        """Enumerate all C(M_h, N) tuples per hotel with M_h >= N."""
        out: list[list] = []
        for t, indices in self.targets2indices.items():
            paths = [self.image_paths[i] for i in indices]
            if len(paths) < self.n:
                continue
            for subset in itertools.combinations(paths, self.n):
                out.append([subset, t])
        return out

    # ------------------------------------------------------------------ pytorch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        target = self.targets[index]
        if self.train:
            possible = self.targets2indices[target]
            if len(possible) <= self.n:
                paths = [self.image_paths[i] for i in possible]
                unique_requirement = False
            else:
                paths = [self.image_paths[index]]
                unique_requirement = True

            while len(paths) < self.n:
                selection = int(np.random.choice(possible))
                path = self.image_paths[selection]
                if path not in paths or not unique_requirement:
                    paths.append(path)
        else:
            paths = list(self.samples[index][0])

        target_tensor = torch.ones((self.n,), dtype=torch.long) * target
        images = torch.stack([self.transform(_safe_open(p)) for p in paths])
        return images, target_tensor, paths
