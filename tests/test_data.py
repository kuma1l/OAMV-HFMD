"""Tests for oamv_hfmd.data."""
import pytest


def test_train_transform_no_color_jitter():
    """Match MV-HFMD: train aug uses RandomCrop+HFlip ONLY, no ColorJitter."""
    from oamv_hfmd.data import build_train_transform
    t = build_train_transform()
    names = [type(tt).__name__ for tt in t.transforms]
    assert "RandomHorizontalFlip" in names
    assert "RandomCrop" in names
    assert "ColorJitter" not in names, "MV-HFMD does not use ColorJitter — PLAN.md §5.3"


def test_eval_transform_is_deterministic():
    """Eval transform must be CenterCrop, not RandomCrop."""
    from oamv_hfmd.data import build_eval_transform
    t = build_eval_transform()
    names = [type(tt).__name__ for tt in t.transforms]
    assert "CenterCrop" in names
    assert "RandomCrop" not in names


def test_hotelsdataset_smoke():
    """Wrapper instantiates, __len__ works, __getitem__ returns expected shapes."""
    import os
    from oamv_hfmd.data import HotelsDataset

    data_dir = "D:/Research-WS/PIVOT/mvhfmd_data"
    if not os.path.exists(os.path.join(data_dir, "train.npy")):
        pytest.skip(f"Fallback dataset not found at {data_dir}")

    ds = HotelsDataset(data_dir, split="train", n=4, train=True)
    assert len(ds) > 0
    assert ds.num_classes > 0
    imgs, target, paths = ds[0]
    assert imgs.shape == (4, 3, 224, 224)
    assert target.shape == (4,)
    assert len(paths) == 4

    # eval mode: exhaustive C(M, N) combos.
    ds_val = HotelsDataset(data_dir, split="val", n=4, train=False, classes=ds.classes)
    assert len(ds_val) > 0
    imgs_v, target_v, paths_v = ds_val[0]
    assert imgs_v.shape == (4, 3, 224, 224)
