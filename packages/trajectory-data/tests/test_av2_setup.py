"""Tests for shared AV2 setup orchestration."""

from __future__ import annotations

from pathlib import Path

from trajectory_data.av2 import setup


def _make_raw_split(directory: Path, split: str, n: int) -> None:
    split_dir = directory / split
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        scenario = split_dir / f"scenario_{i:04d}"
        scenario.mkdir()
        (scenario / f"sample_{i:04d}.pkl").write_bytes(f"{split}-{i}".encode())


def test_run_preprocess_writes_cache_metadata_and_current_symlink(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _make_raw_split(raw_dir, "train", 3)

    calls: list[tuple[str, Path, Path]] = []

    def preprocess_split(input_dir: Path, output_dir: Path, split: str) -> tuple[int, int]:
        calls.append((split, input_dir, output_dir))
        for src in sorted(input_dir.rglob("*.pkl")):
            dest = output_dir / src.relative_to(input_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
        return 3, 3

    cache_dir = setup.run_preprocess(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dataset_etag="etag-1",
        preprocessor_fingerprint="fingerprint-1",
        preprocess_split=preprocess_split,
        splits=("train", "val"),
        use_cache=False,
    )

    assert calls == [("train", raw_dir / "train", cache_dir / "train")]
    assert (cache_dir / ".done.json").exists()
    assert (cache_dir / "train" / "scenario_0000" / "sample_0000.pkl").exists()
    assert (processed_dir / "current").resolve() == cache_dir


def test_run_preprocess_cache_hit_skips_callback(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _make_raw_split(raw_dir, "train", 1)

    def preprocess_split(input_dir: Path, output_dir: Path, split: str) -> None:
        raise AssertionError("cache hit should not preprocess")

    cache_key = setup.preprocess_cache_key("etag-1", "fingerprint-1")
    cache_dir = processed_dir / cache_key
    cache_dir.mkdir(parents=True)
    setup.mark_done(cache_dir, cache_key, "etag-1", "fingerprint-1")

    result = setup.run_preprocess(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dataset_etag="etag-1",
        preprocessor_fingerprint="fingerprint-1",
        preprocess_split=preprocess_split,
        splits=("train",),
        use_cache=True,
    )

    assert result == cache_dir
    assert (processed_dir / "current").resolve() == cache_dir


def test_run_splits_falls_back_to_train_pool_when_val_missing(tmp_path: Path) -> None:
    cache_dir = tmp_path / "processed" / "abc123"
    train_dir = cache_dir / "train"
    train_dir.mkdir(parents=True)
    for i in range(6):
        (train_dir / f"sample_{i}.pkl").write_bytes(str(i).encode())

    train_out = tmp_path / "train_split"
    val_out = tmp_path / "val_split"

    setup.run_splits(
        cache_dir=cache_dir,
        train_size=4,
        val_size=2,
        seed=7,
        train_out=train_out,
        val_out=val_out,
        hash_contents=False,
    )

    train_names = {p.name for p in train_out.glob("*.pkl")}
    val_names = {p.name for p in val_out.glob("*.pkl")}

    assert len(train_names) == 4
    assert len(val_names) == 2
    assert train_names.isdisjoint(val_names)
