"""Unit tests for shared trajectory-data split logic.

These tests exercise both the dual-pool and single-pool (--dev) code paths
using only temporary files.  No AV2 data, no GPU, no network access required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trajectory_data import splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pool(directory: Path, n: int) -> list[Path]:
    """Write n dummy .pkl files into directory; return their paths sorted."""
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        p = directory / f"scenario_{i:04d}.pkl"
        p.write_bytes(f"dummy-{i}".encode())
        files.append(p)
    return sorted(files)


def _collected_names(out_dir: Path) -> set[str]:
    return {p.name for p in out_dir.rglob("*.pkl")}


# ---------------------------------------------------------------------------
# Single-pool mode (source-dir == source-val-dir)
# ---------------------------------------------------------------------------


class TestSinglePool:
    def test_no_overlap(self, tmp_path: Path) -> None:
        """Train and val sets must be disjoint when drawn from one pool."""
        pool = tmp_path / "pool"
        _make_pool(pool, 20)

        train_out = tmp_path / "train"
        val_out = tmp_path / "val"

        splits.create_splits(
            source_dir=pool,
            source_val_dir=pool,
            train_out=train_out,
            val_out=val_out,
            train_size=12,
            val_size=5,
            seed=0,
        )

        train_names = _collected_names(train_out)
        val_names = _collected_names(val_out)

        assert len(train_names) == 12
        assert len(val_names) == 5
        assert train_names.isdisjoint(val_names), (
            f"Overlap between train and val: {train_names & val_names}"
        )

    def test_correct_counts(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        _make_pool(pool, 30)

        train_out = tmp_path / "train"
        val_out = tmp_path / "val"

        splits.create_splits(
            source_dir=pool,
            source_val_dir=pool,
            train_out=train_out,
            val_out=val_out,
            train_size=20,
            val_size=8,
            seed=1,
        )

        assert len(_collected_names(train_out)) == 20
        assert len(_collected_names(val_out)) == 8

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        """Two runs with the same seed must produce identical splits."""
        pool = tmp_path / "pool"
        _make_pool(pool, 20)

        def run(out_prefix: str) -> tuple[set[str], set[str]]:
            train_out = tmp_path / f"{out_prefix}_train"
            val_out = tmp_path / f"{out_prefix}_val"
            splits.create_splits(
                source_dir=pool,
                source_val_dir=pool,
                train_out=train_out,
                val_out=val_out,
                train_size=10,
                val_size=5,
                seed=42,
            )
            return _collected_names(train_out), _collected_names(val_out)

        train_a, val_a = run("run_a")
        train_b, val_b = run("run_b")

        assert train_a == train_b, "Train split not deterministic across runs"
        assert val_a == val_b, "Val split not deterministic across runs"

    def test_raises_when_pool_too_small(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        _make_pool(pool, 5)

        with pytest.raises(ValueError, match="Not enough files"):
            splits.create_splits(
                source_dir=pool,
                source_val_dir=pool,
                train_out=tmp_path / "train",
                val_out=tmp_path / "val",
                train_size=4,
                val_size=4,
                seed=0,
            )


# ---------------------------------------------------------------------------
# Dual-pool mode (separate train / val directories)
# ---------------------------------------------------------------------------


class TestDualPool:
    def test_basic_split(self, tmp_path: Path) -> None:
        train_src = tmp_path / "train_src"
        val_src = tmp_path / "val_src"
        _make_pool(train_src, 15)
        _make_pool(val_src, 10)

        train_out = tmp_path / "train_out"
        val_out = tmp_path / "val_out"

        splits.create_splits(
            source_dir=train_src,
            source_val_dir=val_src,
            train_out=train_out,
            val_out=val_out,
            train_size=10,
            val_size=5,
            seed=7,
        )

        assert len(_collected_names(train_out)) == 10
        assert len(_collected_names(val_out)) == 5
