"""Unit tests for create_datasets.py split logic.

These tests exercise both the dual-pool and single-pool (--dev) code paths
using only temporary files.  No AV2 data, no GPU, no network access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make create_datasets importable without installing the package.
_SMART_ROOT = Path(__file__).resolve().parents[1]
if str(_SMART_ROOT) not in sys.path:
    sys.path.insert(0, str(_SMART_ROOT))

import create_datasets as cd  # noqa: E402


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

        sys_argv_orig = sys.argv
        try:
            sys.argv = [
                "create_datasets.py",
                "--source-dir",
                str(pool),
                "--source-val-dir",
                str(pool),
                "--train-out",
                str(train_out),
                "--val-out",
                str(val_out),
                "--train-size",
                "12",
                "--val-size",
                "5",
                "--seed",
                "0",
            ]
            cd.main()
        finally:
            sys.argv = sys_argv_orig

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

        sys_argv_orig = sys.argv
        try:
            sys.argv = [
                "create_datasets.py",
                "--source-dir",
                str(pool),
                "--source-val-dir",
                str(pool),
                "--train-out",
                str(train_out),
                "--val-out",
                str(val_out),
                "--train-size",
                "20",
                "--val-size",
                "8",
                "--seed",
                "1",
            ]
            cd.main()
        finally:
            sys.argv = sys_argv_orig

        assert len(_collected_names(train_out)) == 20
        assert len(_collected_names(val_out)) == 8

    def test_deterministic_with_same_seed(self, tmp_path: Path) -> None:
        """Two runs with the same seed must produce identical splits."""
        pool = tmp_path / "pool"
        _make_pool(pool, 20)

        def run(out_prefix: str) -> tuple[set[str], set[str]]:
            train_out = tmp_path / f"{out_prefix}_train"
            val_out = tmp_path / f"{out_prefix}_val"
            sys_argv_orig = sys.argv
            try:
                sys.argv = [
                    "create_datasets.py",
                    "--source-dir",
                    str(pool),
                    "--source-val-dir",
                    str(pool),
                    "--train-out",
                    str(train_out),
                    "--val-out",
                    str(val_out),
                    "--train-size",
                    "10",
                    "--val-size",
                    "5",
                    "--seed",
                    "42",
                ]
                cd.main()
            finally:
                sys.argv = sys_argv_orig
            return _collected_names(train_out), _collected_names(val_out)

        train_a, val_a = run("run_a")
        train_b, val_b = run("run_b")

        assert train_a == train_b, "Train split not deterministic across runs"
        assert val_a == val_b, "Val split not deterministic across runs"

    def test_raises_when_pool_too_small(self, tmp_path: Path) -> None:
        pool = tmp_path / "pool"
        _make_pool(pool, 5)

        sys_argv_orig = sys.argv
        try:
            sys.argv = [
                "create_datasets.py",
                "--source-dir",
                str(pool),
                "--source-val-dir",
                str(pool),
                "--train-out",
                str(tmp_path / "train"),
                "--val-out",
                str(tmp_path / "val"),
                "--train-size",
                "4",
                "--val-size",
                "4",
                "--seed",
                "0",
            ]
            with pytest.raises(ValueError, match="Not enough files"):
                cd.main()
        finally:
            sys.argv = sys_argv_orig


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

        sys_argv_orig = sys.argv
        try:
            sys.argv = [
                "create_datasets.py",
                "--source-dir",
                str(train_src),
                "--source-val-dir",
                str(val_src),
                "--train-out",
                str(train_out),
                "--val-out",
                str(val_out),
                "--train-size",
                "10",
                "--val-size",
                "5",
                "--seed",
                "7",
            ]
            cd.main()
        finally:
            sys.argv = sys_argv_orig

        assert len(_collected_names(train_out)) == 10
        assert len(_collected_names(val_out)) == 5
