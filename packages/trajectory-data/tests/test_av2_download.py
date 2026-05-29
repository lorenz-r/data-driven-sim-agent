"""Tests for AV2 download helpers."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from trajectory_data.av2 import download


def _tar_bytes(split: str, n_scenarios: int) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for i in range(n_scenarios):
            scenario_id = f"{i:04d}"
            base = f"{split}/{scenario_id}"
            for name, content in [
                (f"{base}/scenario_{scenario_id}.parquet", b"parquet"),
                (f"{base}/log_map_archive_{scenario_id}.json", b"{}"),
            ]:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def test_download_dev_streams_until_requested_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tar_data = _tar_bytes("train", 3)
    monkeypatch.setattr(download, "_head", lambda url: ("etag-1", len(tar_data)))
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(tar_data),
    )

    etag = download.download_dev(
        "https://example.test/train.tar", 2, tmp_path, "train"
    )

    assert etag == "etag-1"
    scenarios = sorted(p.name for p in (tmp_path / "train").iterdir() if p.is_dir())
    assert scenarios == ["0000", "0001"]


def test_download_dev_raises_when_archive_has_too_few_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tar_data = _tar_bytes("train", 2)
    monkeypatch.setattr(download, "_head", lambda url: ("etag-1", len(tar_data)))
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(tar_data),
    )

    with pytest.raises(ValueError, match="contains only 2 complete scenarios"):
        download.download_dev("https://example.test/train.tar", 3, tmp_path, "train")


def test_split_dev_scenarios_uses_total_budget() -> None:
    assert download.split_dev_scenarios(1000, 0.2) == (800, 200)
    assert download.split_dev_scenarios(5, 0.2) == (4, 1)


def test_download_dev_splits_uses_official_train_and_val_pools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tar_by_url = {
        "https://example.test/train.tar": _tar_bytes("train", 4),
        "https://example.test/val.tar": _tar_bytes("val", 2),
    }
    monkeypatch.setattr(
        download,
        "_head",
        lambda url: (f"etag-{Path(url).stem}", len(tar_by_url[url])),
    )
    monkeypatch.setattr(
        download.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(tar_by_url[url]),
    )

    etag = download.download_dev_splits(
        "https://example.test/train.tar",
        "https://example.test/val.tar",
        5,
        0.2,
        tmp_path,
    )

    assert etag == "dev:train=etag-train:4:val=etag-val:1"
    train_scenarios = sorted(
        p.name for p in (tmp_path / "train").iterdir() if p.is_dir()
    )
    val_scenarios = sorted(p.name for p in (tmp_path / "val").iterdir() if p.is_dir())
    assert train_scenarios == ["0000", "0001", "0002", "0003"]
    assert val_scenarios == ["0000"]
