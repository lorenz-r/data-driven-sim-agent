#!/usr/bin/env python3
"""
scripts/make_test_fixtures.py

Strips three AV2 scenarios down to ~50 KB parquets for use in regression tests.

Strategy per scenario
---------------------
* Parquet : keep ALL rows for a chosen subset of tracks.
  Track selection order:
    1. FOCAL_TRACK / SCORED_TRACK agents   (required: has_predictable must pass)
    2. AV track                            (tests av_index detection)
    3. UNSCORED_TRACK vehicles at t=49     (tests av fallback heuristic)
    4. Any remaining tracks at t=49        (fills to TARGET_TRACKS)
  Only tracks observed at t=49 are eligible (mirrors Python's include filter).
  We write the stripped rows with the original column set untouched.

* Map JSON : keep KEEP_LANES random lanes (seeded) + ALL pedestrian crossings.
  Predecessor / successor IDs inside the kept set are preserved; references to
  removed lanes are dropped (left as empty lists).  This keeps lane-connectivity
  edges non-trivial without inflating file size.

Output layout
-------------
  tests/fixtures/<sid>/
      scenario_<sid>.parquet   (~50 KB)
      log_map_archive_<sid>.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.parent  # packages/smart
RAW_TRAIN = SCRIPT_DIR / "data" / "raw" / "train"
OUT_ROOT = SCRIPT_DIR / "tests" / "fixtures"

SCENARIOS = [
    "26a0cce7-8388-4aa0-bf30-a3a6dc09c591",  # large  (42 agents, 33 lanes)
    "1b0c5826-1614-4996-8a12-f6df12beaa0e",  # medium (33 agents, 46 lanes)
    "1f457f41-ac72-4376-a3f1-6d0ed7db46fb",  # small  (10 agents, 35 lanes)
]

TARGET_TRACKS = 8  # keep at most this many tracks
KEEP_LANES = 20  # keep this many lane segments
CURRENT_TIME = 49
RNG_SEED = 42

# ---------------------------------------------------------------------------


def pick_tracks(df: pd.DataFrame, target: int) -> list[str]:
    """Return up to `target` track_ids in priority order."""
    kept: list[str] = []

    def add(tids):
        for t in tids:
            if t not in kept:
                kept.append(t)

    at49 = set(df[(df["timestep"] == CURRENT_TIME) & (df["observed"])]["track_id"])

    # 1. focal / scored
    focal = df[df["object_category"].isin([2, 3])]["track_id"].unique()
    add([t for t in focal if t in at49])

    # 2. AV
    add([t for t in ["AV"] if t in at49])

    # 3. unscored vehicle with most history (av fallback candidate)
    cands = (
        df[(df["object_category"] == 1) & (df["object_type"] == "vehicle")]
        .groupby("track_id")
        .apply(lambda g: ((g["timestep"] <= CURRENT_TIME) & g["observed"]).sum())
        .sort_values(ascending=False)
    )
    add([t for t in cands.index if t in at49])

    # 4. anything else at t=49
    add(list(at49))

    return kept[:target]


def strip_parquet(src: Path, out_dir: Path, tracks: list[str]) -> Path:
    df = pd.read_parquet(src)
    df = df[df["track_id"].isin(tracks)].copy()
    out = out_dir / src.name
    # write with snappy compression to keep size down
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out, compression="snappy")
    return out


def strip_map(src: Path, out_dir: Path, keep_lanes: int, seed: int) -> Path:
    with open(src) as f:
        m = json.load(f)

    rng = random.Random(seed)
    all_ls = list(m["lane_segments"].keys())

    # always keep lanes referenced by focal scenario track's connected lanes
    # (no direct link in JSON, so just seed-pick a reproducible subset)
    chosen = rng.sample(all_ls, min(keep_lanes, len(all_ls)))
    chosen_set = set(chosen)

    stripped_ls: dict = {}
    for key in chosen:
        ls = dict(m["lane_segments"][key])
        # keep pred/succ only if they're in the kept set
        ls["predecessors"] = [
            p for p in ls.get("predecessors", []) if str(p) in chosen_set
        ]
        ls["successors"] = [s for s in ls.get("successors", []) if str(s) in chosen_set]
        stripped_ls[key] = ls

    stripped = {
        "drivable_areas": {},
        "lane_segments": stripped_ls,
        "pedestrian_crossings": m["pedestrian_crossings"],
    }

    out = out_dir / src.name
    with open(out, "w") as f:
        json.dump(stripped, f, separators=(",", ":"))  # compact JSON
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for sid in SCENARIOS:
        raw_dir = RAW_TRAIN / sid
        pq_src = next(raw_dir.glob("scenario_*.parquet"))
        map_src = next(raw_dir.glob("log_map_archive_*.json"))

        out_dir = OUT_ROOT / sid
        out_dir.mkdir(exist_ok=True)

        df = pd.read_parquet(pq_src)
        tracks = pick_tracks(df, TARGET_TRACKS)

        pq_out = strip_parquet(pq_src, out_dir, tracks)
        map_out = strip_map(map_src, out_dir, KEEP_LANES, RNG_SEED)

        pq_kb = pq_out.stat().st_size // 1024
        map_kb = map_out.stat().st_size // 1024
        rows = len(pd.read_parquet(pq_out))
        print(
            f"{sid}  tracks={len(tracks)}  rows={rows}  "
            f"parquet={pq_kb}KB  map={map_kb}KB"
        )

    print(f"\nFixtures written to {OUT_ROOT}")


if __name__ == "__main__":
    main()
