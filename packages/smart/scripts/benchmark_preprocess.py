#!/usr/bin/env python3#-#
"""#-#
benchmark_preprocess.py#-#
#-#
Download N AV2 scenarios, run both the Python and Rust preprocessing pipelines#-#
on the same input, compare every output field, and report throughput.#-#
#-#
Usage#-#
-----#-#
    # from repo root#-#
    uv run python packages/smart/scripts/benchmark_preprocess.py#-#
    uv run python packages/smart/scripts/benchmark_preprocess.py --scenarios 100#-#
    uv run python packages/smart/scripts/benchmark_preprocess.py --raw-dir /existing/data/raw/train --skip-download#-#
#-#
Requirements#-#
------------#-#
    smart_preprocess must be built before running:#-#
        bash test.sh#-#
        # or: uv run --with maturin maturin develop --release \\#-#
        #         --manifest-path packages/smart-preprocess/Cargo.toml#-#
"""#-#
#-#
from __future__ import annotations#-#
#-#
import argparse#-#
import io#-#
import pickle#-#
import shutil#-#
import sys#-#
import tarfile#-#
import time#-#
import urllib.request#-#
from pathlib import Path#-#
from typing import NamedTuple#-#
#-#
import torch#-#
#-#
# ---------------------------------------------------------------------------#-#
# Bootstrap: refuse to run if smart_preprocess is not installed#-#
# ---------------------------------------------------------------------------#-#
#-#
try:#-#
    import smart_preprocess#-#
except ImportError:#-#
    print(#-#
        "ERROR: smart_preprocess is not installed.\n"#-#
        "Build it first:\n"#-#
        "    bash test.sh\n"#-#
        "  or:\n"#-#
        "    uv run --with maturin maturin develop --release \\\n"#-#
        "        --manifest-path packages/smart-preprocess/Cargo.toml",#-#
        file=sys.stderr,#-#
    )#-#
    sys.exit(1)#-#
#-#
SMART_ROOT = Path(__file__).parent.parent#-#
sys.path.insert(0, str(SMART_ROOT))#-#
import data_preprocess_av2 as py_pipeline  # noqa: E402 – must come after sys.path edit#-#
#-#
# ---------------------------------------------------------------------------#-#
# Constants#-#
# ---------------------------------------------------------------------------#-#
#-#
TRAIN_TAR_URL = (#-#
    "https://s3.amazonaws.com/argoverse/datasets/av2/tars/motion-forecasting/train.tar"#-#
)#-#
CHUNK = 1024 * 1024  # 1 MiB streaming chunk#-#
PER_SCENARIO_BYTES = 500 * 1024  # conservative estimate per scenario in the tar#-#
#-#
FLOAT_ATOL = 2e-3  # looser than unit-test (1e-3); benchmark characterises, not gates#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Download helpers#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
def _stream_partial_tar(url: str, n_scenarios: int, out_dir: Path) -> None:#-#
    """Stream the head of a remote TAR and extract up to n_scenarios folders."""#-#
    byte_budget = n_scenarios * PER_SCENARIO_BYTES#-#
    buf = io.BytesIO()#-#
#-#
    print(f"Streaming up to {byte_budget // 1024} KiB from {url} …")#-#
    req = urllib.request.urlopen(url, timeout=60)#-#
    downloaded = 0#-#
    while downloaded < byte_budget:#-#
        chunk = req.read(min(CHUNK, byte_budget - downloaded))#-#
        if not chunk:#-#
            break#-#
        buf.write(chunk)#-#
        downloaded += len(chunk)#-#
#-#
    buf.seek(0)#-#
    extracted = 0#-#
    try:#-#
        with tarfile.open(fileobj=buf, mode="r|*") as tf:#-#
            for member in tf:#-#
                if extracted // 2 >= n_scenarios:  # each scenario has 2 files#-#
                    break#-#
                if member.isfile():#-#
                    dest = out_dir / Path(member.name).relative_to(#-#
                        Path(member.name).parts[0]#-#
                    )#-#
                    dest.parent.mkdir(parents=True, exist_ok=True)#-#
                    fobj = tf.extractfile(member)#-#
                    if fobj:#-#
                        dest.write_bytes(fobj.read())#-#
                        extracted += 1#-#
    except tarfile.ReadError:#-#
        pass  # expected: we cut the stream short#-#
#-#
    # Prune any incomplete scenario dirs (need both .parquet and .json)#-#
    kept = 0#-#
    for d in sorted(out_dir.iterdir()):#-#
        if not d.is_dir():#-#
            continue#-#
        has_parquet = any(d.glob("scenario_*.parquet"))#-#
        has_json = any(d.glob("log_map_archive_*.json"))#-#
        if has_parquet and has_json:#-#
            kept += 1#-#
            if kept > n_scenarios:#-#
                shutil.rmtree(d)#-#
        else:#-#
            shutil.rmtree(d)#-#
#-#
    print(f"Kept {min(kept, n_scenarios)} complete scenario directories in {out_dir}")#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Field comparison (mirrors test_preprocess_regression._compare)#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
class FieldDiff(NamedTuple):#-#
    key: str#-#
    kind: str  # "exact" | "float" | "shape" | "missing" | "scalar"#-#
    max_diff: float  # 0.0 for exact/missing; abs max for float#-#
    n_differ: int  # element count that differ (0 for float within tol)#-#
#-#
#-#
def _compare_dicts(py: dict, rs: dict) -> list[FieldDiff]:#-#
    diffs: list[FieldDiff] = []#-#
#-#
    def _scalar(k: str, pv, rv) -> None:#-#
        if pv != rv:#-#
            diffs.append(FieldDiff(k, "scalar", 0.0, 1))#-#
#-#
    def _tensors(prefix: str, pd_: dict, rd: dict) -> None:#-#
        for k, pv in pd_.items():#-#
            rv = rd.get(k)#-#
            if not isinstance(pv, torch.Tensor):#-#
                continue#-#
            if rv is None:#-#
                diffs.append(FieldDiff(f"{prefix}.{k}", "missing", 0.0, 1))#-#
                continue#-#
            if rv.shape != pv.shape:#-#
                diffs.append(FieldDiff(f"{prefix}.{k}", "shape", 0.0, 0))#-#
                continue#-#
            if pv.is_floating_point():#-#
                md = (rv.float() - pv.float()).abs().max().item()#-#
                if md > FLOAT_ATOL:#-#
                    diffs.append(FieldDiff(f"{prefix}.{k}", "float", md, 0))#-#
            else:#-#
                nd = int((~torch.eq(rv, pv)).sum().item())#-#
                if nd:#-#
                    diffs.append(FieldDiff(f"{prefix}.{k}", "exact", 0.0, nd))#-#
#-#
    _scalar("scenario_id", py["scenario_id"], rs["scenario_id"])#-#
    _tensors("agent", py["agent"], rs["agent"])#-#
    _tensors("map_polygon", py["map_polygon"], rs["map_polygon"])#-#
    _tensors("map_point", py["map_point"], rs["map_point"])#-#
    for ek in [#-#
        ("map_point", "to", "map_polygon"),#-#
        ("map_polygon", "to", "map_polygon"),#-#
    ]:#-#
        _tensors(str(ek), py[ek], rs[ek])#-#
#-#
    return diffs#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Single-scenario runners (return wall time in seconds)#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
def _run_py(scenario_dir: Path, out_dir: Path) -> float:#-#
    t0 = time.perf_counter()#-#
    py_pipeline.process_scenario(scenario_dir, out_dir)#-#
    return time.perf_counter() - t0#-#
#-#
#-#
def _run_rust(scenario_dir: Path, out_dir: Path) -> float:#-#
    t0 = time.perf_counter()#-#
    smart_preprocess.preprocess_scenario(str(scenario_dir), str(out_dir))#-#
    return time.perf_counter() - t0#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# Main benchmark#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
def run_benchmark(raw_dir: Path, n_scenarios: int, skip_download: bool) -> None:#-#
    raw_dir.mkdir(parents=True, exist_ok=True)#-#
#-#
    # ── 1. Acquire data ──────────────────────────────────────────────────────#-#
    scenario_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())#-#
    if not skip_download and len(scenario_dirs) < n_scenarios:#-#
        _stream_partial_tar(TRAIN_TAR_URL, n_scenarios, raw_dir)#-#
        scenario_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())#-#
#-#
    scenario_dirs = scenario_dirs[:n_scenarios]#-#
    actual = len(scenario_dirs)#-#
    if actual == 0:#-#
        print("ERROR: no scenario directories found. Re-run without --skip-download.")#-#
        sys.exit(1)#-#
    if actual < n_scenarios:#-#
        print(f"WARNING: only {actual} scenarios available (requested {n_scenarios}).")#-#
#-#
    print(f"\nBenchmarking {actual} scenarios …\n")#-#
#-#
    # ── 2. Run both pipelines ────────────────────────────────────────────────#-#
    import tempfile#-#
#-#
    with tempfile.TemporaryDirectory() as tmp:#-#
        py_out = Path(tmp) / "py"#-#
        rs_out = Path(tmp) / "rs"#-#
        py_out.mkdir()#-#
        rs_out.mkdir()#-#
#-#
        py_times: list[float] = []#-#
        rs_times: list[float] = []#-#
        skipped_py: list[str] = []#-#
        skipped_rs: list[str] = []#-#
        compare_failures: dict[str, list[FieldDiff]] = {}#-#
#-#
        for i, sd in enumerate(scenario_dirs, 1):#-#
            sid = sd.name#-#
            py_pkl = py_out / f"{sid}.pkl"#-#
            rs_pkl = rs_out / f"{sid}.pkl"#-#
#-#
            # Python#-#
            try:#-#
                dt = _run_py(sd, py_out)#-#
                py_times.append(dt)#-#
            except Exception:#-#
                skipped_py.append(sid)#-#
                print(f"  [{i}/{actual}] PY  SKIP  {sid}")#-#
                continue#-#
#-#
            # Rust#-#
            try:#-#
                dt = _run_rust(sd, rs_out)#-#
                rs_times.append(dt)#-#
            except Exception:#-#
                skipped_rs.append(sid)#-#
                print(f"  [{i}/{actual}] RUST SKIP  {sid}")#-#
                continue#-#
#-#
            if not py_pkl.exists() or not rs_pkl.exists():#-#
                # scenario was skipped by one pipeline (no predictable tracks, etc.)#-#
                continue#-#
#-#
            # Compare#-#
            with open(py_pkl, "rb") as f:#-#
                py_data = pickle.load(f)#-#
            with open(rs_pkl, "rb") as f:#-#
                rs_data = pickle.load(f)#-#
#-#
            fails = _compare_dicts(py_data, rs_data)#-#
            if fails:#-#
                compare_failures[sid] = fails#-#
#-#
            status = "OK  " if not fails else f"DIFF({len(fails)})"#-#
            py_ms = py_times[-1] * 1000#-#
            rs_ms = rs_times[-1] * 1000#-#
            print(#-#
                f"  [{i:3d}/{actual}] {status}  py={py_ms:6.1f}ms  rs={rs_ms:6.1f}ms  {sid}"#-#
            )#-#
#-#
    # ── 3. Report ────────────────────────────────────────────────────────────#-#
    print("\n" + "=" * 70)#-#
    print("BENCHMARK RESULTS")#-#
    print("=" * 70)#-#
#-#
    n_ok = actual - len(skipped_py) - len(skipped_rs)#-#
#-#
    if py_times:#-#
        py_total = sum(py_times)#-#
        py_mean = py_total / len(py_times) * 1000#-#
        py_tp = len(py_times) / py_total#-#
        print(f"\n  Python pipeline ({len(py_times)} scenarios)")#-#
        print(f"    total : {py_total:.2f}s")#-#
        print(f"    mean  : {py_mean:.1f} ms/scenario")#-#
        print(f"    tp    : {py_tp:.1f} scenarios/s")#-#
#-#
    if rs_times:#-#
        rs_total = sum(rs_times)#-#
        rs_mean = rs_total / len(rs_times) * 1000#-#
        rs_tp = len(rs_times) / rs_total#-#
        print(f"\n  Rust pipeline ({len(rs_times)} scenarios)")#-#
        print(f"    total : {rs_total:.2f}s")#-#
        print(f"    mean  : {rs_mean:.1f} ms/scenario")#-#
        print(f"    tp    : {rs_tp:.1f} scenarios/s")#-#
#-#
    if py_times and rs_times:#-#
        speedup = sum(py_times) / sum(rs_times)#-#
        print(f"\n  Speedup (Python / Rust): {speedup:.2f}x")#-#
#-#
    print(f"\n  Scenarios compared  : {n_ok}")#-#
    print(f"  Field-level matches : {n_ok - len(compare_failures)}/{n_ok}")#-#
    if compare_failures:#-#
        print(f"\n  Scenarios with diffs ({len(compare_failures)}):")#-#
        for sid, fails in compare_failures.items():#-#
            for diff in fails:#-#
                print(#-#
                    f"    {sid}  {diff.key}  [{diff.kind}]  max_diff={diff.max_diff:.2e}  n={diff.n_differ}"#-#
                )#-#
    else:#-#
        print("  All outputs match within tolerance.")#-#
#-#
    if skipped_py:#-#
        print(#-#
            f"\n  Python skipped ({len(skipped_py)}): {', '.join(skipped_py[:5])}"#-#
            + (" …" if len(skipped_py) > 5 else "")#-#
        )#-#
    if skipped_rs:#-#
        print(#-#
            f"\n  Rust skipped   ({len(skipped_rs)}): {', '.join(skipped_rs[:5])}"#-#
            + (" …" if len(skipped_rs) > 5 else "")#-#
        )#-#
#-#
    print("=" * 70)#-#
#-#
    if compare_failures:#-#
        sys.exit(1)#-#
#-#
#-#
# ---------------------------------------------------------------------------#-#
# CLI#-#
# ---------------------------------------------------------------------------#-#
#-#
#-#
def main() -> None:#-#
    parser = argparse.ArgumentParser(#-#
        description="Benchmark Python vs Rust AV2 preprocessing on N scenarios."#-#
    )#-#
    parser.add_argument(#-#
        "--scenarios",#-#
        type=int,#-#
        default=100,#-#
        help="Number of scenarios to benchmark (default: 100)",#-#
    )#-#
    parser.add_argument(#-#
        "--raw-dir",#-#
        type=Path,#-#
        default=Path(__file__).parent.parent / "data" / "benchmark" / "raw",#-#
        help="Directory that already contains or will receive raw AV2 scenario folders",#-#
    )#-#
    parser.add_argument(#-#
        "--skip-download",#-#
        action="store_true",#-#
        help="Skip download step; --raw-dir must already contain scenario folders",#-#
    )#-#
    args = parser.parse_args()#-#
    run_benchmark(args.raw_dir, args.scenarios, args.skip_download)#-#
#-#
#-#
if __name__ == "__main__":#-#
    main()#-#
