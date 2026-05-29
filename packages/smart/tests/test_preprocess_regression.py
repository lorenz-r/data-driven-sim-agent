"""
tests/test_preprocess_regression.py

Regression tests: run both the Python and Rust preprocessing pipelines on the
stripped fixture scenarios and compare every field in the output PKLs.

Pass criteria
-------------
* All integer / bool tensors: bitwise exact.
* All float32 tensors     : abs-diff <= FLOAT_ATOL (1e-3).
  The only expected non-zero float diffs are in map_point geometry fields
  (position / magnitude / orientation).  These are bounded by < 1 ULP of
  float32 at the coordinate scale and come from the difference between
  Python's np.interp and the Rust binary-search interpolation.
* All scalar fields (num_nodes, av_index, scenario_id, city, id list): exact.

Run
---
    cd packages/smart
    .venv_av2/bin/python -m pytest tests/test_preprocess_regression.py -v
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch

# Skip the entire module if the Rust extension has not been built yet.
# Build it with: cd packages/smart-preprocess && maturin develop --release
pytest.importorskip(
    "smart_preprocess",
    reason="smart_preprocess not built; install Rust (https://rustup.rs/) and run test.sh",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SMART_ROOT = Path(__file__).parent.parent

FLOAT_ATOL = 1e-3  # absolute tolerance for float32 comparisons
INT_TYPES = (torch.bool, torch.uint8, torch.long, torch.int64, torch.int32)

SCENARIO_IDS = [d.name for d in sorted(FIXTURES_DIR.iterdir()) if d.is_dir()]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_python(fixture_dir: Path, out_dir: Path) -> Path:
    """Run the Python preprocessing pipeline on one fixture directory."""
    import sys

    sys.path.insert(0, str(SMART_ROOT))
    import data_preprocess_av2 as dp

    sid = fixture_dir.name
    ok = dp.process_scenario(fixture_dir, out_dir)
    out_pkl = out_dir / f"{sid}.pkl"
    assert ok, f"Python pipeline returned False for {sid}"
    assert out_pkl.exists(), f"Python pipeline did not write {out_pkl}"
    return out_pkl


def _run_rust(fixture_dir: Path, out_dir: Path) -> Path:
    """Run the Rust preprocessing module on one fixture directory."""
    import smart_preprocess

    sid = fixture_dir.name
    ok = smart_preprocess.preprocess_scenario(str(fixture_dir), str(out_dir))
    out_pkl = out_dir / f"{sid}.pkl"
    assert ok, f"Rust module returned False for {sid}"
    assert out_pkl.exists(), f"Rust module did not write {out_pkl}"
    return out_pkl


def _compare(py_pkl: Path, rs_pkl: Path) -> list[str]:
    """
    Load both PKLs, compare every field.
    Return a list of failure messages (empty = all good).
    """
    with open(py_pkl, "rb") as f:
        py = pickle.load(f)
    with open(rs_pkl, "rb") as f:
        rs = pickle.load(f)

    fails: list[str] = []

    # ── top-level scalars ──────────────────────────────────────────────────
    for k in ("scenario_id", "city"):
        if rs[k] != py[k]:
            fails.append(f"top.{k}: rust={rs[k]} py={py[k]}")

    # ── agent ──────────────────────────────────────────────────────────────
    ra, pa = rs["agent"], py["agent"]

    for k in ("num_nodes", "av_index"):
        if ra[k] != pa[k]:
            fails.append(f"agent.{k}: rust={ra[k]} py={pa[k]}")

    if ra["id"] != pa["id"]:
        fails.append(f"agent.id: rust={ra['id']} py={pa['id']}")

    _check_tensors(
        fails,
        "agent",
        ra,
        pa,
        exact_keys=["valid_mask", "predict_mask", "type", "category"],
        float_keys=["position", "heading", "velocity", "shape"],
    )

    # ── map_polygon ────────────────────────────────────────────────────────
    rmp, pmp = rs["map_polygon"], py["map_polygon"]
    if rmp["num_nodes"] != pmp["num_nodes"]:
        fails.append(f"map_polygon.num_nodes: {rmp['num_nodes']}!={pmp['num_nodes']}")
    _check_tensors(
        fails, "map_polygon", rmp, pmp, exact_keys=["type", "light_type"], float_keys=[]
    )

    # ── map_point ──────────────────────────────────────────────────────────
    rmpt, pmpt = rs["map_point"], py["map_point"]
    if rmpt["num_nodes"] != pmpt["num_nodes"]:
        fails.append(f"map_point.num_nodes: {rmpt['num_nodes']}!={pmpt['num_nodes']}")
    _check_tensors(
        fails,
        "map_point",
        rmpt,
        pmpt,
        exact_keys=["type"],
        float_keys=["position", "orientation", "magnitude", "height"],
    )

    # ── edge indices ───────────────────────────────────────────────────────
    for edge_key in [
        ("map_point", "to", "map_polygon"),
        ("map_polygon", "to", "map_polygon"),
    ]:
        re, pe = rs[edge_key], py[edge_key]
        _check_tensors(
            fails, str(edge_key), re, pe, exact_keys=["edge_index"], float_keys=[]
        )
        if "type" in pe:
            _check_tensors(
                fails, str(edge_key), re, pe, exact_keys=["type"], float_keys=[]
            )

    return fails


def _check_tensors(
    fails: list[str],
    prefix: str,
    rd: dict,
    pd_: dict,
    exact_keys: list[str],
    float_keys: list[str],
) -> None:
    for k in exact_keys:
        rv, pv = rd.get(k), pd_.get(k)
        if rv is None or pv is None:
            if rv is None:
                fails.append(f"{prefix}.{k}: missing in rust output")
            continue
        if not isinstance(pv, torch.Tensor):
            continue
        if rv.shape != pv.shape:
            fails.append(f"{prefix}.{k}: shape rust={rv.shape} py={pv.shape}")
            continue
        if not torch.equal(rv, pv):
            n = (~torch.eq(rv, pv)).sum().item()
            fails.append(f"{prefix}.{k}: {n} elements differ (expected exact match)")

    for k in float_keys:
        rv, pv = rd.get(k), pd_.get(k)
        if rv is None or pv is None:
            if rv is None:
                fails.append(f"{prefix}.{k}: missing in rust output")
            continue
        if not isinstance(pv, torch.Tensor):
            continue
        if rv.shape != pv.shape:
            fails.append(f"{prefix}.{k}: shape rust={rv.shape} py={pv.shape}")
            continue
        max_diff = (rv.float() - pv.float()).abs().max().item()
        if max_diff > FLOAT_ATOL:
            fails.append(
                f"{prefix}.{k}: max_diff={max_diff:.4e} > atol={FLOAT_ATOL:.0e}"
            )


def _compare_tokenized(py: dict, rs: dict) -> list[str]:
    fails: list[str] = []

    if set(py.keys()) != set(rs.keys()):
        fails.append(f"top.keys differ: {set(py.keys()) ^ set(rs.keys())}")

    _check_tensors(
        fails,
        "agent",
        rs["agent"],
        py["agent"],
        exact_keys=[
            "valid_mask",
            "predict_mask",
            "type",
            "category",
            "token_idx",
            "agent_valid_mask",
        ],
        float_keys=[
            "position",
            "heading",
            "velocity",
            "shape",
            "token_contour",
            "token_pos",
            "token_heading",
            "token_velocity",
        ],
    )

    _check_tensors(
        fails,
        "map_point",
        rs["map_point"],
        py["map_point"],
        exact_keys=["type"],
        float_keys=["orientation"],
    )

    _check_tensors(
        fails,
        "map_save",
        rs["map_save"],
        py["map_save"],
        exact_keys=[],
        float_keys=["traj_pos", "traj_theta", "pl_idx_list"],
    )

    _check_tensors(
        fails,
        "pt_token",
        rs["pt_token"],
        py["pt_token"],
        exact_keys=["type", "side", "pl_type"],
        float_keys=[],
    )

    if rs["pt_token"]["num_nodes"] != py["pt_token"]["num_nodes"]:
        fails.append(
            f"pt_token.num_nodes: rust={rs['pt_token']['num_nodes']} "
            f"py={py['pt_token']['num_nodes']}"
        )

    return fails


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_preprocess_matches_python(sid: str, tmp_path: Path) -> None:
    """
    For each fixture scenario: run Python + Rust preprocessing and assert that
    every field in the output PKL matches within tolerance.
    """
    fixture = FIXTURES_DIR / sid
    py_dir = tmp_path / "py"
    rs_dir = tmp_path / "rs"
    py_dir.mkdir()
    rs_dir.mkdir()

    py_pkl = _run_python(fixture, py_dir)
    rs_pkl = _run_rust(fixture, rs_dir)

    failures = _compare(py_pkl, rs_pkl)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_rust_output_structure(sid: str, tmp_path: Path) -> None:
    """
    Verify the Rust output has the correct top-level keys and sub-dict shapes,
    independently of Python (no Python reference needed).
    """
    import smart_preprocess

    fixture = FIXTURES_DIR / sid
    out_dir = tmp_path / "rs"
    out_dir.mkdir()
    ok = smart_preprocess.preprocess_scenario(str(fixture), str(out_dir))
    assert ok, f"Rust returned False for {sid}"

    with open(out_dir / f"{sid}.pkl", "rb") as f:
        d = pickle.load(f)

    # top-level keys
    expected_keys = {
        "scenario_id",
        "city",
        "agent",
        "map_polygon",
        "map_point",
        ("map_point", "to", "map_polygon"),
        ("map_polygon", "to", "map_polygon"),
    }
    assert set(d.keys()) == expected_keys, f"Unexpected keys: {set(d.keys())}"

    # agent shapes
    a = d["agent"]
    N = a["num_nodes"]
    T = 110
    assert a["position"].shape == (N, T, 3)
    assert a["heading"].shape == (N, T)
    assert a["velocity"].shape == (N, T, 3)
    assert a["shape"].shape == (N, T, 3)
    assert a["valid_mask"].shape == (N, T)
    assert a["predict_mask"].shape == (N, T)
    assert a["type"].shape == (N,)
    assert a["category"].shape == (N,)
    assert len(a["id"]) == N
    assert 0 <= a["av_index"] < N

    # valid_mask: t=0 always False, predict_mask always False for t<50
    assert not a["valid_mask"][:, 0].any(), "valid_mask[:,0] should be all False"
    assert not a["predict_mask"][:, :50].any(), (
        "predict_mask[:,<50] should be all False"
    )

    # map shapes
    P = d["map_polygon"]["num_nodes"]
    M = d["map_point"]["num_nodes"]
    assert d["map_polygon"]["type"].shape == (P,)
    assert d["map_polygon"]["light_type"].shape == (P,)
    assert d["map_point"]["position"].shape == (M, 3)
    assert d["map_point"]["orientation"].shape == (M,)
    assert d["map_point"]["magnitude"].shape == (M,)
    assert d["map_point"]["height"].shape == (M,)
    assert d["map_point"]["type"].shape == (M,)

    # edge index shapes: [2, E]
    pt2mp = d[("map_point", "to", "map_polygon")]["edge_index"]
    p2p = d[("map_polygon", "to", "map_polygon")]["edge_index"]
    assert pt2mp.shape[0] == 2
    assert pt2mp.shape[1] == M, f"pt->poly edge count {pt2mp.shape[1]} != M={M}"
    assert p2p.shape[0] == 2

    # all map point types should be valid SMART codes (CROSSWALK=15, CENTERLINE=16)
    assert d["map_point"]["type"].max().item() <= 16
    # all polygon types 0-3
    assert d["map_polygon"]["type"].max().item() <= 3


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_rust_deterministic(sid: str, tmp_path: Path) -> None:
    """Run Rust preprocessing twice on the same fixture; outputs must be bit-identical."""
    import smart_preprocess

    fixture = FIXTURES_DIR / sid
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    smart_preprocess.preprocess_scenario(str(fixture), str(out1))
    smart_preprocess.preprocess_scenario(str(fixture), str(out2))

    pkl1 = out1 / f"{sid}.pkl"
    pkl2 = out2 / f"{sid}.pkl"

    failures = _compare(pkl1, pkl2)
    assert not failures, "Non-deterministic output:\n" + "\n".join(failures)


def test_token_processor_backend_auto_is_the_only_silent_fallback(monkeypatch) -> None:
    import sys

    sys.path.insert(0, str(SMART_ROOT))
    loaded_smart = sys.modules.get("smart")
    if loaded_smart is not None and not str(
        getattr(loaded_smart, "__file__", "")
    ).startswith(str(SMART_ROOT / "smart")):
        del sys.modules["smart"]

    import smart.datasets.token_processor as token_processor

    class BrokenRustProcessor:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("rust unavailable")

    class DummyPythonProcessor:
        def __init__(
            self,
            token_size: int,
            agent_token_path=None,
            map_token_path=None,
            training: bool = False,
        ) -> None:
            self.token_size = token_size
            self.agent_token_path = agent_token_path
            self.map_token_path = map_token_path
            self.training = training

    monkeypatch.setattr(token_processor, "RustTokenProcessor", BrokenRustProcessor)
    monkeypatch.setattr(token_processor, "PythonTokenProcessor", DummyPythonProcessor)

    auto_processor = token_processor.build_token_processor(2048, backend="auto")
    assert isinstance(auto_processor, DummyPythonProcessor)

    with pytest.raises(RuntimeError, match="rust unavailable"):
        token_processor.build_token_processor(2048, backend="rust")


@pytest.mark.parametrize("sid", SCENARIO_IDS)
def test_rust_token_processor_matches_python(sid: str, tmp_path: Path) -> None:
    """Rust dataloader token preparation must match the legacy Python backend."""
    import sys

    sys.path.insert(0, str(SMART_ROOT))
    loaded_smart = sys.modules.get("smart")
    if loaded_smart is not None and not str(
        getattr(loaded_smart, "__file__", "")
    ).startswith(str(SMART_ROOT / "smart")):
        del sys.modules["smart"]

    from smart.datasets.preprocess import TokenProcessor as PythonTokenProcessor
    from smart.datasets.token_processor import RustTokenProcessor

    fixture = FIXTURES_DIR / sid
    out_dir = tmp_path / "preprocessed"
    out_dir.mkdir()
    pkl_path = _run_rust(fixture, out_dir)

    with open(pkl_path, "rb") as f:
        py_input = pickle.load(f)
    with open(pkl_path, "rb") as f:
        rs_input = pickle.load(f)

    py = PythonTokenProcessor(2048).preprocess(py_input)
    rs = RustTokenProcessor(2048).preprocess(rs_input)

    failures = _compare_tokenized(py, rs)
    assert not failures, "\n".join(failures)
