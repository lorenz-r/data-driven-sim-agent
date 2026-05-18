//! lib.rs - PyO3 module: SMART AV2 preprocessing in Rust
//!
//! Exposed functions:
//!   preprocess_scenario(scenario_dir: str, output_dir: str) -> bool
//!   process_directory(input_dir: str, output_dir: str) -> (int, int)

mod agent;
mod map;

use agent::build_agent_data;
use map::build_map_data;

use anyhow::{Context, Result};
use numpy::{PyArray1, PyArray2, PyArray3};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};
use std::path::{Path, PathBuf};

// ─── helpers: numpy → torch tensor ──────────────────────────────────────────

/// Call `torch.from_numpy(arr).clone()` — clone detaches from numpy buffer so
/// the Python object does not borrow Rust memory after this function returns.
macro_rules! as_tensor {
    ($py:expr, $torch:expr, $arr:expr) => {{
        let t = $torch.call_method1("from_numpy", ($arr,))?;
        t.call_method0("clone")?
    }};
}

fn make_f32_1d<'py>(py: Python<'py>, data: Vec<f32>) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let arr = PyArray1::from_vec(py, data);
    Ok(as_tensor!(py, torch, arr))
}

fn make_f32_2d<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let arr = PyArray2::from_vec2(py, &reshape_f32_2d(data, rows, cols))?;
    Ok(as_tensor!(py, torch, arr))
}

fn make_f32_3d<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    d0: usize,
    d1: usize,
    d2: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let arr = PyArray3::from_vec3(py, &reshape_f32_3d(data, d0, d1, d2))?;
    Ok(as_tensor!(py, torch, arr))
}

fn make_bool_2d<'py>(
    py: Python<'py>,
    data: Vec<bool>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let u8_data: Vec<u8> = data.into_iter().map(|b| b as u8).collect();
    let arr = PyArray2::from_vec2(py, &reshape_u8_2d(u8_data, rows, cols))?;
    // numpy uint8 → torch bool
    let t = as_tensor!(py, torch, arr);
    t.call_method1("to", (torch.getattr("bool")?,))
}

fn make_uint8_1d<'py>(py: Python<'py>, data: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let arr = PyArray1::from_vec(py, data);
    Ok(as_tensor!(py, torch, arr))
}

fn make_int64_2d<'py>(
    py: Python<'py>,
    data: Vec<i64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let torch = py.import("torch")?;
    let arr = PyArray2::from_vec2(py, &reshape_i64_2d(data, rows, cols))?;
    Ok(as_tensor!(py, torch, arr))
}

// ─── reshape helpers (zero-copy rearrangement into nested Vecs for numpy) ────

fn reshape_f32_2d(data: Vec<f32>, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    assert_eq!(data.len(), rows * cols);
    if cols == 0 {
        return vec![Vec::new(); rows];
    }
    data.chunks(cols).map(|c| c.to_vec()).collect()
}

fn reshape_f32_3d(data: Vec<f32>, d0: usize, d1: usize, d2: usize) -> Vec<Vec<Vec<f32>>> {
    assert_eq!(data.len(), d0 * d1 * d2);
    (0..d0)
        .map(|i| {
            (0..d1)
                .map(|j| {
                    let off = (i * d1 + j) * d2;
                    data[off..off + d2].to_vec()
                })
                .collect()
        })
        .collect()
}

fn reshape_u8_2d(data: Vec<u8>, rows: usize, cols: usize) -> Vec<Vec<u8>> {
    assert_eq!(data.len(), rows * cols);
    if cols == 0 {
        return vec![Vec::new(); rows];
    }
    data.chunks(cols).map(|c| c.to_vec()).collect()
}

fn reshape_i64_2d(data: Vec<i64>, rows: usize, cols: usize) -> Vec<Vec<i64>> {
    assert_eq!(data.len(), rows * cols);
    if cols == 0 {
        return vec![Vec::new(); rows];
    }
    data.chunks(cols).map(|c| c.to_vec()).collect()
}

// ─── build the full Python dict for one scenario ─────────────────────────────

fn build_python_dict<'py>(
    py: Python<'py>,
    scenario_id: &str,
    agent: agent::AgentData,
    map: map::MapData,
) -> PyResult<Bound<'py, PyDict>> {
    let n = agent.num_nodes;
    let t = agent::NUM_TIMESTEPS;

    // ── agent sub-dict ────────────────────────────────────────────────────────
    let agent_dict = PyDict::new(py);
    agent_dict.set_item("num_nodes", n)?;
    agent_dict.set_item("av_index", agent.av_index)?;

    agent_dict.set_item("valid_mask", make_bool_2d(py, agent.valid_mask, n, t)?)?;
    agent_dict.set_item("predict_mask", make_bool_2d(py, agent.predict_mask, n, t)?)?;

    let id_list = PyList::new(py, agent.ids.iter())?;
    agent_dict.set_item("id", id_list)?;

    agent_dict.set_item("type", make_uint8_1d(py, agent.agent_type)?)?;
    agent_dict.set_item("category", make_uint8_1d(py, agent.agent_category)?)?;

    agent_dict.set_item("position", make_f32_3d(py, agent.position, n, t, 3)?)?;
    agent_dict.set_item("heading", make_f32_2d(py, agent.heading, n, t)?)?;
    agent_dict.set_item("velocity", make_f32_3d(py, agent.velocity, n, t, 3)?)?;
    agent_dict.set_item("shape", make_f32_3d(py, agent.shape, n, t, 3)?)?;

    // ── map_polygon sub-dict ──────────────────────────────────────────────────
    let mp_dict = PyDict::new(py);
    mp_dict.set_item("num_nodes", map.num_polygons)?;
    mp_dict.set_item("type", make_uint8_1d(py, map.polygon_type)?)?;
    mp_dict.set_item("light_type", make_uint8_1d(py, map.polygon_light)?)?;

    // ── map_point sub-dict ────────────────────────────────────────────────────
    let mpt_dict = PyDict::new(py);
    let m = map.num_points;
    mpt_dict.set_item("num_nodes", m)?;
    mpt_dict.set_item("position", make_f32_2d(py, map.point_position, m, 3)?)?;
    mpt_dict.set_item("orientation", make_f32_1d(py, map.point_orientation)?)?;
    mpt_dict.set_item("magnitude", make_f32_1d(py, map.point_magnitude)?)?;
    mpt_dict.set_item("height", make_f32_1d(py, map.point_height)?)?;
    mpt_dict.set_item("type", make_uint8_1d(py, map.point_type)?)?;

    // ── point→polygon edge_index ──────────────────────────────────────────────
    let e_pts = m;
    let pt2poly_data: Vec<i64> = map.pt2poly_src.into_iter().chain(map.pt2poly_dst).collect();
    let pt2poly_edge = make_int64_2d(py, pt2poly_data, 2, e_pts)?;
    let pt2poly_dict = PyDict::new(py);
    pt2poly_dict.set_item("edge_index", pt2poly_edge)?;

    // ── polygon→polygon edge_index ────────────────────────────────────────────
    let e_poly = map.poly2poly_src.len();
    let poly2poly_data: Vec<i64> = map
        .poly2poly_src
        .into_iter()
        .chain(map.poly2poly_dst)
        .collect();
    let poly2poly_edge = make_int64_2d(py, poly2poly_data, 2, e_poly)?;
    let poly2poly_type_t = make_uint8_1d(py, map.poly2poly_type)?;
    let poly2poly_dict = PyDict::new(py);
    poly2poly_dict.set_item("edge_index", poly2poly_edge)?;
    poly2poly_dict.set_item("type", poly2poly_type_t)?;

    // ── top-level dict ────────────────────────────────────────────────────────
    let out = PyDict::new(py);
    out.set_item("scenario_id", scenario_id)?;
    out.set_item("city", 10086.0f64)?; // CITY_SENTINEL
    out.set_item("agent", agent_dict)?;
    out.set_item("map_polygon", mp_dict)?;
    out.set_item("map_point", mpt_dict)?;

    let pt2poly_key = PyTuple::new(py, ["map_point", "to", "map_polygon"])?;
    out.set_item(pt2poly_key, pt2poly_dict)?;

    let poly2poly_key = PyTuple::new(py, ["map_polygon", "to", "map_polygon"])?;
    out.set_item(poly2poly_key, poly2poly_dict)?;

    Ok(out)
}

// ─── pickle helper ────────────────────────────────────────────────────────────

fn pickle_to_file(py: Python<'_>, obj: &Bound<'_, PyAny>, path: &Path) -> PyResult<()> {
    let pickle = py.import("pickle")?;
    let builtins = py.import("builtins")?;
    let path_str = path.to_string_lossy();
    let fh = builtins.call_method1("open", (&*path_str, "wb"))?;
    pickle.call_method1("dump", (obj, &fh, 4i32))?; // protocol 4
    fh.call_method0("close")?;
    Ok(())
}

// ─── single-scenario function ─────────────────────────────────────────────────

fn process_scenario_inner(scenario_dir: &Path, output_dir: &Path) -> Result<bool> {
    let scenario_id = scenario_dir
        .file_name()
        .context("no dir name")?
        .to_string_lossy()
        .to_string();

    // Find parquet and map files
    let mut parquet_path: Option<PathBuf> = None;
    let mut map_path: Option<PathBuf> = None;

    for entry in std::fs::read_dir(scenario_dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("scenario_") && name.ends_with(".parquet") {
            parquet_path = Some(entry.path());
        } else if name.starts_with("log_map_archive_") && name.ends_with(".json") {
            map_path = Some(entry.path());
        }
    }

    let parquet_path = match parquet_path {
        Some(p) => p,
        None => {
            eprintln!("[WARN] {scenario_id}: no scenario parquet — skipping.");
            return Ok(false);
        }
    };
    let map_path = match map_path {
        Some(p) => p,
        None => {
            eprintln!("[WARN] {scenario_id}: no map json — skipping.");
            return Ok(false);
        }
    };

    let agent_data = match build_agent_data(&parquet_path) {
        Ok(d) => d,
        Err(e) => {
            // "no predictable tracks" is expected for some scenarios
            if e.to_string().contains("no predictable") {
                return Ok(false);
            }
            return Err(e);
        }
    };
    let map_data = build_map_data(&map_path)?;

    let out_path = output_dir.join(format!("{}.pkl", scenario_id));

    Python::with_gil(|py| -> Result<()> {
        let dict = build_python_dict(py, &scenario_id, agent_data, map_data)
            .map_err(|e| anyhow::anyhow!("build dict: {e}"))?;
        pickle_to_file(py, dict.as_any(), &out_path).map_err(|e| anyhow::anyhow!("pickle: {e}"))?;
        Ok(())
    })?;

    Ok(true)
}

// ─── PyO3 exposed functions ───────────────────────────────────────────────────

/// Process a single scenario directory, write one .pkl to output_dir.
/// Returns True on success, False if scenario was skipped (no predictable tracks).
#[pyfunction]
fn preprocess_scenario(scenario_dir: &str, output_dir: &str) -> PyResult<bool> {
    let sd = Path::new(scenario_dir);
    let od = Path::new(output_dir);
    std::fs::create_dir_all(od)?;
    process_scenario_inner(sd, od)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Process all scenario sub-directories in input_dir, write .pkl files to output_dir.
/// Returns (n_success, n_total).
#[pyfunction]
fn process_directory(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
) -> PyResult<(usize, usize)> {
    let input = Path::new(input_dir);
    let output = Path::new(output_dir);
    std::fs::create_dir_all(output)?;

    let mut entries: Vec<PathBuf> = std::fs::read_dir(input)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.path())
        .collect();
    entries.sort();

    let total = entries.len();
    let mut ok = 0usize;

    // Import tqdm for progress display (optional — skip if unavailable)
    let tqdm_opt = py.import("tqdm.auto").or_else(|_| py.import("tqdm")).ok();

    // Wrap entries in tqdm if available
    let use_tqdm = tqdm_opt.is_some();
    let tqdm_iter: Option<Bound<'_, PyAny>> = if use_tqdm {
        let tqdm_mod = tqdm_opt.unwrap();
        let kwargs = PyDict::new(py);
        kwargs.set_item("total", total)?;
        kwargs.set_item("desc", "Processing scenarios")?;
        kwargs.set_item("unit", "scenario")?;
        // Build a Python list of strings so tqdm can iterate
        let path_list: Vec<String> = entries
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        let py_list = PyList::new(py, path_list)?;
        Some(tqdm_mod.call_method("tqdm", (py_list,), Some(&kwargs))?)
    } else {
        None
    };

    for (i, scenario_dir) in entries.iter().enumerate() {
        let scenario_id = scenario_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        match process_scenario_inner(scenario_dir, output) {
            Ok(true) => {
                ok += 1;
            }
            Ok(false) => {}
            Err(e) => {
                eprintln!("[ERROR] {scenario_id}: {e}");
            }
        }

        // Update tqdm
        if let Some(ref it) = tqdm_iter {
            let _ = it.call_method1("update", (1,));
        } else if (i + 1) % 50 == 0 || i + 1 == total {
            println!("  [{}/{}]", i + 1, total);
        }
    }

    if let Some(ref it) = tqdm_iter {
        let _ = it.call_method0("close");
    }

    println!("Done: {ok}/{total} scenarios processed.");
    Ok((ok, total))
}

// ─── module definition ────────────────────────────────────────────────────────

#[pymodule]
fn smart_preprocess(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(preprocess_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(process_directory, m)?)?;
    Ok(())
}
