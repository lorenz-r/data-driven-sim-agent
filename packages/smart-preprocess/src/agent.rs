//! agent.rs - Parse AV2 scenario parquet → AgentData
//!
//! Parquet columns used:
//!   observed (bool), track_id (Utf8), object_type (Utf8), object_category (Int64),
//!   timestep (Int64), position_x (Float64), position_y (Float64), heading (Float64),
//!   velocity_x (Float64), velocity_y (Float64)

use anyhow::{bail, Context, Result};
use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::{collections::HashMap, fs::File, path::Path};

// ─── constants ───────────────────────────────────────────────────────────────

pub const NUM_TIMESTEPS: usize = 110;
pub const CURRENT_TIME_INDEX: usize = 49;
pub const OBSERVED_STEPS: usize = CURRENT_TIME_INDEX + 1; // 50

// ─── mapping tables ──────────────────────────────────────────────────────────

fn object_type_to_smart(s: &str) -> u8 {
    match s {
        "vehicle" | "bus" => 0,
        "pedestrian" => 1,
        "motorcyclist" | "cyclist" | "riderless_bicycle" => 2,
        _ => 3, // background, static, construction, unknown
    }
}

/// Returns (length, width, height) in metres.
fn default_box(object_type: &str) -> (f32, f32, f32) {
    match object_type {
        "vehicle" => (4.50, 2.00, 1.50),
        "bus" => (12.0, 2.50, 3.50),
        "pedestrian" => (0.50, 0.50, 1.70),
        "motorcyclist" => (2.20, 0.70, 1.50),
        "cyclist" => (1.80, 0.60, 1.50),
        "riderless_bicycle" => (1.80, 0.60, 1.00),
        "construction" => (1.50, 1.50, 2.00),
        _ => (1.00, 1.00, 1.00), // static / background / unknown
    }
}

/// SMART category: 3 for focal/scored, 1 for everything else.
fn smart_category(object_category: i64) -> u8 {
    if object_category == 2 || object_category == 3 {
        3
    } else {
        1
    }
}

// ─── row-level structs ───────────────────────────────────────────────────────

#[derive(Clone)]
struct TrackRow {
    observed: bool,
    object_type: String,
    object_category: i64,
    timestep: usize,
    px: f64,
    py: f64,
    heading: f64,
    vx: f64,
    vy: f64,
}

// ─── public output ───────────────────────────────────────────────────────────

pub struct AgentData {
    pub num_nodes: usize,
    pub av_index: usize,
    /// Row-major [N, T]
    pub valid_mask: Vec<bool>,
    /// Row-major [N, T]
    pub predict_mask: Vec<bool>,
    pub ids: Vec<String>,
    pub agent_type: Vec<u8>,
    pub agent_category: Vec<u8>,
    /// Row-major [N, T, 3]  x,y,z
    pub position: Vec<f32>,
    /// Row-major [N, T]
    pub heading: Vec<f32>,
    /// Row-major [N, T, 3]  vx,vy,vz
    pub velocity: Vec<f32>,
    /// Row-major [N, T, 3]  length,width,height
    pub shape: Vec<f32>,
}

// ─── parquet reading ─────────────────────────────────────────────────────────

type ParquetRows = (Vec<String>, HashMap<String, Vec<TrackRow>>);

/// Read all rows from the scenario parquet into a map  track_id → Vec<TrackRow>.
/// Rows are pushed in the order they appear in the file (preserves Python ordering).
fn read_parquet(parquet_path: &Path) -> Result<ParquetRows> {
    let file = File::open(parquet_path)
        .with_context(|| format!("open parquet: {}", parquet_path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;

    // track_id insertion order
    let mut order: Vec<String> = Vec::new();
    let mut map: HashMap<String, Vec<TrackRow>> = HashMap::new();

    for batch_result in reader {
        let batch = batch_result?;

        let observed_col = batch
            .column_by_name("observed")
            .context("no 'observed' column")?
            .as_any()
            .downcast_ref::<BooleanArray>()
            .context("observed not bool")?;
        let track_id_col = batch
            .column_by_name("track_id")
            .context("no 'track_id' column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("track_id not Utf8")?;
        let object_type_col = batch
            .column_by_name("object_type")
            .context("no 'object_type' column")?
            .as_any()
            .downcast_ref::<StringArray>()
            .context("object_type not Utf8")?;
        let object_category_col = batch
            .column_by_name("object_category")
            .context("no 'object_category' column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("object_category not Int64")?;
        let timestep_col = batch
            .column_by_name("timestep")
            .context("no 'timestep' column")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .context("timestep not Int64")?;
        let px_col = batch
            .column_by_name("position_x")
            .context("no 'position_x'")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("position_x not Float64")?;
        let py_col = batch
            .column_by_name("position_y")
            .context("no 'position_y'")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("position_y not Float64")?;
        let heading_col = batch
            .column_by_name("heading")
            .context("no 'heading'")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("heading not Float64")?;
        let vx_col = batch
            .column_by_name("velocity_x")
            .context("no 'velocity_x'")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("velocity_x not Float64")?;
        let vy_col = batch
            .column_by_name("velocity_y")
            .context("no 'velocity_y'")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("velocity_y not Float64")?;

        let n = batch.num_rows();
        for i in 0..n {
            let tid = track_id_col.value(i).to_owned();
            let t = timestep_col.value(i) as usize;
            if t >= NUM_TIMESTEPS {
                continue;
            }

            let row = TrackRow {
                observed: observed_col.value(i),
                object_type: object_type_col.value(i).to_owned(),
                object_category: object_category_col.value(i),
                timestep: t,
                px: if px_col.is_null(i) {
                    0.0
                } else {
                    px_col.value(i)
                },
                py: if py_col.is_null(i) {
                    0.0
                } else {
                    py_col.value(i)
                },
                heading: if heading_col.is_null(i) {
                    0.0
                } else {
                    heading_col.value(i)
                },
                vx: if vx_col.is_null(i) {
                    0.0
                } else {
                    vx_col.value(i)
                },
                vy: if vy_col.is_null(i) {
                    0.0
                } else {
                    vy_col.value(i)
                },
            };

            if !map.contains_key(&tid) {
                order.push(tid.clone());
                map.insert(tid.clone(), Vec::new());
            }
            map.get_mut(&tid).unwrap().push(row);
        }
    }

    Ok((order, map))
}

// ─── public entry point ───────────────────────────────────────────────────────

pub fn build_agent_data(parquet_path: &Path) -> Result<AgentData> {
    let (order, map) = read_parquet(parquet_path)?;

    // ── filter: tracks observed at t=49 ──────────────────────────────────────
    let included_ids: Vec<String> = order
        .iter()
        .filter(|tid| {
            map[*tid]
                .iter()
                .any(|r| r.timestep == CURRENT_TIME_INDEX && r.observed)
        })
        .cloned()
        .collect();

    // Fallback: all tracks if filter is empty
    let included_ids = if included_ids.is_empty() {
        order.clone()
    } else {
        included_ids
    };

    // ── check has_predictable ─────────────────────────────────────────────────
    let has_predictable = included_ids.iter().any(|tid| {
        map[tid]
            .first()
            .map(|r| r.object_category == 2 || r.object_category == 3)
            .unwrap_or(false)
    });
    if !has_predictable {
        bail!("no predictable tracks");
    }

    let n = included_ids.len();
    let t = NUM_TIMESTEPS;

    let mut valid_mask = vec![false; n * t];
    let mut predict_mask = vec![false; n * t];
    let mut position = vec![0.0f32; n * t * 3];
    let mut heading_arr = vec![0.0f32; n * t];
    let mut velocity = vec![0.0f32; n * t * 3];
    let mut shape_arr = vec![0.0f32; n * t * 3];

    let mut ids: Vec<String> = Vec::with_capacity(n);
    let mut agent_type: Vec<u8> = Vec::with_capacity(n);
    let mut agent_category: Vec<u8> = Vec::with_capacity(n);

    let mut av_index = 0usize;

    for (i, tid) in included_ids.iter().enumerate() {
        ids.push(tid.clone());

        let rows = &map[tid];
        // Use first row for per-track metadata
        let first = &rows[0];

        if tid == "AV" {
            av_index = i;
        }

        agent_type.push(object_type_to_smart(&first.object_type));
        agent_category.push(smart_category(first.object_category));

        let (length, width, height) = default_box(&first.object_type);

        // ── raw fill ──────────────────────────────────────────────────────────
        for row in rows {
            let ts = row.timestep;
            // For t < 50 skip if not observed; for t >= 50 always write
            if ts < OBSERVED_STEPS && !row.observed {
                continue;
            }

            let vi = i * t + ts;
            valid_mask[vi] = true;

            let pi = vi * 3;
            position[pi] = row.px as f32;
            position[pi + 1] = row.py as f32;
            position[pi + 2] = 0.0; // AV2 has no z in parquet

            heading_arr[vi] = row.heading as f32;

            velocity[pi] = row.vx as f32;
            velocity[pi + 1] = row.vy as f32;
            velocity[pi + 2] = 0.0;

            shape_arr[pi] = length;
            shape_arr[pi + 1] = width;
            shape_arr[pi + 2] = height;
        }

        // ── valid_mask post-processing ────────────────────────────────────────
        // Python: valid[1:50] = orig[0:49] & orig[1:50]  (single-step vectorised,
        // NOT a rolling AND).  We must snapshot orig before modifying.
        let base = i * t;
        let orig: Vec<bool> = valid_mask[base..base + OBSERVED_STEPS].to_vec();
        for ts in 1..OBSERVED_STEPS {
            valid_mask[base + ts] = orig[ts - 1] & orig[ts];
        }
        valid_mask[base] = false; // t=0 always false

        // ── predict_mask ──────────────────────────────────────────────────────
        let valid_at_49 = valid_mask[i * t + CURRENT_TIME_INDEX];
        if valid_at_49 {
            for ts in OBSERVED_STEPS..t {
                predict_mask[i * t + ts] = valid_mask[i * t + ts];
            }
        }
        // predict_mask for t<50 stays false (already initialised to false)
    }

    // ── find AV index (fallback logic if "AV" not found) ─────────────────────
    // Check if "AV" was actually in included_ids; if not, use heuristic.
    if !included_ids.contains(&"AV".to_string()) {
        let mut best_idx = 0;
        let mut best_count = -1i64;

        // Priority 1: UNSCORED_TRACK vehicle with most historical observations
        for (i, tid) in included_ids.iter().enumerate() {
            let rows = &map[tid];
            let first = &rows[0];
            if first.object_type.as_str() != "vehicle" {
                continue;
            }
            if first.object_category != 1 {
                continue;
            } // UNSCORED_TRACK
            let n_obs = rows
                .iter()
                .filter(|r| r.timestep <= CURRENT_TIME_INDEX && r.observed)
                .count() as i64;
            if n_obs > best_count {
                best_count = n_obs;
                best_idx = i;
            }
        }

        // Priority 2: any vehicle with most historical observations
        if best_count < 0 {
            for (i, tid) in included_ids.iter().enumerate() {
                let rows = &map[tid];
                if rows[0].object_type.as_str() != "vehicle" {
                    continue;
                }
                let n_obs = rows
                    .iter()
                    .filter(|r| r.timestep <= CURRENT_TIME_INDEX && r.observed)
                    .count() as i64;
                if n_obs > best_count {
                    best_count = n_obs;
                    best_idx = i;
                }
            }
        }

        av_index = best_idx;
    }

    Ok(AgentData {
        num_nodes: n,
        av_index,
        valid_mask,
        predict_mask,
        ids,
        agent_type,
        agent_category,
        position,
        heading: heading_arr,
        velocity,
        shape: shape_arr,
    })
}
