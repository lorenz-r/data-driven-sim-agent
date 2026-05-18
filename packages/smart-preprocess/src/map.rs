//! map.rs - Parse AV2 map JSON → MapData
//!
//! Reads log_map_archive_*.json directly (no av2 Python dependency).
//! Replicates Python's build_map_dict() exactly:
//!   - Lane segments: resample left+right boundaries to 20 pts, average → centerline
//!   - Pedestrian crossings: 4-corner polygon → 3 midpoints
//!   - Point vectors from consecutive centerline pairs (centerline[:-1], N-1 per poly)

use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::Deserialize;
use std::path::Path;

// ─── constants ────────────────────────────────────────────────────────────────

const LANE_CENTERLINE_POINTS: usize = 20;

// SMART type indices (must match Python constants exactly)
pub const POLYGON_TYPE_VEHICLE: u8 = 0;
pub const POLYGON_TYPE_BIKE: u8 = 1;
pub const POLYGON_TYPE_BUS: u8 = 2;
pub const POLYGON_TYPE_PEDESTRIAN: u8 = 3;
pub const LIGHT_TYPE_UNKNOWN: u8 = 3;
pub const POINT_TYPE_CROSSWALK: u8 = 15;
pub const POINT_TYPE_CENTERLINE: u8 = 16;
pub const EDGE_TYPE_PRED: u8 = 1;
pub const EDGE_TYPE_SUCC: u8 = 2;

// ─── JSON schema ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Xyz {
    x: f64,
    y: f64,
    #[serde(default)]
    z: f64,
}

#[derive(Deserialize)]
struct LaneSegment {
    id: i64,
    lane_type: String,
    left_lane_boundary: Vec<Xyz>,
    right_lane_boundary: Vec<Xyz>,
    #[serde(default)]
    predecessors: Vec<serde_json::Value>,
    #[serde(default)]
    successors: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct PedestrianCrossing {
    id: i64,
    edge1: Vec<Xyz>,
    edge2: Vec<Xyz>,
}

#[derive(Deserialize)]
struct MapJson {
    lane_segments: IndexMap<String, LaneSegment>,
    pedestrian_crossings: IndexMap<String, PedestrianCrossing>,
}

// ─── public output ────────────────────────────────────────────────────────────

pub struct MapData {
    pub num_polygons: usize,
    pub polygon_type: Vec<u8>,
    pub polygon_light: Vec<u8>,

    pub num_points: usize,
    /// Row-major [M, 3]
    pub point_position: Vec<f32>,
    pub point_orientation: Vec<f32>,
    pub point_magnitude: Vec<f32>,
    pub point_height: Vec<f32>,
    pub point_type: Vec<u8>,

    /// point→polygon: edge_index row 0 (source point indices 0..M)
    pub pt2poly_src: Vec<i64>,
    /// point→polygon: edge_index row 1 (destination polygon indices)
    pub pt2poly_dst: Vec<i64>,

    pub poly2poly_src: Vec<i64>,
    pub poly2poly_dst: Vec<i64>,
    pub poly2poly_type: Vec<u8>,
}

// ─── geometry helpers ─────────────────────────────────────────────────────────

/// Resample a polyline to exactly `num_pts` equidistant points (3D).
fn resample_polyline(pts: &[[f64; 3]], num_pts: usize) -> Vec<[f32; 3]> {
    if pts.is_empty() {
        return vec![[0.0, 0.0, 0.0]; num_pts];
    }
    if pts.len() == 1 {
        return vec![[pts[0][0] as f32, pts[0][1] as f32, pts[0][2] as f32]; num_pts];
    }

    // cumulative arc lengths
    let n = pts.len();
    let mut cum: Vec<f64> = Vec::with_capacity(n);
    cum.push(0.0);
    for i in 1..n {
        let dx = pts[i][0] - pts[i - 1][0];
        let dy = pts[i][1] - pts[i - 1][1];
        let dz = pts[i][2] - pts[i - 1][2];
        let seg = (dx * dx + dy * dy + dz * dz).sqrt();
        cum.push(cum[i - 1] + seg);
    }
    let total = cum[n - 1];
    if total < 1e-6 {
        return vec![[pts[0][0] as f32, pts[0][1] as f32, pts[0][2] as f32]; num_pts];
    }

    let mut out = Vec::with_capacity(num_pts);
    for k in 0..num_pts {
        let s = total * (k as f64) / ((num_pts - 1) as f64);
        // binary search for segment
        let idx = cum.partition_point(|&c| c < s).saturating_sub(1).min(n - 2);
        let seg_len = cum[idx + 1] - cum[idx];
        let t = if seg_len < 1e-12 {
            0.0
        } else {
            (s - cum[idx]) / seg_len
        };
        let x = (pts[idx][0] + t * (pts[idx + 1][0] - pts[idx][0])) as f32;
        let y = (pts[idx][1] + t * (pts[idx + 1][1] - pts[idx][1])) as f32;
        let z = (pts[idx][2] + t * (pts[idx + 1][2] - pts[idx][2])) as f32;
        out.push([x, y, z]);
    }
    out
}

fn xyz_to_arr3(v: &[Xyz]) -> Vec<[f64; 3]> {
    v.iter().map(|p| [p.x, p.y, p.z]).collect()
}

fn compute_centerline(left: &[Xyz], right: &[Xyz]) -> Vec<[f32; 3]> {
    let l3 = xyz_to_arr3(left);
    let r3 = xyz_to_arr3(right);
    let lr = resample_polyline(&l3, LANE_CENTERLINE_POINTS);
    let rr = resample_polyline(&r3, LANE_CENTERLINE_POINTS);
    lr.iter()
        .zip(rr.iter())
        .map(|(l, r)| {
            [
                (l[0] + r[0]) * 0.5,
                (l[1] + r[1]) * 0.5,
                (l[2] + r[2]) * 0.5,
            ]
        })
        .collect()
}

fn lane_type_to_polygon(s: &str) -> u8 {
    match s.to_ascii_uppercase().as_str() {
        "VEHICLE" => POLYGON_TYPE_VEHICLE,
        "BIKE" => POLYGON_TYPE_BIKE,
        "BUS" => POLYGON_TYPE_BUS,
        "PEDESTRIAN" => POLYGON_TYPE_PEDESTRIAN,
        _ => POLYGON_TYPE_VEHICLE,
    }
}

/// Try to extract an i64 from a JSON Value (handles both number and string).
fn value_to_id(v: &serde_json::Value) -> Option<i64> {
    if let Some(n) = v.as_i64() {
        return Some(n);
    }
    if let Some(s) = v.as_str() {
        return s.parse().ok();
    }
    None
}

// ─── point/polygon accumulation ──────────────────────────────────────────────

struct PolyAccum {
    // per-polygon
    poly_type: Vec<u8>,
    poly_light: Vec<u8>,
    poly_id: Vec<i64>, // original AV2 lane/crossing IDs

    // per-point
    pt_pos: Vec<f32>, // flat [M*3]
    pt_ori: Vec<f32>,
    pt_mag: Vec<f32>,
    pt_ht: Vec<f32>,
    pt_type: Vec<u8>,
    pt_pts_per_poly: Vec<usize>,
    pt_poly_idx: Vec<usize>, // which polygon (output index)
}

impl PolyAccum {
    fn new() -> Self {
        Self {
            poly_type: Vec::new(),
            poly_light: Vec::new(),
            poly_id: Vec::new(),
            pt_pos: Vec::new(),
            pt_ori: Vec::new(),
            pt_mag: Vec::new(),
            pt_ht: Vec::new(),
            pt_type: Vec::new(),
            pt_pts_per_poly: Vec::new(),
            pt_poly_idx: Vec::new(),
        }
    }

    fn push_polygon(&mut self, id: i64, ptype: u8, centerline: &[[f32; 3]], pt_type: u8) {
        let poly_idx = self.poly_type.len();
        self.poly_type.push(ptype);
        self.poly_light.push(LIGHT_TYPE_UNKNOWN);
        self.poly_id.push(id);

        if centerline.len() < 2 {
            return;
        }

        let vecs: Vec<_> = centerline.windows(2).collect();
        let n_pts = vecs.len();

        for w in &vecs {
            let p = w[0];
            let q = w[1];
            let dx = (q[0] - p[0]) as f64;
            let dy = (q[1] - p[1]) as f64;
            let dz = (q[2] - p[2]) as f64;
            // norm in f64, then cast — matches torch.norm(f32) precision
            let mag = (dx * dx + dy * dy).sqrt() as f32;
            let ori = dy.atan2(dx) as f32;
            self.pt_pos.extend_from_slice(&[p[0], p[1], p[2]]);
            self.pt_ori.push(ori);
            self.pt_mag.push(mag);
            self.pt_ht.push(dz as f32);
            self.pt_type.push(pt_type);
        }
        self.pt_pts_per_poly.push(n_pts);
        self.pt_poly_idx.push(poly_idx);
    }
}

// ─── public entry point ───────────────────────────────────────────────────────

pub fn build_map_data(map_path: &Path) -> Result<MapData> {
    let text = std::fs::read_to_string(map_path)
        .with_context(|| format!("read map json: {}", map_path.display()))?;
    let map: MapJson = serde_json::from_str(&text).with_context(|| "parse map json")?;

    let mut acc = PolyAccum::new();

    // ── lane segments — iterate in JSON insertion order, matching Python's
    // ArgoverseStaticMap which also follows insertion order.
    struct LaneMeta {
        id: i64,
        poly_idx: usize,
        predecessors: Vec<i64>,
        successors: Vec<i64>,
    }
    let mut lane_meta: Vec<LaneMeta> = Vec::with_capacity(map.lane_segments.len());

    for (_key, ls) in &map.lane_segments {
        let centerline = compute_centerline(&ls.left_lane_boundary, &ls.right_lane_boundary);
        if centerline.len() < 2 {
            continue;
        }

        let poly_idx = acc.poly_type.len();
        acc.push_polygon(
            ls.id,
            lane_type_to_polygon(&ls.lane_type),
            &centerline,
            POINT_TYPE_CENTERLINE,
        );

        let preds: Vec<i64> = ls.predecessors.iter().filter_map(value_to_id).collect();
        let succs: Vec<i64> = ls.successors.iter().filter_map(value_to_id).collect();
        lane_meta.push(LaneMeta {
            id: ls.id,
            poly_idx,
            predecessors: preds,
            successors: succs,
        });
    }

    // ── pedestrian crossings ─ insertion order, matching Python ──────────────
    for (_key, pc) in &map.pedestrian_crossings {
        if pc.edge1.is_empty() || pc.edge2.is_empty() {
            continue;
        }

        // 4-corner polygon: e1[0], e1[-1], e2[-1], e2[0]
        let e1f = &pc.edge1[0];
        let e1l = pc.edge1.last().unwrap();
        let e2l = pc.edge2.last().unwrap();
        let e2f = &pc.edge2[0];

        // 3 midpoints → 2 vectors (matches Python)
        let mid01 = [
            ((e1f.x + e1l.x) * 0.5) as f32,
            ((e1f.y + e1l.y) * 0.5) as f32,
            ((e1f.z + e1l.z) * 0.5) as f32,
        ];
        let mid12 = [
            ((e1l.x + e2l.x) * 0.5) as f32,
            ((e1l.y + e2l.y) * 0.5) as f32,
            ((e1l.z + e2l.z) * 0.5) as f32,
        ];
        let mid23 = [
            ((e2l.x + e2f.x) * 0.5) as f32,
            ((e2l.y + e2f.y) * 0.5) as f32,
            ((e2l.z + e2f.z) * 0.5) as f32,
        ];

        acc.push_polygon(
            pc.id,
            POLYGON_TYPE_PEDESTRIAN,
            &[mid01, mid12, mid23],
            POINT_TYPE_CROSSWALK,
        );
    }

    let num_polygons = acc.poly_type.len();

    // ── point→polygon edge index ──────────────────────────────────────────────
    let num_points = acc.pt_pos.len() / 3;
    let pt2poly_src: Vec<i64> = (0..num_points as i64).collect();
    let mut pt2poly_dst: Vec<i64> = Vec::with_capacity(num_points);
    for (k, &n_pts) in acc.pt_pts_per_poly.iter().enumerate() {
        let pidx = acc.pt_poly_idx[k] as i64;
        for _ in 0..n_pts {
            pt2poly_dst.push(pidx);
        }
    }

    // ── polygon→polygon edges ─────────────────────────────────────────────────
    // Build id→poly_idx map for lane segments only
    let id_to_idx: std::collections::HashMap<i64, usize> =
        lane_meta.iter().map(|m| (m.id, m.poly_idx)).collect();
    let lane_id_set: std::collections::HashSet<i64> = lane_meta.iter().map(|m| m.id).collect();

    let mut poly2poly_src: Vec<i64> = Vec::new();
    let mut poly2poly_dst: Vec<i64> = Vec::new();
    let mut poly2poly_type: Vec<u8> = Vec::new();

    for meta in &lane_meta {
        let dst = meta.poly_idx as i64;

        // predecessors: PRED edge  (pred → self)
        for &pid in &meta.predecessors {
            if lane_id_set.contains(&pid) {
                let src = id_to_idx[&pid] as i64;
                poly2poly_src.push(src);
                poly2poly_dst.push(dst);
                poly2poly_type.push(EDGE_TYPE_PRED);
            }
        }

        // successors: SUCC edge  (succ → self)
        for &sid in &meta.successors {
            if lane_id_set.contains(&sid) {
                let src = id_to_idx[&sid] as i64;
                poly2poly_src.push(src);
                poly2poly_dst.push(dst);
                poly2poly_type.push(EDGE_TYPE_SUCC);
            }
        }
    }

    Ok(MapData {
        num_polygons,
        polygon_type: acc.poly_type,
        polygon_light: acc.poly_light,
        num_points,
        point_position: acc.pt_pos,
        point_orientation: acc.pt_ori,
        point_magnitude: acc.pt_mag,
        point_height: acc.pt_ht,
        point_type: acc.pt_type,
        pt2poly_src,
        pt2poly_dst,
        poly2poly_src,
        poly2poly_dst,
        poly2poly_type,
    })
}
