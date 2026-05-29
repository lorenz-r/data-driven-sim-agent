use numpy::{PyArray1, PyArray2, PyArray3, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{
    exceptions::{PyKeyError, PyValueError},
    prelude::*,
    types::{PyDict, PyTuple},
};
use std::collections::BTreeSet;
use std::f32::consts::PI;

const TWO_PI: f32 = 2.0 * PI;

#[derive(Clone)]
struct TokenSet {
    token: Vec<f32>,
    token_last: Vec<f32>,
    token_num: usize,
    contour_dim: usize,
}

#[derive(Clone, Copy)]
enum AgentKind {
    Vehicle,
    Pedestrian,
    Cyclist,
}

impl AgentKind {
    fn from_type(agent_type: u8) -> Option<Self> {
        match agent_type {
            0 => Some(Self::Vehicle),
            1 => Some(Self::Pedestrian),
            2 => Some(Self::Cyclist),
            _ => None,
        }
    }
}

#[pyclass]
pub struct TokenProcessor {
    veh: TokenSet,
    ped: TokenSet,
    cyc: TokenSet,
    training: bool,
    shift: usize,
    current_step: usize,
}

#[derive(Clone)]
struct AgentArrays {
    valid: Vec<bool>,
    position: Vec<f32>,
    heading: Vec<f32>,
    velocity: Vec<f32>,
    agent_type: Vec<u8>,
    num_nodes: usize,
    timesteps: usize,
    velocity_dim: usize,
}

struct MatchOutput {
    token_idx: Vec<i64>,
    token_contour: Vec<f32>,
    steps: usize,
}

#[pymethods]
impl TokenProcessor {
    #[new]
    #[pyo3(signature = (agent_token_path, _map_token_traj_path=None, training=false, shift=5, current_step=10))]
    fn new(
        py: Python<'_>,
        agent_token_path: &str,
        _map_token_traj_path: Option<&str>,
        training: bool,
        shift: usize,
        current_step: usize,
    ) -> PyResult<Self> {
        let token_root = load_pickle(py, agent_token_path)?;
        let root_dict = token_root.downcast::<PyDict>()?;
        let token_dict = get_dict(root_dict, "token")?;
        let token_all_dict = get_dict(root_dict, "token_all")?;

        Ok(Self {
            veh: load_token_set(&token_dict, &token_all_dict, "veh")?,
            ped: load_token_set(&token_dict, &token_all_dict, "ped")?,
            cyc: load_token_set(&token_dict, &token_all_dict, "cyc")?,
            training,
            shift,
            current_step,
        })
    }

    fn preprocess<'py>(
        &self,
        py: Python<'py>,
        data: Bound<'py, PyDict>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.tokenize_agent(py, &data)?;
        self.tokenize_map(py, &data)?;

        if data.contains("city")? {
            data.del_item("city")?;
        }

        let map_polygon = get_dict(&data, "map_polygon")?;
        if map_polygon.contains("polygon_is_intersection")? {
            map_polygon.del_item("polygon_is_intersection")?;
        }
        if map_polygon.contains("route_type")? {
            map_polygon.del_item("route_type")?;
        }

        Ok(data)
    }
}

impl TokenProcessor {
    fn token_set(&self, kind: AgentKind) -> &TokenSet {
        match kind {
            AgentKind::Vehicle => &self.veh,
            AgentKind::Pedestrian => &self.ped,
            AgentKind::Cyclist => &self.cyc,
        }
    }

    fn dimensions(&self) -> (usize, usize) {
        (self.veh.token_num, self.veh.contour_dim)
    }

    fn tokenize_agent<'py>(&self, py: Python<'py>, data: &Bound<'py, PyDict>) -> PyResult<()> {
        let agent = get_dict(data, "agent")?;
        let mut arrays = read_agent_arrays(&agent)?;

        if self.current_step >= arrays.timesteps || self.current_step < self.shift {
            return Err(PyValueError::new_err(
                "invalid token processor step settings",
            ));
        }

        for a in 0..arrays.num_nodes {
            let cur = idx2(a, self.current_step, arrays.timesteps);
            let prev = idx2(a, self.current_step - 1, arrays.timesteps);
            let valid_cur = arrays.valid[cur];
            let position_cur_x =
                arrays.position[idx3(a, self.current_step, 0, arrays.timesteps, 3)];
            if !valid_cur && position_cur_x != 0.0 {
                for d in 0..3 {
                    let v = if d < arrays.velocity_dim {
                        arrays.velocity[idx3(
                            a,
                            self.current_step,
                            d,
                            arrays.timesteps,
                            arrays.velocity_dim,
                        )]
                    } else {
                        0.0
                    };
                    let cur_pos =
                        arrays.position[idx3(a, self.current_step, d, arrays.timesteps, 3)];
                    arrays.position[idx3(a, self.current_step - 1, d, arrays.timesteps, 3)] =
                        cur_pos - v * 0.1;
                }
                arrays.valid[prev] = true;
                arrays.valid[cur] = true;
                arrays.heading[prev] = arrays.heading[cur];
                for d in 0..arrays.velocity_dim {
                    let val = arrays.velocity[idx3(
                        a,
                        self.current_step,
                        d,
                        arrays.timesteps,
                        arrays.velocity_dim,
                    )];
                    arrays.velocity[idx3(
                        a,
                        self.current_step - 1,
                        d,
                        arrays.timesteps,
                        arrays.velocity_dim,
                    )] = val;
                }
            }
        }

        clean_heading(
            &mut arrays.heading,
            &arrays.valid,
            arrays.num_nodes,
            arrays.timesteps,
        );

        let mut extra_mask = vec![false; arrays.num_nodes];
        for (a, extra) in extra_mask.iter_mut().enumerate() {
            *extra = arrays.valid[idx2(a, self.current_step, arrays.timesteps)]
                && !arrays.valid[idx2(a, self.current_step - self.shift, arrays.timesteps)];
        }

        for a in 0..arrays.num_nodes {
            let first = idx2(a, 0, arrays.timesteps);
            let first_pos = arrays.position[idx3(a, 0, 0, arrays.timesteps, 3)];
            if !arrays.valid[first] && first_pos != 0.0 {
                arrays.valid[first] = true;
            }
        }

        let steps = token_steps(arrays.timesteps, self.shift);
        let (token_num, contour_dim) = self.dimensions();
        let mut token_valid_mask = vec![false; arrays.num_nodes * steps];
        for a in 0..arrays.num_nodes {
            for s in 0..steps {
                let start = s * self.shift;
                token_valid_mask[idx2(a, s, steps)] = arrays.valid
                    [idx2(a, start, arrays.timesteps)]
                    && arrays.valid[idx2(a, start + self.shift, arrays.timesteps)];
            }
        }

        let mut token_idx = vec![0_i64; arrays.num_nodes * steps];
        let mut token_contour = vec![0.0_f32; arrays.num_nodes * steps * contour_dim * 2];

        for kind in [
            AgentKind::Vehicle,
            AgentKind::Pedestrian,
            AgentKind::Cyclist,
        ] {
            let global_indices: Vec<usize> = arrays
                .agent_type
                .iter()
                .enumerate()
                .filter_map(|(idx, &ty)| {
                    let current = AgentKind::from_type(ty)?;
                    if std::mem::discriminant(&current) == std::mem::discriminant(&kind) {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();

            if global_indices.is_empty() {
                continue;
            }

            let matched = self.match_token(&arrays, &global_indices, &extra_mask, kind);
            for (local_idx, &agent_idx) in global_indices.iter().enumerate() {
                for s in 0..matched.steps {
                    token_idx[idx2(agent_idx, s, steps)] =
                        matched.token_idx[idx2(local_idx, s, matched.steps)];
                    for c in 0..contour_dim {
                        for d in 0..2 {
                            token_contour[idx4(agent_idx, s, c, d, steps, contour_dim, 2)] =
                                matched.token_contour
                                    [idx4(local_idx, s, c, d, matched.steps, contour_dim, 2)];
                        }
                    }
                }
            }
        }

        if !self.training && steps > 1 {
            for (a, &extra) in extra_mask.iter().enumerate() {
                if extra {
                    token_valid_mask[idx2(a, 1, steps)] = true;
                }
            }
        }

        let mut token_pos = vec![0.0_f32; arrays.num_nodes * steps * 2];
        let mut token_heading = vec![0.0_f32; arrays.num_nodes * steps];
        for a in 0..arrays.num_nodes {
            for s in 0..steps {
                let mut mean_x = 0.0_f32;
                let mut mean_y = 0.0_f32;
                for c in 0..contour_dim {
                    mean_x += token_contour[idx4(a, s, c, 0, steps, contour_dim, 2)];
                    mean_y += token_contour[idx4(a, s, c, 1, steps, contour_dim, 2)];
                }
                mean_x /= contour_dim as f32;
                mean_y /= contour_dim as f32;
                token_pos[idx3(a, s, 0, steps, 2)] = mean_x;
                token_pos[idx3(a, s, 1, steps, 2)] = mean_y;

                let dx = token_contour[idx4(a, s, 0, 0, steps, contour_dim, 2)]
                    - token_contour[idx4(a, s, 3, 0, steps, contour_dim, 2)];
                let dy = token_contour[idx4(a, s, 0, 1, steps, contour_dim, 2)]
                    - token_contour[idx4(a, s, 3, 1, steps, contour_dim, 2)];
                token_heading[idx2(a, s, steps)] = dy.atan2(dx);
            }
        }

        let mut token_velocity = vec![0.0_f32; arrays.num_nodes * steps * 2];
        for a in 0..arrays.num_nodes {
            for s in 1..steps {
                if token_valid_mask[idx2(a, s, steps)] && token_valid_mask[idx2(a, s - 1, steps)] {
                    token_velocity[idx3(a, s, 0, steps, 2)] = (token_pos[idx3(a, s, 0, steps, 2)]
                        - token_pos[idx3(a, s - 1, 0, steps, 2)])
                        / (0.1 * self.shift as f32);
                    token_velocity[idx3(a, s, 1, steps, 2)] = (token_pos[idx3(a, s, 1, steps, 2)]
                        - token_pos[idx3(a, s - 1, 1, steps, 2)])
                        / (0.1 * self.shift as f32);
                }
            }
            if steps > 1 && arrays.valid[idx2(a, self.current_step, arrays.timesteps)] {
                token_velocity[idx3(a, 1, 0, steps, 2)] = arrays.velocity[idx3(
                    a,
                    self.current_step,
                    0,
                    arrays.timesteps,
                    arrays.velocity_dim,
                )];
                if arrays.velocity_dim > 1 {
                    token_velocity[idx3(a, 1, 1, steps, 2)] = arrays.velocity[idx3(
                        a,
                        self.current_step,
                        1,
                        arrays.timesteps,
                        arrays.velocity_dim,
                    )];
                }
            }
        }

        agent.set_item(
            "valid_mask",
            make_bool_2d(py, arrays.valid, arrays.num_nodes, arrays.timesteps)?,
        )?;
        agent.set_item(
            "position",
            make_f32_3d(py, arrays.position, arrays.num_nodes, arrays.timesteps, 3)?,
        )?;
        agent.set_item(
            "heading",
            make_f32_2d(py, arrays.heading, arrays.num_nodes, arrays.timesteps)?,
        )?;
        if arrays.velocity_dim == 3 {
            agent.set_item(
                "velocity",
                make_f32_3d(py, arrays.velocity, arrays.num_nodes, arrays.timesteps, 3)?,
            )?;
        }
        agent.set_item("type", make_uint8_1d(py, arrays.agent_type)?)?;
        agent.set_item(
            "token_idx",
            make_i64_2d(py, token_idx, arrays.num_nodes, steps)?,
        )?;
        agent.set_item(
            "token_contour",
            make_f32_4d(py, token_contour, arrays.num_nodes, steps, contour_dim, 2)?,
        )?;
        agent.set_item(
            "token_pos",
            make_f32_3d(py, token_pos, arrays.num_nodes, steps, 2)?,
        )?;
        agent.set_item(
            "token_heading",
            make_f32_2d(py, token_heading, arrays.num_nodes, steps)?,
        )?;
        agent.set_item(
            "agent_valid_mask",
            make_bool_2d(py, token_valid_mask, arrays.num_nodes, steps)?,
        )?;
        agent.set_item(
            "token_velocity",
            make_f32_3d(py, token_velocity, arrays.num_nodes, steps, 2)?,
        )?;

        let _ = token_num;
        Ok(())
    }

    fn match_token(
        &self,
        arrays: &AgentArrays,
        global_indices: &[usize],
        extra_mask: &[bool],
        kind: AgentKind,
    ) -> MatchOutput {
        let token_set = self.token_set(kind);
        let agent_num = global_indices.len();
        let steps = token_steps(arrays.timesteps, self.shift);
        let token_num = token_set.token_num;
        let contour_dim = token_set.contour_dim;
        let mut token_idx = vec![0_i64; agent_num * steps];
        let mut token_contour = vec![0.0_f32; agent_num * steps * contour_dim * 2];
        let (width, length) = dimensions_for_kind(kind, self.shift);

        let mut prev_heading = vec![0.0_f32; agent_num];
        let mut prev_pos = vec![0.0_f32; agent_num * 2];
        for (local, &global) in global_indices.iter().enumerate() {
            prev_heading[local] = arrays.heading[idx2(global, 0, arrays.timesteps)];
            prev_pos[idx2(local, 0, 2)] = arrays.position[idx3(global, 0, 0, arrays.timesteps, 3)];
            prev_pos[idx2(local, 1, 2)] = arrays.position[idx3(global, 0, 1, arrays.timesteps, 3)];
        }

        for (s, frame) in (self.shift..arrays.timesteps)
            .step_by(self.shift)
            .enumerate()
        {
            for (local, &global) in global_indices.iter().enumerate() {
                let theta = prev_heading[local];
                let cos = theta.cos();
                let sin = theta.sin();
                let cur_x = arrays.position[idx3(global, frame, 0, arrays.timesteps, 3)];
                let cur_y = arrays.position[idx3(global, frame, 1, arrays.timesteps, 3)];
                let cur_heading = arrays.heading[idx2(global, frame, arrays.timesteps)];
                let cur_contour = polygon_contour(cur_x, cur_y, cur_heading, width, length);

                let mut best_idx = 0_usize;
                let mut best_dist = f32::INFINITY;
                for tok in 0..token_num {
                    let mut total = 0.0_f32;
                    for c in 0..contour_dim {
                        let local_x = token_set.token[idx3(tok, c, 0, contour_dim, 2)];
                        let local_y = token_set.token[idx3(tok, c, 1, contour_dim, 2)];
                        let world_x = local_x * cos - local_y * sin + prev_pos[idx2(local, 0, 2)];
                        let world_y = local_x * sin + local_y * cos + prev_pos[idx2(local, 1, 2)];
                        let dx = cur_contour[idx2(c, 0, 2)] - world_x;
                        let dy = cur_contour[idx2(c, 1, 2)] - world_y;
                        total += (dx * dx + dy * dy).sqrt();
                    }
                    let dist = total / contour_dim as f32;
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = tok;
                    }
                }

                token_idx[idx2(local, s, steps)] = best_idx as i64;
                let mut mean_x = 0.0_f32;
                let mut mean_y = 0.0_f32;
                for c in 0..contour_dim {
                    let local_x = token_set.token[idx3(best_idx, c, 0, contour_dim, 2)];
                    let local_y = token_set.token[idx3(best_idx, c, 1, contour_dim, 2)];
                    let world_x = local_x * cos - local_y * sin + prev_pos[idx2(local, 0, 2)];
                    let world_y = local_x * sin + local_y * cos + prev_pos[idx2(local, 1, 2)];
                    token_contour[idx4(local, s, c, 0, steps, contour_dim, 2)] = world_x;
                    token_contour[idx4(local, s, c, 1, steps, contour_dim, 2)] = world_y;
                    mean_x += world_x;
                    mean_y += world_y;
                }
                mean_x /= contour_dim as f32;
                mean_y /= contour_dim as f32;

                let diff_x = token_contour[idx4(local, s, 0, 0, steps, contour_dim, 2)]
                    - token_contour[idx4(local, s, 3, 0, steps, contour_dim, 2)];
                let diff_y = token_contour[idx4(local, s, 0, 1, steps, contour_dim, 2)]
                    - token_contour[idx4(local, s, 3, 1, steps, contour_dim, 2)];

                prev_heading[local] = arrays.heading[idx2(global, frame, arrays.timesteps)];
                if arrays.valid[idx2(global, frame - self.shift, arrays.timesteps)] {
                    prev_heading[local] = diff_y.atan2(diff_x);
                }
                prev_pos[idx2(local, 0, 2)] =
                    arrays.position[idx3(global, frame, 0, arrays.timesteps, 3)];
                prev_pos[idx2(local, 1, 2)] =
                    arrays.position[idx3(global, frame, 1, arrays.timesteps, 3)];
                if arrays.valid[idx2(global, frame - self.shift, arrays.timesteps)] {
                    prev_pos[idx2(local, 0, 2)] = mean_x;
                    prev_pos[idx2(local, 1, 2)] = mean_y;
                }
            }
        }

        if !self.training && steps > 1 {
            for (local, &global) in global_indices.iter().enumerate() {
                if !extra_mask[global] {
                    continue;
                }
                let theta = arrays.heading[idx2(global, self.current_step - 1, arrays.timesteps)];
                let cos = theta.cos();
                let sin = theta.sin();
                let prev_x =
                    arrays.position[idx3(global, self.current_step - 1, 0, arrays.timesteps, 3)];
                let prev_y =
                    arrays.position[idx3(global, self.current_step - 1, 1, arrays.timesteps, 3)];
                let cur_x =
                    arrays.position[idx3(global, self.current_step, 0, arrays.timesteps, 3)];
                let cur_y =
                    arrays.position[idx3(global, self.current_step, 1, arrays.timesteps, 3)];
                let cur_heading = arrays.heading[idx2(global, self.current_step, arrays.timesteps)];
                let cur_contour = polygon_contour(cur_x, cur_y, cur_heading, width, length);

                let mut best_idx = 0_usize;
                let mut best_dist = f32::INFINITY;
                for tok in 0..token_num {
                    let mut total = 0.0_f32;
                    for c in 0..contour_dim {
                        let local_x = token_set.token_last[idx3(tok, c, 0, contour_dim, 2)];
                        let local_y = token_set.token_last[idx3(tok, c, 1, contour_dim, 2)];
                        let world_x = local_x * cos - local_y * sin + prev_x;
                        let world_y = local_x * sin + local_y * cos + prev_y;
                        let dx = cur_contour[idx2(c, 0, 2)] - world_x;
                        let dy = cur_contour[idx2(c, 1, 2)] - world_y;
                        total += (dx * dx + dy * dy).sqrt();
                    }
                    let dist = total / contour_dim as f32;
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = tok;
                    }
                }

                token_idx[idx2(local, 1, steps)] = best_idx as i64;
                for c in 0..contour_dim {
                    let local_x = token_set.token_last[idx3(best_idx, c, 0, contour_dim, 2)];
                    let local_y = token_set.token_last[idx3(best_idx, c, 1, contour_dim, 2)];
                    token_contour[idx4(local, 1, c, 0, steps, contour_dim, 2)] =
                        local_x * cos - local_y * sin + prev_x;
                    token_contour[idx4(local, 1, c, 1, steps, contour_dim, 2)] =
                        local_x * sin + local_y * cos + prev_y;
                }
            }
        }

        MatchOutput {
            token_idx,
            token_contour,
            steps,
        }
    }

    fn tokenize_map<'py>(&self, py: Python<'py>, data: &Bound<'py, PyDict>) -> PyResult<()> {
        let map_polygon = get_dict(data, "map_polygon")?;
        let map_point = get_dict(data, "map_point")?;
        let edge_key = PyTuple::new(py, ["map_point", "to", "map_polygon"])?;
        let pt2pl_dict = data
            .get_item(edge_key)?
            .ok_or_else(|| PyKeyError::new_err("missing point-to-polygon edge"))?
            .downcast_into::<PyDict>()?;

        let (polygon_type, polygon_shape) = tensor_to_u8_vec(&get_item(&map_polygon, "type")?)?;
        let (pt_type, pt_type_shape) = tensor_to_u8_vec(&get_item(&map_point, "type")?)?;
        let (pt_pos, pt_pos_shape) = tensor_to_f32_vec(&get_item(&map_point, "position")?)?;
        let (pt_heading_raw, pt_heading_shape) =
            tensor_to_f32_vec(&get_item(&map_point, "orientation")?)?;
        let (pt2pl, edge_shape) = tensor_to_i64_vec(&get_item(&pt2pl_dict, "edge_index")?)?;

        if polygon_shape.len() != 1
            || pt_type_shape.len() != 1
            || pt_heading_shape.len() != 1
            || pt_pos_shape.len() != 2
            || pt_pos_shape[1] < 2
            || edge_shape.len() != 2
            || edge_shape[0] != 2
        {
            return Err(PyValueError::new_err("unexpected map tensor shape"));
        }

        let point_count = pt_type_shape[0];
        let edge_count = edge_shape[1];
        let mut pt_heading = pt_heading_raw;
        for h in &mut pt_heading {
            *h = wrap_angle(*h);
        }

        map_polygon.set_item("type", make_uint8_1d(py, polygon_type.clone())?)?;
        map_point.set_item("type", make_uint8_1d(py, pt_type.clone())?)?;
        map_point.set_item("orientation", make_f32_1d(py, pt_heading.clone())?)?;

        let mut polygon_ids = BTreeSet::new();
        for e in 0..edge_count {
            polygon_ids.insert(pt2pl[idx2(1, e, edge_count)]);
        }

        let mut split_polyline_pos: Vec<f64> = Vec::new();
        let mut split_polyline_theta: Vec<f64> = Vec::new();
        let mut split_polyline_type: Vec<u8> = Vec::new();
        let mut split_polyline_side: Vec<u8> = Vec::new();
        let mut pl_idx_list: Vec<f32> = Vec::new();
        let mut split_polygon_type: Vec<u8> = Vec::new();

        for polygon_id in polygon_ids {
            let poly_idx = polygon_id as usize;
            if poly_idx >= polygon_type.len() {
                continue;
            }
            let point_indices: Vec<usize> = (0..edge_count)
                .filter_map(|e| {
                    if pt2pl[idx2(1, e, edge_count)] == polygon_id {
                        let point_idx = pt2pl[idx2(0, e, edge_count)] as usize;
                        if point_idx < point_count {
                            Some(point_idx)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .collect();

            let mut type_values = BTreeSet::new();
            for &idx in &point_indices {
                type_values.insert(pt_type[idx]);
            }

            for type_val in type_values {
                if type_val == 13 {
                    continue;
                }
                let indices: Vec<usize> = point_indices
                    .iter()
                    .copied()
                    .filter(|&idx| pt_type[idx] == type_val)
                    .collect();
                if indices.len() <= 2 {
                    continue;
                }

                let mut polyline = Vec::with_capacity(indices.len() * 2);
                let mut heading = Vec::with_capacity(indices.len());
                for idx in indices {
                    polyline.push(pt_pos[idx2(idx, 0, pt_pos_shape[1])] as f64);
                    polyline.push(pt_pos[idx2(idx, 1, pt_pos_shape[1])] as f64);
                    heading.push(pt_heading[idx] as f64);
                }

                let pieces = interpolate_polyline(&polyline, &heading);
                for piece in pieces {
                    for point in &piece.points {
                        split_polyline_pos.push(point[0]);
                        split_polyline_pos.push(point[1]);
                    }
                    split_polyline_theta.push(piece.theta0);
                    split_polyline_type.push(type_val);
                    split_polyline_side.push(0);
                    pl_idx_list.push(polygon_id as f32);
                    split_polygon_type.push(polygon_type[poly_idx]);
                }
            }
        }

        let num_tokens = split_polyline_type.len();
        let map_save = PyDict::new(py);
        map_save.set_item(
            "traj_pos",
            make_f64_3d(py, split_polyline_pos, num_tokens, 3, 2)?,
        )?;
        map_save.set_item("traj_theta", make_f64_1d(py, split_polyline_theta)?)?;
        map_save.set_item("pl_idx_list", make_f32_1d(py, pl_idx_list)?)?;

        let pt_token = PyDict::new(py);
        pt_token.set_item("type", make_uint8_1d(py, split_polyline_type)?)?;
        pt_token.set_item("side", make_uint8_1d(py, split_polyline_side)?)?;
        pt_token.set_item("pl_type", make_uint8_1d(py, split_polygon_type)?)?;
        pt_token.set_item("num_nodes", num_tokens)?;

        data.set_item("map_save", map_save)?;
        data.set_item("pt_token", pt_token)?;
        Ok(())
    }
}

struct MapPiece {
    points: [[f64; 2]; 3],
    theta0: f64,
}

fn interpolate_polyline(polyline: &[f64], heading: &[f64]) -> Vec<MapPiece> {
    let mut dist_parts: Vec<Vec<f64>> = vec![vec![0.0]];
    let mut point_parts: Vec<Vec<[f64; 2]>> = vec![vec![[polyline[0], polyline[1]]]];

    for i in 1..heading.len() {
        let x = polyline[idx2(i, 0, 2)];
        let y = polyline[idx2(i, 1, 2)];
        let px = polyline[idx2(i - 1, 0, 2)];
        let py = polyline[idx2(i - 1, 1, 2)];
        let euclidean_dist = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
        let heading_diff = (heading[i].max(heading[i - 1]) - heading[1].min(heading[i - 1]))
            .abs()
            .min(
                (heading[i].max(heading[i - 1]) - heading[1].min(heading[i - 1])
                    + std::f64::consts::PI)
                    .abs(),
            );

        if (heading_diff > 0.1 && euclidean_dist > 3.0) || euclidean_dist > 10.0 {
            dist_parts.push(vec![0.0]);
            point_parts.push(vec![[x, y]]);
        } else {
            let last = dist_parts.last_mut().expect("non-empty dist parts");
            let new_dist = last.last().copied().unwrap_or(0.0) + euclidean_dist;
            last.push(new_dist);
            point_parts
                .last_mut()
                .expect("non-empty point parts")
                .push([x, y]);
        }
    }

    let mut out = Vec::new();
    for (dist_along_path, points) in dist_parts.iter().zip(point_parts.iter()) {
        if dist_along_path.len() < 2 {
            continue;
        }
        let total_dist = *dist_along_path.last().unwrap_or(&0.0);
        let mut new_dist = Vec::new();
        let mut d = 0.0_f64;
        while d < total_dist {
            new_dist.push(d);
            d += 0.5;
        }
        new_dist.push(total_dist);

        let mut new_polyline = Vec::with_capacity(new_dist.len());
        for target_d in new_dist {
            new_polyline.push(interpolate_point(dist_along_path, points, target_d));
        }
        if new_polyline.len() < 2 {
            continue;
        }

        let mut new_heading = Vec::with_capacity(new_polyline.len());
        for i in 0..new_polyline.len() - 1 {
            let dx = new_polyline[i + 1][0] - new_polyline[i][0];
            let dy = new_polyline[i + 1][1] - new_polyline[i][1];
            new_heading.push(dy.atan2(dx));
        }
        if let Some(&last) = new_heading.last() {
            new_heading.push(last);
        }

        let polyline_size = 10_usize;
        let len = new_polyline.len();
        let (padding_size, final_index) = if len > polyline_size {
            (
                (len - (polyline_size + 1)) % polyline_size,
                (len - (polyline_size + 1)) / polyline_size + 1,
            )
        } else {
            (len, 0)
        };

        if len > polyline_size {
            let mut start = 0_usize;
            while start + polyline_size < len {
                out.push(MapPiece {
                    points: [
                        new_polyline[start],
                        new_polyline[start + 5],
                        new_polyline[start + 10],
                    ],
                    theta0: new_heading[start],
                });
                start += polyline_size;
            }
        }

        if padding_size >= 3 {
            let start = final_index * polyline_size;
            if start < len {
                let last_len = len - start;
                let i0 = start;
                let i1 = start + ((last_len - 1) as f64 / 2.0).floor() as usize;
                let i2 = len - 1;
                out.push(MapPiece {
                    points: [new_polyline[i0], new_polyline[i1], new_polyline[i2]],
                    theta0: new_heading[i0],
                });
            }
        }
    }

    out
}

fn interpolate_point(dist: &[f64], points: &[[f64; 2]], target: f64) -> [f64; 2] {
    if target <= dist[0] {
        return points[0];
    }
    let last_idx = dist.len() - 1;
    if target >= dist[last_idx] {
        return points[last_idx];
    }
    let upper = dist.partition_point(|&d| d < target);
    let lower = upper - 1;
    let span = dist[upper] - dist[lower];
    if span == 0.0 {
        return points[lower];
    }
    let ratio = (target - dist[lower]) / span;
    [
        points[lower][0] + (points[upper][0] - points[lower][0]) * ratio,
        points[lower][1] + (points[upper][1] - points[lower][1]) * ratio,
    ]
}

fn read_agent_arrays(agent: &Bound<'_, PyDict>) -> PyResult<AgentArrays> {
    let (valid, valid_shape) = tensor_to_bool_vec(&get_item(agent, "valid_mask")?)?;
    let (position, position_shape) = tensor_to_f32_vec(&get_item(agent, "position")?)?;
    let (heading, heading_shape) = tensor_to_f32_vec(&get_item(agent, "heading")?)?;
    let (velocity, velocity_shape) = tensor_to_f32_vec(&get_item(agent, "velocity")?)?;
    let (agent_type, type_shape) = tensor_to_u8_vec(&get_item(agent, "type")?)?;

    if valid_shape.len() != 2
        || position_shape.len() != 3
        || heading_shape.len() != 2
        || velocity_shape.len() != 3
        || type_shape.len() != 1
        || position_shape[2] != 3
        || valid_shape != heading_shape
        || valid_shape[0] != position_shape[0]
        || valid_shape[1] != position_shape[1]
        || velocity_shape[0] != position_shape[0]
        || velocity_shape[1] != position_shape[1]
        || type_shape[0] != valid_shape[0]
    {
        return Err(PyValueError::new_err("unexpected agent tensor shape"));
    }

    Ok(AgentArrays {
        valid,
        position,
        heading,
        velocity,
        agent_type,
        num_nodes: valid_shape[0],
        timesteps: valid_shape[1],
        velocity_dim: velocity_shape[2],
    })
}

fn load_token_set(
    token_dict: &Bound<'_, PyDict>,
    token_all_dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<TokenSet> {
    let token_any = token_dict
        .get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing token key {key}")))?;
    let token_arr = token_any.downcast::<PyArrayDyn<f64>>()?.readonly();
    let token_shape = token_arr.shape().to_vec();
    if token_shape.len() != 3 || token_shape[2] != 2 {
        return Err(PyValueError::new_err(format!(
            "unexpected token shape for {key}"
        )));
    }
    let token: Vec<f32> = token_arr.as_slice()?.iter().map(|&v| v as f32).collect();

    let token_all_any = token_all_dict
        .get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing token_all key {key}")))?;
    let token_all_arr = token_all_any.downcast::<PyArrayDyn<f32>>()?.readonly();
    let token_all_shape = token_all_arr.shape().to_vec();
    if token_all_shape.len() != 4
        || token_all_shape[0] != token_shape[0]
        || token_all_shape[2] != token_shape[1]
        || token_all_shape[3] != 2
        || token_all_shape[1] < 2
    {
        return Err(PyValueError::new_err(format!(
            "unexpected token_all shape for {key}"
        )));
    }
    let token_all = token_all_arr.as_slice()?;
    let token_num = token_shape[0];
    let contour_dim = token_shape[1];
    let time_dim = token_all_shape[1];
    let mut token_last = vec![0.0_f32; token_num * contour_dim * 2];

    for tok in 0..token_num {
        let first_time = time_dim - 2;
        let second_time = time_dim - 1;
        let first0_x = token_all[idx4(tok, first_time, 0, 0, time_dim, contour_dim, 2)];
        let first0_y = token_all[idx4(tok, first_time, 0, 1, time_dim, contour_dim, 2)];
        let first3_x = token_all[idx4(tok, first_time, 3, 0, time_dim, contour_dim, 2)];
        let first3_y = token_all[idx4(tok, first_time, 3, 1, time_dim, contour_dim, 2)];
        let theta = (first0_y - first3_y).atan2(first0_x - first3_x);
        let cos = theta.cos();
        let sin = theta.sin();
        let mut mean_x = 0.0_f32;
        let mut mean_y = 0.0_f32;
        for c in 0..contour_dim {
            mean_x += token_all[idx4(tok, first_time, c, 0, time_dim, contour_dim, 2)];
            mean_y += token_all[idx4(tok, first_time, c, 1, time_dim, contour_dim, 2)];
        }
        mean_x /= contour_dim as f32;
        mean_y /= contour_dim as f32;

        for c in 0..contour_dim {
            let x = token_all[idx4(tok, second_time, c, 0, time_dim, contour_dim, 2)];
            let y = token_all[idx4(tok, second_time, c, 1, time_dim, contour_dim, 2)];
            token_last[idx3(tok, c, 0, contour_dim, 2)] = x * cos + y * sin - mean_x;
            token_last[idx3(tok, c, 1, contour_dim, 2)] = -x * sin + y * cos - mean_y;
        }
    }

    Ok(TokenSet {
        token,
        token_last,
        token_num,
        contour_dim,
    })
}

fn load_pickle<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let pickle = py.import("pickle")?;
    let builtins = py.import("builtins")?;
    let fh = builtins.call_method1("open", (path, "rb"))?;
    let obj = pickle.call_method1("load", (&fh,))?;
    fh.call_method0("close")?;
    Ok(obj)
}

fn get_item<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing key {key}")))
}

fn get_dict<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyDict>> {
    Ok(get_item(dict, key)?.downcast_into::<PyDict>()?)
}

fn tensor_to_numpy<'py>(tensor: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    tensor.call_method0("contiguous")?.call_method0("numpy")
}

fn tensor_to_f32_vec(tensor: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, Vec<usize>)> {
    let np = tensor_to_numpy(tensor)?;
    let arr = np.downcast::<PyArrayDyn<f32>>()?.readonly();
    Ok((arr.as_slice()?.to_vec(), arr.shape().to_vec()))
}

fn tensor_to_bool_vec(tensor: &Bound<'_, PyAny>) -> PyResult<(Vec<bool>, Vec<usize>)> {
    let np = tensor_to_numpy(tensor)?;
    let arr = np.downcast::<PyArrayDyn<bool>>()?.readonly();
    Ok((arr.as_slice()?.to_vec(), arr.shape().to_vec()))
}

fn tensor_to_u8_vec(tensor: &Bound<'_, PyAny>) -> PyResult<(Vec<u8>, Vec<usize>)> {
    let np = tensor_to_numpy(tensor)?;
    let arr = np.downcast::<PyArrayDyn<u8>>()?.readonly();
    Ok((arr.as_slice()?.to_vec(), arr.shape().to_vec()))
}

fn tensor_to_i64_vec(tensor: &Bound<'_, PyAny>) -> PyResult<(Vec<i64>, Vec<usize>)> {
    let np = tensor_to_numpy(tensor)?;
    let arr = np.downcast::<PyArrayDyn<i64>>()?.readonly();
    Ok((arr.as_slice()?.to_vec(), arr.shape().to_vec()))
}

macro_rules! as_tensor {
    ($py:expr, $arr:expr) => {{
        let torch = $py.import("torch")?;
        let t = torch.call_method1("from_numpy", ($arr,))?;
        t.call_method0("clone")?
    }};
}

fn make_f32_1d<'py>(py: Python<'py>, data: Vec<f32>) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray1::from_vec(py, data);
    Ok(as_tensor!(py, arr))
}

fn make_f64_1d<'py>(py: Python<'py>, data: Vec<f64>) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray1::from_vec(py, data);
    Ok(as_tensor!(py, arr))
}

fn make_uint8_1d<'py>(py: Python<'py>, data: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray1::from_vec(py, data);
    Ok(as_tensor!(py, arr))
}

fn make_f32_2d<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray2::from_vec2(py, &reshape_2d(data, rows, cols))?;
    Ok(as_tensor!(py, arr))
}

fn make_bool_2d<'py>(
    py: Python<'py>,
    data: Vec<bool>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray2::from_vec2(py, &reshape_2d(data, rows, cols))?;
    Ok(as_tensor!(py, arr))
}

fn make_i64_2d<'py>(
    py: Python<'py>,
    data: Vec<i64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray2::from_vec2(py, &reshape_2d(data, rows, cols))?;
    Ok(as_tensor!(py, arr))
}

fn make_f32_3d<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    d0: usize,
    d1: usize,
    d2: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray3::from_vec3(py, &reshape_3d(data, d0, d1, d2))?;
    Ok(as_tensor!(py, arr))
}

fn make_f64_3d<'py>(
    py: Python<'py>,
    data: Vec<f64>,
    d0: usize,
    d1: usize,
    d2: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray3::from_vec3(py, &reshape_3d(data, d0, d1, d2))?;
    Ok(as_tensor!(py, arr))
}

fn make_f32_4d<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = PyArray1::from_vec(py, data);
    let shape = PyTuple::new(py, [d0, d1, d2, d3])?;
    let arr = arr.call_method1("reshape", (shape,))?;
    Ok(as_tensor!(py, arr))
}

fn reshape_2d<T: Clone>(data: Vec<T>, rows: usize, cols: usize) -> Vec<Vec<T>> {
    assert_eq!(data.len(), rows * cols);
    if cols == 0 {
        return vec![Vec::new(); rows];
    }
    data.chunks(cols).map(|c| c.to_vec()).collect()
}

fn reshape_3d<T: Clone>(data: Vec<T>, d0: usize, d1: usize, d2: usize) -> Vec<Vec<Vec<T>>> {
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

fn clean_heading(heading: &mut [f32], valid: &[bool], num_nodes: usize, timesteps: usize) {
    for a in 0..num_nodes {
        for t in 0..timesteps - 1 {
            if !valid[idx2(a, t, timesteps)] || !valid[idx2(a, t + 1, timesteps)] {
                continue;
            }
            let diff =
                wrap_angle(heading[idx2(a, t, timesteps)] - heading[idx2(a, t + 1, timesteps)]);
            if diff.abs() > 1.0 {
                heading[idx2(a, t + 1, timesteps)] = heading[idx2(a, t, timesteps)];
            }
        }
    }
}

fn polygon_contour(x: f32, y: f32, theta: f32, width: f32, length: f32) -> [f32; 8] {
    let cos = theta.cos();
    let sin = theta.sin();
    [
        x + 0.5 * length * cos - 0.5 * width * sin,
        y + 0.5 * length * sin + 0.5 * width * cos,
        x + 0.5 * length * cos + 0.5 * width * sin,
        y + 0.5 * length * sin - 0.5 * width * cos,
        x - 0.5 * length * cos + 0.5 * width * sin,
        y - 0.5 * length * sin - 0.5 * width * cos,
        x - 0.5 * length * cos - 0.5 * width * sin,
        y - 0.5 * length * sin + 0.5 * width * cos,
    ]
}

fn dimensions_for_kind(kind: AgentKind, shift: usize) -> (f32, f32) {
    if shift <= 2 {
        match kind {
            AgentKind::Vehicle => (1.0, 2.4),
            AgentKind::Cyclist => (0.5, 1.5),
            AgentKind::Pedestrian => (0.5, 0.5),
        }
    } else {
        match kind {
            AgentKind::Vehicle => (2.0, 4.8),
            AgentKind::Cyclist => (1.0, 2.0),
            AgentKind::Pedestrian => (1.0, 1.0),
        }
    }
}

fn token_steps(timesteps: usize, shift: usize) -> usize {
    if timesteps <= shift {
        0
    } else {
        ((timesteps - shift - 1) / shift) + 1
    }
}

fn wrap_angle(v: f32) -> f32 {
    (v + PI).rem_euclid(TWO_PI) - PI
}

fn idx2(i: usize, j: usize, d1: usize) -> usize {
    i * d1 + j
}

fn idx3(i: usize, j: usize, k: usize, d1: usize, d2: usize) -> usize {
    (i * d1 + j) * d2 + k
}

fn idx4(i: usize, j: usize, k: usize, l: usize, d1: usize, d2: usize, d3: usize) -> usize {
    ((i * d1 + j) * d2 + k) * d3 + l
}
