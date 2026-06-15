import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple, Iterable
from torch import Tensor

import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline
from tqdm import tqdm

from smart.utils import wrap_angle


def transform_to_local(
    pos_global: Tensor,  # [n_agent, n_step, 2]
    head_global: Optional[Tensor],  # [n_agent, n_step]
    pos_now: Tensor,  # [n_agent, 2]
    head_now: Tensor,  # [n_agent]
) -> Tuple[Tensor, Optional[Tensor]]:
    cos, sin = head_now.cos(), head_now.sin()
    rot_mat = torch.zeros((head_now.shape[0], 2, 2), device=head_now.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = -sin
    rot_mat[:, 1, 0] = sin
    rot_mat[:, 1, 1] = cos

    pos_local = pos_global - pos_now.unsqueeze(1)
    pos_local = torch.bmm(pos_local, rot_mat)  # [n_agent, n_step, 2]*[n_agent, 2, 2]
    if head_global is None:
        head_local = None
    else:
        head_local = head_global - head_now.unsqueeze(1)
    return pos_local, head_local


def _iter_pickle_files(raw_dirs: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []

    for raw_dir in raw_dirs:
        path = Path(raw_dir).expanduser()

        if path.is_file() and path.suffix == ".pkl":
            files.append(path)
            continue

        files.extend(sorted(path.glob("*.pkl")))

    return files


def _to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


class TrajTokTokenBuilder:
    def __init__(
        self,
        *,
        shift: int = 5,
        max_workers: int = 16,
        max_file_nums: int | None = 50_000,
        max_traj_nums: int | None = 12_000_000,
        flip_trajs: bool = True,
    ):
        self.shift = shift
        self.t = 0.1 * self.shift
        self.agent_classes = ["veh", "ped", "cyc"]
        self.flip_trajs = flip_trajs

        # grid settings
        self.x_max = {"veh": 20, "ped": 4.5, "cyc": 8}
        self.x_min = {"veh": -5, "ped": -1.5, "cyc": -1}
        self.y_max = {"veh": 3, "ped": 2, "cyc": 1}
        self.y_min = {"veh": -2, "ped": -2, "cyc": -1}
        self.x_binnum = {"veh": 250, "ped": 120, "cyc": 180}
        self.y_binnum = {"veh": 80, "ped": 80, "cyc": 40}

        # filter settings
        self.valid_count_threshold = {"veh": 3, "ped": 1, "cyc": 1}
        self.filter_range = {"veh": 4, "ped": 4, "cyc": 4}
        self.filter_threshold_add = {"veh": 20, "ped": 20, "cyc": 20}
        self.filter_threshold_remove = {"veh": 20, "ped": 20, "cyc": 20}

        self.max_workers = max_workers
        self.max_file_nums = max_file_nums
        self.max_traj_nums = max_traj_nums

        self.traj_data: dict[str, np.ndarray] | None = None
        self.vocab: dict | None = None

    def _init_bin_stats(self) -> dict[str, dict[str, np.ndarray]]:
        stats: dict[str, dict[str, np.ndarray]] = {}

        for agent_class in self.agent_classes:
            x_binnum = self.x_binnum[agent_class]
            y_binnum = self.y_binnum[agent_class]

            stats[agent_class] = {
                "count": np.zeros((x_binnum, y_binnum), dtype=np.int64),
                "sum": np.zeros(
                    (x_binnum, y_binnum, self.shift + 1, 3),
                    dtype=np.float64,
                ),
            }

        return stats

    def _update_bin_stats_with_trajs(
        self,
        stats: dict[str, dict[str, np.ndarray]],
        agent_class: str,
        trajs: np.ndarray,
    ) -> None:
        x_binnum = self.x_binnum[agent_class]
        y_binnum = self.y_binnum[agent_class]

        x_min = self.x_min[agent_class]
        x_max = self.x_max[agent_class]

        y_min = self.y_min[agent_class]
        y_max = self.y_max[agent_class]

        grid_end_x = np.round(
            (trajs[:, self.shift, 0] - x_min) / (x_max - x_min) * x_binnum
        ).astype(np.int32)

        grid_end_y = np.round(
            (trajs[:, self.shift, 1] - y_min) / (y_max - y_min) * y_binnum
        ).astype(np.int32)

        mask = (
            (grid_end_x >= 0)
            & (grid_end_x < x_binnum)
            & (grid_end_y >= 0)
            & (grid_end_y < y_binnum)
            & (np.abs(trajs[:, :, 0]).mean(axis=-1) < x_max)
            & (np.abs(trajs[:, :, 1]).mean(axis=-1) < y_max)
        )

        grid_end_x = grid_end_x[mask]
        grid_end_y = grid_end_y[mask]
        trajs = trajs[mask]

        for i in range(len(trajs)):
            x = grid_end_x[i]
            y = grid_end_y[i]
            stats[agent_class]["count"][x, y] += 1
            stats[agent_class]["sum"][x, y] += trajs[i]

    def _update_bin_stats_for_class(
        self,
        stats: dict[str, dict[str, np.ndarray]],
        agent_class: str,
        agent_trajs: np.ndarray,
    ) -> None:
        if agent_trajs.shape[0] == 0:
            return

        origin = np.zeros(
            (agent_trajs.shape[0], 1, 3),
            dtype=agent_trajs.dtype,
        )
        trajs = np.concatenate([origin, agent_trajs], axis=1)

        if self.max_traj_nums is not None:
            current_count = int(stats[agent_class]["count"].sum())
            remaining = self.max_traj_nums - current_count

            if remaining <= 0:
                return

            trajs = trajs[:remaining]

        self._update_bin_stats_with_trajs(stats, agent_class, trajs)

        if self.flip_trajs:
            flipped_trajs = trajs.copy()
            flipped_trajs[:, :, 1] = -flipped_trajs[:, :, 1]
            flipped_trajs[:, :, 2] = -flipped_trajs[:, :, 2]
            self._update_bin_stats_with_trajs(stats, agent_class, flipped_trajs)

    def collect_bin_stats(
        self,
        raw_dirs: Iterable[str | Path],
    ) -> dict[str, dict[str, np.ndarray]]:
        stats = self._init_bin_stats()

        file_paths = _iter_pickle_files(raw_dirs)

        if self.max_file_nums is not None:
            file_paths = file_paths[: self.max_file_nums]
        print(f"Found {len(file_paths)} TrajTok source files.")
        print(
            "TrajTok settings: "
            f"shift={self.shift}, "
            f"max_workers={self.max_workers}, "
            f"max_file_nums={self.max_file_nums}, "
            f"max_traj_nums={self.max_traj_nums}, "
            f"flip_trajs={self.flip_trajs}"
        )

        def print_progress(processed_files: int) -> None:
            counts = {
                agent_class: int(stats[agent_class]["count"].sum())
                for agent_class in self.agent_classes
            }

            print(
                f"Processed {processed_files}/{len(file_paths)} files. "
                f"traj counts: veh={counts['veh']}, "
                f"ped={counts['ped']}, "
                f"cyc={counts['cyc']}"
            )

        def update_from_result(result: dict[str, list[torch.Tensor]]) -> None:
            for agent_class in self.agent_classes:
                non_empty = [
                    x
                    for x in result[agent_class]
                    if isinstance(x, torch.Tensor) and x.numel() > 0
                ]

                if not non_empty:
                    continue

                values = torch.cat(non_empty).cpu().numpy().astype(np.float32)
                self._update_bin_stats_for_class(stats, agent_class, values)

        if self.max_workers == 0:
            for processed_files, file_path in enumerate(
                tqdm(
                    file_paths,
                    desc="Extracting TrajTok trajectory data",
                ),
                start=1,
            ):
                result = self._get_traj_data(file_path)
                update_from_result(result)

                if processed_files % 5000 == 0:
                    print_progress(processed_files)
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for processed_files, result in enumerate(
                    tqdm(
                        executor.map(self._get_traj_data, file_paths),
                        total=len(file_paths),
                        desc="Extracting TrajTok trajectory data",
                    ),
                    start=1,
                ):
                    update_from_result(result)

                    if processed_files % 5000 == 0:
                        print_progress(processed_files)

        for agent_class in self.agent_classes:
            print(
                f"traj num of {agent_class}: {int(stats[agent_class]['count'].sum())}"
            )

        return stats

    def _get_traj_data(self, file_path: str | Path) -> dict[str, list[torch.Tensor]]:
        with Path(file_path).open("rb") as f:
            data = pickle.load(f)

        pos = _to_tensor(data["agent"]["position"])[..., 0:2]
        masks = _to_tensor(data["agent"]["valid_mask"]).bool()
        types = _to_tensor(data["agent"]["type"]).long()
        headings = wrap_angle(_to_tensor(data["agent"]["heading"]))

        _, n_step, _ = pos.shape

        traj_data: dict[str, list[torch.Tensor]] = {
            "veh": [],
            "ped": [],
            "cyc": [],
        }

        for i in range(0, n_step - self.shift, self.shift):
            pos_local, head_local = transform_to_local(
                pos_global=pos[:, i + 1 : i + self.shift + 1],
                head_global=headings[:, i + 1 : i + self.shift + 1],
                pos_now=pos[:, i],
                head_now=headings[:, i],
            )
            assert head_local is not None

            trajs = torch.cat([pos_local, head_local.unsqueeze(-1)], dim=-1)
            valid_mask = masks[:, i : i + self.shift + 1].all(dim=-1)

            headings_local = trajs[:, :, -1]
            heading_diffs = torch.abs(
                wrap_angle(headings_local[:, 1:] - headings_local[:, :-1])
            )
            head_valid = heading_diffs.max(dim=-1).values < 30 * np.pi / 180

            final_mask = valid_mask & head_valid

            traj_data["veh"].append(trajs[(types == 0) & final_mask])
            traj_data["ped"].append(trajs[(types == 1) & final_mask])
            traj_data["cyc"].append(trajs[(types == 2) & final_mask])

        return traj_data

    def cal_polygon_contour(
        self,
        pos,
        head,
        width_length,
    ):
        x, y = pos[..., 0], pos[..., 1]
        width, length = width_length[..., 0], width_length[..., 1]

        half_cos = np.cos(head) * 0.5
        half_sin = np.sin(head) * 0.5

        length_cos = length * half_cos
        length_sin = length * half_sin
        width_cos = width * half_cos
        width_sin = width * half_sin

        left_front_x = x + length_cos - width_sin
        left_front_y = y + length_sin + width_cos
        left_front = np.stack((left_front_x, left_front_y), axis=-1)

        right_front_x = x + length_cos + width_sin
        right_front_y = y + length_sin - width_cos
        right_front = np.stack((right_front_x, right_front_y), axis=-1)

        right_back_x = x - length_cos + width_sin
        right_back_y = y - length_sin - width_cos
        right_back = np.stack((right_back_x, right_back_y), axis=-1)

        left_back_x = x - length_cos - width_sin
        left_back_y = y - length_sin + width_cos
        left_back = np.stack((left_back_x, left_back_y), axis=-1)

        polygon_contour = np.stack(
            (left_front, right_front, right_back, left_back),
            axis=-2,
        )

        return polygon_contour

    def interpolate_curve(
        self,
        x,
        y,
        theta,
        weight_factor0=0,
        weight_factor1=0,
        num_points=None,
    ):
        if num_points is None:
            num_points = self.shift + 1

        p0 = np.array([0, 0])
        p1 = np.array([x, y])
        dist = np.linalg.norm(p1 - p0)

        t0 = np.array([1, 0]) * dist * weight_factor0
        t1 = np.array([np.cos(theta), np.sin(theta)]) * dist * weight_factor1

        t_vals = [0, 1]
        points = np.vstack((p0, p1))
        tangents = np.vstack((t0, t1))

        spline = CubicHermiteSpline(t_vals, points, tangents)
        t_curve = np.linspace(0, 1, num_points)

        derivatives = spline(t_curve, nu=1)
        xys = spline(t_curve)
        headings = np.arctan2(derivatives[:, 1], derivatives[:, 0])

        curve_points = np.concatenate([xys, headings[:, None]], axis=-1)

        return curve_points

    def get_nearest_traj_from_stats(
        self,
        x,
        y,
        grid_mask,
        traj_sum,
        grid_mask_count,
    ):
        valid_pos = np.argwhere(grid_mask)

        if len(valid_pos) == 0:
            raise ValueError(
                "No valid grid positions available for nearest trajectory lookup."
            )

        distances = np.abs(valid_pos[:, 0] - x) + np.abs(valid_pos[:, 1] - y)
        nearest_idx = np.argmin(distances)
        nearest_x, nearest_y = valid_pos[nearest_idx]

        return traj_sum[nearest_x, nearest_y] / grid_mask_count[nearest_x, nearest_y]

    def build_vocab_from_bin_stats(
        self,
        stats: dict[str, dict[str, np.ndarray]],
    ) -> dict:
        vocab: dict = {
            "token": {},
            "traj": {},
            "token_all": {},
            "grid_mask": {},
            "grid_mask_filtered": {},
            "raw_ep": {},
        }

        for agent_class in self.agent_classes:
            x_binnum = self.x_binnum[agent_class]
            y_binnum = self.y_binnum[agent_class]

            x_min = self.x_min[agent_class]
            x_max = self.x_max[agent_class]

            y_min = self.y_min[agent_class]
            y_max = self.y_max[agent_class]

            filter_range = self.filter_range[agent_class]
            filter_threshold_add = self.filter_threshold_add[agent_class]
            filter_threshold_remove = self.filter_threshold_remove[agent_class]
            valid_count_threshold = self.valid_count_threshold[agent_class]

            grid_mask_count = stats[agent_class]["count"]
            traj_sum = stats[agent_class]["sum"]

            if grid_mask_count.sum() == 0:
                raise ValueError(
                    f"No valid {agent_class} trajectories found for TrajTok vocab build."
                )

            raw_eps = []

            for x in range(x_binnum):
                for y in range(y_binnum):
                    raw_eps.append(
                        [
                            x * (x_max - x_min) / x_binnum + x_min,
                            y * (y_max - y_min) / y_binnum + y_min,
                        ]
                    )

            vocab["raw_ep"][agent_class] = np.array(raw_eps)

            grid_mask = grid_mask_count >= valid_count_threshold
            grid_mask_filtered = grid_mask.copy()

            for x in range(x_binnum):
                for y in range(y_binnum):
                    neighbors = grid_mask[
                        max(0, x - filter_range) : min(x_binnum, x + filter_range + 1),
                        max(0, y - filter_range) : min(y_binnum, y + filter_range + 1),
                    ]

                    if grid_mask[x, y] and neighbors.sum() < filter_threshold_remove:
                        grid_mask_filtered[x, y] = False

                    if not grid_mask[x, y] and neighbors.sum() > filter_threshold_add:
                        grid_mask_filtered[x, y] = True

            token_trajs = []

            valid_mean_mask = grid_mask & grid_mask_filtered

            for x in range(x_binnum):
                for y in range(y_binnum):
                    if not grid_mask_filtered[x, y]:
                        continue

                    grid_end_x_value = x * (x_max - x_min) / x_binnum + x_min
                    grid_end_y_value = y * (y_max - y_min) / y_binnum + y_min

                    if grid_mask[x, y]:
                        token_traj = traj_sum[x, y] / grid_mask_count[x, y]
                        token_traj = token_traj.astype(np.float32)
                        token_traj[-1, 0] = grid_end_x_value
                        token_traj[-1, 1] = grid_end_y_value

                        yaws = token_traj[:, -1]

                        if np.abs(yaws[1:] - yaws[:-1]).max() > 10 * np.pi / 180:
                            nearest_traj = self.get_nearest_traj_from_stats(
                                x,
                                y,
                                valid_mean_mask,
                                traj_sum,
                                grid_mask_count,
                            )

                            token_traj = self.interpolate_curve(
                                grid_end_x_value,
                                grid_end_y_value,
                                nearest_traj[-1, 2],
                            )
                    else:
                        nearest_traj = self.get_nearest_traj_from_stats(
                            x,
                            y,
                            valid_mean_mask,
                            traj_sum,
                            grid_mask_count,
                        )

                        token_traj = self.interpolate_curve(
                            grid_end_x_value,
                            grid_end_y_value,
                            nearest_traj[-1, 2],
                        )

                    token_trajs.append(token_traj)

            if not token_trajs:
                raise ValueError(
                    f"No TrajTok tokens generated for agent class {agent_class}."
                )

            token_trajs = np.stack(token_trajs).astype(np.float32)

            if agent_class == "veh":
                width_length = np.array([2.0, 4.8])
            elif agent_class == "ped":
                width_length = np.array([1.0, 1.0])
            elif agent_class == "cyc":
                width_length = np.array([1.0, 2.0])
            else:
                raise ValueError(f"Unknown agent class: {agent_class}")

            token_contour = self.cal_polygon_contour(
                token_trajs[:, :, 0:2],
                token_trajs[:, :, 2],
                width_length=width_length,
            )

            token = token_contour[:, -1, :, :]

            vocab["traj"][agent_class] = token_trajs
            vocab["token"][agent_class] = token
            vocab["token_all"][agent_class] = token_contour
            vocab["grid_mask"][agent_class] = grid_mask
            vocab["grid_mask_filtered"][agent_class] = grid_mask_filtered

            print(agent_class, token_contour.shape)

        self.vocab = vocab
        return vocab


def build_trajtok_agent_tokens(
    raw_dirs: Iterable[str | Path],
    output_path: str | Path,
    shift: int = 5,
    max_file_nums: int | None = 50_000,
    max_traj_nums: int | None = 12_000_000,
    max_workers: int = 16,
    use_traj_cache: bool = False,
    traj_cache_path: str | Path | None = None,
    flip_trajs: bool = True,
) -> dict:
    builder = TrajTokTokenBuilder(
        shift=shift,
        max_workers=max_workers,
        max_file_nums=max_file_nums,
        max_traj_nums=max_traj_nums,
        flip_trajs=flip_trajs,
    )

    if use_traj_cache:
        raise NotImplementedError(
            "use_traj_cache=True is not supported with streaming TrajTok bin stats yet."
        )

    stats = builder.collect_bin_stats(raw_dirs)

    for agent_class in builder.agent_classes:
        if stats[agent_class]["count"].sum() == 0:
            raise ValueError(
                f"No valid {agent_class} trajectories found for TrajTok token build"
            )

    vocab = builder.build_vocab_from_bin_stats(stats)
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("wb") as f:
        pickle.dump(vocab, f)

    print(f"Wrote TrajTok agent tokens to {output}")

    return vocab
