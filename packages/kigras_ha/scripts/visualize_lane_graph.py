"""
Visualize the KiGRAS lane graph.

Usage:
    python scripts/visualize_lane_graph.py [--scenario_id ID] [--dataroot path]
    python scripts/visualize_lane_graph.py --side-by-side  # single combined figure
"""

import matplotlib

matplotlib.use("Agg")
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kigras_ha.dataloader.dataset import ArgoDataset
from av2.geometry.interpolate import compute_midpoint_line
from av2.map.lane_segment import LaneType, LaneMarkType
from av2.map.map_api import ArgoverseStaticMap


LANE_TYPE_COLORS = {
    LaneType.VEHICLE: "tab:blue",
    LaneType.BUS: "tab:orange",
    LaneType.BIKE: "tab:green",
}
LANE_TYPE_LABELS = {
    LaneType.VEHICLE: "VEHICLE",
    LaneType.BUS: "BUS",
    LaneType.BIKE: "BIKE",
}

MARK_IS_YELLOW = {
    LaneMarkType.SOLID_YELLOW,
    LaneMarkType.DASHED_YELLOW,
    LaneMarkType.DOUBLE_SOLID_YELLOW,
    LaneMarkType.DOUBLE_DASH_YELLOW,
    LaneMarkType.DASH_SOLID_YELLOW,
    LaneMarkType.SOLID_DASH_YELLOW,
}
MARK_IS_DASHED = {
    LaneMarkType.DASHED_WHITE,
    LaneMarkType.DASHED_YELLOW,
    LaneMarkType.DOUBLE_DASH_WHITE,
    LaneMarkType.DOUBLE_DASH_YELLOW,
    LaneMarkType.DASH_SOLID_WHITE,
    LaneMarkType.DASH_SOLID_YELLOW,
    LaneMarkType.SOLID_DASH_WHITE,
    LaneMarkType.SOLID_DASH_YELLOW,
}


def _mark_color(mt):
    return "gold" if mt in MARK_IS_YELLOW else "white"


def _mark_style(mt):
    return (0, (6, 4)) if mt in MARK_IS_DASHED else "solid"


_BG = "#1a1a1a"


def draw_map(
    ax,
    K,
    bounds_left,
    bounds_right,
    centerlines,
    lane_type_raw,
    left_mark_raw,
    right_mark_raw,
):
    """Draw physical map layout on *ax*."""
    ax.set_facecolor(_BG)
    for i in range(K):
        lt = lane_type_raw[i]
        lc = LANE_TYPE_COLORS.get(lt, "gray")
        lb = bounds_left[i]
        rb = bounds_right[i]
        poly = np.vstack([lb, rb[::-1]])
        ax.fill(poly[:, 0], poly[:, 1], color=lc, alpha=0.2, edgecolor="none")

        lmt = left_mark_raw[i]
        rmt = right_mark_raw[i]
        ax.plot(
            lb[:, 0],
            lb[:, 1],
            color=_mark_color(lmt),
            alpha=0.8,
            linestyle=_mark_style(lmt),
            linewidth=1.5,
        )
        ax.plot(
            rb[:, 0],
            rb[:, 1],
            color=_mark_color(rmt),
            alpha=0.8,
            linestyle=_mark_style(rmt),
            linewidth=1.5,
        )

        cl = centerlines[i]
        ax.plot(
            cl[:, 0],
            cl[:, 1],
            color="gray",
            linestyle=(0, (4, 6)),
            linewidth=1,
            alpha=0.5,
        )

        mid = len(cl) // 2
        ax.annotate(
            f"S{i}",
            cl[mid],
            fontsize=8,
            alpha=0.7,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )

    for lt, c in LANE_TYPE_COLORS.items():
        ax.fill([], [], color=c, alpha=0.3, label=LANE_TYPE_LABELS.get(lt, str(lt)))
    legend_elements = [
        Line2D([0], [0], color="white", linewidth=2, label="SOLID_WHITE"),
        Line2D([0], [0], color="gold", linewidth=2, label="SOLID_YELLOW"),
        Line2D(
            [0], [0], color="white", linewidth=2, linestyle=(0, (6, 4)), label="DASHED"
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=1,
            linestyle=(0, (4, 6)),
            label="centerline",
        ),
    ]
    leg = ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    for text in leg.get_texts():
        text.set_color("white")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.3, color="gray")


def draw_nodes(ax, K, centerlines, pos, dvec, edge_idx, edge_type):
    """Draw node graph on *ax*."""
    ax.set_facecolor(_BG)
    for i in range(K):
        cl = centerlines[i]
        ax.plot(cl[:, 0], cl[:, 1], color="gray", linewidth=1.2, alpha=0.25, zorder=1)

    fwd_mask = edge_type == 0
    if fwd_mask.any():
        for f, t in edge_idx[fwd_mask]:
            ax.annotate(
                "",
                xy=pos[t],
                xytext=pos[f],
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2, alpha=0.5),
                zorder=2,
            )

    for et, color, label in [
        (1, "tab:green", "left neighbor (->)"),
        (2, "tab:red", "right neighbor (->)"),
    ]:
        mask = edge_type == et
        if not mask.any():
            continue
        for f, t in edge_idx[mask]:
            ax.annotate(
                "",
                xy=pos[t],
                xytext=pos[f],
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.6),
                zorder=3,
            )
        ax.plot([], [], color=color, linewidth=2.5, label=label)

    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        color="tab:blue",
        s=12,
        zorder=4,
        edgecolors="k",
        linewidths=0.3,
    )

    zero_vec = np.all(dvec == 0, axis=1)
    ax.scatter(
        pos[zero_vec, 0],
        pos[zero_vec, 1],
        color="orange",
        s=60,
        zorder=5,
        marker="o",
        edgecolors="k",
        linewidths=0.8,
        label="segment end",
    )

    leg = ax.legend(fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color("white")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.3, color="gray")


def main():
    parser = argparse.ArgumentParser(description="Visualize KiGRAS lane graph")
    parser.add_argument("--scenario_id", type=str, default=None)
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="Render both plots side-by-side in a single figure",
    )
    args = parser.parse_args()

    kigras_root = Path(__file__).resolve().parent.parent
    dataroot = args.dataroot or (kigras_root / "data" / "train")
    dataroot = Path(dataroot)
    if not dataroot.exists():
        print(f"Data directory not found: {dataroot}")
        sys.exit(1)

    if args.scenario_id:
        scenario_dir = dataroot / args.scenario_id
    else:
        entries = sorted(os.listdir(dataroot))
        if not entries:
            print(f"No scenario directories found in {dataroot}")
            sys.exit(1)
        scenario_dir = dataroot / entries[0]

    file_id = scenario_dir.name
    map_path = scenario_dir / f"log_map_archive_{file_id}.json"
    if not map_path.exists():
        print(f"Missing map: {map_path}")
        sys.exit(1)

    # Load graph via dataset
    config = {"root": str(dataroot), "preprocess": False}
    ds = ArgoDataset(config, train=True)
    idx = ds.filenames.index(file_id)
    data = ds[idx]
    graph = data["graph"]

    pos = graph["position"]
    dvec = graph["direction_vector"]
    graph["segment_idx"]
    graph["lane_type"]
    edge_idx = graph["edge_index"]
    edge_type = graph["edge_type"]
    K = graph["num_segments"]

    # Reload AVM for geometry
    avm = ArgoverseStaticMap.from_json(Path(map_path))
    lane_ids = list(avm.get_scenario_lane_segment_ids())
    bounds_left, bounds_right, centerlines = [], [], []
    lane_type_raw = []
    left_mark_raw = []
    right_mark_raw = []
    for lid in lane_ids:
        seg = avm.vector_lane_segments[lid]
        ctr, _ = compute_midpoint_line(
            seg.left_lane_boundary.xyz, seg.right_lane_boundary.xyz
        )
        bounds_left.append(seg.left_lane_boundary.xyz[:, :2])
        bounds_right.append(seg.right_lane_boundary.xyz[:, :2])
        centerlines.append(ctr[:, :2])
        lane_type_raw.append(seg.lane_type)
        left_mark_raw.append(seg.left_mark_type)
        right_mark_raw.append(seg.right_mark_type)

    print(f"Scenario: {data['scenario_id']}")
    print(f"  Segments: {K}, Nodes: {graph['num_nodes']}, Edges: {graph['num_edges']}")
    for et, label in [(0, "forward"), (1, "left neighbor"), (2, "right neighbor")]:
        count = int((edge_type == et).sum())
        print(f"    {label}: {count}")

    # Plot
    out_dir = kigras_root / "visualization"
    out_dir.mkdir(exist_ok=True)
    dpi = 500

    if args.side_by_side:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(36, 14), facecolor=_BG)
        fig.suptitle(f"Lane Graph - {data['scenario_id']}", fontsize=18, color="white")
        draw_map(
            ax1,
            K,
            bounds_left,
            bounds_right,
            centerlines,
            lane_type_raw,
            left_mark_raw,
            right_mark_raw,
        )
        ax1.set_title("Physical Map", color="white", fontsize=14)
        draw_nodes(ax2, K, centerlines, pos, dvec, edge_idx, edge_type)
        ax2.set_title("Nodes & Edges", color="white", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "side_by_side.png", dpi=dpi, bbox_inches="tight")
        print(f"Saved side-by-side to {out_dir}/side_by_side.png")
    else:
        fig1, ax1 = plt.subplots(figsize=(20, 14), facecolor=_BG)
        fig1.suptitle(
            "Physical Map Layout - Lane Types & Markings", fontsize=16, color="white"
        )
        draw_map(
            ax1,
            K,
            bounds_left,
            bounds_right,
            centerlines,
            lane_type_raw,
            left_mark_raw,
            right_mark_raw,
        )
        fig1.tight_layout()
        fig1.savefig(out_dir / "01_physical_map.png", dpi=dpi, bbox_inches="tight")

        fig2, ax2 = plt.subplots(figsize=(22, 16), facecolor=_BG)
        fig2.suptitle(
            "Nodes, Forward Paths & Neighbor Connections", fontsize=16, color="white"
        )
        draw_nodes(ax2, K, centerlines, pos, dvec, edge_idx, edge_type)
        fig2.tight_layout()
        fig2.savefig(out_dir / "02_nodes_and_edges.png", dpi=dpi, bbox_inches="tight")

        print(f"Saved figures to {out_dir}/")


if __name__ == "__main__":
    main()
