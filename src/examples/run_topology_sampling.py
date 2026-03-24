"""
Interactive Spring-Mass Topology Explorer  —  Scalable N×N
===========================================================
Works for any N >= 3.

Validity rules
--------------
  1. All perimeter edges are FIXED (the outer ring is always present).
  2. Each inner node must keep at least 2 active springs; if exactly 2,
     they must be colinear: {left, right} or {up, down}.
  3. Global connectivity is automatically guaranteed by rules 1 + 2.

Descriptor
----------
  Only average path length (L_avg) is computed.
  Topologies are clustered by finding the 4 largest gaps between
  consecutive sorted L_avg values and cutting there, so boundaries always
  fall in the actual empty spaces between point clouds.  Clusters are sorted by ascending mean
  L_avg so Cluster 1 = shortest paths, Cluster 5 = longest.

Sampling strategy
-----------------
  "Next Sample" draws from clusters round-robin (one topology per cluster,
  cycling when n_sample > 5), so every sample spans the full L_avg range.
  "LHS Sample" does one strict draw (one per cluster, always 5 topologies).
  "All" shows every valid topology.

Usage
-----
  python topology_explorer.py            # default N = 4
  python topology_explorer.py --N 5      # 5x5 grid
  python topology_explorer.py --N 3      # 3x3 grid

Controls
--------
  Next Sample  —  cluster-stratified random sample (seed increments each click)
  LHS Sample   —  one topology per cluster (strict, always 5 results)
  All          —  display every valid topology
  Reset        —  clear all selections, reset seed counter to -1
  N slider     —  how many topologies to draw per sample
"""

import argparse
import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

C_ACTUATOR = "#E8760A"
C_PILLAR   = "#C0392B"
C_INNER    = "#444444"
C_INTER_ABSENT = "#CCCCCC"

# 5 bin colours: low L_avg (blue) → high L_avg (red)
BIN_COLORS = ["#1565C0", "#2E7D32", "#F9A825", "#E65100", "#B71C1C"]
BIN_LABELS = ["C1\n(shortest)", "C2", "C3", "C4",
              "C5\n(longest)"]


# ─────────────────────────────────────────────────────────────────────────────
#  GRID BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(N: int) -> dict:
    n_nodes = N * N
    n_max   = N - 1

    edges = []
    for r in range(N):
        for c in range(N):
            i = r * N + c
            if c < n_max:
                edges.append((i, i + 1))
            if r < n_max:
                edges.append((i, i + N))
    edges   = np.array(edges, dtype=int)
    n_edges = len(edges)

    node_pos = np.array([[c, N - 1 - r]
                         for r in range(N) for c in range(N)], dtype=float)

    def on_perim(nd):
        r, c = nd // N, nd % N
        return r == 0 or r == n_max or c == 0 or c == n_max

    inner_nodes = [nd for nd in range(n_nodes) if not on_perim(nd)]

    perim_edges, inter_edges = [], []
    for ei, (u, v) in enumerate(edges):
        (perim_edges if on_perim(u) and on_perim(v) else inter_edges).append(ei)

    inner_springs = {}
    for nd in inner_nodes:
        nbr_to_dir = {nd - 1: "left", nd + 1: "right",
                      nd - N: "up",   nd + N: "down"}
        dir_map = {}
        for ei, (u, v) in enumerate(edges):
            other = v if u == nd else (u if v == nd else None)
            if other is not None and other in nbr_to_dir:
                dir_map[nbr_to_dir[other]] = ei
        inner_springs[nd] = dir_map

    corners  = [0, n_max, N * n_max, N * N - 1]
    actuator = 0
    pillars  = corners[1:]

    return dict(
        N=N, n_nodes=n_nodes, edges=edges, n_edges=n_edges,
        node_pos=node_pos, inner_nodes=inner_nodes,
        inner_springs=inner_springs, perim_edges=perim_edges,
        inter_edges=inter_edges, on_perim=on_perim,
        actuator=actuator, pillars=pillars, corners=corners,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def satisfies_rules(removed_set: set, grid: dict) -> bool:
    for nd, dir_map in grid["inner_springs"].items():
        active_dirs = {d for d, ei in dir_map.items() if ei not in removed_set}
        n_active = len(active_dirs)
        if n_active < 2:
            return False
        if n_active == 2:
            if active_dirs not in ({"left", "right"}, {"up", "down"}):
                return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  DESCRIPTOR  (L_avg only)
# ─────────────────────────────────────────────────────────────────────────────

def compute_lavg(active_idx: list, grid: dict) -> float:
    """Return average shortest path length for this topology."""
    G = nx.Graph()
    G.add_nodes_from(range(grid["n_nodes"]))
    for ei in active_idx:
        G.add_edge(int(grid["edges"][ei, 0]), int(grid["edges"][ei, 1]))
    return float(nx.average_shortest_path_length(G))


# ─────────────────────────────────────────────────────────────────────────────
#  BIN CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

N_BINS = 5

def classify_bins(Lavgs: np.ndarray):
    """
    Cluster topologies by finding the N_BINS-1 largest gaps between
    consecutive sorted L_avg values, then cutting there.

    This is gap-based clustering: boundaries fall in the actual empty
    spaces between point clouds rather than splitting dense regions the
    way K-Means does.  Clusters are naturally ordered (0 = shortest).

    Returns
    -------
    bins       : int array of shape (n_topos,), values in {0, ..., N_BINS-1}
    thresholds : float array of shape (N_BINS+1,) — the cut points
                 (midpoints of the largest gaps, plus min/max sentinels).
    """
    sorted_vals = np.sort(np.unique(Lavgs))

    if len(sorted_vals) <= N_BINS:
        # Degenerate case: fewer unique values than clusters
        bins = np.searchsorted(sorted_vals, Lavgs)
        bins = np.clip(bins, 0, N_BINS - 1)
        thresholds = np.concatenate([[Lavgs.min()], sorted_vals, [Lavgs.max() + 1e-9]])
        return bins, thresholds

    # Gaps between consecutive unique values
    gaps = np.diff(sorted_vals)                          # shape (n_unique-1,)
    cut_indices = np.argsort(gaps)[-(N_BINS - 1):]       # N_BINS-1 largest gaps
    cut_indices = np.sort(cut_indices)                   # keep left-to-right order

    # Cut points sit in the middle of each gap
    cut_points = (sorted_vals[cut_indices] + sorted_vals[cut_indices + 1]) / 2
    thresholds = np.concatenate([[Lavgs.min()], cut_points, [Lavgs.max() + 1e-9]])

    bins = np.digitize(Lavgs, thresholds) - 1
    bins = np.clip(bins, 0, N_BINS - 1)
    return bins, thresholds


# ─────────────────────────────────────────────────────────────────────────────
#  ENUMERATION
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_topologies(grid: dict):
    inter = grid["inter_edges"]
    N     = grid["N"]
    print(f"\nEnumerating valid topologies for {N}×{N} grid …")
    print(f"  Fixed perimeter edges  : {len(grid['perim_edges'])}")
    print(f"  Variable interior edges: {len(inter)}")
    print()

    all_active, Lavgs, n_sp, removed_list = [], [], [], []
    total = 0

    for n_rem in range(len(inter) + 1):
        level = 0
        for combo in combinations(inter, n_rem):
            removed_set = set(combo)
            if not satisfies_rules(removed_set, grid):
                continue
            active_idx = [i for i in range(grid["n_edges"])
                          if i not in removed_set]
            la = compute_lavg(active_idx, grid)
            all_active.append(active_idx)
            Lavgs.append(la)
            n_sp.append(len(active_idx))
            removed_list.append(sorted(removed_set))
            level += 1
            total += 1
        print(f"  {n_rem} interior edge(s) removed: {level:5d}  "
              f"(running total: {total})")
        if level == 0 and n_rem > len(grid["inter_edges"]):
            break

    Lavgs = np.array(Lavgs)
    bins, thresholds = classify_bins(Lavgs)

    print(f"\n  ✓  Total valid topologies: {total}")
    print(f"\n  L_avg range: [{Lavgs.min():.4f}, {Lavgs.max():.4f}]")
    print(f"  K-Means cluster boundaries: {np.round(thresholds, 4)}")
    for b in range(N_BINS):
        count = int((bins == b).sum())
        print(f"    Cluster {b+1}  [{thresholds[b]:.4f}, {thresholds[b+1]:.4f})  "
              f"→  {count} topologies")
    print()

    return all_active, Lavgs, np.array(n_sp), removed_list, bins, thresholds


# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLING  (bin-stratified)
# ─────────────────────────────────────────────────────────────────────────────

def bin_stratified_sample(n: int, seed: int, bins: np.ndarray) -> list:
    """
    Draw `n` topologies by cycling through bins 0→4→0→… and picking one
    topology uniformly at random from each bin in turn.
    Non-empty bins only; if a bin is empty it is skipped.
    """
    rng = np.random.default_rng(seed)
    bin_indices = [np.where(bins == b)[0] for b in range(N_BINS)]
    non_empty   = [b for b in range(N_BINS) if len(bin_indices[b]) > 0]

    result = []
    cycle  = 0
    while len(result) < n:
        b   = non_empty[cycle % len(non_empty)]
        idx = int(rng.choice(bin_indices[b]))
        if idx not in result:           # avoid duplicates within small grids
            result.append(idx)
        cycle += 1
        if cycle > n * N_BINS * 2:     # safety break if too many collisions
            break
    return result


def lhs_bin_sample(seed: int, bins: np.ndarray) -> list:
    """
    Strict LHS: exactly one topology per non-empty bin, chosen at random.
    """
    rng = np.random.default_rng(seed)
    result = []
    for b in range(N_BINS):
        idxs = np.where(bins == b)[0]
        if len(idxs) > 0:
            result.append(int(rng.choice(idxs)))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  NODE / DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE_HUES = np.arange(40) * 137.508
_PALETTE = [
    "#{:02x}{:02x}{:02x}".format(
        int(matplotlib.cm.hsv(h / 360)[0] * 190),
        int(matplotlib.cm.hsv(h / 360)[1] * 190),
        int(matplotlib.cm.hsv(h / 360)[2] * 190),
    )
    for h in _PALETTE_HUES % 360
]


def node_style(nd: int, grid: dict, sel_color: str):
    if nd == grid["actuator"]:
        return C_ACTUATOR, "D", 12
    if nd in grid["pillars"]:
        return C_PILLAR, "s", 12
    if nd in grid["inner_nodes"]:
        return C_INNER, "o", 10
    return sel_color, "o", 10


def draw_topology(ax, active_idx, removed, grid, title="", sel_color="#222222"):
    N   = grid["N"]
    pos = grid["node_pos"]
    act_set = set(active_idx)

    ax.set_xlim(-0.65, N - 0.35)
    ax.set_ylim(-0.85, N - 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    def seg(ei):
        u, v = grid["edges"][ei]
        return [pos[u], pos[v]]

    absent = [seg(ei) for ei in grid["inter_edges"] if ei not in act_set]
    if absent:
        ax.add_collection(LineCollection(
            absent, colors=C_INTER_ABSENT, linewidths=1.2,
            linestyles="dashed", alpha=0.9, zorder=1))

    active_segs = [seg(ei) for ei in range(grid["n_edges"]) if ei in act_set]
    if active_segs:
        ax.add_collection(LineCollection(
            active_segs, colors=sel_color, linewidths=3.0,
            alpha=0.92, zorder=2))

    for nd in range(grid["n_nodes"]):
        x, y      = pos[nd]
        c, mk, ms = node_style(nd, grid, sel_color)
        ax.plot(x, y, marker=mk, color=c, markersize=ms,
                markeredgecolor="white", markeredgewidth=1.8, zorder=5)

    if removed:
        rm_pairs = [f"{grid['edges'][ei][0]}-{grid['edges'][ei][1]}"
                    for ei in removed]
        ax.text((N - 1) / 2, -0.72,
                "rm: [" + ", ".join(rm_pairs) + "]",
                ha="center", va="top", fontsize=9,
                color="#666666", fontfamily="monospace")

    if title:
        ax.set_title(title, fontsize=9, color="#111111",
                     pad=5, fontfamily="monospace", fontweight="bold",
                     linespacing=1.4)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

class TopologyExplorer:

    def __init__(self, grid, all_active, Lavgs, n_springs,
                 removed_list, bins, thresholds):
        self.grid         = grid
        self.all_active   = all_active
        self.Lavgs        = Lavgs
        self.n_springs    = n_springs
        self.removed_list = removed_list
        self.bins         = bins
        self.thresholds   = thresholds
        self.N_TOPOS      = len(Lavgs)

        # Normalise L_avg to [0,1] for scatter x-axis
        rng = max(Lavgs.max() - Lavgs.min(), 1e-12)
        self.Lavgs_norm = (Lavgs - Lavgs.min()) / rng

        self.current_seed     = -1
        self.selected_indices = []
        self.n_sample         = min(10, self.N_TOPOS)

        self._build_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor":   "white",
            "text.color":       "#111111",
            "axes.labelcolor":  "#111111",
            "xtick.color":      "#333333",
            "ytick.color":      "#333333",
            "axes.edgecolor":   "#888888",
            "grid.color":       "#DDDDDD",
            "font.family":      "monospace",
            "font.size":        12,
        })

        N   = self.grid["N"]
        ttl = (f"Topology Explorer  —  {N}×{N} grid  "
               f"({self.N_TOPOS} valid topologies)")

        self.fig = plt.figure(figsize=(22, 11), facecolor="white")
        self.fig.canvas.manager.set_window_title(ttl)

        outer = gridspec.GridSpec(
            1, 2, figure=self.fig,
            width_ratios=[1.15, 1],
            left=0.04, right=0.98,
            top=0.93, bottom=0.12,
            wspace=0.06)

        left_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0],
            height_ratios=[0.055, 1], hspace=0.04)
        self.ax_title   = self.fig.add_subplot(left_gs[0])
        self.ax_scatter = self.fig.add_subplot(left_gs[1])
        self.ax_title.axis("off")

        self.right_spec = outer[1]
        self.topo_axes  = []

        self._setup_scatter()
        self._setup_controls()
        self._draw_header()
        self._render_right_panel()

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        plt.show()

    def _draw_header(self):
        ax = self.ax_title
        ax.clear(); ax.axis("off")
        N = self.grid["N"]
        ax.text(0.0, 0.5,
                f"Spring-Mass Topology Explorer  —  {N}×{N} grid",
                transform=ax.transAxes, fontsize=16,
                color="#111111", fontweight="bold",
                fontfamily="monospace", va="center")
        lo, hi = self.Lavgs.min(), self.Lavgs.max()
        ax.text(0.48, 0.5,
                (f"{self.N_TOPOS} valid topologies  |  "
                 f"perimeter fixed  |  inner degree ≥ 2 (colinear if =2)  |  "
                 f"x = avg path length  [{lo:.3f}, {hi:.3f}]  |  "
                 f"y = gap-based cluster  (1–{N_BINS})"),
                transform=ax.transAxes, fontsize=10,
                color="#555555", va="center", fontfamily="monospace")

    def _setup_scatter(self):
        ax = self.ax_scatter
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.35, linewidth=0.7)

        # Plot each topology coloured by its bin
        for b in range(N_BINS):
            mask = self.bins == b
            if mask.any():
                # Jitter y slightly so points don't all stack on integer lines
                ys = b + np.random.default_rng(42).uniform(-0.18, 0.18,
                                                            mask.sum())
                ax.scatter(
                    self.Lavgs_norm[mask], ys,
                    s=70, color=BIN_COLORS[b], alpha=0.75,
                    linewidths=0.5, edgecolors="#666666", zorder=3,
                    label=f"C{b+1}  [{self.thresholds[b]:.3f}, "
                          f"{self.thresholds[b+1]:.3f})")

        # Bin boundary lines
        lo, hi = self.Lavgs.min(), self.Lavgs.max()
        rng = max(hi - lo, 1e-12)
        for t in self.thresholds[1:-1]:
            xn = (t - lo) / rng
            ax.axvline(xn, color="#AAAAAA", linewidth=1.0,
                       linestyle="--", zorder=2)

        # Bin labels on y-axis
        ax.set_yticks(range(N_BINS))
        ax.set_yticklabels([f"C{b+1}" for b in range(N_BINS)],
                           fontsize=11)
        ax.set_ylim(-0.55, N_BINS - 0.45)

        # X-axis ticks in original L_avg units
        xt = np.linspace(0, 1, 6)
        ax.set_xticks(xt)
        ax.set_xticklabels(
            [f"{lo + v * rng:.3f}" for v in xt], fontsize=10)
        ax.set_xlim(-0.04, 1.04)

        ax.set_xlabel("Average Path Length  ( L_avg )  →",
                      fontsize=13, color="#111111", labelpad=8)
        ax.set_ylabel("Cluster  (gap-based, largest jumps in L_avg)",
                      fontsize=13, color="#111111", labelpad=8)

        ax.legend(loc="upper left", fontsize=9, facecolor="white",
                  edgecolor="#CCCCCC", framealpha=0.95)

        # Overlay scatter for selected topologies
        self.sel_scatter = ax.scatter(
            [], [], s=280, zorder=10,
            facecolors="none", edgecolors="black", linewidths=2.5)
        self.sel_texts = []

        self.annot = ax.annotate(
            "", xy=(0, 0), xytext=(16, 16),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.6", fc="white",
                      ec="#888888", alpha=0.97),
            fontsize=11, color="#111111",
            arrowprops=dict(arrowstyle="->", color="#555555"),
            zorder=20)
        self.annot.set_visible(False)

        self.info_text = ax.text(
            0.01, 0.99,
            f"Seed: —  |  Selected: 0  |  {self.N_TOPOS} topologies",
            transform=ax.transAxes, fontsize=12,
            color="#333333", va="top", ha="left", fontfamily="monospace")

    def _setup_controls(self):
        y, h = 0.028, 0.058
        pos = dict(
            next  = [0.040, y, 0.115, h],
            lhs   = [0.162, y, 0.115, h],
            all   = [0.284, y, 0.068, h],
            reset = [0.359, y, 0.068, h],
            sld   = [0.435, y + 0.010, 0.135, h - 0.018],
            seed  = [0.578, y, 0.150, h],
        )
        kw = dict(color="white", hovercolor="#F0F0F0")
        self.ax_next  = self.fig.add_axes(pos["next"])
        self.ax_lhs   = self.fig.add_axes(pos["lhs"])
        self.ax_all   = self.fig.add_axes(pos["all"])
        self.ax_reset = self.fig.add_axes(pos["reset"])
        self.ax_sld   = self.fig.add_axes(pos["sld"])
        self.ax_seed  = self.fig.add_axes(pos["seed"])

        self.btn_next  = Button(self.ax_next,  "▶  Next Sample", **kw)
        self.btn_lhs   = Button(self.ax_lhs,   "⊞  LHS Sample",  **kw)
        self.btn_all   = Button(self.ax_all,   "⊠  All",         **kw)
        self.btn_reset = Button(self.ax_reset, "↺  Reset",       **kw)

        for btn, col, wt in [
            (self.btn_next,  "#0D47A1", "bold"),
            (self.btn_lhs,   "#1B5E20", "bold"),
            (self.btn_all,   "#4A148C", "bold"),
            (self.btn_reset, "#444444", "normal"),
        ]:
            btn.label.set_color(col)
            btn.label.set_fontsize(13)
            btn.label.set_fontfamily("monospace")
            btn.label.set_fontweight(wt)
            for sp in btn.ax.spines.values():
                sp.set_edgecolor("#AAAAAA"); sp.set_linewidth(1.2)

        self.slider = Slider(
            self.ax_sld, "N", 1, min(40, self.N_TOPOS),
            valinit=min(10, self.N_TOPOS), valstep=1,
            color="#0D47A1")
        self.slider.label.set_color("#111111")
        self.slider.label.set_fontsize(13)
        self.slider.valtext.set_color("#0D47A1")
        self.slider.valtext.set_fontsize(13)
        self.ax_sld.set_facecolor("white")

        self.ax_seed.axis("off")
        self.ax_seed.set_facecolor("#F7F7F7")
        for sp in self.ax_seed.spines.values():
            sp.set_visible(True); sp.set_edgecolor("#AAAAAA")
            sp.set_linewidth(1.5)
        self.seed_label = self.ax_seed.text(
            0.5, 0.52, "Seed:  —",
            ha="center", va="center",
            transform=self.ax_seed.transAxes,
            fontsize=14, color="#111111",
            fontfamily="monospace", fontweight="bold")

        self.btn_next.on_clicked(self._on_next)
        self.btn_lhs.on_clicked(self._on_lhs)
        self.btn_all.on_clicked(self._on_all)
        self.btn_reset.on_clicked(self._on_reset)
        self.slider.on_changed(lambda v: setattr(self, "n_sample", int(v)))

    # ── Right panel ──────────────────────────────────────────────────────────

    def _render_right_panel(self):
        for ax in self.topo_axes:
            self.fig.delaxes(ax)
        self.topo_axes = []

        n = len(self.selected_indices)

        if n == 0:
            ax = self.fig.add_subplot(self.right_spec)
            ax.set_facecolor("white"); ax.axis("off")
            ax.text(0.5, 0.62,
                    "Click  ▶ Next Sample\n"
                    "or  ⊞ LHS Sample  (one per bin)\n"
                    "or  ⊠ All",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=15, color="#333333",
                    fontfamily="monospace", linespacing=2.2)

            # Legend
            leg_items = [
                plt.Line2D([0], [0], marker="D", color="w",
                           markerfacecolor=C_ACTUATOR, markersize=13,
                           label="Actuator  (node 0)"),
                plt.Line2D([0], [0], marker="s", color="w",
                           markerfacecolor=C_PILLAR, markersize=13,
                           label="Pinned pillar  (3 far corners)"),
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=C_INNER, markersize=12,
                           label="Inner node  (degree ≥ 2, colinear if =2)"),
                plt.Line2D([0], [0], color="#555555", linewidth=4,
                           label="Active spring  (panel colour = bin colour)"),
                plt.Line2D([0], [0], color=C_INTER_ABSENT, linewidth=2,
                           linestyle="dashed", label="Removed interior spring"),
            ]
            for b in range(N_BINS):
                lo = self.thresholds[b]; hi = self.thresholds[b + 1]
                count = int((self.bins == b).sum())
                leg_items.append(
                    plt.Line2D([0], [0], marker="o", color="w",
                               markerfacecolor=BIN_COLORS[b], markersize=11,
                               label=(f"C{b+1}  [{lo:.3f}, {hi:.3f})  "
                                      f"({count} topos)")))
            ax.legend(handles=leg_items, loc="lower center",
                      fontsize=11, facecolor="white",
                      edgecolor="#CCCCCC", labelcolor="#111111",
                      framealpha=1.0, bbox_to_anchor=(0.5, 0.01),
                      borderpad=1.0, labelspacing=0.75)
            self.topo_axes = [ax]
            self.fig.canvas.draw_idle()
            return

        cols = min(n, 5)
        rows = int(np.ceil(n / cols))
        gs   = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=self.right_spec,
            hspace=0.60, wspace=0.10)

        for pos_i, topo_idx in enumerate(self.selected_indices):
            r, c = divmod(pos_i, cols)
            ax   = self.fig.add_subplot(gs[r, c])
            ax.set_facecolor("white")
            self.topo_axes.append(ax)

            active  = self.all_active[topo_idx]
            removed = self.removed_list[topo_idx]
            b       = int(self.bins[topo_idx])
            color   = BIN_COLORS[b]          # colour = bin, not palette index

            ns  = self.n_springs[topo_idx]
            n_r = self.grid["n_edges"] - ns

            title = (f"#{pos_i+1}  C{b+1}  ({ns}/{self.grid['n_edges']} springs,"
                     f" {n_r} removed)\n"
                     f"L_avg = {self.Lavgs[topo_idx]:.4f}")
            draw_topology(ax, active, removed, self.grid,
                          title=title, sel_color=color)

        self.fig.canvas.draw_idle()

    # ── Scatter overlay ───────────────────────────────────────────────────────

    def _update_scatter(self):
        for t in self.sel_texts:
            t.remove()
        self.sel_texts = []

        if self.selected_indices:
            xs = self.Lavgs_norm[self.selected_indices]
            # y = bin index (with small jitter for visibility)
            ys = np.array([self.bins[i] for i in self.selected_indices],
                          dtype=float)
            self.sel_scatter.set_offsets(np.column_stack([xs, ys]))
            self.sel_scatter.set_sizes([280] * len(self.selected_indices))
            for i, (x, y) in enumerate(zip(xs, ys)):
                col = BIN_COLORS[int(self.bins[self.selected_indices[i]])]
                t   = self.ax_scatter.text(
                    x, y + 0.22, str(i + 1),
                    ha="center", va="bottom", fontsize=11,
                    color=col, fontweight="bold",
                    fontfamily="monospace", zorder=15)
                self.sel_texts.append(t)
        else:
            self.sel_scatter.set_offsets(np.empty((0, 2)))

        seed_str = str(self.current_seed) if self.current_seed >= 0 else "—"
        self.info_text.set_text(
            f"Seed: {seed_str}  |  "
            f"Selected: {len(self.selected_indices)}  |  "
            f"{self.N_TOPOS} topologies")
        self.seed_label.set_text(f"Seed:  {seed_str}")
        self.fig.canvas.draw_idle()

    # ── Button callbacks ──────────────────────────────────────────────────────

    def _on_next(self, _):
        self.current_seed = max(0, self.current_seed + 1)
        self.selected_indices = bin_stratified_sample(
            int(self.slider.val), self.current_seed, self.bins)
        self._update_scatter()
        self._render_right_panel()

    def _on_lhs(self, _):
        self.current_seed = max(0, self.current_seed + 1)
        self.selected_indices = lhs_bin_sample(self.current_seed, self.bins)
        self._update_scatter()
        self._render_right_panel()

    def _on_all(self, _):
        self.selected_indices = list(range(self.N_TOPOS))
        self._update_scatter()
        self._render_right_panel()

    def _on_reset(self, _):
        self.current_seed     = -1
        self.selected_indices = []
        self._update_scatter()
        self._render_right_panel()

    # ── Hover tooltip ─────────────────────────────────────────────────────────

    def _on_hover(self, event):
        if event.inaxes != self.ax_scatter:
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return
        x, y = event.xdata, event.ydata
        if x is None:
            return

        # Distance in normalised x + bin-y space
        ys_all = self.bins.astype(float)
        d2 = (self.Lavgs_norm - x) ** 2 + (ys_all - y) ** 2
        idx = int(np.argmin(d2))
        thr = 0.06 ** 2 + 0.5 ** 2      # generous tolerance in mixed units
        if d2[idx] > thr:
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        rem = self.removed_list[idx]
        b   = int(self.bins[idx])
        txt = (f"id        {idx}\n"
               f"L_avg     {self.Lavgs[idx]:.4f}\n"
               f"Cluster   {b+1}  [{self.thresholds[b]:.3f}, "
               f"{self.thresholds[b+1]:.3f})\n"
               f"springs   {self.n_springs[idx]}/{self.grid['n_edges']}\n"
               f"removed   {rem if rem else 'none'}")
        self.annot.set_text(txt)
        self.annot.xy = (self.Lavgs_norm[idx], float(b))
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive N×N spring-mass topology explorer")
    parser.add_argument(
        "--N", type=int, default=4,
        help="Grid size (must be ≥ 3, default = 4)")
    args = parser.parse_args()

    if args.N < 3:
        print("Error: N must be ≥ 3"); return

    grid = build_grid(args.N)
    all_active, Lavgs, n_springs, removed_list, bins, thresholds = \
        enumerate_topologies(grid)
    TopologyExplorer(grid, all_active, Lavgs, n_springs,
                     removed_list, bins, thresholds)


if __name__ == "__main__":
    main()