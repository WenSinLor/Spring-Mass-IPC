"""
Interactive Spring-Mass Topology Explorer  —  Scalable N×N
===========================================================
Works for any N >= 3.

Validity rules
--------------
  1. All perimeter edges are FIXED (the outer ring is always present).
  2. Each inner node must keep at least 3 of its 4 springs (degree >= 3).
  3. Global connectivity is automatically guaranteed by rules 1 + 2.

Variable edges = the interior edges only (those not on the perimeter ring).
At most one interior spring per inner node may be removed.

Visual design
-------------
  - All active springs in a topology panel share the same panel colour.
  - Actuator node 0 (top-left corner)  : bright orange diamond
  - Pinned pillars  (other 3 corners)  : red square
  - Inner nodes     (strictly interior): dark grey circle
  - Other perimeter nodes              : same colour as that topology's springs
  - Removed interior springs           : dashed light grey

Usage
-----
  python topology_explorer.py            # default N = 4
  python topology_explorer.py --N 5      # 5x5 grid
  python topology_explorer.py --N 3      # 3x3 grid

Controls
--------
  Next Sample  —  random sample at next seed (seed increments each click)
  LHS Sample   —  Latin Hypercube sample covering spectral-gap axis evenly
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

# Functional node roles
C_ACTUATOR = "#E8760A"   # actuator (node 0)
C_PILLAR   = "#C0392B"   # other 3 pinned corners
C_INNER    = "#444444"   # strictly interior nodes

# Springs
C_INTER_ACTIVE = "#222222"   # default; overridden by selection colour
C_INTER_ABSENT = "#CCCCCC"   # dashed grey for removed spring


# ─────────────────────────────────────────────────────────────────────────────
#  GRID BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(N: int) -> dict:
    """Return a dict describing the N×N spring-mass grid topology."""
    n_nodes = N * N
    n_max   = N - 1

    # Row-major edges: horizontal first, then vertical within each cell
    edges = []
    for r in range(N):
        for c in range(N):
            i = r * N + c
            if c < n_max:
                edges.append((i, i + 1))      # →
            if r < n_max:
                edges.append((i, i + N))      # ↓
    edges   = np.array(edges, dtype=int)
    n_edges = len(edges)

    node_pos = np.array([[c, N - 1 - r] for r in range(N) for c in range(N)],
                    dtype=float)

    def on_perim(nd):
        r, c = nd // N, nd % N
        return r == 0 or r == n_max or c == 0 or c == n_max

    inner_nodes = [nd for nd in range(n_nodes) if not on_perim(nd)]

    # Classify edges by perimeter / interior
    perim_edges, inter_edges = [], []
    for ei, (u, v) in enumerate(edges):
        (perim_edges if on_perim(u) and on_perim(v) else inter_edges).append(ei)

    # For each inner node: map incident edge indices by direction
    inner_springs = {}
    for nd in inner_nodes:
        r, c = divmod(nd, N)
        nbr_to_dir = {
            nd - 1: "left",
            nd + 1: "right",
            nd - N: "up",
            nd + N: "down",
        }

        dir_map = {}
        for ei, (u, v) in enumerate(edges):
            if u == nd:
                other = v
            elif v == nd:
                other = u
            else:
                continue

            if other in nbr_to_dir:
                dir_map[nbr_to_dir[other]] = ei

        inner_springs[nd] = dir_map

    corners   = [0, n_max, N * n_max, N * N - 1]
    actuator  = 0
    pillars   = corners[1:]

    return dict(
        N=N,
        n_nodes=n_nodes,
        edges=edges,
        n_edges=n_edges,
        node_pos=node_pos,
        inner_nodes=inner_nodes,
        inner_springs=inner_springs,
        perim_edges=perim_edges,
        inter_edges=inter_edges,
        on_perim=on_perim,
        actuator=actuator,
        pillars=pillars,
        corners=corners,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  VALIDITY CHECK
# ─────────────────────────────────────────────────────────────────────────────

def satisfies_rules(removed_set: set, grid: dict) -> bool:
    """
    Validity rule for each inner node:
      - at least 2 active incident springs must remain
      - if exactly 2 remain, they must be colinear:
            {left, right} or {up, down}
      - perpendicular 2-spring cases are invalid
    Perimeter is always intact by construction (we only remove inter edges).
    """
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
#  SPECTRAL DESCRIPTORS
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptors(active_idx: list, grid: dict):
    """Return (spectral_gap, lambda2, eff_rank, L_avg)."""
    G = nx.Graph()
    G.add_nodes_from(range(grid["n_nodes"]))
    for ei in active_idx:
        G.add_edge(int(grid["edges"][ei, 0]), int(grid["edges"][ei, 1]))

    L = nx.laplacian_matrix(G).toarray().astype(float)
    ev, V = np.linalg.eigh(L)
    ev = np.abs(ev)
    lam2 = float(ev[1])
    lmax = float(ev[-1])
    gap  = lam2 / max(lmax, 1e-12)

    phi = V[grid["actuator"], 1:]     # node-0 projections on non-trivial modes
    w   = phi ** 2
    t   = w.sum()
    eff = float(t**2 / (w**2).sum()) / (grid["n_nodes"] - 1) if t > 1e-12 else 0.0

    L_avg = float(nx.average_shortest_path_length(G))
    return gap, lam2, eff, L_avg


# ─────────────────────────────────────────────────────────────────────────────
#  ENUMERATION
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_topologies(grid: dict):
    inter = grid["inter_edges"]
    N     = grid["N"]
    print(f"\nEnumerating valid topologies for {N}×{N} grid …")
    print(f"  Fixed perimeter edges  : {len(grid['perim_edges'])}")
    print(f"  Variable interior edges: {len(inter)}")
    print(f"  Inner nodes (min degree 2; if degree=2 then colinear) : {grid['inner_nodes']}")
    print()

    all_active, gaps, lam2s, effs, Lavgs, n_sp, removed_list = \
        [], [], [], [], [], [], []
    total = 0

    for n_rem in range(len(inter) + 1):
        level = 0
        for combo in combinations(inter, n_rem):
            removed_set = set(combo)
            if not satisfies_rules(removed_set, grid):
                continue
            active_idx = [i for i in range(grid["n_edges"]) if i not in removed_set]
            g, l2, er, la = compute_descriptors(active_idx, grid)
            all_active.append(active_idx)
            gaps.append(g)
            lam2s.append(l2)
            effs.append(er)
            Lavgs.append(la)
            n_sp.append(len(active_idx))
            removed_list.append(sorted(removed_set))
            level += 1
            total += 1
        print(f"  {n_rem} interior edge(s) removed: {level:5d}  "
              f"(running total: {total})")
        if level == 0 and n_rem > len(grid["inter_edges"]):
            break

    print(f"\n  ✓  Total valid topologies: {total}\n")
    return (
        all_active,
        np.array(gaps),
        np.array(lam2s),
        np.array(effs),
        np.array(Lavgs),
        np.array(n_sp),
        removed_list,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def random_sample(n: int, seed: int, total: int) -> list:
    rng = np.random.default_rng(seed)
    return rng.choice(total, size=min(n, total), replace=False).tolist()


def lhs_sample(n: int, seed: int, gaps_norm: np.ndarray) -> list:
    """1-D Latin Hypercube along spectral-gap axis."""
    rng   = np.random.default_rng(seed)
    order = np.argsort(gaps_norm)
    total = len(gaps_norm)
    n     = min(n, total)
    bsz   = total / n
    result = []
    for b in range(n):
        lo = int(b * bsz)
        hi = min(int((b + 1) * bsz), total)
        result.append(int(order[rng.integers(lo, max(lo + 1, hi))]))
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  NODE STYLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def node_style(nd: int, grid: dict, sel_color: str):
    """
    Return (facecolor, marker, markersize) for a node.

    Color logic:
      - actuator        -> C_ACTUATOR
      - pinned pillars  -> C_PILLAR
      - inner nodes     -> C_INNER
      - other perimeter nodes -> sel_color
    """
    if nd == grid["actuator"]:
        return C_ACTUATOR, "D", 12
    if nd in grid["pillars"]:
        return C_PILLAR, "s", 12
    if nd in grid["inner_nodes"]:
        return C_INNER, "o", 10
    return sel_color, "o", 10


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE-PANEL TOPOLOGY DRAWING
# ─────────────────────────────────────────────────────────────────────────────

def draw_topology(ax, active_idx: list, removed: list,
                  grid: dict, title: str = "", sel_color: str = C_INTER_ACTIVE):
    N       = grid["N"]
    pos     = grid["node_pos"]
    act_set = set(active_idx)

    ax.set_xlim(-0.65, N - 0.35)
    ax.set_ylim(-0.85, N - 0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    def seg(ei):
        u, v = grid["edges"][ei]
        return [pos[u], pos[v]]

    # 1. Absent interior springs
    absent = [seg(ei) for ei in grid["inter_edges"] if ei not in act_set]
    if absent:
        ax.add_collection(LineCollection(
            absent,
            colors=C_INTER_ABSENT,
            linewidths=1.2,
            linestyles="dashed",
            alpha=0.9,
            zorder=1
        ))

    # 2. All active springs use the topology selection colour
    active_segs = [seg(ei) for ei in range(grid["n_edges"]) if ei in act_set]
    if active_segs:
        ax.add_collection(LineCollection(
            active_segs,
            colors=sel_color,
            linewidths=3.0,
            alpha=0.92,
            zorder=2
        ))

    # 3. Nodes
    for nd in range(grid["n_nodes"]):
        x, y      = pos[nd]
        c, mk, ms = node_style(nd, grid, sel_color)
        ax.plot(x, y, marker=mk, color=c, markersize=ms,
                markeredgecolor="white", markeredgewidth=1.8, zorder=5)

    # 4. Removed-edge annotation
    if removed:
        rm_pairs = [f"{grid['edges'][ei][0]}-{grid['edges'][ei][1]}" for ei in removed]
        ax.text((N - 1) / 2, -0.72,
                "rm: [" + ", ".join(rm_pairs) + "]",
                ha="center", va="top", fontsize=9,
                color="#666666", fontfamily="monospace")

    if title:
        ax.set_title(title, fontsize=10, color="#111111",
                     pad=5, fontfamily="monospace", fontweight="bold",
                     linespacing=1.4)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN EXPLORER CLASS
# ─────────────────────────────────────────────────────────────────────────────

# Selection palette: 40 distinct dark colours (readable on white)
_PALETTE_HUES = np.arange(40) * 137.508
_PALETTE = [
    "#{:02x}{:02x}{:02x}".format(
        int(matplotlib.cm.hsv(h / 360)[0] * 190),
        int(matplotlib.cm.hsv(h / 360)[1] * 190),
        int(matplotlib.cm.hsv(h / 360)[2] * 190),
    )
    for h in _PALETTE_HUES % 360
]


class TopologyExplorer:

    def __init__(self, grid, all_active, gaps, lam2s, effs, Lavgs,
                 n_springs, removed_list):
        self.grid         = grid
        self.all_active   = all_active
        self.gaps         = gaps
        self.lam2s        = lam2s
        self.effs         = effs
        self.Lavgs        = Lavgs
        self.n_springs    = n_springs
        self.removed_list = removed_list
        self.N_TOPOS      = len(gaps)

        g_rng = max(gaps.max() - gaps.min(), 1e-12)
        e_rng = max(effs.max() - effs.min(), 1e-12)
        self.gaps_norm = (gaps - gaps.min()) / g_rng
        self.effs_norm = (effs - effs.min()) / e_rng

        self.current_seed     = -1
        self.selected_indices = []
        self.n_sample         = min(10, self.N_TOPOS)

        self._build_ui()

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
            wspace=0.06,
        )

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
        ax.clear()
        ax.axis("off")
        N = self.grid["N"]
        ax.text(0.0, 0.5,
                f"Spring-Mass Topology Explorer  —  {N}×{N} grid",
                transform=ax.transAxes, fontsize=16,
                color="#111111", fontweight="bold",
                fontfamily="monospace", va="center")
        ax.text(0.50, 0.5,
                (f"{self.N_TOPOS} valid topologies  |  "
                 f"perimeter fixed  |  inner degree ≥ 3  |  "
                 f"x = spectral gap  |  y = eff rank"),
                transform=ax.transAxes, fontsize=11,
                color="#555555", va="center", fontfamily="monospace")

    def _setup_scatter(self):
        ax = self.ax_scatter
        ax.set_facecolor("white")
        ax.grid(True, alpha=0.35, linewidth=0.7)

        n_rem = self.grid["n_edges"] - self.n_springs
        sc = ax.scatter(
            self.gaps_norm, self.effs_norm,
            s=80, c=n_rem, cmap="plasma_r",
            alpha=0.80, linewidths=0.7, edgecolors="#666666", zorder=3)

        cb = self.fig.colorbar(sc, ax=ax, fraction=0.024, pad=0.01)
        cb.set_label("interior springs removed", fontsize=12, color="#111111")
        cb.ax.tick_params(labelsize=11, colors="#333333")

        self.sel_scatter = ax.scatter(
            [], [], s=220, zorder=10,
            c=[], cmap="tab10", vmin=0, vmax=1,
            edgecolors="black", linewidths=2.0)
        self.sel_texts = []

        g_min, g_max = self.gaps.min(), self.gaps.max()
        e_min, e_max = self.effs.min(), self.effs.max()
        xt = np.linspace(0, 1, 5)
        ax.set_xticks(xt)
        ax.set_xticklabels(
            [f"{g_min + v*(g_max - g_min):.4f}" for v in xt], fontsize=11)
        yt = np.linspace(0, 1, 5)
        ax.set_yticks(yt)
        ax.set_yticklabels(
            [f"{e_min + v*(e_max - e_min):.3f}" for v in yt], fontsize=11)

        ax.set_xlabel("Spectral gap  ( λ₂ / λ_max )  →",
                      fontsize=13, color="#111111", labelpad=8)
        ax.set_ylabel("← Effective rank  ( modal participation )",
                      fontsize=13, color="#111111", labelpad=8)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)

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

        btn_styles = [
            (self.btn_next,  "#0D47A1", "bold"),
            (self.btn_lhs,   "#1B5E20", "bold"),
            (self.btn_all,   "#4A148C", "bold"),
            (self.btn_reset, "#444444", "normal"),
        ]
        for btn, col, wt in btn_styles:
            btn.label.set_color(col)
            btn.label.set_fontsize(13)
            btn.label.set_fontfamily("monospace")
            btn.label.set_fontweight(wt)
            for sp in btn.ax.spines.values():
                sp.set_edgecolor("#AAAAAA")
                sp.set_linewidth(1.2)

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
            sp.set_visible(True)
            sp.set_edgecolor("#AAAAAA")
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

    def _render_right_panel(self):
        for ax in self.topo_axes:
            self.fig.delaxes(ax)
        self.topo_axes = []

        n = len(self.selected_indices)

        if n == 0:
            ax = self.fig.add_subplot(self.right_spec)
            ax.set_facecolor("white")
            ax.axis("off")
            ax.text(0.5, 0.62,
                    "Click  ▶ Next Sample\n"
                    "or  ⊞ LHS Sample\n"
                    "or  ⊠ All",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=15, color="#333333",
                    fontfamily="monospace", linespacing=2.2)

            leg_items = [
                plt.Line2D([0], [0], marker="D", color="w",
                           markerfacecolor=C_ACTUATOR, markersize=13,
                           label="Actuator  (node 0, x-direction only)"),
                plt.Line2D([0], [0], marker="s", color="w",
                           markerfacecolor=C_PILLAR, markersize=13,
                           label="Pinned pillar  (3 far corners)"),
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=C_INNER, markersize=12,
                           label="Inner node  (degree ≥ 3)"),
                plt.Line2D([0], [0], color="#555555", linewidth=4,
                           label="Active spring  (panel colour)"),
                plt.Line2D([0], [0], color=C_INTER_ABSENT, linewidth=2,
                           linestyle="dashed",
                           label="Removed interior spring"),
            ]
            ax.legend(handles=leg_items, loc="lower center",
                      fontsize=12, facecolor="white",
                      edgecolor="#CCCCCC", labelcolor="#111111",
                      framealpha=1.0, bbox_to_anchor=(0.5, 0.02),
                      borderpad=1.0, labelspacing=0.8)
            self.topo_axes = [ax]
            self.fig.canvas.draw_idle()
            return

        cols = min(n, 5)
        rows = int(np.ceil(n / cols))
        gs   = gridspec.GridSpecFromSubplotSpec(
            rows, cols, subplot_spec=self.right_spec,
            hspace=0.55, wspace=0.10)

        for pos_i, topo_idx in enumerate(self.selected_indices):
            r, c = divmod(pos_i, cols)
            ax   = self.fig.add_subplot(gs[r, c])
            ax.set_facecolor("white")
            self.topo_axes.append(ax)

            active  = self.all_active[topo_idx]
            removed = self.removed_list[topo_idx]
            color   = _PALETTE[pos_i % len(_PALETTE)]
            ns      = self.n_springs[topo_idx]
            n_r     = self.grid["n_edges"] - ns

            title = (f"#{pos_i+1}   {ns}/{self.grid['n_edges']} springs"
                     f"  ({n_r} removed)\n"
                     f"gap={self.gaps[topo_idx]:.4f}   "
                     f"λ₂={self.lam2s[topo_idx]:.3f}   "
                     f"eff={self.effs[topo_idx]:.3f}")
            draw_topology(ax, active, removed, self.grid,
                          title=title, sel_color=color)

        self.fig.canvas.draw_idle()

    def _update_scatter(self):
        for t in self.sel_texts:
            t.remove()
        self.sel_texts = []

        if self.selected_indices:
            xs   = self.gaps_norm[self.selected_indices]
            ys   = self.effs_norm[self.selected_indices]
            hues = np.linspace(0, 0.9, len(self.selected_indices))
            self.sel_scatter.set_offsets(np.column_stack([xs, ys]))
            self.sel_scatter.set_array(hues)
            self.sel_scatter.set_sizes([220] * len(self.selected_indices))
            for i, (x, y) in enumerate(zip(xs, ys)):
                col = _PALETTE[i % len(_PALETTE)]
                t   = self.ax_scatter.text(
                    x, y + 0.032, str(i + 1),
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

    def _on_next(self, _):
        self.current_seed = max(0, self.current_seed + 1)
        self.selected_indices = random_sample(
            int(self.slider.val), self.current_seed, self.N_TOPOS)
        self._update_scatter()
        self._render_right_panel()

    def _on_lhs(self, _):
        self.current_seed = max(0, self.current_seed + 1)
        self.selected_indices = lhs_sample(
            int(self.slider.val), self.current_seed, self.gaps_norm)
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

    def _on_hover(self, event):
        if event.inaxes != self.ax_scatter:
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return
        x, y = event.xdata, event.ydata
        if x is None:
            return
        d2  = (self.gaps_norm - x) ** 2 + (self.effs_norm - y) ** 2
        idx = int(np.argmin(d2))
        xlim = self.ax_scatter.get_xlim()
        ylim = self.ax_scatter.get_ylim()
        thr  = ((xlim[1] - xlim[0]) * 0.04) ** 2 + \
               ((ylim[1] - ylim[0]) * 0.04) ** 2
        if d2[idx] > thr:
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        rem = self.removed_list[idx]
        txt = (f"id        {idx}\n"
               f"gap       {self.gaps[idx]:.5f}\n"
               f"λ₂        {self.lam2s[idx]:.4f}\n"
               f"eff rank  {self.effs[idx]:.4f}\n"
               f"L_avg     {self.Lavgs[idx]:.4f}\n"
               f"springs   {self.n_springs[idx]}/{self.grid['n_edges']}\n"
               f"removed   {rem if rem else 'none'}")
        self.annot.set_text(txt)
        self.annot.xy = (self.gaps_norm[idx], self.effs_norm[idx])
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
        print("Error: N must be ≥ 3")
        return

    grid = build_grid(args.N)
    data = enumerate_topologies(grid)
    TopologyExplorer(grid, *data)


if __name__ == "__main__":
    main()