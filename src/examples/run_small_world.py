"""
Small-World Explorer — Grid-Based Rewiring
==========================================

This version keeps your grid topology and fixes the main issues:

1. Uses MAX sigma as the "most small-world-like" point
   sigma = (C / C_rand) / (L / L_rand)
   so larger sigma is better, not smaller.

2. Makes the graph definition consistent with the labels.

3. Keeps a grid base graph (not a ring), because that is your actual system.

4. Lets you choose whether the base grid includes diagonals.

5. Preserves edge count during rewiring.

Notes
-----
- This is NOT the classic Watts–Strogatz ring lattice.
- It is a grid-based rewiring experiment with a small-worldness score.
- For very small graphs (like 4x4), sigma can be noisy and weak.

Usage
-----
    python small_world_explorer_grid.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

N_SIDE = 4
N_NODES = N_SIDE ** 2

INCLUDE_DIAGONALS = True   # set False for only horizontal+vertical base grid

N_P_STEPS = 40             # number of p values in [0, 1]
N_REWIRE = 30              # rewired realisations per p
N_RAND = 300               # random baseline samples per p
RNG_SEED = 42

TOPO_COLS = 8              # columns in topology mini-grid

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────

C_SIGMA = "#0D47A1"
C_C_RATIO = "#2E7D32"
C_L_RATIO = "#B71C1C"
C_STAR = "#FF6F00"
C_NODE_DEF = "#455A64"
C_EDGE_KEPT = "#37474F"
C_EDGE_REM = "#CFD8DC"
C_EDGE_NEW = "#C62828"
C_SEL_BORDER = "#FF6F00"

SIGMA_CMAP = matplotlib.colormaps["coolwarm"]

# ─────────────────────────────────────────────────────────────────────────────
# GRID GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_base_graph(n_side=N_SIDE, include_diagonals=INCLUDE_DIAGONALS):
    """
    Build an n_side x n_side grid graph.
    If include_diagonals=True, add down-right and down-left diagonals.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_side * n_side))

    for r in range(n_side):
        for c in range(n_side):
            i = r * n_side + c

            # horizontal / vertical
            if c + 1 < n_side:
                G.add_edge(i, i + 1)       # right
            if r + 1 < n_side:
                G.add_edge(i, i + n_side)  # down

            # diagonals
            if include_diagonals:
                if r + 1 < n_side and c + 1 < n_side:
                    G.add_edge(i, i + n_side + 1)   # diag down-right
                if r + 1 < n_side and c - 1 >= 0:
                    G.add_edge(i, i + n_side - 1)   # diag down-left

    pos = {
        r * n_side + c: np.array([c, n_side - 1 - r], dtype=float)
        for r in range(n_side)
        for c in range(n_side)
    }
    return G, pos


# ─────────────────────────────────────────────────────────────────────────────
# REWIRING
# ─────────────────────────────────────────────────────────────────────────────

def rewire_graph_preserve_edge_count(G_base, p, rng):
    """
    Rewire each base edge independently with probability p.

    For edge (u, v):
      - with probability p, remove (u, v)
      - reconnect u to a randomly chosen non-neighbor w != u

    This preserves the total number of edges, but does not preserve all degrees.
    That is okay for this grid-based experiment.
    """
    G = G_base.copy()
    nodes = list(G.nodes())

    for u, v in list(G_base.edges()):
        if not G.has_edge(u, v):
            continue

        if rng.random() < p:
            forbidden = set(G.neighbors(u)) | {u}
            candidates = [w for w in nodes if w not in forbidden]

            if not candidates:
                continue

            w = int(rng.choice(candidates))
            G.remove_edge(u, v)
            G.add_edge(u, w)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def safe_metrics(G):
    """
    Return clustering C and average path length L.
    If disconnected, path length is computed on the largest connected component.
    """
    C = nx.average_clustering(G)

    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G)
        connected_frac = 1.0
    else:
        comps = list(nx.connected_components(G))
        lcc_nodes = max(comps, key=len)
        lcc = G.subgraph(lcc_nodes).copy()
        L = nx.average_shortest_path_length(lcc)
        connected_frac = len(lcc_nodes) / G.number_of_nodes()

    return C, L, connected_frac


def random_baseline(n_nodes, n_edges, n_samples, rng):
    """
    Matched Erdos-Renyi/G(n,m) baseline:
    same number of nodes and same number of edges.
    """
    Cs, Ls, fracs = [], [], []

    for _ in range(n_samples):
        seed = int(rng.integers(0, 2**31 - 1))
        Gr = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
        C, L, frac = safe_metrics(Gr)
        Cs.append(C)
        Ls.append(L)
        fracs.append(frac)

    return float(np.mean(Cs)), float(np.mean(Ls)), float(np.mean(fracs))


# ─────────────────────────────────────────────────────────────────────────────
# SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep():
    rng = np.random.default_rng(RNG_SEED)
    G_base, pos = build_base_graph()
    n_edges = G_base.number_of_edges()
    p_values = np.linspace(0.0, 1.0, N_P_STEPS)

    diag_text = "with diagonals" if INCLUDE_DIAGONALS else "without diagonals"
    print(f"\nBase graph: {N_NODES} nodes, {n_edges} edges ({diag_text})")
    print(f"Sweeping {N_P_STEPS} p values x {N_REWIRE} rewired graphs")
    print(f"Using {N_RAND} matched random baselines per p\n")

    results = []

    for p in p_values:
        Cs, Ls, fracs, graphs = [], [], [], []

        for _ in range(N_REWIRE):
            Gr = rewire_graph_preserve_edge_count(G_base, p, rng)
            C, L, frac = safe_metrics(Gr)
            Cs.append(C)
            Ls.append(L)
            fracs.append(frac)
            graphs.append(Gr)

        C_mean = float(np.mean(Cs))
        L_mean = float(np.mean(Ls))
        frac_mean = float(np.mean(fracs))

        C_rand, L_rand, frac_rand = random_baseline(N_NODES, n_edges, N_RAND, rng)

        eps = 1e-12
        C_ratio = C_mean / (C_rand + eps)
        L_ratio = L_mean / (L_rand + eps)
        sigma = C_ratio / (L_ratio + eps)

        results.append(
            dict(
                p=float(p),
                C=C_mean,
                L=L_mean,
                connected_frac=frac_mean,
                C_rand=C_rand,
                L_rand=L_rand,
                connected_frac_rand=frac_rand,
                C_ratio=C_ratio,
                L_ratio=L_ratio,
                sigma=sigma,
                graph=graphs[len(graphs) // 2],
            )
        )

        print(
            f"p={p:0.3f}   sigma={sigma:0.4f}   "
            f"C/C_rand={C_ratio:0.3f}   L/L_rand={L_ratio:0.3f}   "
            f"LCC_frac={frac_mean:0.3f}"
        )

    sigmas = np.array([r["sigma"] for r in results])
    idx_best = int(np.argmax(sigmas))

    print(
        f"\nBest sigma = {sigmas[idx_best]:.4f} at p = {results[idx_best]['p']:.3f}\n"
    )

    return p_values, results, G_base, pos, idx_best


# ─────────────────────────────────────────────────────────────────────────────
# DRAW NETWORK
# ─────────────────────────────────────────────────────────────────────────────

def draw_network(
    ax,
    G,
    G_base,
    pos,
    title="",
    node_size=6,
    lw_kept=1.2,
    lw_new=1.6,
):
    ax.set_facecolor("white")
    ax.axis("off")
    ax.set_aspect("equal")

    base_edges = set(map(frozenset, G_base.edges()))
    current_edges = set(map(frozenset, G.edges()))

    kept = base_edges & current_edges
    removed = base_edges - current_edges
    added = current_edges - base_edges

    xy = np.array([pos[n] for n in range(len(pos))])
    pad = 0.5
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)

    def segs(es):
        return [[pos[list(e)[0]], pos[list(e)[1]]] for e in es]

    if removed:
        ax.add_collection(
            LineCollection(
                segs(removed),
                colors=C_EDGE_REM,
                linewidths=0.8,
                linestyles="dashed",
                alpha=0.6,
                zorder=1,
            )
        )

    if kept:
        ax.add_collection(
            LineCollection(
                segs(kept),
                colors=C_EDGE_KEPT,
                linewidths=lw_kept,
                alpha=0.85,
                zorder=2,
            )
        )

    if added:
        ax.add_collection(
            LineCollection(
                segs(added),
                colors=C_EDGE_NEW,
                linewidths=lw_new,
                alpha=0.95,
                zorder=3,
            )
        )

    for n in G.nodes():
        x, y = pos[n]
        ax.plot(
            x,
            y,
            "o",
            color=C_NODE_DEF,
            markersize=node_size,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=5,
        )

    if title:
        ax.set_title(
            title,
            fontsize=9.5,
            fontfamily="monospace",
            fontweight="bold",
            color="#111111",
            pad=3,
            linespacing=1.35,
        )


# ─────────────────────────────────────────────────────────────────────────────
# EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

class SmallWorldExplorer:
    def __init__(self, p_values, results, G_base, pos, idx_best):
        self.p_values = p_values
        self.results = results
        self.G_base = G_base
        self.pos = pos
        self.idx_best = idx_best
        self.sel_idx = idx_best
        self.rng = np.random.default_rng(RNG_SEED + 999)

        self.ps = np.array([r["p"] for r in results])
        self.sigmas = np.array([r["sigma"] for r in results])
        self.C_rats = np.array([r["C_ratio"] for r in results])
        self.L_rats = np.array([r["L_ratio"] for r in results])

        self.sig_norm = Normalize(vmin=self.sigmas.min(), vmax=self.sigmas.max())
        self._build_ui()

    def _build_ui(self):
        plt.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "text.color": "#111111",
                "axes.labelcolor": "#111111",
                "xtick.color": "#333333",
                "ytick.color": "#333333",
                "axes.edgecolor": "#888888",
                "grid.color": "#DDDDDD",
                "font.family": "monospace",
                "font.size": 13,
            }
        )

        base_desc = "H+V+diag" if INCLUDE_DIAGONALS else "H+V only"

        self.fig = plt.figure(figsize=(26, 14), facecolor="white")
        self.fig.canvas.manager.set_window_title(
            f"Small-World Explorer — {N_SIDE}x{N_SIDE} grid ({base_desc})"
        )

        master = gridspec.GridSpec(
            1,
            2,
            figure=self.fig,
            left=0.04,
            right=0.98,
            top=0.92,
            bottom=0.10,
            wspace=0.30,
            width_ratios=[1, 1.55],
        )

        left_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=master[0], hspace=0.50, height_ratios=[1.2, 1]
        )
        self.ax_sigma = self.fig.add_subplot(left_gs[0])
        self.ax_bar = self.fig.add_subplot(left_gs[1])

        n_topos = len(self.results)
        topo_rows = int(np.ceil(n_topos / TOPO_COLS))

        right_gs = gridspec.GridSpecFromSubplotSpec(
            2,
            1,
            subplot_spec=master[1],
            hspace=0.38,
            height_ratios=[topo_rows, 1.6],
        )

        self.topo_gs = gridspec.GridSpecFromSubplotSpec(
            topo_rows,
            TOPO_COLS,
            subplot_spec=right_gs[0],
            hspace=0.55,
            wspace=0.08,
        )

        self.topo_axes = []
        for i in range(n_topos):
            r, c = divmod(i, TOPO_COLS)
            ax = self.fig.add_subplot(self.topo_gs[r, c])
            self.topo_axes.append(ax)

        for i in range(n_topos, topo_rows * TOPO_COLS):
            r, c = divmod(i, TOPO_COLS)
            self.fig.add_subplot(self.topo_gs[r, c]).axis("off")

        self.ax_detail = self.fig.add_subplot(right_gs[1])

        self._draw_header()
        self._plot_sigma_curve()
        self._plot_bar_chart()
        self._draw_all_topo_panels()
        self._draw_detail_panel()
        self._setup_controls()
        self._draw_edge_legend()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)
        plt.show()

    def _draw_header(self):
        r = self.results[self.idx_best]
        base_desc = "horiz+vert+diagonals" if INCLUDE_DIAGONALS else "horiz+vert only"

        self.fig.text(
            0.5,
            0.965,
            f"Small-World Explorer — {N_SIDE}x{N_SIDE} Grid  (base = {base_desc})",
            ha="center",
            fontsize=18,
            fontweight="bold",
            fontfamily="monospace",
            color="#111111",
        )

        self.fig.text(
            0.5,
            0.945,
            (
                f"sigma = (C/C_rand) / (L/L_rand)   |   sigma > 1 suggests small-world-like   |   "
                f"best sigma = {r['sigma']:.3f} at p = {r['p']:.3f}"
            ),
            ha="center",
            fontsize=13,
            fontfamily="monospace",
            color="#555555",
        )

    def _plot_sigma_curve(self):
        ax = self.ax_sigma
        ax2 = ax.twinx()
        self.ax_sigma2 = ax2

        ax.grid(True, alpha=0.3, linewidth=0.7)
        ax.plot(self.ps, self.sigmas, color=C_SIGMA, linewidth=2.5, zorder=4, label="sigma (left)")
        ax.fill_between(self.ps, 1, self.sigmas, where=self.sigmas >= 1, color=C_SIGMA, alpha=0.07)
        ax.axhline(1.0, color=C_SIGMA, linewidth=0.9, linestyle="--", alpha=0.45)
        ax.text(
            0.01,
            1.01,
            "sigma=1",
            transform=ax.get_yaxis_transform(),
            fontsize=11,
            color=C_SIGMA,
            alpha=0.6,
        )

        # mark best sigma
        p_star = self.results[self.idx_best]["p"]
        s_star = self.results[self.idx_best]["sigma"]
        ax.scatter(
            [p_star],
            [s_star],
            s=200,
            color=C_STAR,
            marker="*",
            zorder=6,
            label=f"best sigma={s_star:.3f} @ p={p_star:.3f}",
        )
        ax.annotate(
            f" p={p_star:.3f}\n sigma={s_star:.3f}",
            xy=(p_star, s_star),
            xytext=(min(p_star + 0.06, 0.88), s_star + 0.04),
            fontsize=11,
            color=C_STAR,
            arrowprops=dict(arrowstyle="->", color=C_STAR, lw=1.0),
        )

        ax2.plot(self.ps, self.C_rats, color=C_C_RATIO, linewidth=1.6, linestyle="--", alpha=0.8, label="C/C_rand (right)")
        ax2.plot(self.ps, self.L_rats, color=C_L_RATIO, linewidth=1.6, linestyle=":", alpha=0.8, label="L/L_rand (right)")
        ax2.set_ylabel("C/C_rand and L/L_rand", fontsize=13, color="#666666")
        ax2.tick_params(axis="y", labelcolor="#666666", labelsize=12)

        self.vline_sigma = ax.axvline(
            self.ps[self.sel_idx],
            color=C_SEL_BORDER,
            linewidth=1.8,
            linestyle="-",
            alpha=0.7,
            zorder=5,
        )

        ax.set_xlabel("Rewiring probability p", fontsize=14, labelpad=6)
        ax.set_ylabel("Small-worldness sigma", fontsize=14, color=C_SIGMA)
        ax.tick_params(axis="y", labelcolor=C_SIGMA, labelsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title("sigma vs Rewiring Probability (click to select p)", fontsize=14, fontweight="bold", pad=6)

        self.hover_dot = ax.scatter([], [], s=100, color=C_SIGMA, zorder=8, edgecolors="white", linewidths=1.5)
        self.annot = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(14, 14),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#888888", alpha=0.97),
            fontsize=12,
            color="#111111",
            arrowprops=dict(arrowstyle="->", color="#555555"),
            zorder=20,
        )
        self.annot.set_visible(False)

        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, loc="upper right", fontsize=11,
                  facecolor="white", edgecolor="#CCCCCC", framealpha=0.95)

    def _plot_bar_chart(self):
        ax = self.ax_bar
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.7)

        w = 0.011
        ax.bar(self.ps - w / 2, self.C_rats, width=w, color=C_C_RATIO, alpha=0.78, label="C / C_rand")
        ax.bar(self.ps + w / 2, self.L_rats, width=w, color=C_L_RATIO, alpha=0.78, label="L / L_rand")
        ax.axhline(1.0, color="#888888", linewidth=0.9, linestyle="--")

        self.vline_bar = ax.axvline(
            self.ps[self.sel_idx],
            color=C_SEL_BORDER,
            linewidth=1.8,
            linestyle="-",
            alpha=0.7,
            zorder=5,
        )

        ax.set_xlabel("Rewiring probability p", fontsize=14, labelpad=6)
        ax.set_ylabel("Ratio (actual / random)", fontsize=14)
        ax.set_xlim(-0.02, 1.02)
        ax.set_title("Clustering & Path-Length Ratios vs p", fontsize=14, fontweight="bold", pad=6)
        ax.legend(fontsize=12, facecolor="white", edgecolor="#CCCCCC", framealpha=0.95)

    def _draw_all_topo_panels(self):
        for i, ax in enumerate(self.topo_axes):
            ax.clear()
            r = self.results[i]
            draw_network(
                ax,
                r["graph"],
                self.G_base,
                self.pos,
                title=f"p={r['p']:.2f}\nσ={r['sigma']:.2f}",
                node_size=4,
                lw_kept=0.9,
                lw_new=1.2,
            )

        self._highlight_selected()

    def _highlight_selected(self):
        for i, ax in enumerate(self.topo_axes):
            color = SIGMA_CMAP(self.sig_norm(self.results[i]["sigma"]))
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color)
                sp.set_linewidth(2.8)

            if i == self.sel_idx:
                for sp in ax.spines.values():
                    sp.set_edgecolor(C_SEL_BORDER)
                    sp.set_linewidth(4.2)

    def _draw_detail_panel(self):
        ax = self.ax_detail
        ax.clear()

        r = self.results[self.sel_idx]
        draw_network(
            ax,
            r["graph"],
            self.G_base,
            self.pos,
            title=(
                f"SELECTED p = {r['p']:.3f}\n"
                f"sigma = {r['sigma']:.4f}   "
                f"C = {r['C']:.4f}   L = {r['L']:.4f}\n"
                f"C/C_rand = {r['C_ratio']:.4f}   "
                f"L/L_rand = {r['L_ratio']:.4f}   "
                f"LCC frac = {r['connected_frac']:.3f}"
            ),
            node_size=9,
            lw_kept=1.6,
            lw_new=2.1,
        )

    def _setup_controls(self):
        ax_btn = self.fig.add_axes([0.04, 0.028, 0.14, 0.042])
        self.btn = Button(ax_btn, "Resample at selected p", color="white", hovercolor="#F0F0F0")
        self.btn.label.set_color("#0D47A1")
        self.btn.label.set_fontsize(13)
        self.btn.label.set_fontweight("bold")
        self.btn.label.set_fontfamily("monospace")
        for sp in self.btn.ax.spines.values():
            sp.set_edgecolor("#AAAAAA")
            sp.set_linewidth(1.2)
        self.btn.on_clicked(self._on_resample)

        self.info_ax = self.fig.add_axes([0.20, 0.028, 0.58, 0.042])
        self.info_ax.axis("off")
        self.info_text = self.info_ax.text(
            0.0,
            0.5,
            "Click any point on the sigma curve or any topology thumbnail to select it. Then Resample for a fresh draw.",
            transform=self.info_ax.transAxes,
            fontsize=12,
            color="#555555",
            va="center",
            fontfamily="monospace",
        )

    def _draw_edge_legend(self):
        items = [
            matplotlib.lines.Line2D([0], [0], color=C_EDGE_KEPT, linewidth=2.5, label="Edge kept from base grid"),
            matplotlib.lines.Line2D([0], [0], color=C_EDGE_REM, linewidth=1.5, linestyle="dashed", label="Edge removed by rewiring"),
            matplotlib.lines.Line2D([0], [0], color=C_EDGE_NEW, linewidth=2.5, label="New edge added by rewiring"),
            matplotlib.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_NODE_DEF, markersize=9, label="Node"),
        ]
        self.fig.legend(
            handles=items,
            loc="lower right",
            ncol=4,
            fontsize=12,
            facecolor="white",
            edgecolor="#CCCCCC",
            framealpha=0.95,
            bbox_to_anchor=(0.99, 0.005),
            borderpad=0.8,
        )

    def _select(self, idx):
        self.sel_idx = idx
        self._highlight_selected()
        self._draw_detail_panel()

        self.vline_sigma.set_xdata([self.ps[idx], self.ps[idx]])
        self.vline_bar.set_xdata([self.ps[idx], self.ps[idx]])

        self.info_text.set_text(
            f"Selected p = {self.ps[idx]:.3f}   "
            f"sigma = {self.sigmas[idx]:.4f}   "
            f"C/C_rand = {self.C_rats[idx]:.4f}   "
            f"L/L_rand = {self.L_rats[idx]:.4f}"
        )
        self.fig.canvas.draw_idle()

    def _on_resample(self, _event):
        idx = self.sel_idx
        p = self.ps[idx]
        G_new = rewire_graph_preserve_edge_count(self.G_base, p, self.rng)

        self.results[idx]["graph"] = G_new

        ax = self.topo_axes[idx]
        ax.clear()
        r = self.results[idx]
        draw_network(
            ax,
            r["graph"],
            self.G_base,
            self.pos,
            title=f"p={r['p']:.2f}\nσ={r['sigma']:.2f}",
            node_size=4,
            lw_kept=0.9,
            lw_new=1.2,
        )

        self._highlight_selected()
        self._draw_detail_panel()
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes in (self.ax_sigma, self.ax_sigma2):
            if event.xdata is not None:
                idx = int(np.argmin(np.abs(self.ps - event.xdata)))
                self._select(idx)
            return

        for i, ax in enumerate(self.topo_axes):
            if event.inaxes is ax:
                self._select(i)
                return

    def _on_hover(self, event):
        if event.inaxes not in (self.ax_sigma, self.ax_sigma2):
            self.annot.set_visible(False)
            self.hover_dot.set_offsets(np.empty((0, 2)))
            self.fig.canvas.draw_idle()
            return

        if event.xdata is None:
            return

        idx = int(np.argmin(np.abs(self.ps - event.xdata)))
        r = self.results[idx]

        self.hover_dot.set_offsets([[self.ps[idx], self.sigmas[idx]]])
        txt = (
            f"p         {r['p']:.3f}\n"
            f"sigma     {r['sigma']:.4f}\n"
            f"C         {r['C']:.4f}\n"
            f"L         {r['L']:.4f}\n"
            f"C/C_rand  {r['C_ratio']:.4f}\n"
            f"L/L_rand  {r['L_ratio']:.4f}\n"
            f"LCC frac  {r['connected_frac']:.4f}"
        )
        self.annot.set_text(txt)
        self.annot.xy = (self.ps[idx], self.sigmas[idx])
        self.annot.set_visible(True)
        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p_values, results, G_base, pos, idx_best = run_sweep()
    SmallWorldExplorer(p_values, results, G_base, pos, idx_best)


if __name__ == "__main__":
    main()