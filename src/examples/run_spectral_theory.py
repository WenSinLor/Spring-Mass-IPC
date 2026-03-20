"""
Interactive Capacity Matrix Explorer
=====================================
Implements Eq. 35 from:
  "A Spectral Theory for Mechanical Reservoir Computing" — Yogesh Phalak

C(d, m) = SA(P*) · SD(Q*) / |G(d·n_max, m)|

  SA(P*) = (1 - r^{2P*}) / (1 - r^2)            amplitude term
  SD(Q*) = (1 - e^{-2γ(Q*+1)}) / (1 - e^{-2γ})  damping term
  |G|    = d · n_max · (m + 1)                    harmonic-delay grid size

Usage:
  python capacity_matrix_explorer.py
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # robust backend for interactive sliders
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'font.family':       'monospace',
    'font.size':         12,
    'axes.titlesize':    13,
    'axes.titleweight':  'bold',
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'text.color':        '#111111',
    'axes.labelcolor':   '#111111',
    'xtick.color':       '#333333',
    'ytick.color':       '#333333',
    'axes.edgecolor':    '#888888',
})

CMAP = LinearSegmentedColormap.from_list(
    'paper', ['white', '#B3D4F0', '#5B9BD5', '#1F5FA6', '#0A2A5E'], N=256)


# ─────────────────────────────────────────────────────────────────────────────
#  MATH  (Eq. 35)
# ─────────────────────────────────────────────────────────────────────────────

def capacity_matrix(d_max, M, r, gamma, N_res, M_actual, n_max=1):
    """
    Returns C array of shape (d_max, M+1).
    Rows    = degree d = 1 … d_max   (y-axis, bottom=1, top=d_max)
    Columns = memory m = 0 … M       (x-axis)
    """
    C = np.zeros((d_max, M + 1))
    for d in range(1, d_max + 1):
        P_star = min(d * n_max, N_res)
        # Amplitude term SA
        if abs(r - 1.0) < 1e-10:
            SA = float(P_star)
        else:
            SA = (1.0 - r**(2*P_star)) / (1.0 - r**2)

        for m in range(M + 1):
            Q_star = min(m, M_actual)
            G_size = d * n_max * (m + 1)
            # Damping term SD
            if abs(gamma) < 1e-10:
                SD = float(Q_star + 1)
            else:
                SD = (1.0 - np.exp(-2*gamma*(Q_star+1))) / (1.0 - np.exp(-2*gamma))
            C[d-1, m] = (SA * SD) / G_size

    return np.clip(C, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  PRESETS  (Fig 6 of paper)
# ─────────────────────────────────────────────────────────────────────────────
PRESETS = {
    'Perfect':            dict(r=1.00, gamma=0.00, N_res=50, M_actual=50),
    'Amplitude-limited':  dict(r=0.55, gamma=0.00, N_res=50, M_actual=50),
    'Few resonators':     dict(r=1.00, gamma=0.00, N_res=3,  M_actual=50),
    'High damping':       dict(r=1.00, gamma=0.28, N_res=50, M_actual=50),
    'Combined':           dict(r=0.55, gamma=0.28, N_res=3,  M_actual=10),
}
PRESET_COLORS = ['#1F5FA6', '#E8760A', '#7D3C98', '#1E8449', '#C0392B']


# ─────────────────────────────────────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────────────────────────────────────

class CapacityExplorer:

    def __init__(self):
        # Current parameter values (set directly, not read from sliders during preset load)
        self.r        = 1.00
        self.gamma    = 0.00
        self.N_res    = 50
        self.M_actual = 50
        self.n_max    = 1
        self.d_max    = 6
        self.M        = 10

        self._suppress_callbacks = False
        self._build_figure()
        self._build_sliders()
        self._build_presets()
        self._redraw()
        plt.show()

    # ── Figure skeleton ───────────────────────────────────────────────────────

    def _build_figure(self):
        # Large figure: top 75% = plots, bottom 25% = controls
        self.fig = plt.figure(figsize=(22, 12), facecolor='white')
        self.fig.canvas.manager.set_window_title(
            'Capacity Matrix Explorer — Spectral Theory of Mechanical RC')

        # Split figure vertically: plots region vs controls region
        main_gs = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[3.2, 1],
            left=0.04, right=0.98,
            top=0.95, bottom=0.03,
            hspace=0.05)

        # ── Plots region: heatmap (left) + profiles (right) ──────────────────
        plots_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=main_gs[0],
            width_ratios=[1.1, 1],
            wspace=0.10)

        # Heatmap
        self.ax_heat = self.fig.add_subplot(plots_gs[0])

        # Three profile plots stacked on right
        profiles_gs = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=plots_gs[1],
            hspace=0.55)
        self.ax_linear  = self.fig.add_subplot(profiles_gs[0])
        self.ax_nonlin  = self.fig.add_subplot(profiles_gs[1])
        self.ax_alldeg  = self.fig.add_subplot(profiles_gs[2])

        # ── Controls region ───────────────────────────────────────────────────
        self.ctrl_gs = main_gs[1]   # reserved for sliders & buttons

    # ── Sliders ───────────────────────────────────────────────────────────────

    def _build_sliders(self):
        """
        Place sliders in the bottom panel using absolute figure coordinates.
        Layout: two columns of 4 sliders each, with preset buttons above them.
        """
        # Vertical positions (figure coords, bottom panel ~ 0.03 to 0.23)
        base   = 0.04    # bottom of slider area
        step   = 0.047   # vertical spacing between sliders
        h_sl   = 0.025   # slider height
        col1_l = 0.06    # left edge of left column
        col2_l = 0.55    # left edge of right column
        sl_w   = 0.38    # slider width

        specs = [
            # (attr,      label,           col, row, vmin, vmax,  vstep, fmt)
            ('r',         'r  (A/a)',       0,   3,  0.01, 1.0,  0.01,  '%.2f'),
            ('gamma',     'γ  (damping)',   0,   2,  0.00, 1.0,  0.01,  '%.2f'),
            ('N_res',     'N_res',          0,   1,  1,    30,   1,     '%d'),
            ('M_actual',  'M_actual',       0,   0,  1,    30,   1,     '%d'),
            ('n_max',     'n_max',          1,   3,  1,    5,    1,     '%d'),
            ('d_max',     'd_max',          1,   2,  1,    12,   1,     '%d'),
            ('M',         'M (max depth)',  1,   1,  1,    30,   1,     '%d'),
        ]

        self.sliders = {}
        for attr, label, col, row, vmin, vmax, vstep, fmt in specs:
            left  = col1_l if col == 0 else col2_l
            bot   = base + row * step
            ax_sl = self.fig.add_axes([left, bot, sl_w, h_sl])
            ax_sl.set_facecolor('white')
            for sp in ax_sl.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor('#CCCCCC')

            init = getattr(self, attr)
            sl = Slider(ax_sl, label, vmin, vmax,
                        valinit=init, valstep=vstep,
                        color='#1F5FA6')
            sl.label.set_fontsize(11)
            sl.label.set_color('#222222')
            sl.label.set_fontfamily('monospace')
            sl.valtext.set_fontsize(11)
            sl.valtext.set_color('#1F5FA6')
            sl.valtext.set_fontfamily('monospace')

            sl.on_changed(lambda val, a=attr: self._on_slider(a, val))
            self.sliders[attr] = sl

    # ── Preset buttons ────────────────────────────────────────────────────────

    def _build_presets(self):
        names  = list(PRESETS.keys())
        n      = len(names)
        btn_w  = 0.14
        gap    = 0.012
        total  = n * btn_w + (n-1) * gap
        start  = (1.0 - total) / 2.0
        btn_y  = 0.245   # just above sliders
        btn_h  = 0.034

        self.preset_btns = []
        for k, (name, col) in enumerate(zip(names, PRESET_COLORS)):
            ax_b = self.fig.add_axes(
                [start + k*(btn_w+gap), btn_y, btn_w, btn_h])
            btn = Button(ax_b, name, color='white', hovercolor='#F5F5F5')
            btn.label.set_fontsize(11)
            btn.label.set_color(col)
            btn.label.set_fontweight('bold')
            btn.label.set_fontfamily('monospace')
            for sp in btn.ax.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(2.0)
            btn.on_clicked(lambda evt, nm=name: self._load_preset(nm))
            self.preset_btns.append(btn)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _redraw(self):
        C = capacity_matrix(
            self.d_max, self.M,
            self.r, self.gamma,
            self.N_res, self.M_actual,
            self.n_max)
        self._draw_heatmap(C)
        self._draw_profiles(C)
        self.fig.canvas.draw_idle()

    def _draw_heatmap(self, C):
        ax = self.ax_heat
        ax.cla()

        d_max = self.d_max
        M     = self.M

        im = ax.imshow(
            C,
            aspect='auto',
            origin='lower',
            cmap=CMAP,
            vmin=0.0, vmax=1.0,
            extent=[-0.5, M + 0.5, 0.5, d_max + 0.5])

        # Remove any existing colorbar axes to avoid overlap
        if hasattr(self, '_cbar_ax') and self._cbar_ax is not None:
            try:
                self._cbar_ax.remove()
            except Exception:
                pass

        self._cbar_ax = self.fig.add_axes(
            [self.ax_heat.get_position().x1 + 0.01,
             self.ax_heat.get_position().y0,
             0.012,
             self.ax_heat.get_position().height])
        cb = self.fig.colorbar(im, cax=self._cbar_ax)
        cb.set_label('C(d, m)', fontsize=12)
        cb.ax.tick_params(labelsize=10)

        # Cell value annotations — font size scales with grid
        cell_fs = max(6, 11 - max(d_max, M) // 3)
        for d_idx in range(d_max):
            for m in range(M + 1):
                val = C[d_idx, m]
                txt_col = '#111111' if val < 0.55 else 'white'
                ax.text(m, d_idx + 1, f'{val:.2f}',
                        ha='center', va='center',
                        fontsize=cell_fs, color=txt_col,
                        fontfamily='monospace')

        # Dashed markers
        ax.axhline(y=1.5, color='#2471A3', lw=1.8, ls='--', alpha=0.75,
                   label='Linear MC  (d=1)')
        ax.axvline(x=0.5, color='#1E8449', lw=1.8, ls='--', alpha=0.75,
                   label='Instant. NL  (m=0)')

        ax.set_xlabel('Memory depth  m', fontsize=13, labelpad=6)
        ax.set_ylabel('Polynomial degree  d', fontsize=13, labelpad=6)

        x_ticks = range(0, M + 1, max(1, M // 8))
        ax.set_xticks(list(x_ticks))
        ax.set_xticklabels([str(x) for x in x_ticks], fontsize=10)
        ax.set_yticks(range(1, d_max + 1))
        ax.set_yticklabels([str(d) for d in range(1, d_max + 1)], fontsize=10)

        ax.set_title(
            f'Capacity Matrix  C(d, m)\n'
            f'r={self.r:.2f}   γ={self.gamma:.2f}   '
            f'N_res={self.N_res}   M_actual={self.M_actual}   '
            f'n_max={self.n_max}   '
            f'Total IPC = {np.sum(C):.2f}',
            fontsize=12, pad=8)

        ax.legend(loc='upper right', fontsize=10,
                  facecolor='white', edgecolor='#CCCCCC',
                  framealpha=0.9)

    def _draw_profiles(self, C):
        m_ax = np.arange(0, self.M + 1)
        d_ax = np.arange(1, self.d_max + 1)

        # ── Linear memory: C(1, m) ────────────────────────────────────────────
        ax = self.ax_linear
        ax.cla()
        ax.plot(m_ax, C[0, :], color='#2471A3', lw=2.2,
                marker='o', ms=5, label='C(d=1, m)')
        ax.fill_between(m_ax, C[0, :], alpha=0.15, color='#2471A3')
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel('Memory depth  m', fontsize=11)
        ax.set_ylabel('C(d, m)', fontsize=11)
        ax.set_title('Linear memory profile  C(d=1, m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        mc_total = float(np.sum(C[0, :]))
        ax.text(0.97, 0.88, f'MC = {mc_total:.2f}',
                transform=ax.transAxes, ha='right', fontsize=11,
                color='#2471A3',
                bbox=dict(boxstyle='round,pad=0.35', fc='white',
                          ec='#2471A3', alpha=0.9))

        # ── Nonlinearity: C(d, 0) ─────────────────────────────────────────────
        ax = self.ax_nonlin
        ax.cla()
        ax.plot(d_ax, C[:, 0], color='#1E8449', lw=2.2,
                marker='s', ms=5)
        ax.fill_between(d_ax, C[:, 0], alpha=0.15, color='#1E8449')
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks(list(d_ax))
        ax.set_xlabel('Polynomial degree  d', fontsize=11)
        ax.set_ylabel('C(d, m)', fontsize=11)
        ax.set_title('Nonlinearity profile  C(d, m=0)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # ── All degree profiles ───────────────────────────────────────────────
        ax = self.ax_alldeg
        ax.cla()
        cmap_deg = plt.cm.plasma
        for d_idx in range(self.d_max):
            d = d_idx + 1
            col = cmap_deg(d_idx / max(self.d_max - 1, 1))
            ax.plot(m_ax, C[d_idx, :], color=col, lw=1.8,
                    marker='.', ms=4, label=f'd={d}')

        # Normalization floor envelope: 1/(d*n_max*(m+1))
        # shown for d=1 only
        env = 1.0 / (1 * self.n_max * (m_ax + 1))
        ax.plot(m_ax, env, color='#999999', lw=1.2, ls='dashed',
                alpha=0.7, label='1/(nmax·(m+1)) envelope')

        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel('Memory depth  m', fontsize=11)
        ax.set_ylabel('C(d, m)', fontsize=11)
        ax.set_title('All degree profiles  C(d, m)', fontsize=12)
        ax.legend(fontsize=8, ncol=3, loc='upper right',
                  facecolor='white', edgecolor='#CCCCCC', framealpha=0.9)
        ax.grid(True, alpha=0.3)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_slider(self, attr, val):
        if self._suppress_callbacks:
            return
        # Cast to correct type
        if attr in ('N_res', 'M_actual', 'n_max', 'd_max', 'M'):
            setattr(self, attr, int(val))
        else:
            setattr(self, attr, float(val))
        self._redraw()

    def _load_preset(self, name):
        """Load preset values without triggering intermediate redraws."""
        p = PRESETS[name]
        self._suppress_callbacks = True

        self.r        = p['r']
        self.gamma    = p['gamma']
        self.N_res    = p['N_res']
        self.M_actual = p['M_actual']

        # Update slider positions to match
        self.sliders['r'].set_val(p['r'])
        self.sliders['gamma'].set_val(p['gamma'])
        self.sliders['N_res'].set_val(p['N_res'])
        self.sliders['M_actual'].set_val(p['M_actual'])

        self._suppress_callbacks = False
        self._redraw()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Capacity Matrix Explorer")
    print("=" * 52)
    print("Based on Eq. 35:  C(d,m) = SA(P*) · SD(Q*) / |G|")
    print()
    print("Sliders:")
    print("  r         — amplitude ratio A/a  → harmonic generation")
    print("  γ         — damping              → memory decay")
    print("  N_res     — resonator count      → hard degree cutoff")
    print("  M_actual  — chain length         → hard memory cutoff")
    print("  n_max     — highest input harmonic")
    print("  d_max     — max polynomial degree to display")
    print("  M         — max memory depth to display")
    print()
    print("Preset buttons reproduce Figure 6 (a)-(e) of the paper.")
    print()

    app = CapacityExplorer()