#!/usr/bin/env python3
"""
EM Wave Propagation Animation
==============================
Animates electromagnetic wavefronts propagating through a multilayer
DGS filter circuit, showing the field on each PCB layer.

Creates:
  - Animated GIF showing time-varying E-field on all layers
  - Static snapshots comparing passband vs stopband propagation
  - Per-layer field distribution at selected time instants

The animation reveals how:
  - Passband signals propagate through with standing-wave ripple
  - Stopband signals are reflected, forming strong standing waves
  - DGS elements on ground layers create evanescent field leakage
  - Coupled patches on inner layers show resonant field enhancement
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
Z_REF = 50.0
ER = 10.2
TAND = 0.0035
H_SUB = 0.254e-3
EEFF = 6.84
W50 = 0.234e-3

N_RES = 5
N_SPATIAL = 400
N_FRAMES = 60
FPS = 12

F_PASS = 4.3e9
F_STOP = 2.0e9
FC = 4.375e9
FBW = 0.286

# ═══════════════════════════════════════════════════════════════════════
#  Simple J-inverter BPF model (self-contained)
# ═══════════════════════════════════════════════════════════════════════
def _m(a, b, c, d):
    return np.array([[a, b], [c, d]], dtype=np.complex128)

def _tline_abcd(z0, f, length):
    beta = 2 * np.pi * f * np.sqrt(EEFF) / C0
    alpha = TAND * np.pi * f * np.sqrt(EEFF) / C0
    gl = (alpha + 1j * beta) * length
    ch, sh = np.cosh(gl), np.sinh(gl)
    return _m(ch, z0 * sh, sh / z0, ch)

def _jinv_abcd(J):
    return _m(0j, -1j / J, -1j * J, 0j)

def _design_bpf():
    g = [1.0, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0]
    J_norm = [np.sqrt(np.pi * FBW / (2 * g[0] * g[1]))]
    for i in range(1, N_RES):
        J_norm.append(np.pi * FBW / (2 * np.sqrt(g[i] * g[i + 1])))
    J_norm.append(np.sqrt(np.pi * FBW / (2 * g[N_RES] * g[N_RES + 1])))
    J_vals = [jn / Z_REF for jn in J_norm]
    lam2 = C0 / (FC * np.sqrt(EEFF)) / 2
    res_lengths = [lam2] * N_RES
    return J_vals, res_lengths

J_VALS, RES_LENGTHS = _design_bpf()

TOTAL_LEN = sum(RES_LENGTHS)
SPACING_IN = 3e-3
SPACING_OUT = 3e-3
FULL_LEN = SPACING_IN + TOTAL_LEN + SPACING_OUT

# Element positions along feedline
ELEM_POSITIONS = []
x = SPACING_IN
for k in range(N_RES):
    ELEM_POSITIONS.append(x)
    x += RES_LENGTHS[k]


def _compute_field_profile(f, n_pts=N_SPATIAL):
    """Compute complex voltage V(x) at n_pts positions along the feedline."""

    sections = []
    sections.append(("tl", SPACING_IN))
    for k in range(N_RES):
        sections.append(("jinv", J_VALS[k]))
        sections.append(("tl", RES_LENGTHS[k]))
    sections.append(("jinv", J_VALS[N_RES]))
    sections.append(("tl", SPACING_OUT))

    M_list = []
    for kind, val in sections:
        if kind == "tl":
            M_list.append(_tline_abcd(Z_REF, f, val))
        else:
            M_list.append(_jinv_abcd(val))

    M_total = np.eye(2, dtype=np.complex128)
    for M in M_list:
        M_total = M_total @ M

    A, B, C, D = M_total[0, 0], M_total[0, 1], M_total[1, 0], M_total[1, 1]
    den = A + B / Z_REF + C * Z_REF + D
    S21 = 2.0 / den
    S11 = (A + B / Z_REF - C * Z_REF - D) / den

    V_out = S21
    I_out = S21 / Z_REF

    node_V = [V_out]
    node_I = [I_out]
    node_x = [FULL_LEN]

    M_from_right = np.eye(2, dtype=np.complex128)
    cur_x = FULL_LEN

    for kind, val in reversed(sections):
        if kind == "tl":
            M_sec = _tline_abcd(Z_REF, f, val)
            cur_x -= val
        else:
            M_sec = _jinv_abcd(val)

        M_from_right = M_sec @ M_from_right
        V_node = M_from_right[0, 0] * V_out + M_from_right[0, 1] * I_out
        I_node = M_from_right[1, 0] * V_out + M_from_right[1, 1] * I_out
        node_V.insert(0, V_node)
        node_I.insert(0, I_node)
        node_x.insert(0, cur_x)

    node_x = np.array(node_x)
    node_V = np.array(node_V)

    x_pts = np.linspace(0, FULL_LEN, n_pts)
    V_profile = np.interp(x_pts, node_x, np.real(node_V)) + 1j * np.interp(
        x_pts, node_x, np.imag(node_V)
    )

    return x_pts, V_profile, float(np.abs(S21)), float(np.abs(S11))


def _transverse_profile(V_line, width_mm=12.0, n_y=60):
    """Expand 1D V(x) into 2D V(x,y) with Gaussian transverse decay."""
    y = np.linspace(-width_mm / 2, width_mm / 2, n_y)
    sigma = 1.5
    envelope = np.exp(-(y ** 2) / (2 * sigma ** 2))
    return np.outer(envelope, V_line)


def _field_on_layer(V_line, layer_type, elem_pos_idx, x_pts, width_mm=12.0, n_y=60):
    """Compute 2D field pattern for a specific layer."""
    y = np.linspace(-width_mm / 2, width_mm / 2, n_y)

    if layer_type == "feedline":
        sigma = 1.2
        env = np.exp(-(y ** 2) / (2 * sigma ** 2))
        return np.outer(env, V_line)

    elif layer_type == "ground_dgs":
        field = np.zeros((n_y, len(V_line)), dtype=np.complex128)
        for idx in elem_pos_idx:
            x_center = ELEM_POSITIONS[idx] if idx < len(ELEM_POSITIONS) else 0
            x_spread = RES_LENGTHS[min(idx, N_RES - 1)] * 0.3
            x_env = np.exp(-((x_pts - x_center) ** 2) / (2 * x_spread ** 2))
            y_spread = 3.0
            y_env = np.exp(-(y ** 2) / (2 * y_spread ** 2)) * (1 - np.exp(-(y ** 2) / 0.5))
            field += 0.3 * np.outer(y_env, V_line * x_env)
        return field

    elif layer_type == "patches":
        field = np.zeros((n_y, len(V_line)), dtype=np.complex128)
        for idx in elem_pos_idx:
            x_center = ELEM_POSITIONS[idx] if idx < len(ELEM_POSITIONS) else 0
            patch_l = RES_LENGTHS[min(idx, N_RES - 1)] * 0.6
            x_env = np.where(
                np.abs(x_pts - x_center) < patch_l / 2,
                np.cos(np.pi * (x_pts - x_center) / patch_l),
                0.0,
            )
            y_env = np.where(np.abs(y) < 2.5, 1.0, 0.0) * np.cos(np.pi * y / 5.0)
            field += 0.5 * np.outer(y_env, V_line * x_env)
        return field

    return np.zeros((n_y, len(V_line)), dtype=np.complex128)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    generated = []
    x_mm = None

    layer_configs = [
        ("L1 Feedline", "feedline", []),
        ("L2 Ground (DGS)", "ground_dgs", [0, 1, 2, 3, 4]),
        ("L3 Coupled Patches", "patches", [0, 2, 4]),
        ("L4 Secondary DGS", "ground_dgs", [1, 3]),
    ]
    n_layers = len(layer_configs)
    width_mm = 10.0
    n_y = 50

    freq_cases = [
        ("Passband (4.3 GHz)", F_PASS, "pass"),
        ("Stopband (2.0 GHz)", F_STOP, "stop"),
    ]

    # ── Static snapshots ─────────────────────────────────────────────
    for freq_label, freq, tag in freq_cases:
        x_pts, V_prof, s21_mag, s11_mag = _compute_field_profile(freq)
        x_mm = x_pts * 1e3

        fig, axes = plt.subplots(n_layers, 1, figsize=(14, 2.5 * n_layers + 1))
        fig.suptitle(
            f"EM Wave Propagation — {freq_label}\n"
            f"|S21| = {20*np.log10(s21_mag+1e-12):.1f} dB   "
            f"|S11| = {20*np.log10(s11_mag+1e-12):.1f} dB",
            fontsize=13, fontweight="bold",
        )

        t_snap = 0.0
        omega = 2 * np.pi * freq

        for li, (lname, ltype, elem_idx) in enumerate(layer_configs):
            field_2d = _field_on_layer(V_prof, ltype, elem_idx, x_pts, width_mm, n_y)
            inst_field = np.real(field_2d * np.exp(1j * omega * t_snap))

            ax = axes[li]
            vmax = max(np.max(np.abs(inst_field)), 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            ax.imshow(
                inst_field,
                aspect="auto",
                extent=[x_mm[0], x_mm[-1], -width_mm / 2, width_mm / 2],
                cmap="RdBu_r",
                norm=norm,
                origin="lower",
                interpolation="bilinear",
            )

            for idx in range(N_RES):
                xp = ELEM_POSITIONS[idx] * 1e3
                ax.axvline(xp, color="#333", lw=0.5, alpha=0.4)

            ax.set_ylabel(f"{lname}\n[mm]", fontsize=8)
            if li < n_layers - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Position along feedline [mm]")

        fig.tight_layout()
        fname = f"em_wave_{tag}.png"
        fig.savefig(fname, dpi=180)
        plt.close(fig)
        generated.append(fname)
        print(f"  [saved] {fname}")

    # ── Animated GIF (passband) ──────────────────────────────────────
    print("  Generating animation frames...", end="", flush=True)
    x_pts, V_prof, _, _ = _compute_field_profile(F_PASS)
    x_mm_anim = x_pts * 1e3
    omega = 2 * np.pi * F_PASS
    period = 1.0 / F_PASS
    times = np.linspace(0, period, N_FRAMES, endpoint=False)

    layer_fields = []
    for lname, ltype, elem_idx in layer_configs:
        layer_fields.append(
            _field_on_layer(V_prof, ltype, elem_idx, x_pts, width_mm, n_y)
        )

    fig_a, axes_a = plt.subplots(n_layers, 1, figsize=(12, 2.2 * n_layers + 0.8))
    fig_a.suptitle(
        "EM Wave Propagation — Passband (4.3 GHz)", fontsize=12, fontweight="bold"
    )

    ims_list = []
    vmax_global = max(
        np.max(np.abs(np.real(lf))) for lf in layer_fields
    )
    vmax_global = max(vmax_global, 1e-6)

    for li, (lname, ltype, elem_idx) in enumerate(layer_configs):
        axes_a[li].set_ylabel(f"{lname}\n[mm]", fontsize=7)
        if li < n_layers - 1:
            axes_a[li].set_xticks([])
        else:
            axes_a[li].set_xlabel("Position [mm]")
        for idx in range(N_RES):
            xp = ELEM_POSITIONS[idx] * 1e3
            axes_a[li].axvline(xp, color="#333", lw=0.3, alpha=0.3)

    def _make_frame(frame_idx):
        t = times[frame_idx]
        arts = []
        for li in range(n_layers):
            inst = np.real(layer_fields[li] * np.exp(1j * omega * t))
            norm = TwoSlopeNorm(vmin=-vmax_global, vcenter=0, vmax=vmax_global)
            im = axes_a[li].imshow(
                inst,
                aspect="auto",
                extent=[x_mm_anim[0], x_mm_anim[-1], -width_mm / 2, width_mm / 2],
                cmap="RdBu_r",
                norm=norm,
                origin="lower",
                interpolation="bilinear",
                animated=True,
            )
            arts.append(im)
        return arts

    init_arts = _make_frame(0)

    def _update(frame_idx):
        t = times[frame_idx]
        for li in range(n_layers):
            inst = np.real(layer_fields[li] * np.exp(1j * omega * t))
            init_arts[li].set_data(inst)
        return init_arts

    ani = animation.FuncAnimation(
        fig_a, _update, frames=N_FRAMES, interval=1000 // FPS, blit=True
    )

    try:
        ani.save("em_propagation.gif", writer=animation.PillowWriter(fps=FPS))
        generated.append("em_propagation.gif")
        print(" done")
        print(f"  [saved] em_propagation.gif  ({N_FRAMES} frames, {FPS} fps)")
    except Exception as e:
        print(f" GIF save failed: {e}")
        for fi in range(min(8, N_FRAMES)):
            _update(fi * (N_FRAMES // 8))
            fname = f"em_frame_{fi:02d}.png"
            fig_a.savefig(fname, dpi=120)
            generated.append(fname)
        print(f"  [saved] {len(generated)} individual frames instead")
    plt.close(fig_a)

    # ── Combined passband vs stopband comparison ─────────────────────
    fig_c, axes_c = plt.subplots(2, n_layers, figsize=(4 * n_layers, 6))
    fig_c.suptitle(
        "EM Wavefronts: Passband vs Stopband", fontsize=13, fontweight="bold"
    )

    for fi, (freq_label, freq, tag) in enumerate(freq_cases):
        x_pts, V_prof, s21_m, s11_m = _compute_field_profile(freq)

        for li, (lname, ltype, elem_idx) in enumerate(layer_configs):
            field_2d = _field_on_layer(V_prof, ltype, elem_idx, x_pts, width_mm, n_y)
            inst = np.real(field_2d)
            vmax = max(np.max(np.abs(inst)), 1e-6)
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            ax = axes_c[fi, li]
            ax.imshow(
                inst,
                aspect="auto",
                extent=[x_pts[0] * 1e3, x_pts[-1] * 1e3, -width_mm / 2, width_mm / 2],
                cmap="RdBu_r",
                norm=norm,
                origin="lower",
                interpolation="bilinear",
            )
            if fi == 0:
                ax.set_title(lname, fontsize=9)
            if li == 0:
                short_label = freq_label.split("(")[0].strip()
                ax.set_ylabel(f"{short_label}\n[mm]", fontsize=8)
            else:
                ax.set_yticks([])
            if fi < 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("mm")

    fig_c.tight_layout()
    fig_c.savefig("em_wave_comparison.png", dpi=200)
    plt.close(fig_c)
    generated.append("em_wave_comparison.png")
    print(f"  [saved] em_wave_comparison.png")

    # ── Frequency response of the demo filter ────────────────────────
    freqs_plot = np.linspace(0.5e9, 7e9, 500)
    s21_list, s11_list = [], []
    for f in freqs_plot:
        _, _, s21m, s11m = _compute_field_profile(f, n_pts=50)
        s21_list.append(s21m)
        s11_list.append(s11m)

    s21_db = 20 * np.log10(np.array(s21_list) + 1e-12)
    s11_db = 20 * np.log10(np.array(s11_list) + 1e-12)
    fg = freqs_plot / 1e9

    fig_r, ax_r = plt.subplots(figsize=(9, 5))
    ax_r.plot(fg, s21_db, "b", lw=1.5, label="|S21|")
    ax_r.plot(fg, s11_db, "r--", lw=1, label="|S11|")
    ax_r.axvline(F_PASS / 1e9, color="green", ls=":", lw=1, label=f"Passband demo ({F_PASS/1e9:.1f} GHz)")
    ax_r.axvline(F_STOP / 1e9, color="orange", ls=":", lw=1, label=f"Stopband demo ({F_STOP/1e9:.1f} GHz)")
    ax_r.set_title("Demo Filter: BPF 3.75-5.0 GHz (5th-order J-inverter)", fontweight="bold")
    ax_r.set_xlabel("Frequency [GHz]")
    ax_r.set_ylabel("[dB]")
    ax_r.set_ylim(-50, 3)
    ax_r.legend()
    ax_r.grid(True, alpha=0.3)
    fig_r.tight_layout()
    fig_r.savefig("em_demo_filter_response.png", dpi=200)
    plt.close(fig_r)
    generated.append("em_demo_filter_response.png")
    print(f"  [saved] em_demo_filter_response.png")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  OUTPUT FILES")
    print("=" * 60)
    for img in generated:
        sz = os.path.getsize(img) / 1024
        ext = os.path.splitext(img)[1]
        print(f"  {img:40s}  {sz:6.0f} KB  {ext}")
    print()
    print("=" * 60)
    print("  Complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
