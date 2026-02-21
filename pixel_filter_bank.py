#!/usr/bin/env python3
"""
Pixel / DGS Topology-Optimised Multilayer Filter Bank
=====================================================
Designs RF filters by JAX-optimising pixel patterns etched into
PCB ground and resonator layers — the same idea as shaping the
ground plane under a feedline for notch/bandpass behaviour.

Physical model
--------------
A 50-ohm microstrip feedline runs along the x-axis on L1.
Below it, the ground plane (L2) is divided into an Nx x Ny
pixel grid.  Each pixel is either metal (present) or void
(etched away).  Removing ground pixels:
  - raises the local characteristic impedance (less C to ground)
  - creates DGS resonances (series-LC in the shunt path)
  - produces frequency-selective reflections

A second signal layer (L3), separated from L2 by thin prepreg,
carries additional coupled pixel patches.  These couple to the
feedline through L2 apertures, adding resonant poles that can
convert band-stop DGS behaviour into band-pass.

The pixel pattern is optimised end-to-end with JAX autodiff:
  pixel grid  -->  local Z0 + shunt DGS admittance
              -->  cascaded ABCD matrices
              -->  S-parameters  -->  loss function

Manufacturing constraints (minimum feature size, binarisation
pressure, bilateral symmetry) are built into the cost function.

Filter bank
-----------
  1. High-pass   fc = 2500 MHz
  2. Bandpass 1   2500 - 3750 MHz
  3. Bandpass 2   3750 - 5000 MHz
"""

import os

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Constants & substrate
# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
EPS0 = 8.854_187_817e-12
Z_REF = 50.0
ER = 3.55
H_SUB = 0.508e-3
TAND = 0.0027
ER_PP = 3.52
H_COUPLE = 0.100e-3

# Microstrip reference parameters at 50 ohm
_W50 = 1.136e-3
_EEFF50 = (ER + 1) / 2 + (ER - 1) / 2 * (1 + 12 * H_SUB / _W50) ** (-0.5)

# ═══════════════════════════════════════════════════════════════════════
#  Pixel grid parameters
# ═══════════════════════════════════════════════════════════════════════
F_CENTER = 3.5e9
LAM0 = C0 / F_CENTER
DX = LAM0 / 16
DY = DX
NX = 28
NY = 12
NY_HALF = NY // 2
GRID_LEN = NX * DX
GRID_WID = NY * DY

FEED_J = NY // 2
_decay = 1.8
_wj = np.exp(-np.abs(np.arange(NY) - FEED_J + 0.5) / _decay)
_wj = _wj / _wj.sum()
WJ = jnp.array(_wj)

L_DGS_0 = 1.2e-9
C_DGS_0 = 0.08e-12
C_COUPLE_0 = 0.04e-12

# ═══════════════════════════════════════════════════════════════════════
#  ABCD helpers
# ═══════════════════════════════════════════════════════════════════════
def _m(a, b, c, d):
    return jnp.array([[a, b], [c, d]], dtype=jnp.complex128)

def abcd_tline(z0, gl):
    ch, sh = jnp.cosh(gl), jnp.sinh(gl)
    return _m(ch, z0 * sh, sh / z0, ch)

def abcd_shunt(Y):
    return _m(1.0 + 0j, 0j, Y, 1.0 + 0j)

def abcd_to_s(M):
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    den = A + B / Z_REF + C * Z_REF + D
    return 2.0 / den, (A + B / Z_REF - C * Z_REF - D) / den


# ═══════════════════════════════════════════════════════════════════════
#  Pixel EM model
# ═══════════════════════════════════════════════════════════════════════
def _pixel_response_at_f(f, L2, L3, temp):
    """S21, S11 for one frequency given pixel grids."""

    def body(M, x):
        l2_row, l3_row = x

        gc = jnp.dot(l2_row, WJ)
        gc = jnp.clip(gc, 0.05, 1.0)
        void = 1.0 - gc

        z0_i = Z_REF / jnp.sqrt(gc)
        eeff_i = _EEFF50 * gc

        beta = 2 * jnp.pi * f * jnp.sqrt(eeff_i) / C0
        alpha = TAND * jnp.pi * f * jnp.sqrt(eeff_i) / C0
        gl = (alpha + 1j * beta) * DX
        M_tl = abcd_tline(z0_i, gl)

        omega = 2 * jnp.pi * f
        L_dgs = L_DGS_0 * void
        C_dgs = C_DGS_0 * gc * void
        Z_dgs = 1j * omega * L_dgs + 1.0 / (1j * omega * C_dgs + 1e-25)
        Y_dgs = void / (Z_dgs + 1e-20)

        l3_c = jnp.dot(l3_row, WJ)
        Y_couple = 1j * omega * C_COUPLE_0 * l3_c * void

        Y_total = Y_dgs + Y_couple

        return M @ M_tl @ abcd_shunt(Y_total), None

    L2_sig = jax.nn.sigmoid(L2 * temp)
    L3_sig = jax.nn.sigmoid(L3 * temp)

    M0 = jnp.eye(2, dtype=jnp.complex128)
    M_f, _ = jax.lax.scan(body, M0, (L2_sig, L3_sig))
    return abcd_to_s(M_f)


_pixel_batch = jax.vmap(
    lambda f, L2, L3, t: _pixel_response_at_f(f, L2, L3, t),
    in_axes=(0, None, None, None),
)


# ═══════════════════════════════════════════════════════════════════════
#  Symmetry & manufacturing constraints
# ═══════════════════════════════════════════════════════════════════════
def _apply_symmetry(half):
    """Mirror upper half to create full Nx x Ny grid."""
    lower = jnp.flip(half, axis=1)
    return jnp.concatenate([lower, half], axis=1)


def _manufacturing_penalty(pixels_sig):
    """Penalise isolated pixels and non-binary values."""
    p = pixels_sig
    pad = jnp.pad(p, 1, mode="edge")
    avg = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
    isolation = jnp.mean((p - avg) ** 2)
    binarisation = jnp.mean(p * (1 - p))
    return isolation * 0.15 + binarisation * 0.10


# ═══════════════════════════════════════════════════════════════════════
#  Loss function builders
# ═══════════════════════════════════════════════════════════════════════
def _make_loss(target_fn, freqs):
    """Build a loss function for a given target specification."""

    def loss_fn(params, temp):
        L2_half = params[: NX * NY_HALF].reshape(NX, NY_HALF)
        L3_half = params[NX * NY_HALF :].reshape(NX, NY_HALF)

        L2 = _apply_symmetry(L2_half)
        L3 = _apply_symmetry(L3_half)

        s21, s11 = _pixel_batch(freqs, L2, L3, temp)
        s21_db = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11_db = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))

        spec_loss = target_fn(freqs, s21_db, s11_db)

        L2_sig = jax.nn.sigmoid(L2 * temp)
        L3_sig = jax.nn.sigmoid(L3 * temp)
        mfg = _manufacturing_penalty(L2_sig) + _manufacturing_penalty(L3_sig)

        return spec_loss + mfg

    return loss_fn


def _hpf_target(freqs, s21_db, s11_db):
    fc = 2.5e9
    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    loss = jnp.sum(jnp.where(pb, jnp.maximum(-s21_db - 1.0, 0.0) ** 2, 0.0)) * 5
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11_db + 10, 0.0) ** 2, 0.0)) * 2
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21_db + 15, 0.0) ** 2, 0.0)) * 3
    return loss / freqs.shape[0]


def _bpf_target_factory(f_lo, f_hi):
    bw = f_hi - f_lo

    def target(freqs, s21_db, s11_db):
        pb = (freqs >= f_lo) & (freqs <= f_hi)
        sl = freqs < (f_lo - 0.3 * bw)
        sh = freqs > (f_hi + 0.3 * bw)
        loss = jnp.sum(jnp.where(pb, jnp.maximum(-s21_db - 2.0, 0.0) ** 2, 0.0)) * 5
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11_db + 8, 0.0) ** 2, 0.0)) * 2
        loss += jnp.sum(jnp.where(sl, jnp.maximum(s21_db + 12, 0.0) ** 2, 0.0)) * 3
        loss += jnp.sum(jnp.where(sh, jnp.maximum(s21_db + 12, 0.0) ** 2, 0.0)) * 3
        return loss / freqs.shape[0]

    return target


# ═══════════════════════════════════════════════════════════════════════
#  Optimiser with temperature annealing
# ═══════════════════════════════════════════════════════════════════════
def optimise_pixels(loss_fn, n_params, *, n_steps=1500, lr=0.08, seed=0):
    key = jax.random.PRNGKey(seed)
    params = jax.random.normal(key, (n_params,)) * 0.3

    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    state = opt.init(params)

    @jax.jit
    def step(params, state, temp):
        loss, grads = jax.value_and_grad(loss_fn)(params, temp)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, ns = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), ns, loss

    best_p, best_l = params, float("inf")
    for i in range(n_steps):
        temp = 1.0 + 4.0 * i / n_steps
        params, state, lv = step(params, state, temp)
        lv_f = float(lv)
        if jnp.isfinite(lv) and lv_f < best_l:
            best_l, best_p = lv_f, params
        if (i + 1) % 500 == 0 or i == 0:
            print(
                f"  step {i+1:5d}/{n_steps}  loss={lv_f:.4f}  "
                f"best={best_l:.4f}  temp={temp:.1f}"
            )

    return best_p, best_l


def optimise_restarts(loss_fn, n_params, n_restarts=3, **kw):
    best_p, best_l = None, float("inf")
    for r in range(n_restarts):
        if n_restarts > 1:
            print(f"  (restart {r+1}/{n_restarts})")
        p, lv = optimise_pixels(loss_fn, n_params, seed=r * 71, **kw)
        if lv < best_l:
            best_l, best_p = lv, p
    return best_p


# ═══════════════════════════════════════════════════════════════════════
#  Extract binarised pixel grids
# ═══════════════════════════════════════════════════════════════════════
def _extract_grids(params, temp=5.0):
    L2_half = params[: NX * NY_HALF].reshape(NX, NY_HALF)
    L3_half = params[NX * NY_HALF :].reshape(NX, NY_HALF)
    L2 = _apply_symmetry(L2_half)
    L3 = _apply_symmetry(L3_half)
    L2_bin = (jax.nn.sigmoid(L2 * temp) > 0.5).astype(jnp.float64)
    L3_bin = (jax.nn.sigmoid(L3 * temp) > 0.5).astype(jnp.float64)
    return np.array(L2_bin), np.array(L3_bin)


# ═══════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════
_copper_cmap = LinearSegmentedColormap.from_list(
    "copper_pcb", ["#1a1a2e", "#b87333"], N=2
)


def _plot_pixel_grid(ax, grid, title, dx_mm, dy_mm):
    """Draw a pixel grid with physical dimensions."""
    nx, ny = grid.shape
    extent = [0, nx * dx_mm, 0, ny * dy_mm]
    ax.imshow(
        grid.T,
        origin="lower",
        cmap=_copper_cmap,
        aspect="equal",
        extent=extent,
        interpolation="nearest",
    )
    feed_y = ny * dy_mm / 2
    ax.axhline(feed_y, color="#e74c3c", lw=2, ls="--", alpha=0.8, label="Feedline")
    ax.set_xlabel("Along feedline [mm]")
    ax.set_ylabel("Perpendicular [mm]")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=8)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs = jnp.linspace(0.1e9, 7e9, 400)
    n_params = NX * NY_HALF * 2
    generated = []
    dx_mm = DX * 1e3
    dy_mm = DY * 1e3

    filters = [
        ("HPF (fc = 2.5 GHz)", _hpf_target, "pixel_hpf"),
        ("BPF1 (2.5-3.75 GHz)", _bpf_target_factory(2.5e9, 3.75e9), "pixel_bpf1"),
        ("BPF2 (3.75-5.0 GHz)", _bpf_target_factory(3.75e9, 5.0e9), "pixel_bpf2"),
    ]

    print("=" * 68)
    print("  Pixel / DGS Topology-Optimised Multilayer Filter Bank")
    print("=" * 68)
    print(f"  Pixel grid     : {NX} x {NY}  ({dx_mm:.2f} x {dy_mm:.2f} mm each)")
    print(f"  Filter length  : {GRID_LEN*1e3:.1f} mm")
    print(f"  Ground width   : {GRID_WID*1e3:.1f} mm")
    print(f"  Parameters     : {n_params} per filter (bilateral symmetry)")
    print(f"  Layers         : L2 ground DGS + L3 coupled resonators")
    print()

    results = []
    for name, target_fn, prefix in filters:
        print("=" * 68)
        print(f"  Optimising  {name}")
        print("=" * 68)
        loss_fn = _make_loss(target_fn, freqs)
        p_opt = optimise_restarts(
            loss_fn, n_params, n_restarts=2, n_steps=1500, lr=0.08
        )
        L2_bin, L3_bin = _extract_grids(p_opt)
        results.append(dict(name=name, prefix=prefix, params=p_opt,
                            L2=L2_bin, L3=L3_bin))
        print()

    # ── Evaluate with binarised grids ────────────────────────────────
    fp = jnp.linspace(0.1e9, 7e9, 800)
    fg = np.array(fp) / 1e9
    db = lambda s: 20 * np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))

    for res in results:
        L2 = jnp.array(res["L2"])
        L3 = jnp.array(res["L3"])
        s21, s11 = _pixel_batch(fp, L2, L3, 50.0)
        res["s21"] = s21
        res["s11"] = s11

    # ── Per-filter images: pixel artwork + response ──────────────────
    for res in results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Pixel Filter: {res['name']}", fontsize=14, fontweight="bold"
        )

        _plot_pixel_grid(axes[0, 0], res["L2"],
                         "L2 — Ground Plane (DGS pattern)", dx_mm, dy_mm)
        _plot_pixel_grid(axes[0, 1], res["L3"],
                         "L3 — Coupled Resonator Patches", dx_mm, dy_mm)

        ax = axes[1, 0]
        ax.plot(fg, db(res["s21"]), "b", lw=1.5, label="|S21|")
        ax.plot(fg, db(res["s11"]), "r--", lw=1.0, label="|S11|")
        ax.set_title("Frequency Response (binarised)")
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("[dB]")
        ax.set_ylim(-40, 3)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        pix_metal_L2 = float(res["L2"].sum()) / (NX * NY) * 100
        pix_metal_L3 = float(res["L3"].sum()) / (NX * NY) * 100
        info = (
            f"Grid: {NX} x {NY} pixels\n"
            f"Pixel size: {dx_mm:.2f} x {dy_mm:.2f} mm\n"
            f"Board area: {GRID_LEN*1e3:.1f} x {GRID_WID*1e3:.1f} mm\n\n"
            f"L2 metal fill: {pix_metal_L2:.0f}%\n"
            f"L3 metal fill: {pix_metal_L3:.0f}%\n\n"
            f"Substrate: RO4003C, er={ER}\n"
            f"Coupling prepreg: {H_COUPLE*1e3:.1f} mm, er={ER_PP}"
        )
        ax.text(0.1, 0.5, info, transform=ax.transAxes, fontsize=11,
                fontfamily="monospace", va="center")
        ax.set_axis_off()
        ax.set_title("Design Summary")

        fig.tight_layout()
        fname = f"{res['prefix']}_design.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        generated.append(fname)

    # ── Combined filter bank overview ────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Pixel/DGS Filter Bank — Ground Plane Artwork & Responses",
        fontsize=14, fontweight="bold",
    )
    for ci, res in enumerate(results):
        _plot_pixel_grid(axes[0, ci], res["L2"],
                         f"L2 — {res['name']}", dx_mm, dy_mm)
        ax = axes[1, ci]
        ax.plot(fg, db(res["s21"]), "b", lw=1.5, label="|S21|")
        ax.plot(fg, db(res["s11"]), "r--", lw=1.0, label="|S11|")
        ax.set_title(res["name"])
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("[dB]")
        ax.set_ylim(-40, 3)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("pixel_filter_bank.png", dpi=200)
    plt.close(fig)
    generated.append("pixel_filter_bank.png")

    # ── Coupled resonator layer artwork ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("L3 — Coupled Resonator Pixel Patches", fontsize=14, fontweight="bold")
    for ci, res in enumerate(results):
        _plot_pixel_grid(axes[ci], res["L3"],
                         res["name"], dx_mm, dy_mm)
    fig.tight_layout()
    fig.savefig("pixel_L3_artwork.png", dpi=200)
    plt.close(fig)
    generated.append("pixel_L3_artwork.png")

    # ── Layer stackup ────────────────────────────────────────────────
    fig_s, ax_s = plt.subplots(figsize=(10, 4))
    ax_s.set_title("Pixel Filter Stackup", fontsize=13, fontweight="bold")
    layers_vis = [
        ("L1 Feedline (50 ohm microstrip)", 35e-6, "#e67e22"),
        ("RO4003C core (er=3.55)", H_SUB, "#ecf0f1"),
        ("L2 Ground — DGS pixel grid", 35e-6, "#27ae60"),
        ("RO4450F prepreg (er=3.52)", H_COUPLE, "#ecf0f1"),
        ("L3 Coupled resonator pixels", 18e-6, "#e67e22"),
        ("RO4450F prepreg (er=3.52)", 0.175e-3, "#ecf0f1"),
        ("L4 Ground", 35e-6, "#27ae60"),
    ]
    y = 0
    for label, t, c in reversed(layers_vis):
        h_mm = max(t * 1e3, 0.02)
        rect = plt.Rectangle((1, y), 8, h_mm, fc=c, ec="#333", lw=1)
        ax_s.add_patch(rect)
        ax_s.text(9.2, y + h_mm / 2, label, va="center", fontsize=9,
                  fontfamily="monospace")
        y += h_mm
    ax_s.set_xlim(0, 18)
    ax_s.set_ylim(-0.02, y + 0.05)
    ax_s.set_ylabel("Thickness [mm]")
    ax_s.set_xticks([])
    fig_s.tight_layout()
    fig_s.savefig("pixel_stackup.png", dpi=200)
    plt.close(fig_s)
    generated.append("pixel_stackup.png")

    # ── Print dimensions and stats ───────────────────────────────────
    print("=" * 68)
    print("  DESIGN SUMMARY")
    print("=" * 68)
    print(f"  Pixel grid       : {NX} x {NY}")
    print(f"  Pixel size       : {dx_mm:.2f} x {dy_mm:.2f} mm")
    print(f"  Filter footprint : {GRID_LEN*1e3:.1f} x {GRID_WID*1e3:.1f} mm")
    print(f"  Feedline width   : {_W50*1e3:.3f} mm (50 ohm)")
    print(f"  L2-L3 coupling   : {H_COUPLE*1e3:.1f} mm prepreg")
    print()
    for res in results:
        s21db = db(res["s21"])
        n_pix_L2 = int(res["L2"].sum())
        n_void_L2 = NX * NY - n_pix_L2
        n_pix_L3 = int(res["L3"].sum())
        print(f"  {res['name']}:")
        print(f"    L2 metal pixels : {n_pix_L2}/{NX*NY}  "
              f"({n_void_L2} voids = DGS features)")
        print(f"    L3 patch pixels : {n_pix_L3}/{NX*NY}")

    # ── Output image verification ────────────────────────────────────
    print("\n" + "=" * 68)
    print("  OUTPUT IMAGES")
    print("=" * 68)
    for img in generated:
        sz = os.path.getsize(img) / 1024
        print(f"  {img:40s}  {sz:6.0f} KB  OK")

    print("\n" + "=" * 68)
    print("  Complete.")
    print("=" * 68)


if __name__ == "__main__":
    main()
