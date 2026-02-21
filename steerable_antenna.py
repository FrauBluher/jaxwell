#!/usr/bin/env python3
"""
3D Switched Patch Array Antenna Designer
=========================================
Designs a 12-element conformal patch array with PIN diode switches
for 2D beam steering (azimuth 0-360 deg, elevation 10-80 deg).

Physical structure
------------------
4 tilted PCB panels arranged as a truncated pyramid:
  - Each panel holds 3 microstrip patch antennas (stacked vertically)
  - Panels face N / E / S / W, tilted 25 deg from vertical
  - Total: 12 elements, each with ON/OFF + 1-bit phase (0/180 deg)
  - PIN diode switches (e.g. Skyworks SMP1345-079LF)

The 3D arrangement gives inherent elevation diversity: lower elements
on each panel cover near-horizon angles while upper elements (which
converge toward the apex) provide zenith coverage.

Operating freq : 3.5 GHz  (mid-band, compatible with filter bank)
Substrate      : Rogers RO4003C  (er = 3.55, h = 1.524 mm)
Optimiser      : JAX autodiff + optax Adam
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
from matplotlib import cm

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
F0 = 3.5e9
LAM0 = C0 / F0
K0 = 2 * np.pi / LAM0

ER = 3.55
H_SUB = 1.524e-3
TAND = 0.0027

N_PANELS = 4
N_PER_PANEL = 3
N_ELEM = N_PANELS * N_PER_PANEL
N_SWITCH_PARAMS = N_ELEM * 2

PANEL_AZ = np.array([0, 90, 180, 270]) * np.pi / 180
TILT_DEG = 25
TILT = np.radians(TILT_DEG)


# ═══════════════════════════════════════════════════════════════════════
#  Patch dimensions (cavity model)
# ═══════════════════════════════════════════════════════════════════════
def _patch_dims():
    W = C0 / (2 * F0) * np.sqrt(2 / (ER + 1))
    eeff = (ER + 1) / 2 + (ER - 1) / 2 * (1 + 12 * H_SUB / W) ** (-0.5)
    dL = (
        0.412
        * H_SUB
        * ((eeff + 0.3) * (W / H_SUB + 0.264))
        / ((eeff - 0.258) * (W / H_SUB + 0.8))
    )
    L = C0 / (2 * F0 * np.sqrt(eeff)) - 2 * dL
    return W, L, eeff


PATCH_W, PATCH_L, ER_EFF = _patch_dims()


# ═══════════════════════════════════════════════════════════════════════
#  3D panel geometry
# ═══════════════════════════════════════════════════════════════════════
def _build_geometry():
    r_panel = 0.5 * LAM0
    v_spacing = 0.45 * LAM0
    z_center = v_spacing * np.sin(TILT) + 5e-3

    pos = np.zeros((N_ELEM, 3))
    nrm = np.zeros((N_ELEM, 3))
    txv = np.zeros((N_ELEM, 3))
    tyv = np.zeros((N_ELEM, 3))

    for ip in range(N_PANELS):
        az = PANEL_AZ[ip]
        ca, sa = np.cos(az), np.sin(az)

        n = np.array([np.sin(TILT) * ca, np.sin(TILT) * sa, np.cos(TILT)])
        tx = np.array([-sa, ca, 0.0])
        ty = np.cross(n, tx)
        ty /= np.linalg.norm(ty)

        center = np.array([r_panel * ca, r_panel * sa, z_center])

        for ie in range(N_PER_PANEL):
            idx = ip * N_PER_PANEL + ie
            v_off = (ie - 1) * v_spacing
            pos[idx] = center + v_off * ty
            nrm[idx] = n
            txv[idx] = tx
            tyv[idx] = ty

    return pos, nrm, txv, tyv


ELEM_POS, ELEM_NRM, ELEM_TX, ELEM_TY = _build_geometry()

_pos = jnp.array(ELEM_POS)
_nrm = jnp.array(ELEM_NRM)
_tx = jnp.array(ELEM_TX)
_ty = jnp.array(ELEM_TY)


# ═══════════════════════════════════════════════════════════════════════
#  Element & array pattern
# ═══════════════════════════════════════════════════════════════════════
def _sinc_safe(x):
    return jnp.where(jnp.abs(x) < 1e-8, 1.0, jnp.sin(x) / x)


def _elem_pat(k_hat, normal, tx, ty):
    """Patch element gain toward direction k_hat."""
    u_l = jnp.dot(k_hat, tx)
    v_l = jnp.dot(k_hat, ty)
    w_l = jnp.dot(k_hat, normal)

    front = jax.nn.sigmoid(w_l * 40)
    f_L = jnp.cos(K0 * PATCH_L * u_l / 2)
    f_W = _sinc_safe(K0 * PATCH_W * v_l / 2)
    return jnp.maximum(w_l, 0.0) * jnp.abs(f_L) * jnp.abs(f_W) * front


def _one_elem(k_hat, pos, nrm, tx, ty, w):
    f = _elem_pat(k_hat, nrm, tx, ty)
    return w * f * jnp.exp(1j * K0 * jnp.dot(k_hat, pos))


_vmap_elem = jax.vmap(_one_elem, in_axes=(None, 0, 0, 0, 0, 0))


def _pattern_at(theta, phi, weights):
    k = jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ]
    )
    return jnp.sum(_vmap_elem(k, _pos, _nrm, _tx, _ty, weights))


_pattern_batch = jax.vmap(
    lambda tp, w: _pattern_at(tp[0], tp[1], w), in_axes=(0, None)
)


# ═══════════════════════════════════════════════════════════════════════
#  Switch model
# ═══════════════════════════════════════════════════════════════════════
def _switches_to_weights(p):
    amp = jax.nn.sigmoid(p[:N_ELEM] * 5)
    phase = jnp.pi * jax.nn.sigmoid(p[N_ELEM:] * 5)
    return amp * jnp.exp(1j * phase)


def _quantize(p):
    a = (jax.nn.sigmoid(p[:N_ELEM] * 5) > 0.5).astype(jnp.float64)
    ph = (jax.nn.sigmoid(p[N_ELEM:] * 5) > 0.5).astype(jnp.float64)
    return a, ph


def _binary_weights(a, ph):
    return a * jnp.exp(1j * jnp.pi * ph)


# ═══════════════════════════════════════════════════════════════════════
#  Angle grids
# ═══════════════════════════════════════════════════════════════════════
def _make_grid(n_th=25, n_ph=48):
    th = jnp.linspace(0.02, jnp.pi / 2 - 0.02, n_th)
    ph = jnp.linspace(0, 2 * jnp.pi, n_ph, endpoint=False)
    TH, PH = jnp.meshgrid(th, ph, indexing="ij")
    dt, dp = float(th[1] - th[0]), float(ph[1] - ph[0])
    dO = jnp.sin(TH) * dt * dp
    angles = jnp.stack([TH.ravel(), PH.ravel()], axis=-1)
    return angles, dO.ravel()


# ═══════════════════════════════════════════════════════════════════════
#  Beam optimiser
# ═══════════════════════════════════════════════════════════════════════
def _beam_loss(p, th_t, ph_t, grid, dO):
    w = _switches_to_weights(p)
    E_t = _pattern_at(th_t, ph_t, w)
    P_t = jnp.abs(E_t) ** 2
    E_all = _pattern_batch(grid, w)
    P_tot = jnp.sum(jnp.abs(E_all) ** 2 * dO) + 1e-20
    D = 4 * jnp.pi * P_t / P_tot
    return -10 * jnp.log10(D + 1e-20)


def _init_switches(th_t, ph_t):
    """Heuristic initial switch states."""
    p0 = jnp.zeros(N_SWITCH_PARAMS)
    kd = jnp.array(
        [jnp.sin(th_t) * jnp.cos(ph_t), jnp.sin(th_t) * jnp.sin(ph_t), jnp.cos(th_t)]
    )
    for n in range(N_ELEM):
        alignment = float(jnp.dot(kd, _nrm[n]))
        p0 = p0.at[n].set(alignment * 4)
        path_phase = float(K0 * jnp.dot(kd, _pos[n]))
        p0 = p0.at[N_ELEM + n].set(2.0 if (path_phase % (2 * np.pi)) > np.pi else -2.0)
    return p0


def optimise_beam(th_t, ph_t, grid, dO, n_steps=800, lr=0.06):
    p0 = _init_switches(th_t, ph_t)
    loss_fn = lambda p: _beam_loss(p, th_t, ph_t, grid, dO)
    opt = optax.adam(lr)
    state = opt.init(p0)
    params = p0

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, ns = opt.update(grads, state, params)
        return optax.apply_updates(params, updates), ns, loss

    best_p, best_l = params, float("inf")
    for i in range(n_steps):
        params, state, lv = step(params, state)
        lv_f = float(lv)
        if jnp.isfinite(lv) and lv_f < best_l:
            best_l, best_p = lv_f, params
    return best_p, -best_l


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    grid, dO = _make_grid(25, 48)

    beam_dirs = [
        (30, 0, "N mid-el"),
        (60, 0, "N low-el"),
        (30, 90, "E mid-el"),
        (60, 180, "S low-el"),
        (45, 270, "W 45-el"),
        (15, 45, "NE zenith"),
        (50, 135, "SE 50-el"),
        (30, 315, "NW mid-el"),
    ]

    print("=" * 68)
    print("  3D Switched Patch Array Antenna Designer")
    print("=" * 68)
    print(f"  Frequency        : {F0 / 1e9:.2f} GHz")
    print(f"  Wavelength       : {LAM0 * 1e3:.1f} mm")
    print(f"  Patch size       : {PATCH_W * 1e3:.2f} x {PATCH_L * 1e3:.2f} mm")
    print(f"  Elements         : {N_ELEM} ({N_PANELS} panels x {N_PER_PANEL})")
    print(f"  Panel tilt       : {TILT_DEG} deg from vertical")
    print(f"  Switches/element : 2 PIN diodes (ON/OFF + phase 0/180 deg)")
    print()

    results = []
    for th_deg, ph_deg, label in beam_dirs:
        th_r, ph_r = np.radians(th_deg), np.radians(ph_deg)
        tag = f"th={th_deg:2d} ph={ph_deg:3d} ({label})"
        print(f"  Optimising {tag} ...", end="", flush=True)

        p_opt, D_cont = optimise_beam(th_r, ph_r, grid, dO, n_steps=800, lr=0.06)

        a_bits, ph_bits = _quantize(p_opt)
        wq = _binary_weights(a_bits, ph_bits)
        Et = _pattern_at(th_r, ph_r, wq)
        Pt = float(jnp.abs(Et) ** 2)
        Ea = _pattern_batch(grid, wq)
        Ptot = float(jnp.sum(jnp.abs(Ea) ** 2 * dO)) + 1e-20
        Dq = float(10 * np.log10(4 * np.pi * Pt / Ptot + 1e-20))

        print(f"  D = {D_cont:.1f} dBi (cont) / {Dq:.1f} dBi (quantized)")
        results.append(
            dict(
                theta=th_r,
                phi=ph_r,
                label=label,
                params=p_opt,
                amp=np.array(a_bits),
                phase=np.array(ph_bits),
                D_cont=D_cont,
                D_quant=Dq,
            )
        )

    # ── 3D radiation pattern plots ───────────────────────────────────
    n_th_p, n_ph_p = 91, 180
    th_p = np.linspace(0.02, np.pi / 2 - 0.02, n_th_p)
    ph_p = np.linspace(0, 2 * np.pi, n_ph_p, endpoint=False)
    TH_p, PH_p = np.meshgrid(th_p, ph_p, indexing="ij")
    ang_p = jnp.stack([jnp.array(TH_p.ravel()), jnp.array(PH_p.ravel())], axis=-1)

    plot_idx = list(range(min(4, len(results))))

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "3D Switched Patch Array  |  Radiation Patterns (upper hemisphere)",
        fontsize=14,
        fontweight="bold",
    )

    for ai, ri in enumerate(plot_idx):
        res = results[ri]
        wq = _binary_weights(jnp.array(res["amp"]), jnp.array(res["phase"]))
        Ep = np.array(_pattern_batch(ang_p, wq))
        Pp = np.abs(Ep.reshape(n_th_p, n_ph_p)) ** 2
        Pdb = 10 * np.log10(Pp / (Pp.max() + 1e-20) + 1e-20)
        Pdb = np.clip(Pdb, -25, 0)

        R = (Pdb - Pdb.min()) / (Pdb.max() - Pdb.min() + 1e-20)
        X = R * np.sin(TH_p) * np.cos(PH_p)
        Y = R * np.sin(TH_p) * np.sin(PH_p)
        Z = R * np.cos(TH_p)

        ax = fig.add_subplot(2, 2, ai + 1, projection="3d")
        colors = cm.hot((Pdb - Pdb.min()) / (Pdb.max() - Pdb.min() + 1e-20))
        ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.85, antialiased=True)

        tr, pr = res["theta"], res["phi"]
        ax.plot(
            [0, 0.9 * np.sin(tr) * np.cos(pr)],
            [0, 0.9 * np.sin(tr) * np.sin(pr)],
            [0, 0.9 * np.cos(tr)],
            "g-",
            lw=2.5,
        )
        ax.set_title(
            f"{res['label']}  (D = {res['D_quant']:.1f} dBi)", fontsize=11
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig("antenna_3d_patterns.png", dpi=180)
    print(f"\n  [saved] antenna_3d_patterns.png")

    # ── Polar cuts ───────────────────────────────────────────────────
    n_cut = min(len(results), 8)
    rows = (n_cut + 3) // 4
    fig2, axes2 = plt.subplots(
        rows, 4, figsize=(18, 4.5 * rows), subplot_kw={"projection": "polar"}
    )
    axes_flat = np.array(axes2).ravel()
    fig2.suptitle(
        "Azimuth-Cut Patterns (at target elevation)", fontsize=14, fontweight="bold"
    )

    phi_cut = np.linspace(0, 2 * np.pi, 360)
    for ri in range(n_cut):
        res = results[ri]
        wq = _binary_weights(jnp.array(res["amp"]), jnp.array(res["phase"]))
        ac = jnp.stack(
            [jnp.full(360, res["theta"]), jnp.array(phi_cut)], axis=-1
        )
        Ec = np.array(_pattern_batch(ac, wq))
        Pc = np.abs(Ec) ** 2
        Pcd = 10 * np.log10(Pc / (Pc.max() + 1e-20) + 1e-20)
        Pcd = np.clip(Pcd, -30, 0)

        ax = axes_flat[ri]
        ax.plot(phi_cut, Pcd + 30, "b", lw=1.4)
        ax.axvline(res["phi"], color="r", ls="--", lw=1, alpha=0.7)
        ax.set_rmax(30)
        ax.set_rticks([0, 10, 20, 30])
        ax.set_yticklabels(["-30", "-20", "-10", "0 dB"])
        ax.set_title(f"{res['label']}, D={res['D_quant']:.1f} dBi", fontsize=9)

    for ri in range(n_cut, len(axes_flat)):
        axes_flat[ri].set_visible(False)

    plt.tight_layout()
    plt.savefig("antenna_polar_cuts.png", dpi=180)
    print(f"  [saved] antenna_polar_cuts.png")

    # ── Structure visualisation ──────────────────────────────────────
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.set_title("Antenna Element Positions", fontsize=13, fontweight="bold")

    panel_colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for ip in range(N_PANELS):
        idx = range(ip * N_PER_PANEL, (ip + 1) * N_PER_PANEL)
        xs = ELEM_POS[idx, 0] * 1e3
        ys = ELEM_POS[idx, 1] * 1e3
        zs = ELEM_POS[idx, 2] * 1e3
        col = panel_colours[ip]
        labels = ["N", "E", "S", "W"]
        ax3.scatter(xs, ys, zs, s=120, c=col, edgecolors="k", zorder=5)
        for j, i in enumerate(idx):
            ax3.text(
                ELEM_POS[i, 0] * 1e3 + 2,
                ELEM_POS[i, 1] * 1e3 + 2,
                ELEM_POS[i, 2] * 1e3,
                f"{i + 1}",
                fontsize=7,
            )
        cx = np.mean(xs)
        cy = np.mean(ys)
        cz = np.max(zs) + 3
        ax3.text(cx, cy, cz, labels[ip], fontsize=11, fontweight="bold", color=col)

        ax3.plot(xs, ys, zs, "-", color=col, alpha=0.5, lw=1.5)

    ax3.set_xlabel("X [mm]")
    ax3.set_ylabel("Y [mm]")
    ax3.set_zlabel("Z [mm]")
    ax3.set_box_aspect([1, 1, 0.6])
    plt.tight_layout()
    plt.savefig("antenna_structure.png", dpi=180)
    print(f"  [saved] antenna_structure.png")

    # ── Physical dimensions ──────────────────────────────────────────
    print()
    print("=" * 68)
    print("  PHYSICAL DIMENSIONS")
    print("=" * 68)
    print(f"  Operating frequency  : {F0 / 1e9:.2f} GHz  (lambda = {LAM0 * 1e3:.1f} mm)")
    print(f"  Substrate            : Rogers RO4003C, h = {H_SUB * 1e3:.3f} mm")
    print(f"  Patch width  (W)     : {PATCH_W * 1e3:.2f} mm")
    print(f"  Patch length (L)     : {PATCH_L * 1e3:.2f} mm")
    print(f"  Panel tilt           : {TILT_DEG} deg from vertical")
    print(f"  Panel radial offset  : {0.5 * LAM0 * 1e3:.1f} mm")
    print(f"  Element spacing      : {0.45 * LAM0 * 1e3:.1f} mm (along panel surface)")
    print(f"  Switches per element : 2 PIN diodes")
    print(f"  Total PIN diodes     : {N_ELEM * 2}")
    print()

    print("  Element positions (mm):")
    print(f"  {'#':>3s}  {'Panel':>5s}  {'X':>8s}  {'Y':>8s}  {'Z':>8s}")
    print("  " + "-" * 40)
    panel_names = ["N", "E", "S", "W"]
    for n in range(N_ELEM):
        pn = panel_names[n // N_PER_PANEL]
        x, y, z = ELEM_POS[n] * 1e3
        print(f"  {n + 1:3d}  {pn:>5s}  {x:8.1f}  {y:8.1f}  {z:8.1f}")

    # ── Switch configuration table ───────────────────────────────────
    print()
    print("=" * 68)
    print("  SWITCH CONFIGURATION LOOKUP TABLE")
    print("=" * 68)
    print(
        f"  {'Beam':>22s}  {'ON elements':>28s}  "
        f"{'Phase=180 elements':>22s}  {'D(dBi)':>7s}"
    )
    print("  " + "-" * 83)
    for res in results:
        th_d = np.degrees(res["theta"])
        ph_d = np.degrees(res["phi"])
        tag = f"th={th_d:2.0f} ph={ph_d:3.0f} ({res['label']})"
        on = ",".join(str(i + 1) for i in range(N_ELEM) if res["amp"][i] > 0.5)
        p180 = ",".join(str(i + 1) for i in range(N_ELEM) if res["phase"][i] > 0.5)
        print(f"  {tag:>22s}  {on:>28s}  {p180:>22s}  {res['D_quant']:>7.1f}")

    # ── Bill of materials ────────────────────────────────────────────
    print()
    print("=" * 68)
    print("  BILL OF MATERIALS (per antenna)")
    print("=" * 68)
    print(f"  {N_PANELS:2d}x  PCB panel  ({PATCH_W*1e3*2:.0f} x {0.45*LAM0*1e3*2+PATCH_L*1e3:.0f} mm, RO4003C)")
    print(f"  {N_ELEM:2d}x  Patch antenna element (etched on PCB)")
    print(f"  {N_ELEM * 2:2d}x  PIN diode switch (e.g. Skyworks SMP1345)")
    print(f"  {N_ELEM:2d}x  DC block capacitor (100 pF, 0402)")
    print(f"  {N_ELEM:2d}x  RF choke inductor (22 nH, 0402)")
    print(f"   1x  3D-printed mounting frame")
    print(f"   1x  Microcontroller for switch control (e.g. STM32)")
    print(f"   1x  SMA connector + feed network")

    print()
    print("=" * 68)
    print("  Complete.  Plots saved.")
    print("=" * 68)


if __name__ == "__main__":
    main()
