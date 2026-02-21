#!/usr/bin/env python3
"""
Shape-Library DGS Multilayer Filter Bank
=========================================
Uses a library of well-characterised resonator shapes on multiple
PCB layers instead of blind pixel optimisation.

Shape library (each has a closed-form equivalent circuit):
  - Dumbbell DGS  :  two voids + bridge slot   → series LC shunt
  - CSRR          :  ring slot with gap         → coupled series LC
  - H-shape DGS   :  two arms + bridge          → high-C series LC

Each DGS shape is paired with a coupled half-wave patch resonator
on an inner layer (L3).  The DGS creates band-stop notches; the
coupled patch adds transmission poles — together they sculpt
bandpass / highpass responses.

Multilayer stackup:
  L1  Feedline  (50-ohm microstrip)
  L2  Ground    (DGS shapes etched here)
  L3  Signal    (coupled resonator patches)
  L4  Ground    (L3 reference)

Filter bank:
  1. HPF   fc = 2500 MHz   (5 dumbbell DGS elements)
  2. BPF1  2500-3750 MHz   (5 CSRR DGS + 5 coupled patches)
  3. BPF2  3750-5000 MHz   (5 CSRR DGS + 5 coupled patches)

Optimiser: JAX autodiff + optax Adam
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
from matplotlib.patches import Rectangle, FancyBboxPatch, Arc, Wedge
from matplotlib.collections import PatchCollection

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
EPS0 = 8.854_187_817e-12
MU0 = 4e-7 * np.pi
Z_REF = 50.0
ER = 3.55
H_SUB = 0.508e-3
TAND = 0.0027
ER_PP = 3.52
H_COUPLE = 0.100e-3

_W50 = 1.136e-3
_EEFF = (ER + 1) / 2 + (ER - 1) / 2 * (1 + 12 * H_SUB / _W50) ** (-0.5)


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
#  Equivalent-circuit shape models
# ═══════════════════════════════════════════════════════════════════════
def _dumbbell_Y(f, void_a, bridge_w, bridge_l):
    """Dumbbell DGS: two square voids joined by a narrow bridge.
    The bridge provides inductance; the void edges provide capacitance."""
    L = MU0 * bridge_l / jnp.maximum(bridge_w, 0.1e-3)
    C_gap = EPS0 * ER * void_a * bridge_w / (2 * H_SUB)
    C_fringe = EPS0 * ER * void_a ** 2 / (4 * H_SUB)
    C = C_gap + C_fringe
    R = 0.8
    omega = 2 * jnp.pi * f
    Z = R + 1j * omega * L + 1.0 / (1j * omega * C + 1e-30)
    return 1.0 / Z


def _csrr_Y(f, radius, ring_w, gap_frac):
    """Complementary split-ring resonator etched from ground plane.
    Ring slot with a gap — compact, high Q."""
    r_avg = radius
    circ = 2 * jnp.pi * r_avg
    gap_len = jnp.maximum(gap_frac * circ, 0.2e-3)
    slot_len = circ - gap_len
    L = MU0 * slot_len / jnp.maximum(ring_w, 0.1e-3) * 0.5
    C = EPS0 * ER * ring_w * slot_len / H_SUB * 0.3
    C_gap = EPS0 * ER * ring_w ** 2 / gap_len * 0.5
    C_total = C + C_gap
    R = 0.5
    omega = 2 * jnp.pi * f
    Z = R + 1j * omega * L + 1.0 / (1j * omega * C_total + 1e-30)
    Cc = EPS0 * ER * jnp.pi * r_avg ** 2 / H_SUB * 0.15
    return Cc * omega ** 2 / (Z * omega + 1e-20)


def _hshape_Y(f, arm_l, arm_w, bridge_w):
    """H-shaped DGS: two transverse arms connected by a longitudinal bridge.
    Higher capacitance than dumbbell → lower resonant frequency."""
    bridge_l = arm_w
    L = MU0 * bridge_l / jnp.maximum(bridge_w, 0.1e-3)
    C = EPS0 * ER * arm_l * arm_w / H_SUB * 0.6
    R = 0.6
    omega = 2 * jnp.pi * f
    Z = R + 1j * omega * L + 1.0 / (1j * omega * C + 1e-30)
    return 1.0 / Z


def _coupled_patch_Y(f, w_patch, l_patch):
    """Half-wave patch resonator on L3, coupled through L2 aperture."""
    f_res = C0 / (2 * l_patch * jnp.sqrt(ER_PP) + 1e-10)
    Cc = EPS0 * ER_PP * w_patch * l_patch / H_COUPLE * 0.08
    Q = 25.0
    omega = 2 * jnp.pi * f
    omega_0 = 2 * jnp.pi * f_res
    denom = omega_0 ** 2 - omega ** 2 + 1j * omega * omega_0 / Q
    return 1j * omega * Cc * omega_0 ** 2 / (denom + 1e-20)


# ═══════════════════════════════════════════════════════════════════════
#  Bounded parameter transforms
# ═══════════════════════════════════════════════════════════════════════
def _bnd(x, lo, hi):
    t = jax.nn.sigmoid(x)
    return jnp.exp(jnp.log(lo) * (1 - t) + jnp.log(hi) * t)

def _bnd_inv(y, lo, hi):
    t = (jnp.log(y) - jnp.log(lo)) / (jnp.log(hi) - jnp.log(lo))
    t = jnp.clip(t, 1e-6, 1 - 1e-6)
    return jnp.log(t / (1 - t))


# ═══════════════════════════════════════════════════════════════════════
#  HPF model: 5 dumbbell DGS elements
# ═══════════════════════════════════════════════════════════════════════
N_HPF = 5
HPF_NPARAMS = N_HPF * 3 + (N_HPF + 1)  # 3 per dumbbell + spacings

HPF_BOUNDS = [
    (N_HPF, 2e-3, 18e-3),    # void_a
    (N_HPF, 0.2e-3, 4e-3),   # bridge_w
    (N_HPF, 0.5e-3, 8e-3),   # bridge_l
    (N_HPF + 1, 1e-3, 20e-3),  # spacings
]

def _hpf_init():
    fc = 2.5e9
    target_freqs = [0.8e9, 1.2e9, 1.6e9, 2.0e9, 2.4e9]
    void_a, bridge_w, bridge_l = [], [], []
    for ft in target_freqs:
        bw = 0.2e-3 + 1e-3 * (ft / 3e9)
        bl = 2e-3
        L = MU0 * bl / bw
        C_needed = 1.0 / ((2 * np.pi * ft) ** 2 * L)
        a = np.sqrt(4 * C_needed * H_SUB / (EPS0 * ER))
        a = np.clip(a, 2.5e-3, 17e-3)
        void_a.append(a)
        bridge_w.append(bw)
        bridge_l.append(bl)
    spacings = [8e-3] * (N_HPF + 1)
    vals = void_a + bridge_w + bridge_l + spacings
    parts = []
    idx = 0
    for n, lo, hi in HPF_BOUNDS:
        parts.append(jnp.array([float(_bnd_inv(vals[idx + i], lo, hi)) for i in range(n)]))
        idx += n
    return jnp.concatenate(parts)


def _hpf_unpack(p):
    i = 0
    res = []
    for n, lo, hi in HPF_BOUNDS:
        res.append(_bnd(p[i:i + n], lo, hi))
        i += n
    return res  # [void_a, bridge_w, bridge_l, spacings]


def _hpf_at_f(p, f):
    va, bw, bl, sp = _hpf_unpack(p)
    gl_sp = lambda l: (TAND * jnp.pi * f * jnp.sqrt(_EEFF) / C0 +
                        1j * 2 * jnp.pi * f * jnp.sqrt(_EEFF) / C0) * l
    M = abcd_tline(Z_REF, gl_sp(sp[0]))
    for k in range(N_HPF):
        Y = _dumbbell_Y(f, va[k], bw[k], bl[k])
        M = M @ abcd_shunt(Y) @ abcd_tline(Z_REF, gl_sp(sp[k + 1]))
    return abcd_to_s(M)

_hpf_batch = jax.vmap(_hpf_at_f, in_axes=(None, 0))

def hpf_loss(p, freqs):
    s21, s11 = _hpf_batch(p, freqs)
    s21d = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
    s11d = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
    fc = 2.5e9
    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    tb = (freqs > 1.5e9) & (freqs < fc)
    loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 1.0, 0)**2, 0)) * 5
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d + 10, 0)**2, 0)) * 2
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21d + 15, 0)**2, 0)) * 4
    loss += jnp.sum(jnp.where(tb, jnp.maximum(s21d + 8, 0)**2, 0)) * 2
    return loss / freqs.shape[0]


# ═══════════════════════════════════════════════════════════════════════
#  BPF model: 5 CSRR DGS + 5 coupled patches
# ═══════════════════════════════════════════════════════════════════════
N_BPF = 5

BPF_BOUNDS = [
    (N_BPF, 1.5e-3, 12e-3),    # csrr radius
    (N_BPF, 0.2e-3, 3e-3),     # csrr ring_w
    (N_BPF, 0.03, 0.25),       # csrr gap_frac
    (N_BPF, 1e-3, 15e-3),      # patch_w
    (N_BPF, 5e-3, 35e-3),      # patch_l
    (N_BPF + 1, 1e-3, 20e-3),  # spacings
]

def _bpf_init(fc, fbw):
    f_lo = fc * (1 - fbw / 2)
    f_hi = fc * (1 + fbw / 2)
    radii, ring_ws, gap_fracs = [], [], []
    patch_ws, patch_ls = [], []
    target_notch_freqs = [
        f_lo * 0.6, f_lo * 0.85,
        (f_lo + f_hi) / 2,
        f_hi * 1.15, f_hi * 1.4,
    ]
    for ft in target_notch_freqs:
        rw = 0.5e-3
        gf = 0.08
        circ_target = C0 / (ft * np.sqrt(ER)) * 0.3
        r = circ_target / (2 * np.pi)
        r = np.clip(r, 2e-3, 11e-3)
        radii.append(r)
        ring_ws.append(rw)
        gap_fracs.append(gf)
    target_pass_freqs = np.linspace(f_lo * 1.05, f_hi * 0.95, N_BPF)
    for fp in target_pass_freqs:
        lp = C0 / (2 * fp * np.sqrt(ER_PP))
        lp = np.clip(lp, 6e-3, 34e-3)
        patch_ls.append(lp)
        patch_ws.append(3e-3)
    spacings = [6e-3] * (N_BPF + 1)
    vals = radii + ring_ws + gap_fracs + patch_ws + patch_ls + spacings
    parts = []
    idx = 0
    for n, lo, hi in BPF_BOUNDS:
        parts.append(jnp.array([float(_bnd_inv(vals[idx + i], lo, hi)) for i in range(n)]))
        idx += n
    return jnp.concatenate(parts)


def _bpf_unpack(p):
    i = 0
    res = []
    for n, lo, hi in BPF_BOUNDS:
        res.append(_bnd(p[i:i + n], lo, hi))
        i += n
    return res  # [radius, ring_w, gap_frac, patch_w, patch_l, spacings]


def _bpf_at_f(p, f):
    rad, rw, gf, pw, pl, sp = _bpf_unpack(p)
    gl_sp = lambda l: (TAND * jnp.pi * f * jnp.sqrt(_EEFF) / C0 +
                        1j * 2 * jnp.pi * f * jnp.sqrt(_EEFF) / C0) * l
    M = abcd_tline(Z_REF, gl_sp(sp[0]))
    for k in range(N_BPF):
        Y_dgs = _csrr_Y(f, rad[k], rw[k], gf[k])
        Y_patch = _coupled_patch_Y(f, pw[k], pl[k])
        M = M @ abcd_shunt(Y_dgs + Y_patch) @ abcd_tline(Z_REF, gl_sp(sp[k + 1]))
    return abcd_to_s(M)

_bpf_batch = jax.vmap(_bpf_at_f, in_axes=(None, 0))

def make_bpf_loss(f_lo, f_hi):
    bw = f_hi - f_lo
    def loss_fn(p, freqs):
        s21, s11 = _bpf_batch(p, freqs)
        s21d = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11d = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
        pb = (freqs >= f_lo) & (freqs <= f_hi)
        sl = freqs < (f_lo - 0.35 * bw)
        sh = freqs > (f_hi + 0.35 * bw)
        loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 2, 0)**2, 0)) * 6
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d + 8, 0)**2, 0)) * 2
        loss += jnp.sum(jnp.where(sl, jnp.maximum(s21d + 15, 0)**2, 0)) * 3
        loss += jnp.sum(jnp.where(sh, jnp.maximum(s21d + 15, 0)**2, 0)) * 3
        return loss / freqs.shape[0]
    return loss_fn


# ═══════════════════════════════════════════════════════════════════════
#  Optimiser
# ═══════════════════════════════════════════════════════════════════════
def optimise(loss_fn, init_p, freqs, *, n_steps=2000, lr=5e-3):
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    state = opt.init(init_p)
    params = init_p
    @jax.jit
    def step(params, state):
        loss, g = jax.value_and_grad(loss_fn)(params, freqs)
        g = jnp.where(jnp.isfinite(g), g, 0.0)
        u, ns = opt.update(g, state, params)
        return optax.apply_updates(params, u), ns, loss
    best_p, best_l = params, float("inf")
    for i in range(n_steps):
        params, state, lv = step(params, state)
        lv_f = float(lv)
        if jnp.isfinite(lv) and lv_f < best_l:
            best_l, best_p = lv_f, params
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  step {i+1:5d}/{n_steps}  loss={lv_f:.4f}  best={best_l:.4f}")
    return best_p, best_l

def opt_restarts(loss_fn, init_fn, freqs, n_r=3, **kw):
    best_p, best_l = None, float("inf")
    for r in range(n_r):
        p0 = init_fn()
        if r > 0:
            p0 = p0 + jax.random.normal(jax.random.PRNGKey(r * 97), p0.shape) * 0.4
        if n_r > 1:
            print(f"  (restart {r+1}/{n_r})")
        p, lv = optimise(loss_fn, p0, freqs, **kw)
        if lv < best_l:
            best_l, best_p = lv, p
    return best_p


# ═══════════════════════════════════════════════════════════════════════
#  Artwork rendering
# ═══════════════════════════════════════════════════════════════════════
CU_COLOR = "#cc8844"
VOID_COLOR = "#1a1a2e"
PATCH_COLOR = "#dd9955"
FEED_COLOR = "#e74c3c"
GND_BG = "#b87333"


def _draw_dumbbell(ax, x, y, a_mm, bw_mm, bl_mm):
    """Draw dumbbell DGS void on ground plane."""
    hw = a_mm / 2
    ax.add_patch(Rectangle((x - bl_mm / 2 - a_mm, y - hw), a_mm, a_mm,
                            fc=VOID_COLOR, ec="#444", lw=0.5, zorder=2))
    ax.add_patch(Rectangle((x + bl_mm / 2, y - hw), a_mm, a_mm,
                            fc=VOID_COLOR, ec="#444", lw=0.5, zorder=2))
    ax.add_patch(Rectangle((x - bl_mm / 2, y - bw_mm / 2), bl_mm, bw_mm,
                            fc=VOID_COLOR, ec="#444", lw=0.5, zorder=2))


def _draw_csrr(ax, x, y, r_mm, rw_mm, gap_frac):
    """Draw CSRR void on ground plane."""
    gap_deg = gap_frac * 360
    theta1 = 90 + gap_deg / 2
    theta2 = 90 - gap_deg / 2 + 360
    arc_outer = Arc((x, y), 2 * r_mm, 2 * r_mm, angle=0,
                     theta1=theta1, theta2=theta2, lw=rw_mm * 3 + 1,
                     color=VOID_COLOR, zorder=2)
    ax.add_patch(arc_outer)
    n_pts = 60
    th = np.linspace(np.radians(theta1), np.radians(theta2), n_pts)
    for j in range(len(th) - 1):
        x1, y1 = x + r_mm * np.cos(th[j]), y + r_mm * np.sin(th[j])
        x2, y2 = x + r_mm * np.cos(th[j + 1]), y + r_mm * np.sin(th[j + 1])
        ax.plot([x1, x2], [y1, y2], color=VOID_COLOR,
                lw=max(rw_mm * 8, 2), solid_capstyle="round", zorder=2)


def _draw_patch(ax, x, y, w_mm, l_mm):
    """Draw coupled resonator patch on L3."""
    ax.add_patch(Rectangle((x - l_mm / 2, y - w_mm / 2), l_mm, w_mm,
                            fc=PATCH_COLOR, ec="#996633", lw=0.8, zorder=2))


def _draw_layer(ax, elems, draw_fn, total_len_mm, total_wid_mm, title, bg_color):
    """Draw a complete layer with elements."""
    ax.add_patch(Rectangle((0, 0), total_len_mm, total_wid_mm,
                            fc=bg_color, ec="#333", lw=1.5, zorder=1))
    feed_y = total_wid_mm / 2
    ax.plot([0, total_len_mm], [feed_y, feed_y], color=FEED_COLOR,
            lw=2, ls="--", alpha=0.6, zorder=3, label="Feedline")
    for elem in elems:
        draw_fn(ax, elem["x"], feed_y, **elem["params"])
    ax.set_xlim(-2, total_len_mm + 2)
    ax.set_ylim(-2, total_wid_mm + 2)
    ax.set_xlabel("Along feedline [mm]")
    ax.set_ylabel("[mm]")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs = jnp.linspace(0.1e9, 7e9, 500)
    generated = []

    # ── HPF ──────────────────────────────────────────────────────────
    print("=" * 68)
    print("  HPF  (5 dumbbell DGS on L2)")
    print("=" * 68)
    p_hpf = _hpf_init()
    p_hpf = opt_restarts(hpf_loss, _hpf_init, freqs,
                          n_r=3, n_steps=2500, lr=5e-3)

    # ── BPF1 ─────────────────────────────────────────────────────────
    f1_lo, f1_hi = 2.5e9, 3.75e9
    fc1 = (f1_lo + f1_hi) / 2
    fbw1 = (f1_hi - f1_lo) / fc1
    bpf1_loss = make_bpf_loss(f1_lo, f1_hi)
    print("\n" + "=" * 68)
    print("  BPF1  (5 CSRR + 5 coupled patches, L2/L3)")
    print("=" * 68)
    p_b1 = opt_restarts(bpf1_loss, lambda: _bpf_init(fc1, fbw1), freqs,
                         n_r=3, n_steps=3000, lr=5e-3)

    # ── BPF2 ─────────────────────────────────────────────────────────
    f2_lo, f2_hi = 3.75e9, 5.0e9
    fc2 = (f2_lo + f2_hi) / 2
    fbw2 = (f2_hi - f2_lo) / fc2
    bpf2_loss = make_bpf_loss(f2_lo, f2_hi)
    print("\n" + "=" * 68)
    print("  BPF2  (5 CSRR + 5 coupled patches, L2/L3)")
    print("=" * 68)
    p_b2 = opt_restarts(bpf2_loss, lambda: _bpf_init(fc2, fbw2), freqs,
                         n_r=3, n_steps=3000, lr=5e-3)

    # ── Evaluate ─────────────────────────────────────────────────────
    fp = jnp.linspace(0.1e9, 7e9, 1000)
    fg = np.array(fp) / 1e9
    db = lambda s: 20 * np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))

    s21h, s11h = _hpf_batch(p_hpf, fp)
    s21b1, s11b1 = _bpf_batch(p_b1, fp)
    s21b2, s11b2 = _bpf_batch(p_b2, fp)

    all_filters = [
        ("HPF (fc = 2.5 GHz)", s21h, s11h, [2.5], p_hpf, "hpf"),
        ("BPF1 (2.5-3.75 GHz)", s21b1, s11b1, [2.5, 3.75], p_b1, "bpf1"),
        ("BPF2 (3.75-5.0 GHz)", s21b2, s11b2, [3.75, 5.0], p_b2, "bpf2"),
    ]

    # ── Per-filter artwork + response ────────────────────────────────
    for name, s21, s11, vlines, params, prefix in all_filters:
        is_hpf = prefix == "hpf"

        if is_hpf:
            va, bw, bl, sp = (np.array(x) for x in _hpf_unpack(params))
            positions = np.cumsum(np.array(sp))
            dgs_elems = []
            for k in range(N_HPF):
                dgs_elems.append(dict(x=float(positions[k]) * 1e3,
                                      params=dict(a_mm=float(va[k]) * 1e3,
                                                  bw_mm=float(bw[k]) * 1e3,
                                                  bl_mm=float(bl[k]) * 1e3)))
            patch_elems = []
            total_len = float(np.sum(sp)) * 1e3
        else:
            rad, rw, gf, pw, pl, sp = (np.array(x) for x in _bpf_unpack(params))
            positions = np.cumsum(np.array(sp))
            dgs_elems = []
            patch_elems = []
            for k in range(N_BPF):
                x_pos = float(positions[k]) * 1e3
                dgs_elems.append(dict(x=x_pos,
                                      params=dict(r_mm=float(rad[k]) * 1e3,
                                                  rw_mm=float(rw[k]) * 1e3,
                                                  gap_frac=float(gf[k]))))
                patch_elems.append(dict(x=x_pos,
                                        params=dict(w_mm=float(pw[k]) * 1e3,
                                                    l_mm=float(pl[k]) * 1e3)))
            total_len = float(np.sum(sp)) * 1e3

        ground_wid = 30.0

        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(f"Shape-Library Filter: {name}", fontsize=14, fontweight="bold")

        draw_dgs = _draw_dumbbell if is_hpf else _draw_csrr
        _draw_layer(axes[0, 0], dgs_elems, draw_dgs,
                    total_len, ground_wid, "L2 — Ground Plane (DGS shapes)", GND_BG)

        if patch_elems:
            _draw_layer(axes[0, 1], patch_elems, _draw_patch,
                        total_len, ground_wid,
                        "L3 — Coupled Resonator Patches", "#2c3e50")
        else:
            axes[0, 1].text(0.5, 0.5, "No coupled patches\n(HPF uses DGS only)",
                           ha="center", va="center", transform=axes[0, 1].transAxes,
                           fontsize=12)
            axes[0, 1].set_title("L3 — (not used for HPF)")

        ax = axes[1, 0]
        ax.plot(fg, db(s21), "b", lw=1.6, label="|S21|")
        ax.plot(fg, db(s11), "r--", lw=1.1, label="|S11|")
        ax.set_title("Frequency Response")
        ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("[dB]")
        ax.set_ylim(-50, 3); ax.legend(); ax.grid(True, alpha=0.3)
        for fv in vlines:
            ax.axvline(fv, color="gray", ls=":", lw=0.8)

        ax = axes[1, 1]
        if is_hpf:
            info_lines = [f"Shape: Dumbbell DGS x {N_HPF}", ""]
            for k in range(N_HPF):
                f0 = 1 / (2 * np.pi * np.sqrt(
                    MU0 * bl[k] / max(bw[k], 1e-4) *
                    (EPS0 * ER * va[k]**2 / (4*H_SUB) +
                     EPS0 * ER * va[k] * bw[k] / (2*H_SUB))))
                info_lines.append(
                    f"  DGS {k+1}: void={va[k]*1e3:.2f}mm  "
                    f"bridge={bw[k]*1e3:.2f}x{bl[k]*1e3:.2f}mm  "
                    f"f0~{f0/1e9:.2f}GHz")
        else:
            info_lines = [f"Shapes: CSRR x {N_BPF} + Patch x {N_BPF}", ""]
            for k in range(N_BPF):
                info_lines.append(
                    f"  CSRR {k+1}: r={rad[k]*1e3:.2f}mm  "
                    f"w={rw[k]*1e3:.2f}mm  gap={gf[k]*100:.1f}%")
            info_lines.append("")
            for k in range(N_BPF):
                fres = C0 / (2 * pl[k] * np.sqrt(ER_PP))
                info_lines.append(
                    f"  Patch {k+1}: {pw[k]*1e3:.2f}x{pl[k]*1e3:.2f}mm  "
                    f"f_res={fres/1e9:.2f}GHz")
        ax.text(0.05, 0.95, "\n".join(info_lines), transform=ax.transAxes,
                fontsize=9, fontfamily="monospace", va="top")
        ax.set_axis_off()
        ax.set_title("Element Dimensions")

        fig.tight_layout()
        fname = f"shape_{prefix}_design.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        generated.append(fname)

    # ── Combined overview ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Shape-Library DGS Filter Bank — L2 Artwork & Responses",
                 fontsize=14, fontweight="bold")
    for ci, (name, s21, s11, vl, params, prefix) in enumerate(all_filters):
        is_hpf = prefix == "hpf"
        if is_hpf:
            va, bw, bl, sp = (np.array(x) for x in _hpf_unpack(params))
            positions = np.cumsum(np.array(sp))
            elems = [dict(x=float(positions[k])*1e3,
                         params=dict(a_mm=float(va[k])*1e3, bw_mm=float(bw[k])*1e3,
                                     bl_mm=float(bl[k])*1e3)) for k in range(N_HPF)]
            total_l = float(np.sum(sp)) * 1e3
            draw_fn = _draw_dumbbell
        else:
            rad, rw, gf, pw, pl, sp = (np.array(x) for x in _bpf_unpack(params))
            positions = np.cumsum(np.array(sp))
            elems = [dict(x=float(positions[k])*1e3,
                         params=dict(r_mm=float(rad[k])*1e3, rw_mm=float(rw[k])*1e3,
                                     gap_frac=float(gf[k]))) for k in range(N_BPF)]
            total_l = float(np.sum(sp)) * 1e3
            draw_fn = _draw_csrr
        _draw_layer(axes[0, ci], elems, draw_fn, total_l, 30.0,
                    f"L2 — {name}", GND_BG)
        ax = axes[1, ci]
        ax.plot(fg, db(s21), "b", lw=1.5, label="|S21|")
        ax.plot(fg, db(s11), "r--", lw=1, label="|S11|")
        ax.set_title(name); ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("[dB]"); ax.set_ylim(-50, 3)
        ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
        for fv in vl:
            ax.axvline(fv, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig("shape_filter_bank.png", dpi=200)
    plt.close(fig)
    generated.append("shape_filter_bank.png")

    # ── Performance ──────────────────────────────────────────────────
    def _perf(s21, s11, flo, fhi, label):
        s21db, s11db = db(s21), db(s11)
        pb = (fg >= flo) & (fg <= fhi)
        if not np.any(pb):
            return
        print(f"  {label}:")
        print(f"    Passband IL : {float(-np.max(s21db[pb])):.1f} - "
              f"{float(-np.min(s21db[pb])):.1f} dB")
        print(f"    Passband RL : > {float(-np.max(s11db[pb])):.1f} dB")

    print("\n" + "=" * 68)
    print("  PERFORMANCE")
    print("=" * 68)
    _perf(s21h, s11h, 2.5, 5.5, "HPF (dumbbell DGS)")
    _perf(s21b1, s11b1, 2.5, 3.75, "BPF1 (CSRR + patches)")
    _perf(s21b2, s11b2, 3.75, 5.0, "BPF2 (CSRR + patches)")

    # ── Image verification ───────────────────────────────────────────
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
