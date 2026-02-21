#!/usr/bin/env python3
"""
Shape-Library Multilayer DGS Filter Bank
=========================================
Designs compact RF filters using parametric DGS shapes on multiple
PCB layers with Rogers high-er substrates for size reduction and
hard manufacturing constraints enforced throughout.

Shape library (closed-form equivalent circuits):
  Dumbbell DGS  :  two voids + bridge slot    → series LC shunt
  CSRR          :  ring slot with gap          → coupled series LC
  H-shape DGS   :  two arms + bridge           → high-C series LC
  Coupled patch :  half-wave resonator on inner layer

Multilayer stackup  (6-layer, Rogers RO3010 + RO4450F)
------------------------------------------------------
  L1  Signal   50-ohm feedline                  RO3010 core 0.254 mm  er=10.2
  L2  Ground   DGS shapes (primary band-stop)   RO4450F PP  0.100 mm  er=3.52
  L3  Signal   Coupled resonator patches         RO3010 core 0.254 mm  er=10.2
  L4  Ground   Secondary DGS (extra rejection)   RO4450F PP  0.175 mm  er=3.52
  L5  Signal   Second coupled patches (optional)
  L6  Ground   Bottom reference

Using RO3010 (er=10.2) shrinks guided wavelength by ~42%
compared to RO4003C (er=3.55), producing much more compact filters.

Manufacturing constraints  (enforced as hard parameter bounds)
--------------------------------------------------------------
  Minimum trace / slot width   : 0.15 mm  (6 mil)
  Minimum CSRR gap             : 0.15 mm
  Minimum patch dimension      : 0.50 mm
  Minimum via drill            : 0.20 mm

Filter bank:
  1. HPF   fc = 2500 MHz   (7 dumbbell DGS, L2)
  2. BPF1  2500-3750 MHz   (5 CSRR L2 + 5 patches L3 + 5 H-shape L4)
  3. BPF2  3750-5000 MHz   (5 CSRR L2 + 5 patches L3 + 5 H-shape L4)

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
from matplotlib.patches import Rectangle, Arc

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
EPS0 = 8.854_187_817e-12
MU0 = 4e-7 * np.pi
Z_REF = 50.0

# ═══════════════════════════════════════════════════════════════════════
#  Substrate — Rogers RO3010 (high er for compactness)
# ═══════════════════════════════════════════════════════════════════════
ER_CORE = 10.2
H_CORE = 0.254e-3
TAND_CORE = 0.0035

ER_PP = 3.52
H_PP_THIN = 0.100e-3
H_PP_THICK = 0.175e-3
TAND_PP = 0.004

TOTAL_H = (
    35e-6 + H_CORE + 35e-6 + H_PP_THIN + 18e-6
    + H_CORE + 35e-6 + H_PP_THICK + 18e-6 + 35e-6
)

# Microstrip on RO3010 — 50 ohm reference
def _ms_eeff(w, h=H_CORE, er=ER_CORE):
    u = jnp.clip(w / h, 0.01, 1000.0)
    return (er + 1) / 2 + (er - 1) / 2 * jnp.power(1 + 12 / u, -0.5)

def _ms_width(z0, h=H_CORE, er=ER_CORE):
    A = z0 / 60.0 * jnp.sqrt((er + 1) / 2) + (er - 1) / (er + 1) * (0.23 + 0.11 / er)
    B = 377.0 * jnp.pi / (2.0 * z0 * jnp.sqrt(er))
    wh1 = 8 * jnp.exp(A) / (jnp.exp(2 * A) - 2)
    wh2 = (2 / jnp.pi) * (B - 1 - jnp.log(jnp.clip(2 * B - 1, 1e-6))
           + (er - 1) / (2 * er) * (jnp.log(jnp.clip(B - 1, 1e-6)) + 0.39 - 0.61 / er))
    return jnp.where(z0 > 63.0, wh1, wh2) * h

W50 = float(_ms_width(jnp.array(Z_REF)))
EEFF = float(_ms_eeff(jnp.array(W50)))

# ═══════════════════════════════════════════════════════════════════════
#  Manufacturing constraints
# ═══════════════════════════════════════════════════════════════════════
MIN_TRACE = 0.15e-3
MIN_SPACE = 0.15e-3
MIN_PAD = 0.50e-3
MIN_VIA = 0.20e-3

# ═══════════════════════════════════════════════════════════════════════
#  ABCD helpers
# ═══════════════════════════════════════════════════════════════════════
def _m(a, b, c, d):
    return jnp.array([[a, b], [c, d]], dtype=jnp.complex128)

def abcd_tl(z0, gl):
    ch, sh = jnp.cosh(gl), jnp.sinh(gl)
    return _m(ch, z0 * sh, sh / z0, ch)

def abcd_sh(Y):
    return _m(1.0 + 0j, 0j, Y, 1.0 + 0j)

def abcd_to_s(M):
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    den = A + B / Z_REF + C * Z_REF + D
    return 2.0 / den, (A + B / Z_REF - C * Z_REF - D) / den

def _gl_ms(f, length):
    beta = 2 * jnp.pi * f * jnp.sqrt(EEFF) / C0
    alpha = TAND_CORE * jnp.pi * f * jnp.sqrt(EEFF) / C0
    return (alpha + 1j * beta) * length

# ═══════════════════════════════════════════════════════════════════════
#  Shape equivalent circuits
# ═══════════════════════════════════════════════════════════════════════
def _dumbbell_Y(f, void_a, bridge_w, bridge_l):
    L = MU0 * bridge_l / jnp.maximum(bridge_w, MIN_SPACE)
    C_gap = EPS0 * ER_CORE * void_a * bridge_w / (2 * H_CORE)
    C_fringe = EPS0 * ER_CORE * void_a ** 2 / (4 * H_CORE)
    C = C_gap + C_fringe
    R = 0.6
    omega = 2 * jnp.pi * f
    Z = R + 1j * omega * L + 1.0 / (1j * omega * C + 1e-30)
    return 1.0 / Z


def _csrr_Y(f, radius, ring_w, gap_frac):
    circ = 2 * jnp.pi * radius
    gap_len = jnp.maximum(gap_frac * circ, MIN_SPACE)
    slot_len = circ - gap_len
    L = MU0 * slot_len / jnp.maximum(ring_w, MIN_SPACE) * 0.5
    C_slot = EPS0 * ER_CORE * ring_w * slot_len / H_CORE * 0.3
    C_gap = EPS0 * ER_CORE * ring_w ** 2 / jnp.maximum(gap_len, MIN_SPACE) * 0.5
    C = C_slot + C_gap
    R = 0.4
    omega = 2 * jnp.pi * f
    Z_lc = R + 1j * omega * L + 1.0 / (1j * omega * C + 1e-30)
    Cc = EPS0 * ER_CORE * jnp.pi * radius ** 2 / H_CORE * 0.12
    return Cc * omega ** 2 / (Z_lc * omega + 1e-20)


def _hshape_Y(f, arm_l, arm_w, bridge_w):
    L = MU0 * arm_w / jnp.maximum(bridge_w, MIN_SPACE)
    C = EPS0 * ER_CORE * arm_l * arm_w / H_CORE * 0.5
    R = 0.5
    omega = 2 * jnp.pi * f
    Z = R + 1j * omega * L + 1.0 / (1j * omega * C + 1e-30)
    return 1.0 / Z


def _coupled_patch_Y(f, w_patch, l_patch, er_sub, h_couple, coupling_atten=1.0):
    f_res = C0 / (2 * l_patch * jnp.sqrt(er_sub) + 1e-10)
    Cc = EPS0 * er_sub * w_patch * l_patch / h_couple * 0.06 * coupling_atten
    Q = 20.0
    omega = 2 * jnp.pi * f
    omega_0 = 2 * jnp.pi * f_res
    denom = omega_0 ** 2 - omega ** 2 + 1j * omega * omega_0 / Q
    return 1j * omega * Cc * omega_0 ** 2 / (denom + 1e-20)


# ═══════════════════════════════════════════════════════════════════════
#  Bounded transforms  (lower bounds = manufacturing minimums)
# ═══════════════════════════════════════════════════════════════════════
def _bnd(x, lo, hi):
    t = jax.nn.sigmoid(x)
    return jnp.exp(jnp.log(lo) * (1 - t) + jnp.log(hi) * t)

def _bnd_inv(y, lo, hi):
    t = (jnp.log(y) - jnp.log(lo)) / (jnp.log(hi) - jnp.log(lo))
    t = jnp.clip(t, 1e-6, 1 - 1e-6)
    return jnp.log(t / (1 - t))

def _pack(vals, bounds):
    parts, idx = [], 0
    for n, lo, hi in bounds:
        parts.append(jnp.array([float(_bnd_inv(vals[idx + i], lo, hi)) for i in range(n)]))
        idx += n
    return jnp.concatenate(parts)

def _unpack(p, bounds):
    res, i = [], 0
    for n, lo, hi in bounds:
        res.append(_bnd(p[i:i + n], lo, hi))
        i += n
    return res


# ═══════════════════════════════════════════════════════════════════════
#  HPF — 7 dumbbell DGS on L2
# ═══════════════════════════════════════════════════════════════════════
N_HPF = 7

HPF_BOUNDS = [
    (N_HPF, 0.4e-3, 12e-3),         # void_a  (min = 2*MIN_SPACE rounded)
    (N_HPF, MIN_SPACE, 3e-3),        # bridge_w
    (N_HPF, MIN_SPACE * 2, 6e-3),    # bridge_l
    (N_HPF + 1, 0.5e-3, 15e-3),     # spacings
]

def _hpf_init():
    target_f = np.linspace(0.6e9, 2.4e9, N_HPF)
    va, bw, bl = [], [], []
    for ft in target_f:
        b_w = 0.3e-3
        b_l = 1.5e-3
        L = MU0 * b_l / b_w
        C_need = 1.0 / ((2 * np.pi * ft) ** 2 * L)
        a = np.sqrt(max(4 * C_need * H_CORE / (EPS0 * ER_CORE), 0)) + 0.5e-3
        va.append(np.clip(a, 0.5e-3, 11e-3))
        bw.append(b_w)
        bl.append(b_l)
    sp = [4e-3] * (N_HPF + 1)
    return _pack(va + bw + bl + sp, HPF_BOUNDS)


def _hpf_at_f(p, f):
    va, bw, bl, sp = _unpack(p, HPF_BOUNDS)

    def body(M, x):
        va_k, bw_k, bl_k, sp_k = x
        Y = _dumbbell_Y(f, va_k, bw_k, bl_k)
        return M @ abcd_sh(Y) @ abcd_tl(Z_REF, _gl_ms(f, sp_k)), None

    M0 = abcd_tl(Z_REF, _gl_ms(f, sp[0]))
    M_f, _ = jax.lax.scan(body, M0, (va, bw, bl, sp[1:]))
    return abcd_to_s(M_f)

_hpf_batch = jax.vmap(_hpf_at_f, in_axes=(None, 0))

def hpf_loss(p, freqs):
    s21, s11 = _hpf_batch(p, freqs)
    s21d = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
    s11d = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
    fc = 2.5e9
    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    tb = (freqs > 1.5e9) & (freqs < fc)
    loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 1.0, 0.0) ** 2, 0.0)) * 5
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d + 10, 0.0) ** 2, 0.0)) * 2
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21d + 15, 0.0) ** 2, 0.0)) * 4
    loss += jnp.sum(jnp.where(tb, jnp.maximum(s21d + 8, 0.0) ** 2, 0.0)) * 2
    return loss / freqs.shape[0]


# ═══════════════════════════════════════════════════════════════════════
#  BPF — CSRR (L2) + patch (L3) + H-shape (L4)  ×  5 elements
# ═══════════════════════════════════════════════════════════════════════
N_BPF = 5

BPF_BOUNDS = [
    (N_BPF, 1.0e-3, 8e-3),          # csrr radius
    (N_BPF, MIN_SPACE, 2e-3),        # csrr ring_w
    (N_BPF, 0.03, 0.30),            # csrr gap_frac
    (N_BPF, MIN_PAD, 8e-3),         # L3 patch_w
    (N_BPF, 2e-3, 22e-3),           # L3 patch_l
    (N_BPF, 0.3e-3, 6e-3),          # L4 H arm_l
    (N_BPF, 0.3e-3, 4e-3),          # L4 H arm_w
    (N_BPF, MIN_SPACE, 2e-3),        # L4 H bridge_w
    (N_BPF + 1, 0.5e-3, 12e-3),     # spacings
]

def _bpf_init(fc, fbw):
    f_lo, f_hi = fc * (1 - fbw / 2), fc * (1 + fbw / 2)
    notch_f = np.array([f_lo * 0.55, f_lo * 0.8, fc, f_hi * 1.2, f_hi * 1.5])
    rad, rw, gf = [], [], []
    for ft in notch_f:
        circ = C0 / (ft * np.sqrt(ER_CORE)) * 0.25
        r = np.clip(circ / (2 * np.pi), 1.2e-3, 7.5e-3)
        rad.append(r)
        rw.append(0.4e-3)
        gf.append(0.08)
    pass_f = np.linspace(f_lo * 1.05, f_hi * 0.95, N_BPF)
    pw, pl = [], []
    for fp in pass_f:
        lp = C0 / (2 * fp * np.sqrt(ER_CORE))
        pw.append(2e-3)
        pl.append(np.clip(lp, 3e-3, 21e-3))
    al = [1.5e-3] * N_BPF
    aw = [1.0e-3] * N_BPF
    hbw = [0.3e-3] * N_BPF
    sp = [3e-3] * (N_BPF + 1)
    return _pack(rad + rw + gf + pw + pl + al + aw + hbw + sp, BPF_BOUNDS)


def _bpf_at_f(p, f):
    rad, rw, gf, pw, pl, al, aw, hbw, sp = _unpack(p, BPF_BOUNDS)

    def body(M, x):
        rad_k, rw_k, gf_k, pw_k, pl_k, al_k, aw_k, hbw_k, sp_k = x
        Y_L2 = _csrr_Y(f, rad_k, rw_k, gf_k)
        Y_L3 = _coupled_patch_Y(f, pw_k, pl_k, ER_CORE, H_PP_THIN, 1.0)
        Y_L4 = _hshape_Y(f, al_k, aw_k, hbw_k) * 0.4
        return M @ abcd_sh(Y_L2 + Y_L3 + Y_L4) @ abcd_tl(Z_REF, _gl_ms(f, sp_k)), None

    M0 = abcd_tl(Z_REF, _gl_ms(f, sp[0]))
    M_f, _ = jax.lax.scan(body, M0, (rad, rw, gf, pw, pl, al, aw, hbw, sp[1:]))
    return abcd_to_s(M_f)

_bpf_batch = jax.vmap(_bpf_at_f, in_axes=(None, 0))

def make_bpf_loss(f_lo, f_hi):
    bw = f_hi - f_lo
    def loss_fn(p, freqs):
        s21, s11 = _bpf_batch(p, freqs)
        s21d = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11d = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
        pb = (freqs >= f_lo) & (freqs <= f_hi)
        sl = freqs < (f_lo - 0.3 * bw)
        sh = freqs > (f_hi + 0.3 * bw)
        loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 2.0, 0.0) ** 2, 0.0)) * 6
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d + 8, 0.0) ** 2, 0.0)) * 2
        loss += jnp.sum(jnp.where(sl, jnp.maximum(s21d + 15, 0.0) ** 2, 0.0)) * 3
        loss += jnp.sum(jnp.where(sh, jnp.maximum(s21d + 15, 0.0) ** 2, 0.0)) * 3
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
CU = "#b87333"
VOID = "#1a1a2e"
PATCH = "#dd9955"
FEED_C = "#e74c3c"


def _draw_dumbbell(ax, x, y, a, bw, bl):
    for dx in [-bl / 2 - a, bl / 2]:
        ax.add_patch(Rectangle((x + dx, y - a / 2), a, a, fc=VOID, ec="#555", lw=0.5, zorder=2))
    ax.add_patch(Rectangle((x - bl / 2, y - bw / 2), bl, bw, fc=VOID, ec="#555", lw=0.5, zorder=2))


def _draw_csrr(ax, x, y, r, rw, gf):
    gap_deg = gf * 360
    th = np.linspace(np.radians(90 + gap_deg / 2), np.radians(90 - gap_deg / 2 + 360), 80)
    for j in range(len(th) - 1):
        ax.plot(
            [x + r * np.cos(th[j]), x + r * np.cos(th[j + 1])],
            [y + r * np.sin(th[j]), y + r * np.sin(th[j + 1])],
            color=VOID, lw=max(rw * 6, 1.5), solid_capstyle="round", zorder=2,
        )


def _draw_hshape(ax, x, y, al, aw, bw):
    ax.add_patch(Rectangle((x - al / 2, y - aw - bw / 2), al, aw, fc=VOID, ec="#555", lw=0.4, zorder=2))
    ax.add_patch(Rectangle((x - al / 2, y + bw / 2), al, aw, fc=VOID, ec="#555", lw=0.4, zorder=2))
    ax.add_patch(Rectangle((x - bw / 2, y - bw / 2), bw, bw, fc=VOID, ec="#555", lw=0.4, zorder=2))


def _draw_patch(ax, x, y, w, l):
    ax.add_patch(Rectangle((x - l / 2, y - w / 2), l, w, fc=PATCH, ec="#996633", lw=0.6, zorder=2))


def _draw_layer(ax, elems, draw_fn, total_len, total_wid, title, bg):
    ax.add_patch(Rectangle((0, 0), total_len, total_wid, fc=bg, ec="#333", lw=1.2, zorder=1))
    fy = total_wid / 2
    ax.plot([0, total_len], [fy, fy], color=FEED_C, lw=1.8, ls="--", alpha=0.5, zorder=3, label="Feedline")
    for e in elems:
        draw_fn(ax, e["x"], fy, **e["kw"])
    ax.set_xlim(-1, total_len + 1)
    ax.set_ylim(-1, total_wid + 1)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper right")


def _check_mfg(name, dims_dict):
    """Print manufacturing check for a set of dimensions."""
    violations = []
    for label, val, limit in dims_dict:
        if val < limit - 1e-6:
            violations.append(f"    FAIL  {label} = {val*1e3:.3f} mm < {limit*1e3:.3f} mm min")
    if violations:
        print(f"  {name}: {len(violations)} violation(s)")
        for v in violations:
            print(v)
    else:
        print(f"  {name}: ALL PASS")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs = jnp.linspace(0.1e9, 7e9, 500)
    generated = []

    lam_g = C0 / (3.5e9 * np.sqrt(EEFF))
    print("=" * 68)
    print("  Shape-Library Multilayer DGS Filter Bank")
    print("=" * 68)
    print(f"  Substrate      : Rogers RO3010  er={ER_CORE}  h={H_CORE*1e3:.3f} mm")
    print(f"  Prepreg        : RO4450F  er={ER_PP}")
    print(f"  50-ohm width   : {W50*1e3:.3f} mm")
    print(f"  Lambda_g @3.5G : {lam_g*1e3:.1f} mm")
    print(f"  Board thickness: {TOTAL_H*1e3:.3f} mm")
    print(f"  Mfg minimums   : trace {MIN_TRACE*1e3:.2f} mm, space {MIN_SPACE*1e3:.2f} mm")
    print()

    # ── HPF ──────────────────────────────────────────────────────────
    print("=" * 68)
    print(f"  HPF  ({N_HPF} dumbbell DGS on L2)")
    print("=" * 68)
    p_hpf = opt_restarts(hpf_loss, _hpf_init, freqs, n_r=3, n_steps=2500, lr=5e-3)

    # ── BPF1 ─────────────────────────────────────────────────────────
    f1l, f1h = 2.5e9, 3.75e9
    fc1, fbw1 = (f1l + f1h) / 2, (f1h - f1l) / ((f1l + f1h) / 2)
    bpf1_loss = make_bpf_loss(f1l, f1h)
    print("\n" + "=" * 68)
    print(f"  BPF1  ({N_BPF} CSRR L2 + {N_BPF} patch L3 + {N_BPF} H-shape L4)")
    print("=" * 68)
    p_b1 = opt_restarts(bpf1_loss, lambda: _bpf_init(fc1, fbw1), freqs, n_r=3, n_steps=3000, lr=5e-3)

    # ── BPF2 ─────────────────────────────────────────────────────────
    f2l, f2h = 3.75e9, 5.0e9
    fc2, fbw2 = (f2l + f2h) / 2, (f2h - f2l) / ((f2l + f2h) / 2)
    bpf2_loss = make_bpf_loss(f2l, f2h)
    print("\n" + "=" * 68)
    print(f"  BPF2  ({N_BPF} CSRR L2 + {N_BPF} patch L3 + {N_BPF} H-shape L4)")
    print("=" * 68)
    p_b2 = opt_restarts(bpf2_loss, lambda: _bpf_init(fc2, fbw2), freqs, n_r=3, n_steps=3000, lr=5e-3)

    # ── Evaluate ─────────────────────────────────────────────────────
    fp = jnp.linspace(0.1e9, 7e9, 1000)
    fg = np.array(fp) / 1e9
    db = lambda s: 20 * np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))

    s21h, s11h = _hpf_batch(p_hpf, fp)
    s21b1, s11b1 = _bpf_batch(p_b1, fp)
    s21b2, s11b2 = _bpf_batch(p_b2, fp)

    filters = [
        ("HPF fc=2.5GHz", s21h, s11h, [2.5], p_hpf, "hpf"),
        ("BPF1 2.5-3.75GHz", s21b1, s11b1, [2.5, 3.75], p_b1, "bpf1"),
        ("BPF2 3.75-5.0GHz", s21b2, s11b2, [3.75, 5.0], p_b2, "bpf2"),
    ]

    # ── Per-filter design sheets ─────────────────────────────────────
    for name, s21, s11, vlines, params, prefix in filters:
        is_hpf = prefix == "hpf"

        if is_hpf:
            va, bw, bl, sp = (np.array(x) for x in _unpack(params, HPF_BOUNDS))
            pos = np.cumsum(sp) * 1e3
            total_l = float(np.sum(sp)) * 1e3
            gnd_w = 18.0
            dgs_elems = [dict(x=pos[k], kw=dict(a=va[k]*1e3, bw=bw[k]*1e3, bl=bl[k]*1e3)) for k in range(N_HPF)]
            patch_elems = []
            l4_elems = []
        else:
            rad, rw, gf, pw, pl, al, aw, hbw, sp = (np.array(x) for x in _unpack(params, BPF_BOUNDS))
            pos = np.cumsum(sp) * 1e3
            total_l = float(np.sum(sp)) * 1e3
            gnd_w = 18.0
            dgs_elems = [dict(x=pos[k], kw=dict(r=rad[k]*1e3, rw=rw[k]*1e3, gf=gf[k])) for k in range(N_BPF)]
            patch_elems = [dict(x=pos[k], kw=dict(w=pw[k]*1e3, l=pl[k]*1e3)) for k in range(N_BPF)]
            l4_elems = [dict(x=pos[k], kw=dict(al=al[k]*1e3, aw=aw[k]*1e3, bw=hbw[k]*1e3)) for k in range(N_BPF)]

        n_layers = 1 if is_hpf else 3
        fig, axes = plt.subplots(2, max(n_layers, 2), figsize=(7 * max(n_layers, 2), 10))
        fig.suptitle(f"Shape-Library Filter: {name}  (Rogers RO3010)", fontsize=13, fontweight="bold")

        draw_dgs = _draw_dumbbell if is_hpf else _draw_csrr
        _draw_layer(axes[0, 0], dgs_elems, draw_dgs, total_l, gnd_w,
                    "L2 — Ground DGS", CU)
        if not is_hpf:
            _draw_layer(axes[0, 1], patch_elems, _draw_patch, total_l, gnd_w,
                        "L3 — Coupled Patches", "#2c3e50")
            _draw_layer(axes[0, 2], l4_elems, _draw_hshape, total_l, gnd_w,
                        "L4 — Secondary H-shape DGS", CU)
        else:
            axes[0, 1].set_visible(False)

        ax = axes[1, 0]
        ax.plot(fg, db(s21), "b", lw=1.6, label="|S21|")
        ax.plot(fg, db(s11), "r--", lw=1.0, label="|S11|")
        ax.set_title("Frequency Response")
        ax.set_xlabel("Freq [GHz]")
        ax.set_ylabel("[dB]")
        ax.set_ylim(-50, 3)
        ax.legend()
        ax.grid(True, alpha=0.3)
        for fv in vlines:
            ax.axvline(fv, color="gray", ls=":", lw=0.8)

        ax2 = axes[1, 1] if n_layers >= 2 else axes[1, 0]
        lines = [f"Substrate: RO3010 er={ER_CORE}", f"Board: {TOTAL_H*1e3:.2f} mm thick", ""]
        if is_hpf:
            for k in range(N_HPF):
                L_k = MU0 * bl[k] / max(bw[k], 1e-4)
                C_k = EPS0 * ER_CORE * va[k]**2 / (4*H_CORE) + EPS0 * ER_CORE * va[k] * bw[k] / (2*H_CORE)
                f0 = 1 / (2 * np.pi * np.sqrt(L_k * C_k))
                lines.append(f"DGS{k+1}: void={va[k]*1e3:.2f} bridge={bw[k]*1e3:.2f}x{bl[k]*1e3:.2f}mm f0={f0/1e9:.2f}GHz")
        else:
            for k in range(N_BPF):
                fres = C0 / (2 * pl[k] * np.sqrt(ER_CORE))
                lines.append(f"CSRR{k+1}: r={rad[k]*1e3:.2f} rw={rw[k]*1e3:.2f}mm gap={gf[k]*100:.0f}%")
                lines.append(f"Patch{k+1}: {pw[k]*1e3:.2f}x{pl[k]*1e3:.2f}mm fres={fres/1e9:.2f}GHz")
                lines.append(f"H-DGS{k+1}: arm={al[k]*1e3:.2f}x{aw[k]*1e3:.2f} brg={hbw[k]*1e3:.2f}mm")
                lines.append("")
        ax2.text(0.03, 0.97, "\n".join(lines), transform=ax2.transAxes,
                 fontsize=8, fontfamily="monospace", va="top")
        ax2.set_axis_off()
        ax2.set_title("Dimensions & Resonances")

        if n_layers >= 3:
            axes[1, 2].set_visible(False)

        fig.tight_layout()
        fname = f"shape_{prefix}_design.png"
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        generated.append(fname)

    # ── Combined overview ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Shape-Library Multilayer DGS Filter Bank  |  Rogers RO3010",
                 fontsize=14, fontweight="bold")
    for ci, (name, s21, s11, vl, params, prefix) in enumerate(filters):
        is_hpf = prefix == "hpf"
        if is_hpf:
            va, bw, bl, sp = (np.array(x) for x in _unpack(params, HPF_BOUNDS))
            pos = np.cumsum(sp) * 1e3
            total_l = float(np.sum(sp)) * 1e3
            elems = [dict(x=pos[k], kw=dict(a=va[k]*1e3, bw=bw[k]*1e3, bl=bl[k]*1e3)) for k in range(N_HPF)]
            df = _draw_dumbbell
        else:
            rad, rw, gf, pw, pl, al, aw, hbw, sp = (np.array(x) for x in _unpack(params, BPF_BOUNDS))
            pos = np.cumsum(sp) * 1e3
            total_l = float(np.sum(sp)) * 1e3
            elems = [dict(x=pos[k], kw=dict(r=rad[k]*1e3, rw=rw[k]*1e3, gf=gf[k])) for k in range(N_BPF)]
            df = _draw_csrr
        _draw_layer(axes[0, ci], elems, df, total_l, 18.0, f"L2 — {name}", CU)
        ax = axes[1, ci]
        ax.plot(fg, db(s21), "b", lw=1.5, label="|S21|")
        ax.plot(fg, db(s11), "r--", lw=1, label="|S11|")
        ax.set_title(name)
        ax.set_xlabel("Freq [GHz]")
        ax.set_ylabel("[dB]")
        ax.set_ylim(-50, 3)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        for fv in vl:
            ax.axvline(fv, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig("shape_filter_bank.png", dpi=200)
    plt.close(fig)
    generated.append("shape_filter_bank.png")

    # ── Stackup diagram ──────────────────────────────────────────────
    fig_s, ax_s = plt.subplots(figsize=(12, 5))
    ax_s.set_title("6-Layer Stackup  |  Rogers RO3010 + RO4450F", fontsize=13, fontweight="bold")
    stack = [
        ("L1 Signal — Feedline", 35e-6, "#e67e22"),
        ("RO3010 core (er=10.2, 0.254mm)", H_CORE, "#f5e6d3"),
        ("L2 Ground — DGS shapes", 35e-6, "#27ae60"),
        ("RO4450F prepreg (er=3.52, 0.100mm)", H_PP_THIN, "#ecf0f1"),
        ("L3 Signal — Coupled patches", 18e-6, "#e67e22"),
        ("RO3010 core (er=10.2, 0.254mm)", H_CORE, "#f5e6d3"),
        ("L4 Ground — H-shape DGS", 35e-6, "#27ae60"),
        ("RO4450F prepreg (er=3.52, 0.175mm)", H_PP_THICK, "#ecf0f1"),
        ("L5 Signal — (optional 2nd patches)", 18e-6, "#e67e22"),
        ("L6 Ground — Bottom ref", 35e-6, "#27ae60"),
    ]
    y = 0
    for label, t, c in reversed(stack):
        h_mm = max(t * 1e3, 0.015)
        ax_s.add_patch(Rectangle((1, y), 8, h_mm, fc=c, ec="#333", lw=0.8))
        ax_s.text(9.3, y + h_mm / 2, label, va="center", fontsize=8, fontfamily="monospace")
        y += h_mm
    ax_s.set_xlim(0, 20)
    ax_s.set_ylim(-0.02, y + 0.04)
    ax_s.set_ylabel("mm")
    ax_s.set_xticks([])
    ax_s.text(5, y + 0.025, f"Total: {TOTAL_H*1e3:.3f} mm", ha="center", fontsize=11, fontweight="bold")
    fig_s.tight_layout()
    fig_s.savefig("shape_stackup.png", dpi=200)
    plt.close(fig_s)
    generated.append("shape_stackup.png")

    # ── Performance + Manufacturing audit ────────────────────────────
    def _perf(s21, s11, flo, fhi, label):
        s21db, s11db = db(s21), db(s11)
        pb = (fg >= flo) & (fg <= fhi)
        if not np.any(pb):
            return
        print(f"  {label}:")
        print(f"    Passband IL : {float(-np.max(s21db[pb])):.1f} - {float(-np.min(s21db[pb])):.1f} dB")
        print(f"    Passband RL : > {float(-np.max(s11db[pb])):.1f} dB")
        total_len = 0
        return

    print("\n" + "=" * 68)
    print("  PERFORMANCE")
    print("=" * 68)
    _perf(s21h, s11h, 2.5, 5.5, "HPF  (L2 dumbbell DGS)")
    _perf(s21b1, s11b1, 2.5, 3.75, "BPF1 (L2 CSRR + L3 patch + L4 H-shape)")
    _perf(s21b2, s11b2, 3.75, 5.0, "BPF2 (L2 CSRR + L3 patch + L4 H-shape)")

    print("\n" + "=" * 68)
    print("  MANUFACTURING AUDIT")
    print("=" * 68)
    va, bw, bl, sp = (np.array(x) for x in _unpack(p_hpf, HPF_BOUNDS))
    checks = [(f"bridge_w[{k}]", bw[k], MIN_SPACE) for k in range(N_HPF)]
    checks += [(f"bridge_l[{k}]", bl[k], MIN_SPACE * 2) for k in range(N_HPF)]
    _check_mfg("HPF dumbbells", checks)

    for tag, params in [("BPF1", p_b1), ("BPF2", p_b2)]:
        rad, rw, gf, pw, pl, al, aw, hbw, sp = (np.array(x) for x in _unpack(params, BPF_BOUNDS))
        checks = []
        for k in range(N_BPF):
            circ = 2 * np.pi * rad[k]
            gap_mm = gf[k] * circ
            checks.append((f"CSRR ring_w[{k}]", rw[k], MIN_SPACE))
            checks.append((f"CSRR gap[{k}]", gap_mm, MIN_SPACE))
            checks.append((f"patch_w[{k}]", pw[k], MIN_PAD))
            checks.append((f"patch_l[{k}]", pl[k], 2e-3))
            checks.append((f"H bridge_w[{k}]", hbw[k], MIN_SPACE))
        _check_mfg(f"{tag} shapes", checks)

    # ── Compactness comparison ───────────────────────────────────────
    hpf_len = float(np.sum(_unpack(p_hpf, HPF_BOUNDS)[-1])) * 1e3
    b1_len = float(np.sum(_unpack(p_b1, BPF_BOUNDS)[-1])) * 1e3
    b2_len = float(np.sum(_unpack(p_b2, BPF_BOUNDS)[-1])) * 1e3

    print("\n" + "=" * 68)
    print("  COMPACTNESS  (RO3010 er=10.2 vs RO4003C er=3.55)")
    print("=" * 68)
    ratio = np.sqrt(3.55 / 10.2)
    print(f"  Wavelength reduction factor : {ratio:.2f}x  ({(1-ratio)*100:.0f}% shorter)")
    print(f"  HPF  total length : {hpf_len:.1f} mm")
    print(f"  BPF1 total length : {b1_len:.1f} mm")
    print(f"  BPF2 total length : {b2_len:.1f} mm")
    print(f"  Board thickness   : {TOTAL_H*1e3:.2f} mm")

    # ── Output images ────────────────────────────────────────────────
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
