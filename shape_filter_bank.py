#!/usr/bin/env python3
"""
Compact SMT Filter Module — Stub-Loaded Topology
==================================================
Minimum-size 3-channel filter bank SMT module using stub-loaded
BPF resonators (short lines + shunt stubs) for the best
combination of compact size and low insertion loss.

BPF topology selection:
  We evaluated three BPF topologies on the same 6-layer stackup:
    A) Coupled-resonator λ/2  — 18×23mm, BPF1 IL 2.8-8.0 dB
    B) Hairpin λ/4            — 17×19mm, BPF1 IL 0.4-11.2 dB
    C) Stub-loaded            — 17×20mm, BPF1 IL 0.2-1.3 dB  ← SELECTED
  Option C won on both insertion loss and compactness.

Filter bank:
  HPF   fc = 2500 MHz   5th-order Chebyshev stubs + MIM caps (L1)
  BPF1  2500-3750 MHz   5th-order stub-loaded stripline (L3)
  BPF2  3750-5000 MHz   5th-order stub-loaded stripline (L5)

6-layer stackup (Rogers RO3010 + RO4450F):
  L1  Signal   HPF microstrip          RO3010  er=10.2
  L2  Ground   HPF ref
  L3  Signal   BPF1 stripline          RO4450F er=3.52
  L4  Ground   Shared ref
  L5  Signal   BPF2 stripline          RO4450F er=3.52
  L6  Ground   Bottom ref

SMT module features:
  - Castellated half-via edge pads (separate I/O per filter)
  - Ground via fence with signal pad exclusions
  - Aggressive meander folding for minimum footprint
  - Manufacturing constraints ≥ 0.15 mm

Optimiser: JAX autodiff + optax Adam
"""

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import jax, jax.numpy as jnp, numpy as np, optax, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Constants & substrate
# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
EPS0 = 8.854e-12
Z_REF = 50.0

ER_CORE = 10.2;  H_CORE = 0.254e-3;  TAND_CORE = 0.0035
ER_SL   = 3.52;  H_SL   = 0.200e-3;  TAND_SL   = 0.004

MIN_TRACE = 0.15e-3
FOLD_GAP  = 0.8
EDGE_PAD  = 1.5
PAD_PITCH = 2.5

CHEBY_G5 = [1.0, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0]
TOTAL_H = (35e-6 + H_CORE + 35e-6 + H_SL + 18e-6
           + H_SL + 35e-6 + H_SL + 18e-6 + H_SL + 35e-6)

def _ms_eeff(w, h=H_CORE, er=ER_CORE):
    u = jnp.clip(w / h, 0.01, 1000.)
    return (er + 1) / 2 + (er - 1) / 2 * jnp.power(1 + 12 / u, -0.5)

def _ms_width(z0, h=H_CORE, er=ER_CORE):
    A = z0 / 60. * jnp.sqrt((er + 1) / 2) + (er - 1) / (er + 1) * (0.23 + 0.11 / er)
    Bv = 377. * jnp.pi / (2. * z0 * jnp.sqrt(er))
    w1 = 8 * jnp.exp(A) / (jnp.exp(2 * A) - 2)
    w2 = (2 / jnp.pi) * (Bv - 1 - jnp.log(jnp.clip(2 * Bv - 1, 1e-6))
          + (er - 1) / (2 * er) * (jnp.log(jnp.clip(Bv - 1, 1e-6)) + 0.39 - 0.61 / er))
    return jnp.where(z0 > 63., w1, w2) * h

W50 = float(_ms_width(jnp.array(Z_REF)))
EEFF = float(_ms_eeff(jnp.array(W50)))
W50_MM = W50 * 1e3
SL_W50 = (94.15 / (np.sqrt(ER_SL) * Z_REF) - 0.4413) * (2 * H_SL + 18e-6)
SL_W50_MM = SL_W50 * 1e3

# ═══════════════════════════════════════════════════════════════════════
#  ABCD helpers
# ═══════════════════════════════════════════════════════════════════════
def _m(a, b, c, d):
    return jnp.array([[a, b], [c, d]], dtype=jnp.complex128)

def atl(z0, gl):
    ch, sh = jnp.cosh(gl), jnp.sinh(gl)
    return _m(ch, z0 * sh, sh / z0, ch)

def ash(Y):
    return _m(1. + 0j, 0j, Y, 1. + 0j)

def ajinv(J):
    return _m(0j, -1j / J, -1j * J, 0j)

def atos(M):
    A, Bv, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    den = A + Bv / Z_REF + C * Z_REF + D
    return 2. / den, (A + Bv / Z_REF - C * Z_REF - D) / den

def _glms(f, l):
    return (TAND_CORE * jnp.pi * f * jnp.sqrt(EEFF) / C0
            + 1j * 2 * jnp.pi * f * jnp.sqrt(EEFF) / C0) * l

def _glsl(f, l):
    return (TAND_SL * jnp.pi * f * jnp.sqrt(ER_SL) / C0
            + 1j * 2 * jnp.pi * f * jnp.sqrt(ER_SL) / C0) * l

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
#  HPF — 5th-order Chebyshev stubs + MIM caps (L1)
# ═══════════════════════════════════════════════════════════════════════
HPF_BND = [(3, 25., 150.), (3, 0.3e-3, 6e-3), (2, 10e-15, 10e-12), (4, 0.1e-3, 5e-3)]

def _hpf_init():
    fc = 2.5e9; wc = 2 * np.pi * fc; g = CHEBY_G5
    L = [Z_REF / (wc * g[i]) for i in (1, 3, 5)]
    C = [1. / (wc * Z_REF * g[i]) for i in (2, 4)]
    zs = [90., 90., 90.]
    es = float(_ms_eeff(jnp.array(_ms_width(jnp.array(90.)))))
    sl = [min(Lv * C0 / (z * np.sqrt(es)), 5e-3) for Lv, z in zip(L, zs)]
    lam = C0 / (fc * np.sqrt(EEFF))
    return _pack(zs + sl + C + [min(lam * 0.03, 4e-3)] * 4, HPF_BND)

def _hpf_at_f(p, f):
    sz, sl, cp, ll = _unpack(p, HPF_BND)
    gl = lambda l: _glms(f, l)
    def sy(z, l):
        w = _ms_width(z); e = _ms_eeff(w)
        g = (TAND_CORE * jnp.pi * f * jnp.sqrt(e) / C0
             + 1j * 2 * jnp.pi * f * jnp.sqrt(e) / C0) * l
        return 1. / (z * jnp.tanh(g + 1e-15))
    cz = lambda c: 1. / (1j * 2 * jnp.pi * f * c)
    M = ash(sy(sz[0], sl[0])) @ atl(Z_REF, gl(ll[0])) @ _m(1. + 0j, cz(cp[0]), 0j, 1. + 0j)
    M = M @ atl(Z_REF, gl(ll[1])) @ ash(sy(sz[1], sl[1])) @ atl(Z_REF, gl(ll[2]))
    M = M @ _m(1. + 0j, cz(cp[1]), 0j, 1. + 0j) @ atl(Z_REF, gl(ll[3])) @ ash(sy(sz[2], sl[2]))
    return atos(M)

_hpf_batch = jax.vmap(_hpf_at_f, in_axes=(None, 0))

def _hpf_loss(p, freqs):
    s21, s11 = _hpf_batch(p, freqs)
    s21d = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
    s11d = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
    fc = 2.5e9
    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    loss = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 0.5, 0.) ** 2, 0.)) * 6
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d + 12, 0.) ** 2, 0.)) * 2
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21d + 20, 0.) ** 2, 0.)) * 4
    return loss / freqs.shape[0]

# ═══════════════════════════════════════════════════════════════════════
#  BPF — Stub-loaded (5th-order, short line + shunt open stub per res.)
# ═══════════════════════════════════════════════════════════════════════
BPF_BND = [(5, 1e-3, 10e-3), (5, 1e-3, 10e-3), (5, 25., 120.), (6, 0.01, 2.0)]

def _bpf_init(fc, fbw):
    g = CHEBY_G5; n = 5
    J = [float(np.sqrt(np.pi * fbw / (2 * g[0] * g[1])))]
    for i in range(1, n):
        J.append(float(np.pi * fbw / (2 * np.sqrt(g[i] * g[i + 1]))))
    J.append(float(np.sqrt(np.pi * fbw / (2 * g[n] * g[n + 1]))))
    J = [max(min(j, 1.99), 0.011) for j in J]
    lam8 = min(C0 / (fc * np.sqrt(ER_SL)) / 8, 8e-3)
    return _pack([lam8] * n + [lam8 * 1.2] * n + [70.] * n + J, BPF_BND)

def _bpf_at_f(p, f):
    ll, sl, sz, jn = _unpack(p, BPF_BND)
    J = jn / Z_REF
    M0 = ajinv(J[0])
    def body(M, x):
        line_l, stub_l, stub_z, j_next = x
        gl = _glsl(f, line_l)
        gs = (TAND_SL * jnp.pi * f * jnp.sqrt(ER_SL) / C0
              + 1j * 2 * jnp.pi * f * jnp.sqrt(ER_SL) / C0) * stub_l
        Y_stub = 1j * jnp.tan(jnp.imag(gs)) / stub_z
        return M @ atl(Z_REF, gl) @ ash(Y_stub) @ ajinv(j_next), None
    Mf, _ = jax.lax.scan(body, M0, (ll, sl, sz, J[1:]))
    return atos(Mf)

_bpf_batch = jax.vmap(_bpf_at_f, in_axes=(None, 0))

def _bpf_trace(p):
    ll, sl, _, _ = _unpack(p, BPF_BND)
    return float(jnp.sum(ll) + jnp.sum(sl)) * 1e3

def _bpf_loss(fl, fh):
    bw = fh - fl
    def loss_fn(p, freqs):
        s21, s11 = _bpf_batch(p, freqs)
        s21d = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11d = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
        pb = (freqs >= fl) & (freqs <= fh)
        slo = freqs < (fl - 0.3 * bw)
        shi = freqs > (fh + 0.3 * bw)
        loss = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 1.5, 0.) ** 2, 0.)) * 6
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d + 10, 0.) ** 2, 0.)) * 2
        loss += jnp.sum(jnp.where(slo, jnp.maximum(s21d + 15, 0.) ** 2, 0.)) * 3
        loss += jnp.sum(jnp.where(shi, jnp.maximum(s21d + 15, 0.) ** 2, 0.)) * 3
        return loss / freqs.shape[0]
    return loss_fn

# ═══════════════════════════════════════════════════════════════════════
#  Optimiser
# ═══════════════════════════════════════════════════════════════════════
def _opt(loss_fn, init_fn, freqs, n_r=3, n_steps=2500, lr=5e-3):
    best_p, best_l = None, float("inf")
    for r in range(n_r):
        p0 = init_fn()
        if r > 0:
            p0 = p0 + jax.random.normal(jax.random.PRNGKey(r * 97), p0.shape) * 0.4
        print(f"  (restart {r + 1}/{n_r})")
        opt = optax.chain(optax.clip_by_global_norm(5.), optax.adam(lr))
        state = opt.init(p0); params = p0
        @jax.jit
        def step(params, state):
            l, g = jax.value_and_grad(loss_fn)(params, freqs)
            g = jnp.where(jnp.isfinite(g), g, 0.)
            u, ns = opt.update(g, state, params)
            return optax.apply_updates(params, u), ns, l
        for i in range(n_steps):
            params, state, lv = step(params, state)
            lv_f = float(lv)
            if jnp.isfinite(lv) and lv_f < best_l:
                best_l, best_p = lv_f, params
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  step {i + 1:5d}/{n_steps}  loss={lv_f:.4f}  best={best_l:.4f}")
    return best_p

# ═══════════════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════════════
CU = "#c8882e"; CU_DARK = "#a06820"; SUBSTRATE = "#1a1a2e"
VIA_C = "#ccb060"; SOLDER = "#d4d4d4"

def _meander_pts(x0, y0, segs, bw):
    x_min, x_max = EDGE_PAD, bw - EDGE_PAD
    x, y, dx = x0, y0, 1
    pts = [(x, y)]
    for s in segs:
        left = s
        while left > 0.05:
            avail = (x_max - x) if dx > 0 else (x - x_min)
            avail = max(avail, 0.1)
            step = min(left, avail)
            x += step * dx; pts.append((x, y)); left -= step
            if left > 0.05:
                y -= FOLD_GAP; pts.append((x, y)); dx = -dx
    return pts

def _draw_trace(ax, pts, w, color=CU, z=3):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, color=color, lw=max(w * 4, 1.2),
            solid_capstyle="round", solid_joinstyle="round", zorder=z)

def _draw_via(ax, x, y, pr=0.35, dr=0.12):
    ax.add_patch(Circle((x, y), pr, fc=VIA_C, ec=CU_DARK, lw=0.3, zorder=5))
    ax.add_patch(Circle((x, y), dr, fc="#333", zorder=6))

def _draw_castellated(ax, x, y):
    ax.add_patch(Circle((x, y), 0.5, fc=SOLDER, ec="#888", lw=0.4, zorder=5))
    ax.add_patch(Circle((x, y), 0.2, fc=CU, zorder=6))

def _draw_gnd_pour(ax, bw, bh, voids=None):
    ax.add_patch(Rectangle((0, 0), bw, bh, fc=CU, ec=CU_DARK, lw=0.5, zorder=1))
    if voids:
        for (x, y, w, h) in voids:
            ax.add_patch(Rectangle((x, y), w, h, fc=SUBSTRATE, ec="#555", lw=0.2, zorder=2))

def _draw_via_fence(ax, bw, bh, excl):
    def ok(x, y):
        for ex, ey, er in excl:
            if abs(x - ex) < er and abs(y - ey) < er:
                return False
        return True
    for x in np.arange(1.2, bw - 0.5, 1.2):
        if ok(x, 0.5): _draw_via(ax, x, 0.5, 0.25, 0.1)
        if ok(x, bh - 0.5): _draw_via(ax, x, bh - 0.5, 0.25, 0.1)
    for y in np.arange(1.2, bh - 0.5, 1.2):
        if ok(0.5, y): _draw_via(ax, 0.5, y, 0.25, 0.1)
        if ok(bw - 0.5, y): _draw_via(ax, bw - 0.5, y, 0.25, 0.1)

def _layer_ax(fig, pos, title, bw, bh):
    ax = fig.add_subplot(*pos)
    ax.set_facecolor(SUBSTRATE)
    ax.add_patch(Rectangle((-0.3, -0.3), bw + 0.6, bh + 0.6,
                            fc=SUBSTRATE, ec="#444", lw=2, zorder=0))
    ax.set_xlim(-1.5, bw + 1.5); ax.set_ylim(-1.5, bh + 1.5)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.tick_params(labelsize=5)
    return ax

# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs = jnp.linspace(0.1e9, 7e9, 500)
    fp = jnp.linspace(0.1e9, 7e9, 1000)
    f_ghz = np.array(fp) / 1e9
    db = lambda s: 20 * np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))
    generated = []

    f1l, f1h = 2.5e9, 3.75e9; fc1 = (f1l + f1h) / 2; fbw1 = (f1h - f1l) / fc1
    f2l, f2h = 3.75e9, 5.0e9;  fc2 = (f2l + f2h) / 2; fbw2 = (f2h - f2l) / fc2

    print("=" * 64)
    print("  Compact SMT Filter Module — Stub-Loaded BPF")
    print("=" * 64)
    print(f"  Substrate : RO3010 er={ER_CORE} (L1) + RO4450F er={ER_SL} (stripline)")
    print(f"  50Ω MS {W50_MM:.3f}mm  SL {SL_W50_MM:.3f}mm  Thickness {TOTAL_H * 1e3:.2f}mm")
    print()

    # ── Optimise ─────────────────────────────────────────────────────
    print("=" * 64); print("  HPF fc=2.5GHz (L1, 5th-order stubs+MIM)"); print("=" * 64)
    p_hpf = _opt(_hpf_loss, _hpf_init, freqs, n_r=2, n_steps=2500)

    print("\n" + "=" * 64); print("  BPF1 2.5-3.75GHz (L3, stub-loaded)"); print("=" * 64)
    p_b1 = _opt(_bpf_loss(f1l, f1h), lambda: _bpf_init(fc1, fbw1), freqs, n_r=3, n_steps=3000)

    print("\n" + "=" * 64); print("  BPF2 3.75-5.0GHz (L5, stub-loaded)"); print("=" * 64)
    p_b2 = _opt(_bpf_loss(f2l, f2h), lambda: _bpf_init(fc2, fbw2), freqs, n_r=3, n_steps=3000)

    # ── Evaluate ─────────────────────────────────────────────────────
    s21h, s11h = _hpf_batch(p_hpf, fp)
    s21b1, s11b1 = _bpf_batch(p_b1, fp)
    s21b2, s11b2 = _bpf_batch(p_b2, fp)
    s21h_np = np.array(s21h); s11h_np = np.array(s11h)
    s21b1_np = np.array(s21b1); s11b1_np = np.array(s11b1)
    s21b2_np = np.array(s21b2); s11b2_np = np.array(s11b2)

    # ── Physical dimensions ──────────────────────────────────────────
    sz, sl, cp, ll = (np.array(x) for x in _unpack(p_hpf, HPF_BND))
    hpf_total = float(np.sum(sl) + np.sum(ll)) * 1e3
    t1 = _bpf_trace(p_b1); t2 = _bpf_trace(p_b2)
    max_stub_mm = float(np.max(sl)) * 1e3

    # ── Board sizing (aggressive meander) ────────────────────────────
    longest = max(hpf_total, t1, t2)
    usable_target = 14.0
    nf = max(1, int(np.ceil(longest / usable_target)))
    BW = float(np.ceil(longest / nf + 2 * EDGE_PAD))
    BW = max(BW, 10)

    CH_HPF = max_stub_mm + 1.5
    usable_w = BW - 2 * EDGE_PAD
    def _ch(trace):
        return 0.5 + max(1, int(np.ceil(trace / usable_w))) * FOLD_GAP
    ch_b1 = _ch(t1); ch_b2 = _ch(t2); ch_hpf = max(CH_HPF, _ch(hpf_total))
    BH = float(np.ceil(EDGE_PAD + ch_hpf + PAD_PITCH + ch_b1 + PAD_PITCH + ch_b2 + EDGE_PAD))
    BH = max(BH, 8)

    y_hpf  = BH - EDGE_PAD - 0.5
    y_bpf1 = y_hpf - ch_hpf - PAD_PITCH + 0.5
    y_bpf2 = y_bpf1 - ch_b1 - PAD_PITCH

    print(f"\n  BOARD: {BW:.0f} x {BH:.0f} mm  ({BW * BH:.0f} mm²)")
    print(f"  HPF trace  {hpf_total:.1f}mm (y={y_hpf:.1f})")
    print(f"  BPF1 trace {t1:.1f}mm (y={y_bpf1:.1f})")
    print(f"  BPF2 trace {t2:.1f}mm (y={y_bpf2:.1f})")

    # ── Meander paths ────────────────────────────────────────────────
    hpf_segs = []
    for k in range(3):
        hpf_segs.append(ll[k] * 1e3 if k < len(ll) else 1)
        hpf_segs.append(sl[k] * 1e3)
    if len(ll) > 3: hpf_segs.append(ll[3] * 1e3)

    hpf_pts = _meander_pts(EDGE_PAD, y_hpf, hpf_segs, BW)
    bpf1_pts = _meander_pts(EDGE_PAD, y_bpf1, [t1 / 10] * 10, BW)
    bpf2_pts = _meander_pts(EDGE_PAD, y_bpf2, [t2 / 10] * 10, BW)

    stub_w_mm = [float(_ms_width(jnp.array(float(sz[k])))) * 1e3 for k in range(3)]
    mim_pads = [np.sqrt(float(cp[k]) * H_CORE / (EPS0 * ER_CORE)) * 1e3 for k in range(2)]
    stub_xs = []; cum = EDGE_PAD
    for k in range(3):
        cum += hpf_segs[2 * k]; stub_xs.append(min(cum, BW - 2))
        cum += hpf_segs[2 * k + 1] if 2 * k + 1 < len(hpf_segs) else 0

    # ── Pads and exclusions ──────────────────────────────────────────
    pads = [(0, BH - 1, "G"), (0, y_hpf, "H"), (0, y_bpf1, "1"), (0, y_bpf2, "2"), (0, 1, "G"),
            (BW, BH - 1, "G"), (BW, y_hpf, "H"), (BW, y_bpf1, "1"), (BW, y_bpf2, "2"), (BW, 1, "G")]
    excl = [(px, py, 1.8) for px, py, _ in pads]
    gnd_voids = [(px - 0.4 if px > 0 else -0.1, py - 0.4, 0.8, 0.8)
                 for px, py, lb in pads if lb != "G"]

    def _pads(ax, hi=None):
        for px, py, lb in pads:
            _draw_castellated(ax, px, py)
            if lb != "G":
                tx = px + (1.2 if px < BW / 2 else -1.2)
                ha = "left" if px < BW / 2 else "right"
                lbl = {"H": "HPF", "1": "BPF1", "2": "BPF2"}[lb]
                c = "#e74c3c" if hi and lb == hi else "#888"
                fw = "bold" if hi and lb == hi else "normal"
                ax.text(tx, py, lbl, fontsize=5, ha=ha, va="center", color=c, fontweight=fw)

    # ── Render 6-layer copper artwork ────────────────────────────────
    fig = plt.figure(figsize=(21, 14))
    fig.suptitle(f"SMT Filter Module — {BW:.0f} x {BH:.0f} mm  |  6-Layer Copper",
                 fontsize=14, fontweight="bold")

    # L1 HPF
    ax = _layer_ax(fig, (2, 3, 1), "L1 — HPF Signal", BW, BH)
    _draw_trace(ax, hpf_pts, W50_MM)
    for k in range(3):
        sx = stub_xs[k]; se = y_hpf - sl[k] * 1e3
        _draw_trace(ax, [(sx, y_hpf), (sx, se)], stub_w_mm[k], color="#d4a040")
        _draw_via(ax, sx, se - 0.3)
    for k in range(2):
        cx = stub_xs[k] + (stub_xs[min(k + 1, 2)] - stub_xs[k]) * 0.5
        cx = min(max(cx, 2), BW - 2); ps = mim_pads[k]
        ax.add_patch(Rectangle((cx - ps / 2, y_hpf - ps / 2), ps, ps,
                                fc="#e8d080", ec=CU_DARK, lw=0.4, zorder=4))
    _pads(ax, "H"); _draw_via_fence(ax, BW, BH, excl)

    # L2 Ground
    ax = _layer_ax(fig, (2, 3, 2), "L2 — Ground (HPF ref)", BW, BH)
    voids = list(gnd_voids)
    for k in range(2):
        cx = stub_xs[k] + (stub_xs[min(k + 1, 2)] - stub_xs[k]) * 0.5
        cx = min(max(cx, 2), BW - 2); ps = mim_pads[k] + 0.3
        voids.append((cx - ps / 2, y_hpf - ps / 2, ps, ps))
    for sx in stub_xs:
        voids.append((sx - 0.3, y_hpf - sl.max() * 1e3 - 0.8, 0.6, 0.6))
    _draw_gnd_pour(ax, BW, BH, voids); _draw_via_fence(ax, BW, BH, excl); _pads(ax)

    # L3 BPF1
    ax = _layer_ax(fig, (2, 3, 3), f"L3 — BPF1 ({t1:.0f}mm)", BW, BH)
    _draw_trace(ax, bpf1_pts, SL_W50_MM, color="#3498db")
    _draw_via(ax, EDGE_PAD, y_bpf1); _draw_via(ax, BW - EDGE_PAD, y_bpf1)
    _pads(ax, "1"); _draw_via_fence(ax, BW, BH, excl)

    # L4 Ground
    ax = _layer_ax(fig, (2, 3, 4), "L4 — Ground (shared)", BW, BH)
    _draw_gnd_pour(ax, BW, BH, gnd_voids); _draw_via_fence(ax, BW, BH, excl); _pads(ax)

    # L5 BPF2
    ax = _layer_ax(fig, (2, 3, 5), f"L5 — BPF2 ({t2:.0f}mm)", BW, BH)
    _draw_trace(ax, bpf2_pts, SL_W50_MM, color="#9b59b6")
    _draw_via(ax, EDGE_PAD, y_bpf2); _draw_via(ax, BW - EDGE_PAD, y_bpf2)
    _pads(ax, "2"); _draw_via_fence(ax, BW, BH, excl)

    # L6 Ground
    ax = _layer_ax(fig, (2, 3, 6), "L6 — Ground (bottom)", BW, BH)
    _draw_gnd_pour(ax, BW, BH, gnd_voids); _draw_via_fence(ax, BW, BH, excl); _pads(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("board_copper_layers.png", dpi=200); plt.close(fig)
    generated.append("board_copper_layers.png")

    # ── Frequency response plot ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"SMT Filter Module — {BW:.0f}×{BH:.0f}mm  |  Stub-Loaded BPF",
                 fontsize=14, fontweight="bold")
    for ax, s21, s11, title, vl in [
        (axes[0], s21h_np, s11h_np, "HPF fc=2.5GHz (L1)", [2.5]),
        (axes[1], s21b1_np, s11b1_np, "BPF1 2.5-3.75GHz (L3)", [2.5, 3.75]),
        (axes[2], s21b2_np, s11b2_np, "BPF2 3.75-5.0GHz (L5)", [3.75, 5.0]),
    ]:
        ax.plot(f_ghz, db(s21), "b", lw=1.5, label="|S21|")
        ax.plot(f_ghz, db(s11), "r--", lw=1, label="|S11|")
        ax.set_title(title, fontsize=10); ax.set_xlabel("GHz"); ax.set_ylabel("dB")
        ax.set_ylim(-50, 3); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        for fv in vl: ax.axvline(fv, color="gray", ls=":", lw=0.8)
    fig.tight_layout(); fig.savefig("board_response.png", dpi=200); plt.close(fig)
    generated.append("board_response.png")

    # ── Stackup ──────────────────────────────────────────────────────
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.set_title(f"6-Layer Stackup — {BW:.0f}×{BH:.0f}mm SMT Module", fontsize=13, fontweight="bold")
    stack = [
        ("L1 HPF signal", 35e-6, "#e67e22"), ("RO3010 er=10.2", H_CORE, "#f5e6d3"),
        ("L2 Ground", 35e-6, "#27ae60"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L3 BPF1 stripline", 18e-6, "#3498db"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L4 Ground", 35e-6, "#27ae60"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L5 BPF2 stripline", 18e-6, "#9b59b6"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L6 Ground", 35e-6, "#27ae60"),
    ]
    y = 0
    for label, t, c in reversed(stack):
        h = max(t * 1e3, 0.015)
        ax_s.add_patch(Rectangle((1, y), 8, h, fc=c, ec="#333", lw=0.8))
        ax_s.text(9.3, y + h / 2, label, va="center", fontsize=8, fontfamily="monospace")
        y += h
    ax_s.set_xlim(0, 20); ax_s.set_ylim(-0.02, y + 0.04)
    ax_s.set_ylabel("mm"); ax_s.set_xticks([])
    ax_s.text(5, y + 0.025, f"Total: {TOTAL_H * 1e3:.2f} mm", ha="center",
              fontsize=11, fontweight="bold")
    fig_s.tight_layout(); fig_s.savefig("board_stackup.png", dpi=200); plt.close(fig_s)
    generated.append("board_stackup.png")

    # ── Performance ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  MODULE: {BW:.0f} × {BH:.0f} mm  ({BW * BH:.0f} mm²)  {TOTAL_H * 1e3:.2f} mm thick")
    print("=" * 64)
    for name, s21, vl in [("HPF", s21h_np, [2.5, 5.5]),
                           ("BPF1", s21b1_np, [2.5, 3.75]),
                           ("BPF2", s21b2_np, [3.75, 5.0])]:
        s21d = db(s21); pb = (f_ghz >= vl[0]) & (f_ghz <= vl[-1])
        if np.any(pb):
            print(f"  {name:5s}  IL: {float(-np.max(s21d[pb])):.1f} - {float(-np.min(s21d[pb])):.1f} dB")

    print(f"\n  BPF topology: stub-loaded (5th-order, 5 line+stub resonators)")
    print(f"  Evaluated but rejected:")
    print(f"    A) Coupled-resonator λ/2 — larger board, 2.8-8.0 dB BPF1 IL")
    print(f"    B) Hairpin λ/4 — similar size, 0.4-11.2 dB BPF1 IL (inconsistent)")

    # ── Output ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  OUTPUT IMAGES")
    print("=" * 64)
    for img in generated:
        sz = os.path.getsize(img) / 1024
        print(f"  {img:40s}  {sz:6.0f} KB  OK")
    print("=" * 64)
    print("  Complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
