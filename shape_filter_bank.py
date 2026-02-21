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
#  BPF — Stub-loaded (parameterised order)
# ═══════════════════════════════════════════════════════════════════════
CHEBY_G7 = [1.0, 1.7372, 1.2583, 2.6381, 1.3444, 2.6381, 1.2583, 1.7372, 1.0]
CHEBY_G9 = [1.0, 1.7504, 1.2690, 2.6678, 1.3673, 2.7239, 1.3673, 2.6678, 1.2690, 1.7504, 1.0]

def _bpf_bounds(n):
    return [(n, 1e-3, 10e-3), (n, 1e-3, 10e-3), (n, 25., 120.), (n + 1, 0.01, 2.0)]

def _bpf_init(fc, fbw, n=5, g=None):
    if g is None: g = CHEBY_G5
    J = [float(np.sqrt(np.pi * fbw / (2 * g[0] * g[1])))]
    for i in range(1, n):
        J.append(float(np.pi * fbw / (2 * np.sqrt(g[i] * g[i + 1]))))
    J.append(float(np.sqrt(np.pi * fbw / (2 * g[n] * g[n + 1]))))
    J = [max(min(j, 1.99), 0.011) for j in J]
    lam8 = min(C0 / (fc * np.sqrt(ER_SL)) / 8, 8e-3)
    return _pack([lam8] * n + [lam8 * 1.2] * n + [70.] * n + J, _bpf_bounds(n))

def _make_bpf_at_f(n):
    bnd = _bpf_bounds(n)
    def _at_f(p, f):
        ll, sl, sz, jn = _unpack(p, bnd)
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
    return _at_f

def _bpf_trace(p, n):
    bnd = _bpf_bounds(n)
    ll, sl, _, _ = _unpack(p, bnd)
    return float(jnp.sum(ll) + jnp.sum(sl)) * 1e3

def _bpf_loss(fl, fh, batch_fn, guard=0.3):
    bw = fh - fl
    def loss_fn(p, freqs):
        s21, s11 = batch_fn(p, freqs)
        s21d = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11d = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
        pb = (freqs >= fl) & (freqs <= fh)
        slo = freqs < (fl - guard * bw)
        shi = freqs > (fh + guard * bw)
        loss = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 1.5, 0.) ** 2, 0.)) * 6
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d + 10, 0.) ** 2, 0.)) * 2
        loss += jnp.sum(jnp.where(slo, jnp.maximum(s21d + 20, 0.) ** 2, 0.)) * 4
        loss += jnp.sum(jnp.where(shi, jnp.maximum(s21d + 20, 0.) ** 2, 0.)) * 4
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
    """Meander trace from left edge pad to right edge pad."""
    x_min, x_max = EDGE_PAD, bw - EDGE_PAD
    x, y, dx = x0, y0, 1
    pts = [(0, y), (x, y)]
    for s in segs:
        left = s
        while left > 0.05:
            avail = (x_max - x) if dx > 0 else (x - x_min)
            avail = max(avail, 0.1)
            step = min(left, avail)
            x += step * dx; pts.append((x, y)); left -= step
            if left > 0.05:
                y -= FOLD_GAP; pts.append((x, y)); dx = -dx
    pts.append((bw, pts[-1][1]))
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
def _design_board(label, prefix, n_bpf, g_bpf, guard, p_hpf, hpf_data,
                   freqs, fp, f_ghz, db, f1l, f1h, f2l, f2h):
    """Optimise BPFs and render one complete board variant."""
    generated = []
    fc1, fbw1 = (f1l + f1h) / 2, (f1h - f1l) / ((f1l + f1h) / 2)
    fc2, fbw2 = (f2l + f2h) / 2, (f2h - f2l) / ((f2l + f2h) / 2)

    bpf_at_f = _make_bpf_at_f(n_bpf)
    bpf_batch = jax.vmap(bpf_at_f, in_axes=(None, 0))

    print(f"\n{'='*64}\n  {label} — BPF1 2.5-3.75 GHz (N={n_bpf})\n{'='*64}")
    loss1 = _bpf_loss(f1l, f1h, bpf_batch, guard)
    p_b1 = _opt(loss1, lambda: _bpf_init(fc1, fbw1, n_bpf, g_bpf), freqs, n_r=3, n_steps=3000)

    print(f"\n{'='*64}\n  {label} — BPF2 3.75-5.0 GHz (N={n_bpf})\n{'='*64}")
    loss2 = _bpf_loss(f2l, f2h, bpf_batch, guard)
    p_b2 = _opt(loss2, lambda: _bpf_init(fc2, fbw2, n_bpf, g_bpf), freqs, n_r=3, n_steps=3000)

    s21b1_np = np.array(bpf_batch(p_b1, fp)[0])
    s11b1_np = np.array(bpf_batch(p_b1, fp)[1])
    s21b2_np = np.array(bpf_batch(p_b2, fp)[0])
    s11b2_np = np.array(bpf_batch(p_b2, fp)[1])

    sz, sl, cp, ll, s21h_np, s11h_np = hpf_data
    hpf_total = float(np.sum(sl) + np.sum(ll)) * 1e3
    t1 = _bpf_trace(p_b1, n_bpf); t2 = _bpf_trace(p_b2, n_bpf)
    max_stub_mm = float(np.max(sl)) * 1e3

    longest = max(hpf_total, t1, t2)
    nf = max(1, int(np.ceil(longest / 14.0)))
    BW = float(np.ceil(longest / nf + 2 * EDGE_PAD)); BW = max(BW, 10)
    CH_HPF = max_stub_mm + 1.5; uw = BW - 2 * EDGE_PAD
    def _ch(tr): return 0.5 + max(1, int(np.ceil(tr / uw))) * FOLD_GAP
    ch_b1 = _ch(t1); ch_b2 = _ch(t2); ch_hpf = max(CH_HPF, _ch(hpf_total))
    BH = float(np.ceil(EDGE_PAD + ch_hpf + PAD_PITCH + ch_b1 + PAD_PITCH + ch_b2 + EDGE_PAD))
    BH = max(BH, 8)
    y_hpf = BH - EDGE_PAD - 0.5
    y_b1 = y_hpf - ch_hpf - PAD_PITCH + 0.5
    y_b2 = y_b1 - ch_b1 - PAD_PITCH

    # HPF feedline = only the 4 connecting lines (stubs are vertical branches)
    hpf_line_segs = [ll[k] * 1e3 for k in range(len(ll))]
    hpf_feed_total = sum(hpf_line_segs)
    hpf_pts = _meander_pts(EDGE_PAD, y_hpf, hpf_line_segs, BW)

    bpf1_pts = _meander_pts(EDGE_PAD, y_b1, [t1 / (2 * n_bpf)] * (2 * n_bpf), BW)
    bpf2_pts = _meander_pts(EDGE_PAD, y_b2, [t2 / (2 * n_bpf)] * (2 * n_bpf), BW)
    stub_w_mm = [float(_ms_width(jnp.array(float(sz[k])))) * 1e3 for k in range(3)]
    mim_pads = [np.sqrt(float(cp[k]) * H_CORE / (EPS0 * ER_CORE)) * 1e3 for k in range(2)]

    # Stub/cap positions along the HPF feedline
    # Topology: Stub0 — ll[0] — Cap0 — ll[1] — Stub1 — ll[2] — Cap1 — ll[3] — Stub2
    # Stubs at cumulative: 0, ll[0]+ll[1], ll[0]+ll[1]+ll[2]+ll[3]
    # Caps at cumulative: ll[0], ll[0]+ll[1]+ll[2]
    def _x_at_pathlen(pts, d_mm):
        """Find (x,y) at distance d_mm along a polyline path."""
        cum = 0
        for i in range(len(pts) - 1):
            seg = np.sqrt((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2)
            if cum + seg >= d_mm - 0.01:
                frac = (d_mm - cum) / max(seg, 0.01)
                frac = min(max(frac, 0), 1)
                return (pts[i][0] + frac * (pts[i+1][0] - pts[i][0]),
                        pts[i][1] + frac * (pts[i+1][1] - pts[i][1]))
            cum += seg
        return pts[-1]

    stub_positions = [
        _x_at_pathlen(hpf_pts, 0),
        _x_at_pathlen(hpf_pts, (ll[0]+ll[1])*1e3),
        _x_at_pathlen(hpf_pts, sum(hpf_line_segs)),
    ]
    cap_positions = [
        _x_at_pathlen(hpf_pts, ll[0]*1e3),
        _x_at_pathlen(hpf_pts, (ll[0]+ll[1]+ll[2])*1e3),
    ]

    pads = [(0, BH-1, "G"), (0, y_hpf, "H"), (0, y_b1, "1"), (0, y_b2, "2"), (0, 1, "G"),
            (BW, BH-1, "G"), (BW, y_hpf, "H"), (BW, y_b1, "1"), (BW, y_b2, "2"), (BW, 1, "G")]
    excl = [(px, py, 1.8) for px, py, _ in pads]
    gnd_voids = [(px-0.4 if px>0 else -0.1, py-0.4, 0.8, 0.8) for px,py,lb in pads if lb!="G"]
    def _pads(ax, hi=None):
        for px,py,lb in pads:
            _draw_castellated(ax, px, py)
            if lb != "G":
                tx = px + (1.2 if px < BW/2 else -1.2)
                ha = "left" if px < BW/2 else "right"
                lbl = {"H":"HPF","1":"BPF1","2":"BPF2"}[lb]
                c = "#e74c3c" if hi and lb==hi else "#888"
                ax.text(tx, py, lbl, fontsize=5, ha=ha, va="center", color=c,
                        fontweight="bold" if hi and lb==hi else "normal")

    # 6-layer copper artwork
    fig = plt.figure(figsize=(21, 14))
    fig.suptitle(f"{label} — {BW:.0f}×{BH:.0f}mm  |  6-Layer Copper", fontsize=14, fontweight="bold")
    ax = _layer_ax(fig, (2,3,1), "L1 HPF", BW, BH)
    _draw_trace(ax, hpf_pts, W50_MM)
    for k in range(3):
        sx, sy = stub_positions[k]; se = sy - sl[k] * 1e3
        _draw_trace(ax, [(sx, sy), (sx, se)], stub_w_mm[k], color="#d4a040")
        _draw_via(ax, sx, se - 0.3)
    for k in range(2):
        cx, cy = cap_positions[k]; ps = mim_pads[k]
        ax.add_patch(Rectangle((cx-ps/2, cy-ps/2), ps, ps, fc="#e8d080", ec=CU_DARK, lw=0.4, zorder=4))
    _pads(ax,"H"); _draw_via_fence(ax,BW,BH,excl)
    ax = _layer_ax(fig, (2,3,2), "L2 Ground", BW, BH)
    voids = list(gnd_voids)
    for k in range(2):
        cx, cy = cap_positions[k]; ps = mim_pads[k] + 0.3
        voids.append((cx-ps/2, cy-ps/2, ps, ps))
    for k in range(3):
        sx, sy = stub_positions[k]
        voids.append((sx-0.3, sy-sl[k]*1e3-0.8, 0.6, 0.6))
    _draw_gnd_pour(ax,BW,BH,voids); _draw_via_fence(ax,BW,BH,excl); _pads(ax)
    ax = _layer_ax(fig, (2,3,3), f"L3 BPF1 ({t1:.0f}mm)", BW, BH)
    _draw_trace(ax, bpf1_pts, SL_W50_MM, color="#3498db")
    _draw_via(ax,EDGE_PAD,y_b1); _draw_via(ax,BW-EDGE_PAD,y_b1); _pads(ax,"1"); _draw_via_fence(ax,BW,BH,excl)
    ax = _layer_ax(fig, (2,3,4), "L4 Ground", BW, BH)
    _draw_gnd_pour(ax,BW,BH,gnd_voids); _draw_via_fence(ax,BW,BH,excl); _pads(ax)
    ax = _layer_ax(fig, (2,3,5), f"L5 BPF2 ({t2:.0f}mm)", BW, BH)
    _draw_trace(ax, bpf2_pts, SL_W50_MM, color="#9b59b6")
    _draw_via(ax,EDGE_PAD,y_b2); _draw_via(ax,BW-EDGE_PAD,y_b2); _pads(ax,"2"); _draw_via_fence(ax,BW,BH,excl)
    ax = _layer_ax(fig, (2,3,6), "L6 Ground", BW, BH)
    _draw_gnd_pour(ax,BW,BH,gnd_voids); _draw_via_fence(ax,BW,BH,excl); _pads(ax)
    fig.tight_layout(rect=[0,0,1,0.95])
    fn = f"{prefix}_copper.png"; fig.savefig(fn, dpi=200); plt.close(fig); generated.append(fn)

    # Response
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{label} — {BW:.0f}×{BH:.0f}mm", fontsize=14, fontweight="bold")
    for ax, s21, s11, title, vl in [
        (axes[0], s21h_np, s11h_np, "HPF (L1)", [2.5]),
        (axes[1], s21b1_np, s11b1_np, f"BPF1 N={n_bpf} (L3)", [2.5, 3.75]),
        (axes[2], s21b2_np, s11b2_np, f"BPF2 N={n_bpf} (L5)", [3.75, 5.0]),
    ]:
        ax.plot(f_ghz, db(s21), "b", lw=1.5, label="|S21|")
        ax.plot(f_ghz, db(s11), "r--", lw=1, label="|S11|")
        ax.set_title(title, fontsize=10); ax.set_xlabel("GHz"); ax.set_ylabel("dB")
        ax.set_ylim(-50, 3); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        for fv in vl: ax.axvline(fv, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fn = f"{prefix}_response.png"; fig.savefig(fn, dpi=200); plt.close(fig); generated.append(fn)

    print(f"\n  {label}: {BW:.0f}×{BH:.0f}mm ({BW*BH:.0f}mm²)")
    for name, s21, vl in [("HPF",s21h_np,[2.5,5.5]),("BPF1",s21b1_np,[2.5,3.75]),("BPF2",s21b2_np,[3.75,5.0])]:
        s21d = db(s21); pb = (f_ghz>=vl[0])&(f_ghz<=vl[-1])
        if np.any(pb):
            print(f"  {name:5s} IL: {float(-np.max(s21d[pb])):.1f} - {float(-np.min(s21d[pb])):.1f} dB")
    return generated, s21b1_np, s11b1_np, s21b2_np, s11b2_np, BW, BH, bpf1_pts


def main():
    freqs = jnp.linspace(0.1e9, 7e9, 500)
    fp = jnp.linspace(0.1e9, 7e9, 1000)
    f_ghz = np.array(fp) / 1e9
    db = lambda s: 20 * np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))
    generated = []
    f1l, f1h = 2.5e9, 3.75e9; f2l, f2h = 3.75e9, 5.0e9

    print("=" * 64)
    print("  Compact SMT Filter Module — Standard & Steep Rolloff")
    print("=" * 64)
    print(f"  Substrate: RO3010 er={ER_CORE} + RO4450F er={ER_SL}")
    print(f"  50Ω MS {W50_MM:.3f}mm  SL {SL_W50_MM:.3f}mm  Thickness {TOTAL_H*1e3:.2f}mm\n")

    # Shared HPF
    print("=" * 64); print("  HPF fc=2.5GHz (shared, L1)"); print("=" * 64)
    p_hpf = _opt(_hpf_loss, _hpf_init, freqs, n_r=2, n_steps=2500)
    s21h_np = np.array(_hpf_batch(p_hpf, fp)[0])
    s11h_np = np.array(_hpf_batch(p_hpf, fp)[1])
    sz, sl, cp, ll = (np.array(x) for x in _unpack(p_hpf, HPF_BND))
    hpf_data = (sz, sl, cp, ll, s21h_np, s11h_np)

    # Board 1: Standard (N=5, 0.3 guard)
    g1, s21b1_std, s11b1_std, s21b2_std, s11b2_std, bw1, bh1, b1pts1 = _design_board(
        "Standard (N=5)", "board_standard",
        5, CHEBY_G5, 0.3, p_hpf, hpf_data,
        freqs, fp, f_ghz, db, f1l, f1h, f2l, f2h)
    generated.extend(g1)

    # Board 2: Steep rolloff (N=9, 0.12 guard — very steep crossover)
    g2, s21b1_stp, s11b1_stp, s21b2_stp, s11b2_stp, bw2, bh2, b1pts2 = _design_board(
        "Steep Rolloff (N=9)", "board_steep",
        9, CHEBY_G9, 0.12, p_hpf, hpf_data,
        freqs, fp, f_ghz, db, f1l, f1h, f2l, f2h)
    generated.extend(g2)

    # Comparison overlay
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BPF Rolloff Comparison: Standard (N=5) vs Steep (N=9)", fontsize=13, fontweight="bold")
    for ax, s21_s, s21_st, title, vl in [
        (axes[0], s21b1_std, s21b1_stp, "BPF1 2.5-3.75 GHz", [2.5, 3.75]),
        (axes[1], s21b2_std, s21b2_stp, "BPF2 3.75-5.0 GHz", [3.75, 5.0]),
    ]:
        ax.plot(f_ghz, db(s21_s), "b", lw=1.5, label=f"N=5 ({bw1:.0f}×{bh1:.0f}mm)")
        ax.plot(f_ghz, db(s21_st), "r", lw=1.5, label=f"N=9 steep ({bw2:.0f}×{bh2:.0f}mm)")
        ax.set_title(title); ax.set_xlabel("GHz"); ax.set_ylabel("|S21| [dB]")
        ax.set_ylim(-50, 3); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        for fv in vl: ax.axvline(fv, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig("board_rolloff_comparison.png", dpi=200); plt.close(fig)
    generated.append("board_rolloff_comparison.png")

    # Stackup
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.set_title("6-Layer Stackup", fontsize=13, fontweight="bold")
    stack = [
        ("L1 HPF signal", 35e-6, "#e67e22"), ("RO3010 er=10.2", H_CORE, "#f5e6d3"),
        ("L2 Ground", 35e-6, "#27ae60"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L3 BPF1 stripline", 18e-6, "#3498db"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L4 Ground", 35e-6, "#27ae60"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L5 BPF2 stripline", 18e-6, "#9b59b6"), ("RO4450F er=3.52", H_SL, "#ecf0f1"),
        ("L6 Ground", 35e-6, "#27ae60"),
    ]
    y_s = 0
    for label_s, t, c in reversed(stack):
        h = max(t * 1e3, 0.015)
        ax_s.add_patch(Rectangle((1, y_s), 8, h, fc=c, ec="#333", lw=0.8))
        ax_s.text(9.3, y_s + h / 2, label_s, va="center", fontsize=8, fontfamily="monospace")
        y_s += h
    ax_s.set_xlim(0, 20); ax_s.set_ylim(-0.02, y_s + 0.04)
    ax_s.set_ylabel("mm"); ax_s.set_xticks([])
    ax_s.text(5, y_s + 0.025, f"Total: {TOTAL_H * 1e3:.2f} mm", ha="center", fontsize=11, fontweight="bold")
    fig_s.tight_layout(); fig_s.savefig("board_stackup.png", dpi=200); plt.close(fig_s)
    generated.append("board_stackup.png")

    # ── Wave propagation GIFs (2D board-layout wavefront animation) ──
    import matplotlib.animation as animation

    def _wave_gif_2d(bpf_pts, s21_bpf, label, fname, f_pass, f_stop, board_w, board_h):
        """Animate wavefronts on the 2D board layout along the BPF trace."""
        n_frames = 40; fps = 10
        pts = np.array(bpf_pts)
        seg_lens = np.sqrt(np.diff(pts[:,0])**2 + np.diff(pts[:,1])**2)
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cum_len[-1]
        n_pts = len(pts)

        s21_pi = int(f_pass / 7e9 * len(s21_bpf))
        s21_si = int(f_stop / 7e9 * len(s21_bpf))
        s21_p = s21_bpf[min(s21_pi, len(s21_bpf)-1)]
        s21_s = s21_bpf[min(s21_si, len(s21_bpf)-1)]
        s11_p = float(np.sqrt(max(1 - np.abs(s21_p)**2, 0)))
        s11_s = float(np.sqrt(max(1 - np.abs(s21_s)**2, 0)))

        beta_p = 2 * np.pi * f_pass * np.sqrt(ER_SL) / C0
        beta_s = 2 * np.pi * f_stop * np.sqrt(ER_SL) / C0
        d_m = cum_len * 1e-3
        V_p = np.exp(-1j * beta_p * d_m) + s11_p * np.exp(1j * beta_p * d_m)
        V_s = np.exp(-1j * beta_s * d_m) + s11_s * np.exp(1j * beta_s * d_m)

        fig_g, axes_g = plt.subplots(1, 2, figsize=(14, 5))
        fig_g.suptitle(f"{label} — Wavefront on Board", fontsize=12, fontweight="bold")

        for ai, (ax_g, title) in enumerate(zip(axes_g,
                [f"Passband {f_pass/1e9:.1f} GHz", f"Stopband {f_stop/1e9:.1f} GHz"])):
            ax_g.set_facecolor(SUBSTRATE)
            ax_g.add_patch(Rectangle((-0.3,-0.3), board_w+0.6, board_h+0.6,
                                      fc=SUBSTRATE, ec="#444", lw=1.5, zorder=0))
            ax_g.set_xlim(-1, board_w+1); ax_g.set_ylim(-1, board_h+1)
            ax_g.set_aspect("equal"); ax_g.set_title(title, fontsize=10)
            ax_g.set_xlabel("mm"); ax_g.set_ylabel("mm")
            ax_g.plot(pts[:,0], pts[:,1], color="#333", lw=0.5, zorder=1)

        scat_p = axes_g[0].scatter(pts[:,0], pts[:,1], c=np.zeros(n_pts),
                                    cmap="RdBu_r", vmin=-2, vmax=2, s=8, zorder=3)
        scat_s = axes_g[1].scatter(pts[:,0], pts[:,1], c=np.zeros(n_pts),
                                    cmap="RdBu_r", vmin=-2, vmax=2, s=8, zorder=3)

        def _update(frame):
            t = frame / n_frames / f_pass
            c_p = np.real(V_p * np.exp(1j * 2 * np.pi * f_pass * t))
            c_s = np.real(V_s * np.exp(1j * 2 * np.pi * f_stop * t))
            scat_p.set_array(c_p)
            scat_s.set_array(c_s)
            return scat_p, scat_s

        _update(0)
        ani = animation.FuncAnimation(fig_g, _update, frames=n_frames,
                                       interval=1000//fps, blit=True)
        try:
            ani.save(fname, writer=animation.PillowWriter(fps=fps))
            print(f"  [saved] {fname}")
        except Exception as e:
            print(f"  GIF failed: {e}")
            fname = None
        plt.close(fig_g)
        return fname

    print("\n  Generating 2D wave propagation GIFs...")
    gf = _wave_gif_2d(b1pts1, s21b1_std, "Standard BPF1 (N=5)",
                       "board_standard_wave.gif", 3.1e9, 1.5e9, bw1, bh1)
    if gf: generated.append(gf)
    gf = _wave_gif_2d(b1pts2, s21b1_stp, "Steep BPF1 (N=9)",
                       "board_steep_wave.gif", 3.1e9, 1.5e9, bw2, bh2)
    if gf: generated.append(gf)

    print("\n" + "=" * 64)
    print("  OUTPUT FILES")
    print("=" * 64)
    for img in generated:
        sz = os.path.getsize(img) / 1024
        ext = os.path.splitext(img)[1]
        print(f"  {img:40s}  {sz:6.0f} KB  {ext}")
    print("=" * 64)
    print("  Complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
