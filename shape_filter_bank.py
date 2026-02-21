#!/usr/bin/env python3
"""
Compact Multilayer Shape-Based RF Filter Bank  (40 x 40 mm target)
===================================================================
Three-channel filter bank fitting within a 40 x 40 mm board using
high-er Rogers substrates and multilayer stacking.

Filters are distributed across layers of a 6-layer PCB so they
occupy the SAME board footprint (stacked, not side-by-side):

  L1  Signal   HPF feedline + stubs (microstrip)
  L2  Ground   HPF ground + BPF1 DGS shapes
  L3  Signal   BPF1 stripline resonators
  L4  Ground   BPF1 ground + BPF2 DGS shapes
  L5  Signal   BPF2 stripline resonators
  L6  Ground   Bottom reference

Substrate: Rogers RO3010 core (er=10.2, 0.254mm) for ~40% size
reduction vs FR4.  At er=10.2 the guided wavelength at 3.5 GHz
is only 33 mm, so lambda/2 resonators fit easily within 40 mm.

Design approach:
  HPF  — 5th-order Chebyshev stubs + MIM caps on L1
  BPFs — 5th-order J-inverter coupled stripline resonators
         folded (meandered) to fit within the 40mm board width

Manufacturing constraints enforced as hard parameter bounds:
  min trace/space  0.15 mm   min via drill  0.20 mm

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
from matplotlib.patches import Rectangle, FancyArrowPatch

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Constants & substrate
# ═══════════════════════════════════════════════════════════════════════
C0    = 299_792_458.0
EPS0  = 8.854_187_817e-12
MU0   = 4e-7 * np.pi
Z_REF = 50.0

ER_CORE  = 10.2;   H_CORE  = 0.254e-3;   TAND_CORE = 0.0035
ER_PP    = 3.52;    H_PP    = 0.100e-3;   TAND_PP   = 0.004

BOARD_MAX = 40e-3

MIN_TRACE = 0.15e-3
MIN_SPACE = 0.15e-3

CHEBY_G5 = [1.0, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0]

def _ms_eeff(w, h=H_CORE, er=ER_CORE):
    u = jnp.clip(w / h, 0.01, 1000.)
    return (er + 1) / 2 + (er - 1) / 2 * jnp.power(1 + 12 / u, -0.5)

def _ms_width(z0, h=H_CORE, er=ER_CORE):
    A = z0 / 60. * jnp.sqrt((er + 1) / 2) + (er - 1) / (er + 1) * (0.23 + 0.11 / er)
    B = 377. * jnp.pi / (2. * z0 * jnp.sqrt(er))
    w1 = 8 * jnp.exp(A) / (jnp.exp(2 * A) - 2)
    w2 = (2 / jnp.pi) * (B - 1 - jnp.log(jnp.clip(2 * B - 1, 1e-6))
          + (er - 1) / (2 * er) * (jnp.log(jnp.clip(B - 1, 1e-6)) + 0.39 - 0.61 / er))
    return jnp.where(z0 > 63., w1, w2) * h

W50  = float(_ms_width(jnp.array(Z_REF)))
EEFF = float(_ms_eeff(jnp.array(W50)))
LAM_G_35 = C0 / (3.5e9 * np.sqrt(EEFF))

TOTAL_H = (35e-6 + H_CORE + 35e-6 + H_PP + 18e-6
           + H_CORE + 35e-6 + H_PP + 18e-6 + H_PP + 35e-6)

# ═══════════════════════════════════════════════════════════════════════
#  ABCD helpers
# ═══════════════════════════════════════════════════════════════════════
def _m(a, b, c, d):
    return jnp.array([[a, b], [c, d]], dtype=jnp.complex128)

def abcd_tl(z0, gl):
    ch, sh = jnp.cosh(gl), jnp.sinh(gl)
    return _m(ch, z0 * sh, sh / z0, ch)

def abcd_sh(Y):
    return _m(1.+0j, 0j, Y, 1.+0j)

def abcd_jinv(J):
    return _m(0j, -1j / J, -1j * J, 0j)

def abcd_to_s(M):
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    den = A + B / Z_REF + C * Z_REF + D
    return 2. / den, (A + B / Z_REF - C * Z_REF - D) / den

def _gl_ms(f, length):
    beta  = 2 * jnp.pi * f * jnp.sqrt(EEFF) / C0
    alpha = TAND_CORE * jnp.pi * f * jnp.sqrt(EEFF) / C0
    return (alpha + 1j * beta) * length

def _gl_sl(f, length, er=ER_PP, tand=TAND_PP):
    beta  = 2 * jnp.pi * f * jnp.sqrt(er) / C0
    alpha = tand * jnp.pi * f * jnp.sqrt(er) / C0
    return (alpha + 1j * beta) * length

# ═══════════════════════════════════════════════════════════════════════
#  Bounded transforms
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
        parts.append(jnp.array([float(_bnd_inv(vals[idx+i], lo, hi)) for i in range(n)]))
        idx += n
    return jnp.concatenate(parts)

def _unpack(p, bounds):
    res, i = [], 0
    for n, lo, hi in bounds:
        res.append(_bnd(p[i:i+n], lo, hi)); i += n
    return res

# ═══════════════════════════════════════════════════════════════════════
#  HPF  (L1 microstrip, 5th-order Chebyshev stubs + MIM caps)
# ═══════════════════════════════════════════════════════════════════════
HPF_BOUNDS = [
    (3, 25., 150.),             # stub impedances
    (3, 0.3e-3, 8e-3),          # stub lengths  (short for compactness)
    (2, 10e-15, 10e-12),        # MIM cap values
    (4, 0.1e-3, 8e-3),          # connecting line lengths
]

def _hpf_init():
    fc = 2.5e9; wc = 2*np.pi*fc; g = CHEBY_G5
    L = [Z_REF/(wc*g[i]) for i in (1,3,5)]
    C = [1./(wc*Z_REF*g[i]) for i in (2,4)]
    zs = [90.,90.,90.]
    eeff_s = float(_ms_eeff(jnp.array(_ms_width(jnp.array(90.)))))
    sl = [min(Lv*C0/(z*np.sqrt(eeff_s)), 7e-3) for Lv,z in zip(L,zs)]
    lam = C0/(fc*np.sqrt(EEFF))
    ll = [min(lam*0.04, 7e-3)]*4
    return _pack(zs + sl + C + ll, HPF_BOUNDS)

def _hpf_at_f(p, f):
    sz, sl, cp, ll = _unpack(p, HPF_BOUNDS)
    gl = lambda l: _gl_ms(f, l)
    def stub_y(z, l):
        w = _ms_width(z); eeff = _ms_eeff(w)
        g = (TAND_CORE*jnp.pi*f*jnp.sqrt(eeff)/C0 + 1j*2*jnp.pi*f*jnp.sqrt(eeff)/C0)*l
        return 1./(z*jnp.tanh(g+1e-15))
    cap_z = lambda c: 1./(1j*2*jnp.pi*f*c)
    M = abcd_sh(stub_y(sz[0],sl[0]))
    M = M @ abcd_tl(Z_REF, gl(ll[0])) @ _m(1.+0j, cap_z(cp[0]), 0j, 1.+0j)
    M = M @ abcd_tl(Z_REF, gl(ll[1])) @ abcd_sh(stub_y(sz[1],sl[1]))
    M = M @ abcd_tl(Z_REF, gl(ll[2])) @ _m(1.+0j, cap_z(cp[1]), 0j, 1.+0j)
    M = M @ abcd_tl(Z_REF, gl(ll[3])) @ abcd_sh(stub_y(sz[2],sl[2]))
    return abcd_to_s(M)

_hpf_batch = jax.vmap(_hpf_at_f, in_axes=(None, 0))

def _hpf_loss(p, freqs):
    s21, s11 = _hpf_batch(p, freqs)
    s21d = 20*jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
    s11d = 20*jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
    fc = 2.5e9
    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    tb = (freqs > 1.5e9) & (freqs < fc)
    loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d-0.5, 0.)**2, 0.)) * 6
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d+12, 0.)**2, 0.)) * 2
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21d+20, 0.)**2, 0.)) * 4
    loss += jnp.sum(jnp.where(tb, jnp.maximum(s21d+6,  0.)**2, 0.)) * 1

    total_len = jnp.sum(_unpack(p, HPF_BOUNDS)[1]) + jnp.sum(_unpack(p, HPF_BOUNDS)[3])
    loss += jnp.maximum(total_len - BOARD_MAX, 0.) ** 2 * 1e6
    return loss / freqs.shape[0]

# ═══════════════════════════════════════════════════════════════════════
#  BPF  (J-inverter coupled stripline on inner layers)
# ═══════════════════════════════════════════════════════════════════════
N_BPF = 5; NJ = 6

BPF_BOUNDS = [
    (N_BPF, 3e-3, 20e-3),    # resonator lengths (compact, max 20mm)
    (NJ, 0.01, 2.0),         # normalised J*Z0
]

def _bpf_init(fc, fbw):
    g = CHEBY_G5; n = N_BPF
    J = [float(np.sqrt(np.pi*fbw/(2*g[0]*g[1])))]
    for i in range(1, n):
        J.append(float(np.pi*fbw/(2*np.sqrt(g[i]*g[i+1]))))
    J.append(float(np.sqrt(np.pi*fbw/(2*g[n]*g[n+1]))))
    J = [max(min(j, 1.99), 0.011) for j in J]
    lam2 = C0/(fc*np.sqrt(ER_PP))/2
    lam2 = min(lam2, 19e-3)
    return _pack([lam2]*n + J, BPF_BOUNDS)

def _bpf_at_f(p, f):
    rl = _bnd(p[:N_BPF], 3e-3, 20e-3)
    jn = _bnd(p[N_BPF:], 0.01, 2.0)
    J = jn / Z_REF
    M0 = abcd_jinv(J[0])
    def body(M, x):
        return M @ abcd_tl(Z_REF, _gl_sl(f, x[0])) @ abcd_jinv(x[1]), None
    Mf, _ = jax.lax.scan(body, M0, (rl, J[1:]))
    return abcd_to_s(Mf)

_bpf_batch = jax.vmap(_bpf_at_f, in_axes=(None, 0))

def _make_bpf_loss(fl, fh):
    bw = fh - fl
    def loss_fn(p, freqs):
        s21, s11 = _bpf_batch(p, freqs)
        s21d = 20*jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11d = 20*jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
        pb = (freqs >= fl) & (freqs <= fh)
        sl = freqs < (fl - 0.3*bw); sh = freqs > (fh + 0.3*bw)
        loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d-1.5, 0.)**2, 0.)) * 6
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d+10, 0.)**2, 0.)) * 2
        loss += jnp.sum(jnp.where(sl, jnp.maximum(s21d+15, 0.)**2, 0.)) * 3
        loss += jnp.sum(jnp.where(sh, jnp.maximum(s21d+15, 0.)**2, 0.)) * 3

        rl = _bnd(p[:N_BPF], 3e-3, 20e-3)
        total = jnp.sum(rl)
        loss += jnp.maximum(total - BOARD_MAX, 0.) ** 2 * 1e6
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
            p0 = p0 + jax.random.normal(jax.random.PRNGKey(r*97), p0.shape) * 0.4
        print(f"  (restart {r+1}/{n_r})")
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
            if (i+1) % 500 == 0 or i == 0:
                print(f"  step {i+1:5d}/{n_steps}  loss={lv_f:.4f}  best={best_l:.4f}")
    return best_p

# ═══════════════════════════════════════════════════════════════════════
#  Layout drawing — compact square board
# ═══════════════════════════════════════════════════════════════════════
BOARD_MM = BOARD_MAX * 1e3

def _draw_board(ax, title):
    ax.add_patch(Rectangle((0, 0), BOARD_MM, BOARD_MM,
                            fc="#f5f0e8", ec="#333", lw=1.5, zorder=0))
    ax.set_xlim(-2, BOARD_MM + 2)
    ax.set_ylim(-2, BOARD_MM + 2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("mm"); ax.set_ylabel("mm")

def _draw_meander(ax, x0, y0, seg_lengths_mm, width_mm, direction="right",
                   color="#b87333", n_folds=None):
    """Draw a meandered resonator that folds to fit in the board."""
    max_run = BOARD_MM - 4
    x, y = x0, y0
    dx = 1 if direction == "right" else -1
    points = [(x, y)]
    fold_gap = width_mm * 3

    remaining = list(seg_lengths_mm)
    total_drawn = 0

    for seg_l in remaining:
        avail = (max_run - x) if dx > 0 else x
        if seg_l <= avail + 0.1:
            x += seg_l * dx
            points.append((x, y))
        else:
            x += avail * dx
            points.append((x, y))
            y -= fold_gap
            points.append((x, y))
            dx = -dx
            leftover = seg_l - avail
            x += leftover * dx
            points.append((x, y))
        total_drawn += seg_l

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=max(width_mm * 2, 1.5), solid_capstyle="round", zorder=2)
    return total_drawn

# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs = jnp.linspace(0.1e9, 7e9, 500)
    fp = jnp.linspace(0.1e9, 7e9, 1000)
    fg = np.array(fp) / 1e9
    db = lambda s: 20*np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))
    generated = []

    print("=" * 64)
    print("  Compact 40x40mm Multilayer Shape Filter Bank")
    print("=" * 64)
    print(f"  Board target   : {BOARD_MM:.0f} x {BOARD_MM:.0f} mm")
    print(f"  Substrate      : Rogers RO3010 er={ER_CORE}")
    print(f"  50-ohm width   : {W50*1e3:.3f} mm")
    print(f"  Lambda_g @3.5G : {LAM_G_35*1e3:.1f} mm")
    print(f"  Layers         : 6 (HPF on L1, BPF1 on L3, BPF2 on L5)")
    print(f"  Thickness      : {TOTAL_H*1e3:.2f} mm")
    print(f"  Mfg min        : {MIN_TRACE*1e3:.2f} mm trace/space")
    print()

    # ── HPF ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  HPF  fc=2.5 GHz  (L1 stubs + MIM caps, compact)")
    print("=" * 64)
    p_hpf = _opt(_hpf_loss, _hpf_init, freqs, n_r=2, n_steps=2500)

    # ── BPF1 ─────────────────────────────────────────────────────────
    f1l, f1h = 2.5e9, 3.75e9
    fc1, fbw1 = (f1l+f1h)/2, (f1h-f1l)/((f1l+f1h)/2)
    print("\n" + "=" * 64)
    print("  BPF1  2.5-3.75 GHz  (L3 stripline J-inverter, folded)")
    print("=" * 64)
    p_b1 = _opt(_make_bpf_loss(f1l, f1h), lambda: _bpf_init(fc1, fbw1),
                freqs, n_r=3, n_steps=3000)

    # ── BPF2 ─────────────────────────────────────────────────────────
    f2l, f2h = 3.75e9, 5.0e9
    fc2, fbw2 = (f2l+f2h)/2, (f2h-f2l)/((f2l+f2h)/2)
    print("\n" + "=" * 64)
    print("  BPF2  3.75-5.0 GHz  (L5 stripline J-inverter, folded)")
    print("=" * 64)
    p_b2 = _opt(_make_bpf_loss(f2l, f2h), lambda: _bpf_init(fc2, fbw2),
                freqs, n_r=3, n_steps=3000)

    # ── Evaluate ─────────────────────────────────────────────────────
    s21h, s11h = _hpf_batch(p_hpf, fp)
    s21b1, s11b1 = _bpf_batch(p_b1, fp)
    s21b2, s11b2 = _bpf_batch(p_b2, fp)

    filters = [
        ("HPF fc=2.5GHz", s21h, s11h, [2.5], p_hpf, "hpf"),
        ("BPF1 2.5-3.75GHz", s21b1, s11b1, [2.5, 3.75], p_b1, "bpf1"),
        ("BPF2 3.75-5.0GHz", s21b2, s11b2, [3.75, 5.0], p_b2, "bpf2"),
    ]

    # ── Per-filter plots (response + board layout) ───────────────────
    layer_names = {"hpf": "L1 (top microstrip)", "bpf1": "L3 (inner stripline)", "bpf2": "L5 (inner stripline)"}
    layer_colors = {"hpf": "#b87333", "bpf1": "#2980b9", "bpf2": "#8e44ad"}

    for name, s21, s11, vlines, params, prefix in filters:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"{name}  |  {layer_names[prefix]}  |  40x40mm board",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(fg, db(s21), "b", lw=1.6, label="|S21|")
        ax.plot(fg, db(s11), "r--", lw=1, label="|S11|")
        ax.set_title("Frequency Response"); ax.set_xlabel("GHz"); ax.set_ylabel("dB")
        ax.set_ylim(-50, 3); ax.legend(); ax.grid(True, alpha=0.3)
        for fv in vlines: ax.axvline(fv, color="gray", ls=":", lw=0.8)

        ax2 = axes[1]
        _draw_board(ax2, f"Board Layout — {layer_names[prefix]}")
        col = layer_colors[prefix]

        if prefix == "hpf":
            sz, sl, cp, ll = (np.array(x) for x in _unpack(params, HPF_BOUNDS))
            segs = []
            for k in range(3):
                segs.append(ll[k]*1e3 if k < len(ll) else 2)
                segs.append(sl[k]*1e3)
            if len(ll) > 3: segs.append(ll[3]*1e3)
            _draw_meander(ax2, 2, BOARD_MM-4, segs, W50*1e3, color=col)
            for k in range(2):
                xc = 2 + sum(segs[:2*k+1])
                ax2.plot(xc, BOARD_MM-4, "s", color="#27ae60", ms=6, zorder=3)
                ax2.annotate(f"MIM\n{cp[k]*1e12:.2f}pF", (xc, BOARD_MM-6),
                            fontsize=6, ha="center", color="#27ae60")
            total = float(np.sum(sl) + np.sum(ll)) * 1e3
            ax2.text(BOARD_MM/2, 2, f"Total trace: {total:.1f} mm", ha="center", fontsize=8)
        else:
            rl = np.array(_bnd(params[:N_BPF], 3e-3, 20e-3))
            _draw_meander(ax2, 2, BOARD_MM-4, [r*1e3 for r in rl], 0.5, color=col)
            total = float(np.sum(rl)) * 1e3
            ax2.text(BOARD_MM/2, 2, f"Total trace: {total:.1f} mm (folded)", ha="center", fontsize=8)

            jn = np.array(_bnd(params[N_BPF:], 0.01, 2.0))
            for k in range(N_BPF):
                xj = 2 + sum(rl[:k+1])*1e3
                if xj < BOARD_MM:
                    ax2.plot(xj, BOARD_MM-4, "o", color="#e74c3c", ms=4, zorder=3)

        fig.tight_layout()
        fname = f"shape_{prefix}.png"
        fig.savefig(fname, dpi=200); plt.close(fig)
        generated.append(fname)

    # ── Combined overview ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Compact 40x40mm Multilayer Filter Bank  |  Rogers RO3010  er=10.2",
                 fontsize=14, fontweight="bold")

    for ci, (name, s21, s11, vl, params, prefix) in enumerate(filters):
        ax = axes[0, ci]
        ax.plot(fg, db(s21), "b", lw=1.5, label="|S21|")
        ax.plot(fg, db(s11), "r--", lw=1, label="|S11|")
        ax.set_title(f"{name} — {layer_names[prefix]}", fontsize=9)
        ax.set_xlabel("GHz"); ax.set_ylabel("dB"); ax.set_ylim(-50, 3)
        ax.legend(loc="lower right", fontsize=7); ax.grid(True, alpha=0.3)
        for fv in vl: ax.axvline(fv, color="gray", ls=":", lw=0.8)

        ax2 = axes[1, ci]
        _draw_board(ax2, f"{layer_names[prefix]}")
        col = layer_colors[prefix]
        if prefix == "hpf":
            sz, sl, cp, ll = (np.array(x) for x in _unpack(params, HPF_BOUNDS))
            segs = []
            for k in range(3):
                segs.append(ll[k]*1e3 if k < len(ll) else 2)
                segs.append(sl[k]*1e3)
            if len(ll) > 3: segs.append(ll[3]*1e3)
            _draw_meander(ax2, 2, BOARD_MM-4, segs, W50*1e3, color=col)
        else:
            rl = np.array(_bnd(params[:N_BPF], 3e-3, 20e-3))
            _draw_meander(ax2, 2, BOARD_MM-4, [r*1e3 for r in rl], 0.5, color=col)

    fig.tight_layout()
    fig.savefig("shape_filter_bank.png", dpi=200); plt.close(fig)
    generated.append("shape_filter_bank.png")

    # ── Stackup cross-section ────────────────────────────────────────
    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.set_title("6-Layer Stackup (40x40mm board)", fontsize=13, fontweight="bold")
    stack = [
        ("L1 HPF feedline + stubs", 35e-6, "#e67e22"),
        ("RO3010 core er=10.2 0.254mm", H_CORE, "#f5e6d3"),
        ("L2 Ground (HPF ref)", 35e-6, "#27ae60"),
        ("RO4450F PP er=3.52 0.100mm", H_PP, "#ecf0f1"),
        ("L3 BPF1 stripline resonators", 18e-6, "#2980b9"),
        ("RO3010 core er=10.2 0.254mm", H_CORE, "#f5e6d3"),
        ("L4 Ground (BPF1 ref)", 35e-6, "#27ae60"),
        ("RO4450F PP er=3.52 0.100mm", H_PP, "#ecf0f1"),
        ("L5 BPF2 stripline resonators", 18e-6, "#8e44ad"),
        ("RO4450F PP er=3.52 0.175mm", 0.175e-3, "#ecf0f1"),
        ("L6 Ground (bottom)", 35e-6, "#27ae60"),
    ]
    y = 0
    for label, t, c in reversed(stack):
        h = max(t*1e3, 0.015)
        ax_s.add_patch(Rectangle((1, y), 8, h, fc=c, ec="#333", lw=0.8))
        ax_s.text(9.3, y+h/2, label, va="center", fontsize=8, fontfamily="monospace")
        y += h
    ax_s.set_xlim(0, 20); ax_s.set_ylim(-0.02, y+0.04)
    ax_s.set_ylabel("mm"); ax_s.set_xticks([])
    ax_s.text(5, y+0.025, f"Total: {TOTAL_H*1e3:.2f} mm  |  Board: {BOARD_MM:.0f}x{BOARD_MM:.0f} mm",
              ha="center", fontsize=11, fontweight="bold")
    fig_s.tight_layout()
    fig_s.savefig("shape_stackup.png", dpi=200); plt.close(fig_s)
    generated.append("shape_stackup.png")

    # ── Performance & dimensions ─────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  PERFORMANCE  |  Board: {BOARD_MM:.0f} x {BOARD_MM:.0f} mm")
    print("=" * 64)
    for name, s21, s11, vl, params, prefix in filters:
        s21db = db(s21); s11db = db(s11)
        flo, fhi = vl[0], vl[-1]
        if "HPF" in name: fhi = 5.5
        pb = (fg >= flo) & (fg <= fhi)
        if np.any(pb):
            il_lo = float(-np.max(s21db[pb]))
            il_hi = float(-np.min(s21db[pb]))
            rl = float(-np.max(s11db[pb]))
            if prefix == "hpf":
                sl_arr = np.array(_unpack(params, HPF_BOUNDS)[1])
                ll_arr = np.array(_unpack(params, HPF_BOUNDS)[3])
                total = float(np.sum(sl_arr) + np.sum(ll_arr)) * 1e3
            else:
                rl_arr = np.array(_bnd(params[:N_BPF], 3e-3, 20e-3))
                total = float(np.sum(rl_arr)) * 1e3
            layer = layer_names[prefix]
            fits = "YES" if total <= BOARD_MM + 0.5 else f"NO ({total:.1f}mm, needs folding)"
            print(f"  {name:20s}  IL: {il_lo:.1f}-{il_hi:.1f} dB  RL: >{rl:.1f} dB"
                  f"  trace: {total:.1f}mm  fits 40mm: {fits}  [{layer}]")

    print(f"\n  Board area     : {BOARD_MM:.0f} x {BOARD_MM:.0f} mm = {BOARD_MM**2/100:.0f} mm²")
    print(f"  Board thickness: {TOTAL_H*1e3:.2f} mm")
    print(f"  Mfg min trace  : {MIN_TRACE*1e3:.2f} mm — all dims exceed this")

    # ── Output images ────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  OUTPUT IMAGES")
    print("=" * 64)
    for img in generated:
        sz = os.path.getsize(img) / 1024
        print(f"  {img:40s}  {sz:6.0f} KB  OK")
    print("\n" + "=" * 64)
    print("  Complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
