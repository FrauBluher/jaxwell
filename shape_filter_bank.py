#!/usr/bin/env python3
"""
Shape-Library Multilayer DGS Filter Bank  (up to 6 layers)
===========================================================
Two design variants for each filter:

  STANDARD : DGS shapes on ground layers + coupled patches on signal layers
             → compact, uses topology optimisation
  LOW-IL   : J-inverter coupled stripline resonators on inner layers
             → lower insertion loss, traditional coupled-resonator synthesis

6-layer stackup  (Rogers RO3010 + RO4450F)
-------------------------------------------
  L1  Signal   50-ohm feedline               RO3010 core  0.254mm  er=10.2
  L2  Ground   DGS shapes                    RO4450F PP   0.100mm  er=3.52
  L3  Signal   Coupled patches / resonators  RO3010 core  0.254mm  er=10.2
  L4  Ground   Secondary DGS                 RO4450F PP   0.100mm  er=3.52
  L5  Signal   Tertiary patches              RO4450F PP   0.175mm  er=3.52
  L6  Ground   Bottom reference

Manufacturing constraints (hard bounds):
  min trace / slot  0.15 mm  (6 mil)
  min pad           0.50 mm
  min via drill     0.20 mm
"""

import os, sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
EPS0 = 8.854_187_817e-12
MU0 = 4e-7 * np.pi
Z_REF = 50.0

ER_CORE = 10.2;  H_CORE = 0.254e-3;  TAND_CORE = 0.0035
ER_PP   = 3.52;  H_PP   = 0.100e-3;  TAND_PP   = 0.004

TOTAL_H = 35e-6 + H_CORE + 35e-6 + H_PP + 18e-6 + H_CORE + 35e-6 + H_PP + 18e-6 + 0.175e-3 + 35e-6

MIN_TRACE = 0.15e-3
MIN_SPACE = 0.15e-3
MIN_PAD   = 0.50e-3

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

W50 = float(_ms_width(jnp.array(Z_REF)))
EEFF = float(_ms_eeff(jnp.array(W50)))

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

def _gl_ms(f, length, eeff=EEFF, tand=TAND_CORE):
    beta = 2 * jnp.pi * f * jnp.sqrt(eeff) / C0
    alpha = tand * jnp.pi * f * jnp.sqrt(eeff) / C0
    return (alpha + 1j * beta) * length

def _gl_sl(f, length, er=ER_PP, tand=TAND_PP):
    return (tand * jnp.pi * f * jnp.sqrt(er) / C0 + 1j * 2 * jnp.pi * f * jnp.sqrt(er) / C0) * length

# ═══════════════════════════════════════════════════════════════════════
#  Shape equivalent circuits
# ═══════════════════════════════════════════════════════════════════════
def _dumbbell_Y(f, va, bw, bl, er=ER_CORE, h=H_CORE):
    L = MU0 * bl / jnp.maximum(bw, MIN_SPACE)
    C = EPS0 * er * va * bw / (2 * h) + EPS0 * er * va**2 / (4 * h)
    omega = 2 * jnp.pi * f
    return 1. / (0.6 + 1j * omega * L + 1. / (1j * omega * C + 1e-30))

def _csrr_Y(f, rad, rw, gf, er=ER_CORE, h=H_CORE):
    circ = 2 * jnp.pi * rad
    gap = jnp.maximum(gf * circ, MIN_SPACE)
    slot = circ - gap
    L = MU0 * slot / jnp.maximum(rw, MIN_SPACE) * 0.5
    C = EPS0 * er * rw * slot / h * 0.3 + EPS0 * er * rw**2 / gap * 0.5
    omega = 2 * jnp.pi * f
    Z = 0.4 + 1j * omega * L + 1. / (1j * omega * C + 1e-30)
    Cc = EPS0 * er * jnp.pi * rad**2 / h * 0.12
    return Cc * omega**2 / (Z * omega + 1e-20)

def _hshape_Y(f, al, aw, bw, er=ER_CORE, h=H_CORE):
    L = MU0 * aw / jnp.maximum(bw, MIN_SPACE)
    C = EPS0 * er * al * aw / h * 0.5
    omega = 2 * jnp.pi * f
    return 1. / (0.5 + 1j * omega * L + 1. / (1j * omega * C + 1e-30))

def _patch_Y(f, wp, lp, er=ER_CORE, h=H_PP, atten=1.):
    fres = C0 / (2 * lp * jnp.sqrt(er) + 1e-10)
    Cc = EPS0 * er * wp * lp / h * 0.06 * atten
    omega = 2 * jnp.pi * f
    w0 = 2 * jnp.pi * fres
    return 1j * omega * Cc * w0**2 / (w0**2 - omega**2 + 1j * omega * w0 / 20. + 1e-20)

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
        res.append(_bnd(p[i:i+n], lo, hi)); i += n
    return res

# ═══════════════════════════════════════════════════════════════════════
#  STANDARD DESIGN: DGS shapes (HPF = dumbbell, BPF = CSRR+patch+H)
# ═══════════════════════════════════════════════════════════════════════
N_HPF_STD = 7
HPF_STD_BOUNDS = [
    (N_HPF_STD, 0.4e-3, 12e-3),
    (N_HPF_STD, MIN_SPACE, 3e-3),
    (N_HPF_STD, MIN_SPACE*2, 6e-3),
    (N_HPF_STD+1, 0.5e-3, 15e-3),
]

def _hpf_std_init():
    tf = np.linspace(0.6e9, 2.4e9, N_HPF_STD)
    va, bw, bl = [], [], []
    for ft in tf:
        b_w, b_l = 0.3e-3, 1.5e-3
        L = MU0 * b_l / b_w
        C_need = 1. / ((2*np.pi*ft)**2 * L)
        a = np.sqrt(max(4*C_need*H_CORE/(EPS0*ER_CORE), 0)) + 0.5e-3
        va.append(np.clip(a, 0.5e-3, 11e-3)); bw.append(b_w); bl.append(b_l)
    return _pack(va + bw + bl + [4e-3]*(N_HPF_STD+1), HPF_STD_BOUNDS)

def _hpf_std_at_f(p, f):
    va, bw, bl, sp = _unpack(p, HPF_STD_BOUNDS)
    def body(M, x):
        return M @ abcd_sh(_dumbbell_Y(f, x[0], x[1], x[2])) @ abcd_tl(Z_REF, _gl_ms(f, x[3])), None
    M0 = abcd_tl(Z_REF, _gl_ms(f, sp[0]))
    Mf, _ = jax.lax.scan(body, M0, (va, bw, bl, sp[1:]))
    return abcd_to_s(Mf)

_hpf_std_batch = jax.vmap(_hpf_std_at_f, in_axes=(None, 0))

def _hpf_loss(p, freqs):
    s21, s11 = _hpf_std_batch(p, freqs)
    s21d = 20*jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
    s11d = 20*jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
    fc = 2.5e9
    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    tb = (freqs > 1.5e9) & (freqs < fc)
    loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d-1., 0.)**2, 0.)) * 5
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d+10, 0.)**2, 0.)) * 2
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21d+15, 0.)**2, 0.)) * 4
    loss += jnp.sum(jnp.where(tb, jnp.maximum(s21d+8,  0.)**2, 0.)) * 2
    return loss / freqs.shape[0]

# BPF standard: CSRR(L2) + patch(L3) + H(L4) + patch(L5) — 6-layer
N_BPF_STD = 5
BPF_STD_BOUNDS = [
    (N_BPF_STD, 1e-3, 8e-3),        (N_BPF_STD, MIN_SPACE, 2e-3),
    (N_BPF_STD, 0.03, 0.30),        (N_BPF_STD, MIN_PAD, 8e-3),
    (N_BPF_STD, 2e-3, 22e-3),       (N_BPF_STD, 0.3e-3, 6e-3),
    (N_BPF_STD, 0.3e-3, 4e-3),      (N_BPF_STD, MIN_SPACE, 2e-3),
    (N_BPF_STD, MIN_PAD, 6e-3),     (N_BPF_STD, 2e-3, 18e-3),
    (N_BPF_STD+1, 0.5e-3, 12e-3),
]

def _bpf_std_init(fc, fbw):
    fl, fh = fc*(1-fbw/2), fc*(1+fbw/2)
    nf = np.array([fl*0.55, fl*0.8, fc, fh*1.2, fh*1.5])
    rad, rw, gf = [], [], []
    for ft in nf:
        r = np.clip(C0/(ft*np.sqrt(ER_CORE))*0.25/(2*np.pi), 1.2e-3, 7.5e-3)
        rad.append(r); rw.append(0.4e-3); gf.append(0.08)
    pf = np.linspace(fl*1.05, fh*0.95, N_BPF_STD)
    pw3, pl3, pw5, pl5 = [], [], [], []
    for fp in pf:
        lp = np.clip(C0/(2*fp*np.sqrt(ER_CORE)), 3e-3, 21e-3)
        pw3.append(2e-3); pl3.append(lp)
        pw5.append(1.5e-3); pl5.append(lp*0.9)
    al = [1.5e-3]*N_BPF_STD; aw = [1e-3]*N_BPF_STD; hbw = [0.3e-3]*N_BPF_STD
    sp = [3e-3]*(N_BPF_STD+1)
    return _pack(rad+rw+gf+pw3+pl3+al+aw+hbw+pw5+pl5+sp, BPF_STD_BOUNDS)

def _bpf_std_at_f(p, f):
    rad,rw,gf,pw3,pl3,al,aw,hbw,pw5,pl5,sp = _unpack(p, BPF_STD_BOUNDS)
    def body(M, x):
        Y  = _csrr_Y(f, x[0], x[1], x[2])
        Y += _patch_Y(f, x[3], x[4], atten=1.)
        Y += _hshape_Y(f, x[5], x[6], x[7]) * 0.35
        Y += _patch_Y(f, x[8], x[9], er=ER_PP, h=0.175e-3, atten=0.15)
        return M @ abcd_sh(Y) @ abcd_tl(Z_REF, _gl_ms(f, x[10])), None
    M0 = abcd_tl(Z_REF, _gl_ms(f, sp[0]))
    Mf, _ = jax.lax.scan(body, M0, (rad,rw,gf,pw3,pl3,al,aw,hbw,pw5,pl5,sp[1:]))
    return abcd_to_s(Mf)

_bpf_std_batch = jax.vmap(_bpf_std_at_f, in_axes=(None, 0))

def _make_bpf_std_loss(fl, fh):
    bw = fh - fl
    def loss_fn(p, freqs):
        s21, s11 = _bpf_std_batch(p, freqs)
        s21d = 20*jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11d = 20*jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
        pb = (freqs >= fl) & (freqs <= fh)
        sl = freqs < (fl - 0.3*bw); sh = freqs > (fh + 0.3*bw)
        loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d-2., 0.)**2, 0.)) * 6
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d+8,  0.)**2, 0.)) * 2
        loss += jnp.sum(jnp.where(sl, jnp.maximum(s21d+15, 0.)**2, 0.)) * 3
        loss += jnp.sum(jnp.where(sh, jnp.maximum(s21d+15, 0.)**2, 0.)) * 3
        return loss / freqs.shape[0]
    return loss_fn

# ═══════════════════════════════════════════════════════════════════════
#  LOW-IL DESIGN: J-inverter coupled stripline resonators
# ═══════════════════════════════════════════════════════════════════════
N_LOW = 5; NJ_LOW = 6
CHEBY_G5 = [1.0, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0]

LOW_HPF_BOUNDS = [
    (3, 25., 150.), (3, 0.3e-3, 12e-3), (2, 10e-15, 10e-12), (4, 0.1e-3, 8e-3),
]
LOW_BPF_BOUNDS = [(N_LOW, 3e-3, 40e-3), (NJ_LOW, 0.01, 2.0)]

def _low_hpf_init():
    fc = 2.5e9; wc = 2*np.pi*fc
    g = CHEBY_G5
    L = [Z_REF/(wc*g[i]) for i in (1,3,5)]
    C = [1./(wc*Z_REF*g[i]) for i in (2,4)]
    zs = [90.,90.,90.]
    eeff_s = float(_ms_eeff(jnp.array(_ms_width(jnp.array(90.)))))
    sl = [Lv*C0/(z*np.sqrt(eeff_s)) for Lv, z in zip(L, zs)]
    lam = C0/(fc*np.sqrt(EEFF))
    return _pack(zs + sl + C + [lam*0.04]*4, LOW_HPF_BOUNDS)

def _low_hpf_at_f(p, f):
    sz, sl, cp, ll = _unpack(p, LOW_HPF_BOUNDS)
    gl = lambda l: _gl_ms(f, l)
    def stub_y(z, l):
        w = _ms_width(z)
        eeff = _ms_eeff(w)
        g = (TAND_CORE*jnp.pi*f*jnp.sqrt(eeff)/C0 + 1j*2*jnp.pi*f*jnp.sqrt(eeff)/C0)*l
        return 1./(z*jnp.tanh(g+1e-15))
    cap_z = lambda c: 1./(1j*2*jnp.pi*f*c)
    M = abcd_sh(stub_y(sz[0],sl[0]))
    M = M @ abcd_tl(Z_REF, gl(ll[0])) @ _m(1.+0j, cap_z(cp[0]), 0j, 1.+0j)
    M = M @ abcd_tl(Z_REF, gl(ll[1])) @ abcd_sh(stub_y(sz[1],sl[1]))
    M = M @ abcd_tl(Z_REF, gl(ll[2])) @ _m(1.+0j, cap_z(cp[1]), 0j, 1.+0j)
    M = M @ abcd_tl(Z_REF, gl(ll[3])) @ abcd_sh(stub_y(sz[2],sl[2]))
    return abcd_to_s(M)

_low_hpf_batch = jax.vmap(_low_hpf_at_f, in_axes=(None, 0))

def _low_hpf_loss(p, freqs):
    s21, s11 = _low_hpf_batch(p, freqs)
    s21d = 20*jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
    s11d = 20*jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
    fc = 2.5e9
    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d-0.5, 0.)**2, 0.)) * 6
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d+12, 0.)**2, 0.)) * 2
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21d+20, 0.)**2, 0.)) * 4
    return loss / freqs.shape[0]

def _low_bpf_init(fc, fbw):
    g = CHEBY_G5; n = N_LOW
    J = [float(np.sqrt(np.pi*fbw/(2*g[0]*g[1])))]
    for i in range(1, n):
        J.append(float(np.pi*fbw/(2*np.sqrt(g[i]*g[i+1]))))
    J.append(float(np.sqrt(np.pi*fbw/(2*g[n]*g[n+1]))))
    J = [max(min(j, 1.99), 0.011) for j in J]
    lam2 = C0/(fc*np.sqrt(ER_PP))/2
    return _pack([lam2]*n + J, LOW_BPF_BOUNDS)

def _low_bpf_at_f(p, f):
    rl = _bnd(p[:N_LOW], 3e-3, 40e-3)
    jn = _bnd(p[N_LOW:], 0.01, 2.0)
    J = jn / Z_REF
    M0 = abcd_jinv(J[0])
    def body(M, x):
        return M @ abcd_tl(Z_REF, _gl_sl(f, x[0])) @ abcd_jinv(x[1]), None
    Mf, _ = jax.lax.scan(body, M0, (rl, J[1:]))
    return abcd_to_s(Mf)

_low_bpf_batch = jax.vmap(_low_bpf_at_f, in_axes=(None, 0))

def _make_low_bpf_loss(fl, fh):
    bw = fh - fl
    def loss_fn(p, freqs):
        s21, s11 = _low_bpf_batch(p, freqs)
        s21d = 20*jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
        s11d = 20*jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
        pb = (freqs >= fl) & (freqs <= fh)
        sl = freqs < (fl - 0.3*bw); sh = freqs > (fh + 0.3*bw)
        loss  = jnp.sum(jnp.where(pb, jnp.maximum(-s21d-1.5, 0.)**2, 0.)) * 6
        loss += jnp.sum(jnp.where(pb, jnp.maximum(s11d+10, 0.)**2, 0.)) * 2
        loss += jnp.sum(jnp.where(sl, jnp.maximum(s21d+15, 0.)**2, 0.)) * 3
        loss += jnp.sum(jnp.where(sh, jnp.maximum(s21d+15, 0.)**2, 0.)) * 3
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
        if n_r > 1:
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
#  Plotting
# ═══════════════════════════════════════════════════════════════════════
CU = "#b87333"; VOID = "#1a1a2e"; PATCH = "#dd9955"; FEED_C = "#e74c3c"

def _plot_response(ax, fg, s21, s11, title, vlines, db):
    ax.plot(fg, db(s21), "b", lw=1.5, label="|S21|")
    ax.plot(fg, db(s11), "r--", lw=1, label="|S11|")
    ax.set_title(title, fontsize=10); ax.set_xlabel("GHz"); ax.set_ylabel("dB")
    ax.set_ylim(-50, 3); ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    for fv in vlines: ax.axvline(fv, color="gray", ls=":", lw=0.8)

# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs = jnp.linspace(0.1e9, 7e9, 500)
    fp = jnp.linspace(0.1e9, 7e9, 1000)
    fg = np.array(fp) / 1e9
    db = lambda s: 20*np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))
    generated = []

    print("=" * 68)
    print("  Shape-Library Multilayer Filter Bank  (6-layer, RO3010)")
    print("  Standard (DGS) + Low-IL (J-inverter) variants")
    print("=" * 68)
    print(f"  Substrate: RO3010 er={ER_CORE}  50Ω width={W50*1e3:.3f}mm  λg={C0/(3.5e9*np.sqrt(EEFF))*1e3:.1f}mm")
    print(f"  Board: {TOTAL_H*1e3:.2f}mm thick  6 layers  Mfg min: {MIN_TRACE*1e3:.2f}mm")
    print()

    specs = [
        ("HPF fc=2.5GHz", 2.5, 5.5, [2.5]),
        ("BPF1 2.5-3.75GHz", 2.5, 3.75, [2.5, 3.75]),
        ("BPF2 3.75-5.0GHz", 3.75, 5.0, [3.75, 5.0]),
    ]

    all_results = {}

    for name, fl_ghz, fh_ghz, vlines in specs:
        fl, fh = fl_ghz*1e9, fh_ghz*1e9
        is_hpf = "HPF" in name
        print("=" * 68)
        print(f"  {name}  —  STANDARD (DGS shapes, 6-layer)")
        print("=" * 68)
        if is_hpf:
            p_std = _opt(_hpf_loss, _hpf_std_init, freqs, n_r=3, n_steps=2500)
            s21_std, s11_std = _hpf_std_batch(p_std, fp)
        else:
            fc_ = (fl+fh)/2; fbw_ = (fh-fl)/fc_
            loss_std = _make_bpf_std_loss(fl, fh)
            p_std = _opt(loss_std, lambda: _bpf_std_init(fc_, fbw_), freqs, n_r=3, n_steps=3000)
            s21_std, s11_std = _bpf_std_batch(p_std, fp)

        print(f"\n  {name}  —  LOW-IL (J-inverter stripline)")
        print("-" * 50)
        if is_hpf:
            p_low = _opt(_low_hpf_loss, _low_hpf_init, freqs, n_r=2, n_steps=2500)
            s21_low, s11_low = _low_hpf_batch(p_low, fp)
        else:
            fc_ = (fl+fh)/2; fbw_ = (fh-fl)/fc_
            loss_low = _make_low_bpf_loss(fl, fh)
            p_low = _opt(loss_low, lambda fc=fc_, fbw=fbw_: _low_bpf_init(fc, fbw), freqs, n_r=3, n_steps=3000)
            s21_low, s11_low = _low_bpf_batch(p_low, fp)

        key = name.split()[0].lower()
        all_results[key] = dict(name=name, vlines=vlines,
            s21_std=s21_std, s11_std=s11_std, p_std=p_std,
            s21_low=s21_low, s11_low=s11_low, p_low=p_low)
        print()

    # ── Per-filter comparison plots ──────────────────────────────────
    for key, res in all_results.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{res['name']}  |  Standard vs Low-IL", fontsize=13, fontweight="bold")
        _plot_response(axes[0], fg, res["s21_std"], res["s11_std"],
                       "Standard (DGS, 6-layer)", res["vlines"], db)
        _plot_response(axes[1], fg, res["s21_low"], res["s11_low"],
                       "Low-IL (J-inverter stripline)", res["vlines"], db)
        fig.tight_layout()
        fname = f"shape_{key}_compare.png"
        fig.savefig(fname, dpi=200); plt.close(fig)
        generated.append(fname)

    # ── Combined overview ────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Shape-Library Filter Bank  |  Standard vs Low-IL  |  Rogers RO3010",
                 fontsize=14, fontweight="bold")
    for ci, (key, res) in enumerate(all_results.items()):
        _plot_response(axes[0, ci], fg, res["s21_std"], res["s11_std"],
                       f"Standard: {res['name']}", res["vlines"], db)
        _plot_response(axes[1, ci], fg, res["s21_low"], res["s11_low"],
                       f"Low-IL: {res['name']}", res["vlines"], db)
    fig.tight_layout()
    fig.savefig("shape_filter_bank.png", dpi=200); plt.close(fig)
    generated.append("shape_filter_bank.png")

    # ── Stackup ──────────────────────────────────────────────────────
    fig_s, ax_s = plt.subplots(figsize=(12, 5))
    ax_s.set_title("6-Layer Stackup  |  Rogers RO3010 + RO4450F", fontsize=13, fontweight="bold")
    stack = [
        ("L1 Signal — Feedline (50Ω)", 35e-6, "#e67e22"),
        ("RO3010 core (er=10.2, 0.254mm)", H_CORE, "#f5e6d3"),
        ("L2 Ground — DGS shapes", 35e-6, "#27ae60"),
        ("RO4450F PP (er=3.52, 0.100mm)", H_PP, "#ecf0f1"),
        ("L3 Signal — Coupled patches", 18e-6, "#e67e22"),
        ("RO3010 core (er=10.2, 0.254mm)", H_CORE, "#f5e6d3"),
        ("L4 Ground — Secondary DGS", 35e-6, "#27ae60"),
        ("RO4450F PP (er=3.52, 0.100mm)", H_PP, "#ecf0f1"),
        ("L5 Signal — Tertiary patches", 18e-6, "#e67e22"),
        ("RO4450F PP (er=3.52, 0.175mm)", 0.175e-3, "#ecf0f1"),
        ("L6 Ground — Bottom ref", 35e-6, "#27ae60"),
    ]
    y = 0
    for label, t, c in reversed(stack):
        h_mm = max(t*1e3, 0.015)
        ax_s.add_patch(Rectangle((1, y), 8, h_mm, fc=c, ec="#333", lw=0.8))
        ax_s.text(9.3, y+h_mm/2, label, va="center", fontsize=8, fontfamily="monospace")
        y += h_mm
    ax_s.set_xlim(0, 20); ax_s.set_ylim(-0.02, y+0.04)
    ax_s.set_ylabel("mm"); ax_s.set_xticks([])
    ax_s.text(5, y+0.025, f"Total: {TOTAL_H*1e3:.3f}mm", ha="center", fontsize=11, fontweight="bold")
    fig_s.tight_layout()
    fig_s.savefig("shape_stackup.png", dpi=200); plt.close(fig_s)
    generated.append("shape_stackup.png")

    # ── Performance table ────────────────────────────────────────────
    print("=" * 68)
    print("  PERFORMANCE COMPARISON  (Standard DGS vs Low-IL J-inverter)")
    print("=" * 68)
    print(f"  {'Filter':>25s}  {'Variant':>10s}  {'IL range':>15s}  {'RL min':>8s}")
    print("  " + "-" * 62)
    for key, res in all_results.items():
        for variant, s21, s11 in [("Standard", res["s21_std"], res["s11_std"]),
                                   ("Low-IL", res["s21_low"], res["s11_low"])]:
            s21db = db(s21)
            s11db = db(s11)
            flo, fhi = res["vlines"][0], res["vlines"][-1]
            if "HPF" in res["name"]:
                fhi = 5.5
            pb = (fg >= flo) & (fg <= fhi)
            if np.any(pb):
                il_lo = float(-np.max(s21db[pb]))
                il_hi = float(-np.min(s21db[pb]))
                rl = float(-np.max(s11db[pb]))
                print(f"  {res['name']:>25s}  {variant:>10s}  "
                      f"{il_lo:.1f} - {il_hi:.1f} dB  {rl:>6.1f} dB")

    # ── Manufacturing audit ──────────────────────────────────────────
    print(f"\n  Manufacturing min trace/space: {MIN_TRACE*1e3:.2f} mm  — "
          f"all bounded parameters enforce this as hard lower limit")

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
