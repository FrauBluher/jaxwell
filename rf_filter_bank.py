#!/usr/bin/env python3
"""
PCB Planar RF Filter Bank Designer
===================================
Generates JAX-optimised microstrip filter physical dimensions for a
3-channel filter bank on Rogers RO4003C:

  1. High-pass filter     fc = 2500 MHz
  2. Bandpass filter #1   2500 - 3750 MHz   (BW = 1.25 GHz)
  3. Bandpass filter #2   3750 - 5000 MHz   (BW = 1.25 GHz)

Substrate : Rogers RO4003C  (er = 3.55, h = 0.508 mm, tan d = 0.0027)
Optimiser : JAX autodiff  +  optax Adam
Output    : frequency-response plot  +  PCB dimension table
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

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
#  Physical constants & substrate
# ═══════════════════════════════════════════════════════════════════════
C0 = 299_792_458.0
EPS0 = 8.854_187_817e-12
Z_REF = 50.0

ER = 3.55
H_SUB = 0.508e-3
TAND = 0.0027

CHEBY_05DB_G = {
    3: [1.0, 1.5963, 1.0967, 1.5963, 1.0],
    5: [1.0, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0],
}

BPF_ORDER = 5
BPF_NRES = BPF_ORDER
BPF_NJ = BPF_ORDER + 1


# ═══════════════════════════════════════════════════════════════════════
#  Microstrip closed-form models  (Hammerstad-Jensen / Wheeler)
# ═══════════════════════════════════════════════════════════════════════
def ms_eeff(w):
    u = jnp.clip(w / H_SUB, 0.01, 1000.0)
    return (ER + 1) / 2 + (ER - 1) / 2 * jnp.power(1 + 12 / u, -0.5)


def ms_width(z0_target):
    A = (
        z0_target / 60.0 * jnp.sqrt((ER + 1) / 2)
        + (ER - 1) / (ER + 1) * (0.23 + 0.11 / ER)
    )
    B = 377.0 * jnp.pi / (2.0 * z0_target * jnp.sqrt(ER))
    wh_narrow = 8 * jnp.exp(A) / (jnp.exp(2 * A) - 2)
    wh_wide = (2 / jnp.pi) * (
        B
        - 1
        - jnp.log(jnp.clip(2 * B - 1, 1e-6))
        + (ER - 1) / (2 * ER) * (jnp.log(jnp.clip(B - 1, 1e-6)) + 0.39 - 0.61 / ER)
    )
    return jnp.where(z0_target > 63.0, wh_narrow, wh_wide) * H_SUB


# ═══════════════════════════════════════════════════════════════════════
#  ABCD matrix primitives
# ═══════════════════════════════════════════════════════════════════════
def _m(a, b, c, d):
    return jnp.array([[a, b], [c, d]], dtype=jnp.complex128)


def abcd_tline(z0, gamma_l):
    ch, sh = jnp.cosh(gamma_l), jnp.sinh(gamma_l)
    return _m(ch, z0 * sh, sh / z0, ch)


def abcd_series(Z):
    return _m(1.0 + 0j, Z, 0j, 1.0 + 0j)


def abcd_shunt(Y):
    return _m(1.0 + 0j, 0j, Y, 1.0 + 0j)


def abcd_jinv(J):
    return _m(0j, -1j / J, -1j * J, 0j)


def cascade(*Ms):
    out = Ms[0]
    for m in Ms[1:]:
        out = out @ m
    return out


def abcd_to_s(M):
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    den = A + B / Z_REF + C * Z_REF + D
    return 2.0 / den, (A + B / Z_REF - C * Z_REF - D) / den


# ═══════════════════════════════════════════════════════════════════════
#  Microstrip element helpers
# ═══════════════════════════════════════════════════════════════════════
def _gamma_l(f, length, z0_elem):
    w = ms_width(z0_elem)
    eeff = ms_eeff(w)
    beta = 2 * jnp.pi * f * jnp.sqrt(eeff) / C0
    filling = ER * (eeff - 1) / (jnp.sqrt(eeff) * (ER - 1) + 1e-20)
    alpha_d = filling * TAND * jnp.pi * f / C0
    return (alpha_d + 1j * beta) * length


def _short_stub_y(f, z0_s, length):
    gl = _gamma_l(f, length, z0_s)
    return 1.0 / (z0_s * jnp.tanh(gl + 1e-15))


def _cap_z(f, cap):
    return 1.0 / (1j * 2 * jnp.pi * f * cap)


# ═══════════════════════════════════════════════════════════════════════
#  Bounded parameter transforms  (smooth bijection R -> (lo, hi))
# ═══════════════════════════════════════════════════════════════════════
def _bounded(x, lo, hi):
    t = jax.nn.sigmoid(x)
    return jnp.exp(jnp.log(lo) * (1 - t) + jnp.log(hi) * t)


def _bounded_inv(y, lo, hi):
    t = (jnp.log(y) - jnp.log(lo)) / (jnp.log(hi) - jnp.log(lo))
    t = jnp.clip(t, 1e-6, 1 - 1e-6)
    return jnp.log(t / (1 - t))


def _pack_bounded(vals, bounds):
    parts = []
    idx = 0
    for n, lo, hi in bounds:
        parts.append(
            jnp.array([float(_bounded_inv(vals[idx + i], lo, hi)) for i in range(n)])
        )
        idx += n
    return jnp.concatenate(parts)


# ═══════════════════════════════════════════════════════════════════════
#  HIGH-PASS FILTER  (5th-order Chebyshev, 0.5 dB ripple)
# ═══════════════════════════════════════════════════════════════════════
#  3 shunt short-stubs + 2 series caps + 4 connecting lines = 12 params
#
#   ──┬── Line0 ──||── Line1 ──┬── Line2 ──||── Line3 ──┬──
#     │          Cap0           │          Cap1           │
#   Stub0                    Stub1                     Stub2
#     │                        │                         │
#    GND                      GND                       GND

HPF_BOUNDS = [
    (3, 25.0, 150.0),      # stub impedances  [ohm]
    (3, 0.3e-3, 15e-3),    # stub lengths     [m]
    (2, 10e-15, 10e-12),   # series caps      [F]
    (4, 0.1e-3, 10e-3),    # connecting lines  [m]
]


def hpf_init():
    fc = 2.5e9
    wc = 2 * np.pi * fc
    g = CHEBY_05DB_G[5]

    L_vals = [Z_REF / (wc * g[i]) for i in (1, 3, 5)]
    C_vals = [1.0 / (wc * Z_REF * g[i]) for i in (2, 4)]

    zs = [90.0, 90.0, 90.0]
    w_s = float(ms_width(jnp.array(90.0)))
    eeff_s = float(ms_eeff(jnp.array(w_s)))
    stub_l = [L * C0 / (z * np.sqrt(eeff_s)) for L, z in zip(L_vals, zs)]

    w50 = float(ms_width(jnp.array(50.0)))
    eeff50 = float(ms_eeff(jnp.array(w50)))
    lam = C0 / (fc * np.sqrt(eeff50))
    line_l = [lam * 0.04] * 4

    vals = zs + stub_l + C_vals + line_l
    return _pack_bounded(vals, HPF_BOUNDS)


def _hpf_unpack(p):
    sz = _bounded(p[0:3], 25.0, 150.0)
    sl = _bounded(p[3:6], 0.3e-3, 15e-3)
    cp = _bounded(p[6:8], 10e-15, 10e-12)
    ll = _bounded(p[8:12], 0.1e-3, 10e-3)
    return sz, sl, cp, ll


def _hpf_at_f(p, f):
    sz, sl, cp, ll = _hpf_unpack(p)
    z50 = Z_REF
    gl = lambda length: _gamma_l(f, length, z50)

    return abcd_to_s(
        cascade(
            abcd_shunt(_short_stub_y(f, sz[0], sl[0])),
            abcd_tline(z50, gl(ll[0])),
            abcd_series(_cap_z(f, cp[0])),
            abcd_tline(z50, gl(ll[1])),
            abcd_shunt(_short_stub_y(f, sz[1], sl[1])),
            abcd_tline(z50, gl(ll[2])),
            abcd_series(_cap_z(f, cp[1])),
            abcd_tline(z50, gl(ll[3])),
            abcd_shunt(_short_stub_y(f, sz[2], sl[2])),
        )
    )


_hpf_batch = jax.vmap(_hpf_at_f, in_axes=(None, 0))


def hpf_loss(p, freqs):
    s21, s11 = _hpf_batch(p, freqs)
    s21_db = 20 * jnp.log10(jnp.clip(jnp.abs(s21), 1e-12, None))
    s11_db = 20 * jnp.log10(jnp.clip(jnp.abs(s11), 1e-12, None))
    fc = 2.5e9

    pb = (freqs >= fc) & (freqs <= 5.5e9)
    sb = freqs <= 1.5e9
    tb = (freqs > 1.5e9) & (freqs < fc)

    loss = jnp.sum(jnp.where(pb, jnp.maximum(-s21_db - 0.5, 0.0) ** 2, 0.0)) * 6.0
    loss += jnp.sum(jnp.where(pb, jnp.maximum(s11_db + 12.0, 0.0) ** 2, 0.0)) * 2.0
    loss += jnp.sum(jnp.where(sb, jnp.maximum(s21_db + 20.0, 0.0) ** 2, 0.0)) * 4.0
    loss += jnp.sum(jnp.where(tb, jnp.maximum(s21_db + 6.0, 0.0) ** 2, 0.0)) * 1.0
    return loss / freqs.shape[0]


# ═══════════════════════════════════════════════════════════════════════
#  BANDPASS FILTER  (5th-order coupled-resonator with J-inverters)
# ═══════════════════════════════════════════════════════════════════════
#  Topology: J-inverters + lambda/2 microstrip resonators
#
#   ──[J0]── Res1 ──[J1]── Res2 ──[J2]── Res3 ──[J3]── Res4 ──[J4]── Res5 ──[J5]──
#
#  11 free parameters: res_l[5] + J_normalised[6]

BPF_BOUNDS = [
    (BPF_NRES, 3e-3, 60e-3),   # resonator lengths  [m]
    (BPF_NJ, 0.01, 2.0),       # normalised J*Z0    [dimensionless]
]


def bpf_init(fc, fbw):
    g = CHEBY_05DB_G[BPF_ORDER]
    n = BPF_ORDER

    J_norm = [float(np.sqrt(np.pi * fbw / (2 * g[0] * g[1])))]
    for i in range(1, n):
        J_norm.append(float(np.pi * fbw / (2 * np.sqrt(g[i] * g[i + 1]))))
    J_norm.append(float(np.sqrt(np.pi * fbw / (2 * g[n] * g[n + 1]))))
    J_norm = [max(min(j, 1.99), 0.011) for j in J_norm]

    w50 = float(ms_width(jnp.array(50.0)))
    eeff = float(ms_eeff(jnp.array(w50)))
    lam2 = C0 / (fc * np.sqrt(eeff)) / 2
    res_l = [lam2] * n

    return _pack_bounded(res_l + J_norm, BPF_BOUNDS)


def _bpf_at_f(p, f):
    rl = _bounded(p[:BPF_NRES], 3e-3, 60e-3)
    jn = _bounded(p[BPF_NRES:], 0.01, 2.0)
    J = jn / Z_REF

    M0 = abcd_jinv(J[0])

    def body(carry, x):
        rl_i, j_i = x
        gl = _gamma_l(f, rl_i, Z_REF)
        return carry @ abcd_tline(Z_REF, gl) @ abcd_jinv(j_i), None

    M_final, _ = jax.lax.scan(body, M0, (rl, J[1:]))
    return abcd_to_s(M_final)


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

        loss = jnp.sum(jnp.where(pb, jnp.maximum(-s21d - 1.5, 0.0) ** 2, 0.0)) * 6.0
        loss += (
            jnp.sum(jnp.where(pb, jnp.maximum(s11d + 10.0, 0.0) ** 2, 0.0)) * 2.0
        )
        loss += (
            jnp.sum(jnp.where(sl, jnp.maximum(s21d + 15.0, 0.0) ** 2, 0.0)) * 3.0
        )
        loss += (
            jnp.sum(jnp.where(sh, jnp.maximum(s21d + 15.0, 0.0) ** 2, 0.0)) * 3.0
        )
        return loss / freqs.shape[0]

    return loss_fn


# ═══════════════════════════════════════════════════════════════════════
#  Optimiser  (Adam + gradient clipping + multi-restart)
# ═══════════════════════════════════════════════════════════════════════
def optimise(loss_fn, init_params, freqs, *, n_steps=2000, lr=3e-3):
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    state = opt.init(init_params)
    params = init_params

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(loss_fn)(params, freqs)
        grads = jnp.where(jnp.isfinite(grads), grads, 0.0)
        updates, new_state = opt.update(grads, state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    best_params, best_loss = params, float("inf")
    for i in range(n_steps):
        params, state, lv = step(params, state)
        lv_f = float(lv)
        if jnp.isfinite(lv) and lv_f < best_loss:
            best_loss, best_params = lv_f, params
        if (i + 1) % 500 == 0 or i == 0:
            print(
                f"  step {i + 1:5d}/{n_steps}  "
                f"loss = {lv_f:.4f}  best = {best_loss:.4f}"
            )
    return best_params, best_loss


def optimise_restarts(loss_fn, init_fn, freqs, n_restarts=3, **kwargs):
    best_p, best_l = None, float("inf")
    for r in range(n_restarts):
        p0 = init_fn()
        if r > 0:
            key = jax.random.PRNGKey(r * 137)
            p0 = p0 + jax.random.normal(key, p0.shape) * 0.5
        if n_restarts > 1:
            print(f"  (restart {r + 1}/{n_restarts})")
        p, lv = optimise(loss_fn, p0, freqs, **kwargs)
        if lv < best_l:
            best_l, best_p = lv, p
    return best_p


# ═══════════════════════════════════════════════════════════════════════
#  Coupled-line dimension helpers
# ═══════════════════════════════════════════════════════════════════════
def _j_to_coupled_z(j_norm):
    x = float(j_norm)
    z0e = Z_REF * (1 + x + x**2)
    z0o = Z_REF * (1 - x + x**2)
    return z0e, z0o


def _cap_to_gap_mm(cap_f, w50_m):
    eeff = float(ms_eeff(jnp.array(w50_m)))
    denom = EPS0 * np.sqrt(eeff) * w50_m
    ratio = float(cap_f) / denom
    if ratio <= 0 or ratio >= 1.0:
        return np.nan
    return max(float(-H_SUB / 1.86 * np.log(ratio)) * 1e3, 0.05)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs = jnp.linspace(0.1e9, 7e9, 500)

    # ── HPF ──────────────────────────────────────────────────────────
    print("=" * 64)
    print("  Optimising High-Pass Filter  (fc = 2500 MHz)")
    print("=" * 64)
    p_hpf = hpf_init()
    p_hpf, _ = optimise(hpf_loss, p_hpf, freqs, n_steps=2500, lr=3e-3)

    # ── BPF 1 ────────────────────────────────────────────────────────
    f1_lo, f1_hi = 2.5e9, 3.75e9
    fc1 = (f1_lo + f1_hi) / 2
    fbw1 = (f1_hi - f1_lo) / fc1

    print("\n" + "=" * 64)
    print("  Optimising Band-Pass Filter 1  (2500 - 3750 MHz)")
    print("=" * 64)
    bpf1_loss = make_bpf_loss(f1_lo, f1_hi)
    p_b1 = optimise_restarts(
        bpf1_loss,
        lambda: bpf_init(fc1, fbw1),
        freqs,
        n_restarts=3,
        n_steps=3000,
        lr=3e-3,
    )

    # ── BPF 2 ────────────────────────────────────────────────────────
    f2_lo, f2_hi = 3.75e9, 5.0e9
    fc2 = (f2_lo + f2_hi) / 2
    fbw2 = (f2_hi - f2_lo) / fc2

    print("\n" + "=" * 64)
    print("  Optimising Band-Pass Filter 2  (3750 - 5000 MHz)")
    print("=" * 64)
    bpf2_loss = make_bpf_loss(f2_lo, f2_hi)
    p_b2 = optimise_restarts(
        bpf2_loss,
        lambda: bpf_init(fc2, fbw2),
        freqs,
        n_restarts=3,
        n_steps=3000,
        lr=3e-3,
    )

    # ── Evaluate ─────────────────────────────────────────────────────
    fp = jnp.linspace(0.1e9, 7e9, 1000)
    s21_hpf, s11_hpf = _hpf_batch(p_hpf, fp)
    s21_b1, s11_b1 = _bpf_batch(p_b1, fp)
    s21_b2, s11_b2 = _bpf_batch(p_b2, fp)

    fg = np.array(fp) / 1e9
    db = lambda s: 20 * np.log10(np.clip(np.abs(np.array(s)), 1e-12, None))

    # ── Helper for single-filter plot ────────────────────────────────
    generated_images = []

    def _plot_filter(s21, s11, title, vlines, filename):
        fig_f, ax_f = plt.subplots(figsize=(8, 5))
        ax_f.plot(fg, db(s21), "b", lw=1.8, label="|S21|")
        ax_f.plot(fg, db(s11), "r--", lw=1.2, label="|S11|")
        ax_f.set_title(title, fontsize=13, fontweight="bold")
        ax_f.set_xlabel("Frequency [GHz]")
        ax_f.set_ylabel("[dB]")
        ax_f.set_ylim(-50, 3)
        ax_f.legend(loc="lower right", fontsize=11)
        ax_f.grid(True, alpha=0.3)
        for fv in vlines:
            ax_f.axvline(fv, color="gray", ls=":", lw=0.8)
        fig_f.tight_layout()
        fig_f.savefig(filename, dpi=200)
        plt.close(fig_f)
        generated_images.append(filename)

    # ── Individual filter plots ──────────────────────────────────────
    _plot_filter(
        s21_hpf, s11_hpf,
        "High-Pass Filter  (fc = 2.5 GHz)  |  Rogers RO4003C",
        [2.5],
        "filter_hpf_response.png",
    )
    _plot_filter(
        s21_b1, s11_b1,
        "Bandpass Filter 1  (2.5 - 3.75 GHz)  |  Rogers RO4003C",
        [2.5, 3.75],
        "filter_bpf1_response.png",
    )
    _plot_filter(
        s21_b2, s11_b2,
        "Bandpass Filter 2  (3.75 - 5.0 GHz)  |  Rogers RO4003C",
        [3.75, 5.0],
        "filter_bpf2_response.png",
    )

    # ── Combined 4-panel overview ────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "PCB Planar RF Filter Bank  |  Rogers RO4003C",
        fontsize=14,
        fontweight="bold",
    )

    specs = [
        (axes[0, 0], s21_hpf, s11_hpf,
         "High-Pass Filter (fc = 2.5 GHz)", [2.5]),
        (axes[0, 1], s21_b1, s11_b1,
         "Bandpass Filter 1 (2.5 - 3.75 GHz)", [2.5, 3.75]),
        (axes[1, 0], s21_b2, s11_b2,
         "Bandpass Filter 2 (3.75 - 5.0 GHz)", [3.75, 5.0]),
    ]
    for ax, s21, s11, title, vlines in specs:
        ax.plot(fg, db(s21), "b", lw=1.5, label="|S21|")
        ax.plot(fg, db(s11), "r--", lw=1.0, label="|S11|")
        ax.set_title(title)
        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel("[dB]")
        ax.set_ylim(-50, 3)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        for fv in vlines:
            ax.axvline(fv, color="gray", ls=":", lw=0.8)

    ax = axes[1, 1]
    ax.plot(fg, db(s21_hpf), "k", lw=1.5, label="HPF")
    ax.plot(fg, db(s21_b1), "b", lw=1.5, label="BPF 1")
    ax.plot(fg, db(s21_b2), "m", lw=1.5, label="BPF 2")
    ax.set_title("Filter Bank  |  Combined |S21|")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("|S21| [dB]")
    ax.set_ylim(-50, 3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    for fv in (2.5, 3.75, 5.0):
        ax.axvline(fv, color="gray", ls=":", lw=0.8)

    plt.tight_layout()
    plt.savefig("filter_bank_response.png", dpi=200)
    plt.close(fig)
    generated_images.append("filter_bank_response.png")

    print()
    for img in generated_images:
        sz = os.path.getsize(img)
        print(f"  [saved] {img}  ({sz / 1024:.0f} KB)")

    # ── Performance summary ──────────────────────────────────────────
    def _perf(s21_arr, s11_arr, flo, fhi, label):
        s21db = db(s21_arr)
        s11db = db(s11_arr)
        pb = (fg >= flo) & (fg <= fhi)
        if not np.any(pb):
            return
        il_max = float(-np.min(s21db[pb]))
        il_min = float(-np.max(s21db[pb]))
        rl_min = float(-np.max(s11db[pb]))
        print(f"  {label}:")
        print(f"    Passband IL   : {il_min:.1f} - {il_max:.1f} dB")
        print(f"    Passband RL   : > {rl_min:.1f} dB")

    print()
    print("=" * 64)
    print("  PERFORMANCE SUMMARY")
    print("=" * 64)
    _perf(s21_hpf, s11_hpf, 2.5, 5.5, "HPF (2.5 - 5.5 GHz passband)")
    _perf(s21_b1, s11_b1, 2.5, 3.75, "BPF 1 (2.5 - 3.75 GHz)")
    _perf(s21_b2, s11_b2, 3.75, 5.0, "BPF 2 (3.75 - 5.0 GHz)")

    # ── Dimension table ──────────────────────────────────────────────
    w50 = float(ms_width(jnp.array(50.0)))

    print()
    print("=" * 64)
    print(f"  PCB DIMENSIONS  |  Rogers RO4003C  (er={ER}, h={H_SUB*1e3:.3f} mm)")
    print("=" * 64)
    print(f"\n  50-ohm microstrip width : {w50 * 1e3:.3f} mm\n")

    # HPF
    sz, sl, cp, ll = _hpf_unpack(p_hpf)
    sz, sl, cp, ll = (np.array(x) for x in (sz, sl, cp, ll))

    print("  --- High-Pass Filter (fc = 2.5 GHz) " + "-" * 26)
    for i in range(3):
        ws = float(ms_width(jnp.array(float(sz[i])))) * 1e3
        print(
            f"  Stub {i + 1}:  Z0 = {sz[i]:6.1f} ohm   "
            f"w = {ws:.3f} mm   l = {sl[i] * 1e3:.3f} mm   (via to GND)"
        )
    for i in range(2):
        gw = _cap_to_gap_mm(cp[i], w50)
        note = "interdigital recommended" if (np.isnan(gw) or gw < 0.10) else ""
        gw_str = f"{gw:.3f}" if not np.isnan(gw) else "N/A"
        print(
            f"  Cap  {i + 1}:  C  = {cp[i] * 1e12:.3f} pF   "
            f"gap ~ {gw_str} mm   {note}"
        )
    for i in range(4):
        print(f"  Line {i + 1}:  l  = {ll[i] * 1e3:.3f} mm   (50 ohm)")
    total = float(np.sum(sl) + np.sum(ll))
    print(f"  Total length ~ {total * 1e3:.1f} mm")

    # BPFs
    for tag, params, flo_ghz, fhi_ghz in [
        ("Band-Pass Filter 1 (2.5 - 3.75 GHz)", p_b1, 2.5, 3.75),
        ("Band-Pass Filter 2 (3.75 - 5.0 GHz)", p_b2, 3.75, 5.0),
    ]:
        rl = np.array(_bounded(params[:BPF_NRES], 3e-3, 60e-3))
        jn = np.array(_bounded(params[BPF_NRES:], 0.01, 2.0))
        fc_ghz = (flo_ghz + fhi_ghz) / 2

        eeff50 = float(ms_eeff(jnp.array(w50)))
        lam4 = C0 / (4 * fc_ghz * 1e9 * np.sqrt(eeff50)) * 1e3

        print(f"\n  --- {tag} " + "-" * max(0, 62 - len(tag) - 7))
        for i in range(BPF_NRES):
            print(
                f"  Resonator {i + 1}:  l = {rl[i] * 1e3:.3f} mm  (50 ohm, ~lambda/2)"
            )

        print()
        for i in range(BPF_NJ):
            z0e, z0o = _j_to_coupled_z(jn[i])
            w_e = float(ms_width(jnp.array(z0e))) * 1e3
            w_o = float(ms_width(jnp.array(z0o))) * 1e3
            w_coupled = (w_e + w_o) / 2
            s_gap = max(abs(w_e - w_coupled), 0.05)
            labels = {0: "input", BPF_NJ - 1: "output"}
            lbl = labels.get(i, f"inter-{i}")
            print(
                f"  Coupled section {i} ({lbl:>7s}):  "
                f"J*Z0 = {jn[i]:.4f}   "
                f"Z0e = {z0e:.1f}   Z0o = {z0o:.1f} ohm"
            )
            print(
                f"{'':32s}w ~ {w_coupled:.3f} mm   "
                f"s ~ {s_gap:.3f} mm   "
                f"len = {lam4:.2f} mm (lambda/4)"
            )

        print(f"  Total resonator length ~ {float(np.sum(rl)) * 1e3:.1f} mm")

    print("\n" + "=" * 64)
    print("  OUTPUT IMAGES")
    print("=" * 64)
    for img in generated_images:
        sz = os.path.getsize(img)
        print(f"  {img:40s}  {sz / 1024:6.0f} KB  OK")
    print()
    print("=" * 64)
    print("  Complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
