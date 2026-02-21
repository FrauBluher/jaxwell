#!/usr/bin/env python3
"""
Compact SMT Filter Module — Minimum-Size Multilayer Design
============================================================
Self-contained filter bank module with castellated edge pads,
designed to be reflow-soldered onto a host PCB.

Size strategy:
  - 3rd-order BPFs (3 resonators, fewer folds)
  - RO3010 core er=10.2 for L1 HPF (smallest stubs/lines)
  - RO3010/RO4450F hybrid for stripline BPFs
  - Aggressive meander folding (0.8 mm fold-to-fold)
  - Board size derived from optimised trace lengths

6-layer stackup:
  L1  Signal   HPF (microstrip, er=10.2)
  L2  Ground   HPF ref
  L3  Signal   BPF1 (stripline, er=3.52 environment)
  L4  Ground   Shared ref
  L5  Signal   BPF2 (stripline, er=3.52 environment)
  L6  Ground   Bottom

SMT module features:
  - Castellated half-via edge pads (input, output, ground)
  - Ground via fence around perimeter
  - Fiducial marks for pick-and-place

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
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

jax.config.update("jax_enable_x64", True)

# ═══════════════════════════════════════════════════════════════════════
C0   = 299_792_458.0
EPS0 = 8.854_187_817e-12
MU0  = 4e-7 * np.pi
Z_REF = 50.0

ER_CORE = 10.2;  H_CORE = 0.254e-3;  TAND_CORE = 0.0035
ER_SL   = 3.52;  H_SL   = 0.200e-3;  TAND_SL   = 0.004

MIN_TRACE = 0.15e-3
FOLD_GAP  = 0.8          # mm, fold-to-fold spacing for meanders
EDGE_PAD  = 1.5          # mm, edge clearance for castellated pads
VIA_FENCE_PITCH = 1.2    # mm

CHEBY_G3 = [1.0, 1.5963, 1.0967, 1.5963, 1.0]
CHEBY_G5 = [1.0, 1.7058, 1.2296, 2.5408, 1.2296, 1.7058, 1.0]

def _ms_eeff(w, h=H_CORE, er=ER_CORE):
    u = jnp.clip(w/h, 0.01, 1000.)
    return (er+1)/2 + (er-1)/2 * jnp.power(1+12/u, -0.5)
def _ms_width(z0, h=H_CORE, er=ER_CORE):
    A = z0/60.*jnp.sqrt((er+1)/2)+(er-1)/(er+1)*(0.23+0.11/er)
    Bv = 377.*jnp.pi/(2.*z0*jnp.sqrt(er))
    w1 = 8*jnp.exp(A)/(jnp.exp(2*A)-2)
    w2 = (2/jnp.pi)*(Bv-1-jnp.log(jnp.clip(2*Bv-1,1e-6))
          +(er-1)/(2*er)*(jnp.log(jnp.clip(Bv-1,1e-6))+0.39-0.61/er))
    return jnp.where(z0>63.,w1,w2)*h

W50  = float(_ms_width(jnp.array(Z_REF)))
EEFF = float(_ms_eeff(jnp.array(W50)))
W50_MM = W50 * 1e3

B_SL = 2 * H_SL + 18e-6
SL_W50_B = 94.15 / (np.sqrt(ER_SL) * Z_REF) - 0.4413
SL_W50 = SL_W50_B * B_SL
SL_W50_MM = SL_W50 * 1e3

TOTAL_H = (35e-6 + H_CORE + 35e-6 + H_SL + 18e-6 + H_SL
           + 35e-6 + H_SL + 18e-6 + H_SL + 35e-6)

# ═══════════════════════════════════════════════════════════════════════
def _m(a,b,c,d): return jnp.array([[a,b],[c,d]], dtype=jnp.complex128)
def abcd_tl(z0,gl):
    ch,sh=jnp.cosh(gl),jnp.sinh(gl); return _m(ch,z0*sh,sh/z0,ch)
def abcd_sh(Y): return _m(1.+0j,0j,Y,1.+0j)
def abcd_jinv(J): return _m(0j,-1j/J,-1j*J,0j)
def abcd_to_s(M):
    A,Bv,C,D=M[0,0],M[0,1],M[1,0],M[1,1]
    den=A+Bv/Z_REF+C*Z_REF+D
    return 2./den,(A+Bv/Z_REF-C*Z_REF-D)/den
def _gl_ms(f,l):
    return (TAND_CORE*jnp.pi*f*jnp.sqrt(EEFF)/C0+1j*2*jnp.pi*f*jnp.sqrt(EEFF)/C0)*l
def _gl_sl(f,l):
    return (TAND_SL*jnp.pi*f*jnp.sqrt(ER_SL)/C0+1j*2*jnp.pi*f*jnp.sqrt(ER_SL)/C0)*l
def _bnd(x,lo,hi):
    t=jax.nn.sigmoid(x); return jnp.exp(jnp.log(lo)*(1-t)+jnp.log(hi)*t)
def _bnd_inv(y,lo,hi):
    t=(jnp.log(y)-jnp.log(lo))/(jnp.log(hi)-jnp.log(lo))
    t=jnp.clip(t,1e-6,1-1e-6); return jnp.log(t/(1-t))
def _pack(vals,bounds):
    parts,idx=[],0
    for n,lo,hi in bounds:
        parts.append(jnp.array([float(_bnd_inv(vals[idx+i],lo,hi)) for i in range(n)]))
        idx+=n
    return jnp.concatenate(parts)
def _unpack(p,bounds):
    res,i=[],0
    for n,lo,hi in bounds:
        res.append(_bnd(p[i:i+n],lo,hi)); i+=n
    return res

# ═══════════════════════════════════════════════════════════════════════
#  HPF — 5th-order stubs + MIM caps on L1
# ═══════════════════════════════════════════════════════════════════════
HPF_BOUNDS = [(3,25.,150.),(3,0.3e-3,6e-3),(2,10e-15,10e-12),(4,0.1e-3,5e-3)]

def _hpf_init():
    fc=2.5e9;wc=2*np.pi*fc;g=CHEBY_G5
    L=[Z_REF/(wc*g[i]) for i in (1,3,5)]; C=[1./(wc*Z_REF*g[i]) for i in (2,4)]
    zs=[90.,90.,90.]
    es=float(_ms_eeff(jnp.array(_ms_width(jnp.array(90.)))))
    sl=[min(Lv*C0/(z*np.sqrt(es)),5e-3) for Lv,z in zip(L,zs)]
    lam=C0/(fc*np.sqrt(EEFF))
    return _pack(zs+sl+C+[min(lam*0.03,4e-3)]*4, HPF_BOUNDS)

def _hpf_at_f(p,f):
    sz,sl,cp,ll=_unpack(p,HPF_BOUNDS)
    gl=lambda l:_gl_ms(f,l)
    def sy(z,l):
        w=_ms_width(z);e=_ms_eeff(w)
        g=(TAND_CORE*jnp.pi*f*jnp.sqrt(e)/C0+1j*2*jnp.pi*f*jnp.sqrt(e)/C0)*l
        return 1./(z*jnp.tanh(g+1e-15))
    cz=lambda c:1./(1j*2*jnp.pi*f*c)
    M=abcd_sh(sy(sz[0],sl[0]))
    M=M@abcd_tl(Z_REF,gl(ll[0]))@_m(1.+0j,cz(cp[0]),0j,1.+0j)
    M=M@abcd_tl(Z_REF,gl(ll[1]))@abcd_sh(sy(sz[1],sl[1]))
    M=M@abcd_tl(Z_REF,gl(ll[2]))@_m(1.+0j,cz(cp[1]),0j,1.+0j)
    M=M@abcd_tl(Z_REF,gl(ll[3]))@abcd_sh(sy(sz[2],sl[2]))
    return abcd_to_s(M)

_hpf_batch=jax.vmap(_hpf_at_f,in_axes=(None,0))

def _hpf_loss(p,freqs):
    s21,s11=_hpf_batch(p,freqs)
    s21d=20*jnp.log10(jnp.clip(jnp.abs(s21),1e-12,None))
    s11d=20*jnp.log10(jnp.clip(jnp.abs(s11),1e-12,None))
    fc=2.5e9;pb=(freqs>=fc)&(freqs<=5.5e9);sb=freqs<=1.5e9;tb=(freqs>1.5e9)&(freqs<fc)
    loss =jnp.sum(jnp.where(pb,jnp.maximum(-s21d-0.5,0.)**2,0.))*6
    loss+=jnp.sum(jnp.where(pb,jnp.maximum(s11d+12,0.)**2,0.))*2
    loss+=jnp.sum(jnp.where(sb,jnp.maximum(s21d+20,0.)**2,0.))*4
    loss+=jnp.sum(jnp.where(tb,jnp.maximum(s21d+6,0.)**2,0.))*1
    return loss/freqs.shape[0]

# ═══════════════════════════════════════════════════════════════════════
#  BPF — 3rd-order J-inverter stripline (compact)
# ═══════════════════════════════════════════════════════════════════════
N_BPF=3; NJ=4
BPF_BOUNDS = [(N_BPF,3e-3,20e-3),(NJ,0.01,2.0)]

def _bpf_init(fc,fbw):
    g=CHEBY_G3
    J=[float(np.sqrt(np.pi*fbw/(2*g[0]*g[1])))]
    for i in range(1,N_BPF):
        J.append(float(np.pi*fbw/(2*np.sqrt(g[i]*g[i+1]))))
    J.append(float(np.sqrt(np.pi*fbw/(2*g[N_BPF]*g[N_BPF+1]))))
    J=[max(min(j,1.99),0.011) for j in J]
    lam2=min(C0/(fc*np.sqrt(ER_SL))/2, 19e-3)
    return _pack([lam2]*N_BPF+J, BPF_BOUNDS)

def _bpf_at_f(p,f):
    rl=_bnd(p[:N_BPF],3e-3,20e-3);jn=_bnd(p[N_BPF:],0.01,2.0);J=jn/Z_REF
    M0=abcd_jinv(J[0])
    def body(M,x): return M@abcd_tl(Z_REF,_gl_sl(f,x[0]))@abcd_jinv(x[1]),None
    Mf,_=jax.lax.scan(body,M0,(rl,J[1:]))
    return abcd_to_s(Mf)

_bpf_batch=jax.vmap(_bpf_at_f,in_axes=(None,0))

def _make_bpf_loss(fl,fh):
    bw=fh-fl
    def loss_fn(p,freqs):
        s21,s11=_bpf_batch(p,freqs)
        s21d=20*jnp.log10(jnp.clip(jnp.abs(s21),1e-12,None))
        s11d=20*jnp.log10(jnp.clip(jnp.abs(s11),1e-12,None))
        pb=(freqs>=fl)&(freqs<=fh);sl_=freqs<(fl-0.3*bw);sh_=freqs>(fh+0.3*bw)
        loss =jnp.sum(jnp.where(pb,jnp.maximum(-s21d-1.5,0.)**2,0.))*6
        loss+=jnp.sum(jnp.where(pb,jnp.maximum(s11d+10,0.)**2,0.))*2
        loss+=jnp.sum(jnp.where(sl_,jnp.maximum(s21d+15,0.)**2,0.))*3
        loss+=jnp.sum(jnp.where(sh_,jnp.maximum(s21d+15,0.)**2,0.))*3
        return loss/freqs.shape[0]
    return loss_fn

# ═══════════════════════════════════════════════════════════════════════
def _opt(loss_fn,init_fn,freqs,n_r=3,n_steps=2500,lr=5e-3):
    best_p,best_l=None,float("inf")
    for r in range(n_r):
        p0=init_fn()
        if r>0: p0=p0+jax.random.normal(jax.random.PRNGKey(r*97),p0.shape)*0.4
        print(f"  (restart {r+1}/{n_r})")
        opt=optax.chain(optax.clip_by_global_norm(5.),optax.adam(lr))
        state=opt.init(p0);params=p0
        @jax.jit
        def step(params,state):
            l,g=jax.value_and_grad(loss_fn)(params,freqs)
            g=jnp.where(jnp.isfinite(g),g,0.)
            u,ns=opt.update(g,state,params)
            return optax.apply_updates(params,u),ns,l
        for i in range(n_steps):
            params,state,lv=step(params,state)
            lv_f=float(lv)
            if jnp.isfinite(lv) and lv_f<best_l: best_l,best_p=lv_f,params
            if (i+1)%500==0 or i==0:
                print(f"  step {i+1:5d}/{n_steps}  loss={lv_f:.4f}  best={best_l:.4f}")
    return best_p

# ═══════════════════════════════════════════════════════════════════════
#  COPPER LAYER RENDERING
# ═══════════════════════════════════════════════════════════════════════
CU="#c8882e"; CU_DARK="#a06820"; SUBSTRATE="#1a1a2e"; VIA_C="#ccb060"; SOLDER="#d4d4d4"

def _meander_pts(x0,y0,segs,board_w,fold_gap=FOLD_GAP):
    mx=board_w-EDGE_PAD; x,y,dx=x0,y0,1; pts=[(x,y)]
    for s in segs:
        avail=(mx-x) if dx>0 else (x-EDGE_PAD)
        avail=max(avail,0.3)
        if s<=avail+0.1:
            x+=s*dx; pts.append((x,y))
        else:
            x+=avail*dx; pts.append((x,y))
            y-=fold_gap; pts.append((x,y)); dx=-dx
            x+=(s-avail)*dx; pts.append((x,y))
    return pts

def _draw_trace(ax,pts,w,color=CU,z=3):
    xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
    ax.plot(xs,ys,color=color,lw=max(w*4,1.2),solid_capstyle="round",solid_joinstyle="round",zorder=z)

def _draw_via(ax,x,y,pr=0.35,dr=0.12):
    ax.add_patch(Circle((x,y),pr,fc=VIA_C,ec=CU_DARK,lw=0.3,zorder=5))
    ax.add_patch(Circle((x,y),dr,fc="#333",zorder=6))

def _draw_castellated(ax,x,y,side="left"):
    """Half-via castellated pad on board edge."""
    r=0.5
    ax.add_patch(Circle((x,y),r,fc=SOLDER,ec="#888",lw=0.4,zorder=5))
    ax.add_patch(Circle((x,y),0.2,fc=CU,zorder=6))

def _draw_gnd_pour(ax,bw,bh,voids=None):
    ax.add_patch(Rectangle((0,0),bw,bh,fc=CU,ec=CU_DARK,lw=0.5,zorder=1))
    if voids:
        for (x,y,w,h) in voids:
            ax.add_patch(Rectangle((x,y),w,h,fc=SUBSTRATE,ec="#555",lw=0.2,zorder=2))

def _draw_via_fence(ax,bw,bh,exclusions=None):
    """Draw perimeter ground via fence, skipping near signal pads."""
    if exclusions is None:
        exclusions = []
    def _ok(x,y):
        for ex,ey,er in exclusions:
            if abs(x-ex)<er and abs(y-ey)<er:
                return False
        return True
    for x in np.arange(VIA_FENCE_PITCH, bw-0.5, VIA_FENCE_PITCH):
        if _ok(x,0.5): _draw_via(ax,x,0.5,0.25,0.1)
        if _ok(x,bh-0.5): _draw_via(ax,x,bh-0.5,0.25,0.1)
    for y in np.arange(VIA_FENCE_PITCH, bh-0.5, VIA_FENCE_PITCH):
        if _ok(0.5,y): _draw_via(ax,0.5,y,0.25,0.1)
        if _ok(bw-0.5,y): _draw_via(ax,bw-0.5,y,0.25,0.1)

def _layer_ax(fig,pos,title,bw,bh):
    ax=fig.add_subplot(*pos)
    ax.set_facecolor(SUBSTRATE)
    ax.add_patch(Rectangle((-0.3,-0.3),bw+0.6,bh+0.6,fc=SUBSTRATE,ec="#444",lw=2,zorder=0))
    ax.set_xlim(-1.5,bw+1.5); ax.set_ylim(-1.5,bh+1.5)
    ax.set_aspect("equal"); ax.set_title(title,fontsize=8,fontweight="bold")
    ax.tick_params(labelsize=5)
    return ax

# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs=jnp.linspace(0.1e9,7e9,500)
    fp=jnp.linspace(0.1e9,7e9,1000); fg=np.array(fp)/1e9
    db=lambda s:20*np.log10(np.clip(np.abs(np.array(s)),1e-12,None))
    generated=[]

    print("="*64)
    print("  Minimum-Size SMT Filter Module")
    print("="*64)
    print(f"  Substrate : RO3010 er={ER_CORE} (L1) + RO4450F er={ER_SL} (stripline)")
    print(f"  50Ω MS    : {W50_MM:.3f} mm   50Ω SL: {SL_W50_MM:.3f} mm")
    print(f"  Thickness : {TOTAL_H*1e3:.2f} mm   6 layers")
    print(f"  Mfg min   : {MIN_TRACE*1e3:.2f} mm trace/space")
    print()

    # ── Optimise ─────────────────────────────────────────────────────
    print("="*64); print("  HPF fc=2.5GHz (L1, 5th-order)"); print("="*64)
    p_hpf=_opt(_hpf_loss,_hpf_init,freqs,n_r=2,n_steps=2500)

    f1l,f1h=2.5e9,3.75e9; fc1,fbw1=(f1l+f1h)/2,(f1h-f1l)/((f1l+f1h)/2)
    print("\n"+"="*64); print("  BPF1 2.5-3.75GHz (L3, 3rd-order)"); print("="*64)
    p_b1=_opt(_make_bpf_loss(f1l,f1h),lambda:_bpf_init(fc1,fbw1),freqs,n_r=3,n_steps=3000)

    f2l,f2h=3.75e9,5.0e9; fc2,fbw2=(f2l+f2h)/2,(f2h-f2l)/((f2l+f2h)/2)
    print("\n"+"="*64); print("  BPF2 3.75-5.0GHz (L5, 3rd-order)"); print("="*64)
    p_b2=_opt(_make_bpf_loss(f2l,f2h),lambda:_bpf_init(fc2,fbw2),freqs,n_r=3,n_steps=3000)

    s21h,s11h=_hpf_batch(p_hpf,fp)
    s21b1,s11b1=_bpf_batch(p_b1,fp)
    s21b2,s11b2=_bpf_batch(p_b2,fp)

    # ── Compute minimum board size ───────────────────────────────────
    sz,sl,cp,ll=(np.array(x) for x in _unpack(p_hpf,HPF_BOUNDS))
    hpf_total = float(np.sum(sl)+np.sum(ll))*1e3

    rl1=np.array(_bnd(p_b1[:N_BPF],3e-3,20e-3))*1e3
    rl2=np.array(_bnd(p_b2[:N_BPF],3e-3,20e-3))*1e3
    bpf1_total=float(np.sum(rl1))
    bpf2_total=float(np.sum(rl2))

    longest=max(hpf_total, bpf1_total, bpf2_total)

    n_folds = 1
    trial_w = longest + 2*EDGE_PAD
    while trial_w > 40:
        n_folds += 1
        trial_w = longest / n_folds + 2*EDGE_PAD
    BW = float(np.ceil(trial_w))

    # Board height: 3 filter channels stacked vertically
    # Each channel needs ~2mm for trace+fold, plus HPF needs stub room
    PAD_PITCH = 2.5
    max_stub_mm = float(np.max(sl)) * 1e3
    CH_HPF  = max_stub_mm + 1.0 + n_folds * FOLD_GAP
    CH_BPF  = 1.0 + n_folds * FOLD_GAP
    BH = float(np.ceil(
        EDGE_PAD + CH_HPF + PAD_PITCH + CH_BPF + PAD_PITCH + CH_BPF + EDGE_PAD
    ))

    # Y-coordinates for each filter's feedline (from top)
    y_hpf  = BH - EDGE_PAD - 0.5
    y_bpf1 = y_hpf - CH_HPF - PAD_PITCH + 0.5
    y_bpf2 = y_bpf1 - CH_BPF - PAD_PITCH

    print(f"\n  BOARD SIZE: {BW:.0f} x {BH:.0f} mm  ({BW*BH:.0f} mm²)")
    print(f"  HPF trace : {hpf_total:.1f} mm  (y={y_hpf:.1f}mm)")
    print(f"  BPF1 trace: {bpf1_total:.1f} mm  (y={y_bpf1:.1f}mm)")
    print(f"  BPF2 trace: {bpf2_total:.1f} mm  (y={y_bpf2:.1f}mm)")

    # ── Build meander paths (each at its own y) ──────────────────────
    hpf_segs = []
    for k in range(3):
        hpf_segs.append(ll[k]*1e3 if k<len(ll) else 1)
        hpf_segs.append(sl[k]*1e3)
    if len(ll)>3: hpf_segs.append(ll[3]*1e3)

    hpf_pts  = _meander_pts(EDGE_PAD, y_hpf,  hpf_segs,  BW)
    bpf1_pts = _meander_pts(EDGE_PAD, y_bpf1, list(rl1), BW)
    bpf2_pts = _meander_pts(EDGE_PAD, y_bpf2, list(rl2), BW)

    stub_w_mm=[float(_ms_width(jnp.array(float(sz[k]))))*1e3 for k in range(3)]
    mim_pads=[np.sqrt(float(cp[k])*H_CORE/(EPS0*ER_CORE))*1e3 for k in range(2)]

    stub_xs=[]; cum=EDGE_PAD
    for k in range(3):
        cum+=hpf_segs[2*k]; stub_xs.append(min(cum,BW-2))
        cum+=hpf_segs[2*k+1] if 2*k+1<len(hpf_segs) else 0

    # ── Castellated pad positions (separate per filter) ──────────────
    # Left edge: GND, HPF_IN, BPF1_IN, BPF2_IN, GND
    # Right edge: GND, HPF_OUT, BPF1_OUT, BPF2_OUT, GND
    pads_L = [(0, BH-1.0, "GND"), (0, y_hpf, "HPF"),
              (0, y_bpf1, "BPF1"), (0, y_bpf2, "BPF2"), (0, 1.0, "GND")]
    pads_R = [(BW, BH-1.0, "GND"), (BW, y_hpf, "HPF"),
              (BW, y_bpf1, "BPF1"), (BW, y_bpf2, "BPF2"), (BW, 1.0, "GND")]
    all_pads = pads_L + pads_R
    # Exclusion zones for via fence around ALL pads
    excl = [(px, py, 1.8) for px, py, _ in all_pads]

    def _draw_all_pads(ax, layer_filter=None):
        """Draw castellated pads, labelling signal pads."""
        for px, py, label in all_pads:
            is_gnd = label == "GND"
            col = "#27ae60" if is_gnd else SOLDER
            _draw_castellated(ax, px, py)
            if not is_gnd:
                side = "left" if px < BW/2 else "right"
                tx = px + (1.2 if side=="left" else -1.2)
                ha = "left" if side=="left" else "right"
                fs = 5
                if layer_filter and label == layer_filter:
                    ax.text(tx, py, label, fontsize=fs, ha=ha, va="center",
                            color="#e74c3c", fontweight="bold")
                else:
                    ax.text(tx, py, label, fontsize=fs, ha=ha, va="center", color="#888")

    # ── Render all 6 layers ──────────────────────────────────────────
    fig=plt.figure(figsize=(21,14))
    fig.suptitle(f"SMT Filter Module — {BW:.0f} x {BH:.0f} mm  |  6-Layer Copper Artwork",
                 fontsize=14, fontweight="bold")

    # L1 — HPF signal
    ax=_layer_ax(fig,(2,3,1),f"L1 — HPF Signal",BW,BH)
    _draw_trace(ax,hpf_pts,W50_MM)
    for k in range(3):
        sx=stub_xs[k]; sy_end=y_hpf-sl[k]*1e3
        _draw_trace(ax,[(sx,y_hpf),(sx,sy_end)],stub_w_mm[k],color="#d4a040")
        _draw_via(ax,sx,sy_end-0.3)
    for k in range(2):
        cx=stub_xs[k]+(stub_xs[min(k+1,2)]-stub_xs[k])*0.5
        cx=min(max(cx,2),BW-2); ps=mim_pads[k]
        ax.add_patch(Rectangle((cx-ps/2,y_hpf-ps/2),ps,ps,fc="#e8d080",ec=CU_DARK,lw=0.4,zorder=4))
    _draw_all_pads(ax, "HPF")
    _draw_via_fence(ax,BW,BH,excl)

    # L2 — Ground
    ax=_layer_ax(fig,(2,3,2),f"L2 — Ground (HPF ref)",BW,BH)
    voids=[]
    for k in range(2):
        cx=stub_xs[k]+(stub_xs[min(k+1,2)]-stub_xs[k])*0.5
        cx=min(max(cx,2),BW-2); ps=mim_pads[k]+0.3
        voids.append((cx-ps/2,y_hpf-ps/2,ps,ps))
    for sx in stub_xs:
        voids.append((sx-0.3,y_hpf-sl.max()*1e3-0.8,0.6,0.6))
    for px,py,lb in all_pads:
        if lb != "GND":
            voids.append((px-0.4 if px>0 else -0.1, py-0.4, 0.8, 0.8))
    _draw_gnd_pour(ax,BW,BH,voids); _draw_via_fence(ax,BW,BH,excl)
    _draw_all_pads(ax)

    # L3 — BPF1 signal
    ax=_layer_ax(fig,(2,3,3),f"L3 — BPF1 (2.5-3.75 GHz)",BW,BH)
    _draw_trace(ax,bpf1_pts,SL_W50_MM,color="#3498db")
    for k in range(N_BPF):
        pt=bpf1_pts[min(k+1,len(bpf1_pts)-1)]
        ax.plot(pt[0],pt[1],"o",color="#e74c3c",ms=3,zorder=5)
    _draw_via(ax,EDGE_PAD,y_bpf1); _draw_via(ax,BW-EDGE_PAD,y_bpf1)
    _draw_all_pads(ax, "BPF1")
    _draw_via_fence(ax,BW,BH,excl)

    # L4 — Ground
    ax=_layer_ax(fig,(2,3,4),f"L4 — Ground (shared)",BW,BH)
    gnd_voids = [(px-0.4 if px>0 else -0.1, py-0.4, 0.8, 0.8) for px,py,lb in all_pads if lb!="GND"]
    _draw_gnd_pour(ax,BW,BH,gnd_voids); _draw_via_fence(ax,BW,BH,excl)
    _draw_all_pads(ax)

    # L5 — BPF2 signal
    ax=_layer_ax(fig,(2,3,5),f"L5 — BPF2 (3.75-5.0 GHz)",BW,BH)
    _draw_trace(ax,bpf2_pts,SL_W50_MM,color="#9b59b6")
    for k in range(N_BPF):
        pt=bpf2_pts[min(k+1,len(bpf2_pts)-1)]
        ax.plot(pt[0],pt[1],"o",color="#e74c3c",ms=3,zorder=5)
    _draw_via(ax,EDGE_PAD,y_bpf2); _draw_via(ax,BW-EDGE_PAD,y_bpf2)
    _draw_all_pads(ax, "BPF2")
    _draw_via_fence(ax,BW,BH,excl)

    # L6 — Ground
    ax=_layer_ax(fig,(2,3,6),f"L6 — Ground (bottom)",BW,BH)
    _draw_gnd_pour(ax,BW,BH,gnd_voids); _draw_via_fence(ax,BW,BH,excl)
    _draw_all_pads(ax)

    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig("shape_copper_layers.png",dpi=200); plt.close(fig)
    generated.append("shape_copper_layers.png")

    # ── Individual layer PNGs ────────────────────────────────────────
    layer_info = [
        ("L1 — HPF Signal","shape_L1.png","hpf"),
        ("L2 — Ground","shape_L2.png","gnd_hpf"),
        ("L3 — BPF1","shape_L3.png","bpf1"),
        ("L4 — Ground","shape_L4.png","gnd"),
        ("L5 — BPF2","shape_L5.png","bpf2"),
        ("L6 — Ground","shape_L6.png","gnd"),
    ]
    for title,fname,kind in layer_info:
        fig_l,ax_l=plt.subplots(figsize=(7,7*BH/BW))
        ax_l.set_facecolor(SUBSTRATE)
        ax_l.add_patch(Rectangle((-0.3,-0.3),BW+0.6,BH+0.6,fc=SUBSTRATE,ec="#444",lw=2,zorder=0))
        ax_l.set_xlim(-1.5,BW+1.5); ax_l.set_ylim(-1.5,BH+1.5)
        ax_l.set_aspect("equal")
        ax_l.set_title(f"{title}  ({BW:.0f}x{BH:.0f}mm)",fontsize=11,fontweight="bold")
        ax_l.set_xlabel("mm"); ax_l.set_ylabel("mm")

        if kind=="hpf":
            _draw_trace(ax_l,hpf_pts,W50_MM)
            for k in range(3):
                sx=stub_xs[k]; se=y_hpf-sl[k]*1e3
                _draw_trace(ax_l,[(sx,y_hpf),(sx,se)],stub_w_mm[k],color="#d4a040")
                _draw_via(ax_l,sx,se-0.3)
            for k in range(2):
                cx=stub_xs[k]+(stub_xs[min(k+1,2)]-stub_xs[k])*0.5
                cx=min(max(cx,2),BW-2); ps=mim_pads[k]
                ax_l.add_patch(Rectangle((cx-ps/2,y_hpf-ps/2),ps,ps,fc="#e8d080",ec=CU_DARK,lw=0.4,zorder=4))
            _draw_all_pads(ax_l,"HPF"); _draw_via_fence(ax_l,BW,BH,excl)
        elif kind=="gnd_hpf":
            _draw_gnd_pour(ax_l,BW,BH,voids); _draw_via_fence(ax_l,BW,BH,excl)
            _draw_all_pads(ax_l)
        elif kind=="bpf1":
            _draw_trace(ax_l,bpf1_pts,SL_W50_MM,color="#3498db")
            for k in range(N_BPF):
                pt=bpf1_pts[min(k+1,len(bpf1_pts)-1)]
                ax_l.plot(pt[0],pt[1],"o",color="#e74c3c",ms=4,zorder=5)
            _draw_via(ax_l,EDGE_PAD,y_bpf1); _draw_via(ax_l,BW-EDGE_PAD,y_bpf1)
            _draw_all_pads(ax_l,"BPF1"); _draw_via_fence(ax_l,BW,BH,excl)
        elif kind=="bpf2":
            _draw_trace(ax_l,bpf2_pts,SL_W50_MM,color="#9b59b6")
            for k in range(N_BPF):
                pt=bpf2_pts[min(k+1,len(bpf2_pts)-1)]
                ax_l.plot(pt[0],pt[1],"o",color="#e74c3c",ms=4,zorder=5)
            _draw_via(ax_l,EDGE_PAD,y_bpf2); _draw_via(ax_l,BW-EDGE_PAD,y_bpf2)
            _draw_all_pads(ax_l,"BPF2"); _draw_via_fence(ax_l,BW,BH,excl)
        else:
            _draw_gnd_pour(ax_l,BW,BH,gnd_voids); _draw_via_fence(ax_l,BW,BH,excl)
            _draw_all_pads(ax_l)

        fig_l.tight_layout()
        fig_l.savefig(fname,dpi=200); plt.close(fig_l)
        generated.append(fname)

    # ── Response plots ───────────────────────────────────────────────
    for name,s21,s11,vl,pref in [
        ("HPF fc=2.5GHz",s21h,s11h,[2.5],"hpf"),
        ("BPF1 2.5-3.75GHz",s21b1,s11b1,[2.5,3.75],"bpf1"),
        ("BPF2 3.75-5.0GHz",s21b2,s11b2,[3.75,5.0],"bpf2"),
    ]:
        fig_r,ax_r=plt.subplots(figsize=(8,5))
        ax_r.plot(fg,db(s21),"b",lw=1.6,label="|S21|")
        ax_r.plot(fg,db(s11),"r--",lw=1,label="|S11|")
        ax_r.set_title(f"{name}  |  {BW:.0f}x{BH:.0f}mm SMT module",fontsize=12,fontweight="bold")
        ax_r.set_xlabel("Frequency [GHz]"); ax_r.set_ylabel("[dB]")
        ax_r.set_ylim(-50,3); ax_r.legend(); ax_r.grid(True,alpha=0.3)
        for fv in vl: ax_r.axvline(fv,color="gray",ls=":",lw=0.8)
        fig_r.tight_layout()
        fn=f"shape_{pref}_response.png"; fig_r.savefig(fn,dpi=200); plt.close(fig_r)
        generated.append(fn)

    # ── Stackup ──────────────────────────────────────────────────────
    fig_s,ax_s=plt.subplots(figsize=(10,5))
    ax_s.set_title(f"6-Layer Stackup — {BW:.0f}x{BH:.0f}mm SMT Module",fontsize=13,fontweight="bold")
    stack=[
        ("L1 HPF signal",35e-6,"#e67e22"),("RO3010 er=10.2 0.254mm",H_CORE,"#f5e6d3"),
        ("L2 Ground",35e-6,"#27ae60"),("RO4450F er=3.52 0.200mm",H_SL,"#ecf0f1"),
        ("L3 BPF1 stripline",18e-6,"#3498db"),("RO4450F er=3.52 0.200mm",H_SL,"#ecf0f1"),
        ("L4 Ground",35e-6,"#27ae60"),("RO4450F er=3.52 0.200mm",H_SL,"#ecf0f1"),
        ("L5 BPF2 stripline",18e-6,"#9b59b6"),("RO4450F er=3.52 0.200mm",H_SL,"#ecf0f1"),
        ("L6 Ground",35e-6,"#27ae60"),
    ]
    y=0
    for label,t,c in reversed(stack):
        h=max(t*1e3,0.015)
        ax_s.add_patch(Rectangle((1,y),8,h,fc=c,ec="#333",lw=0.8))
        ax_s.text(9.3,y+h/2,label,va="center",fontsize=8,fontfamily="monospace")
        y+=h
    ax_s.set_xlim(0,20); ax_s.set_ylim(-0.02,y+0.04)
    ax_s.set_ylabel("mm"); ax_s.set_xticks([])
    ax_s.text(5,y+0.025,f"Total: {TOTAL_H*1e3:.2f}mm  |  Module: {BW:.0f}x{BH:.0f}mm",
              ha="center",fontsize=11,fontweight="bold")
    fig_s.tight_layout()
    fig_s.savefig("shape_stackup.png",dpi=200); plt.close(fig_s)
    generated.append("shape_stackup.png")

    # ── Performance ──────────────────────────────────────────────────
    print("\n"+"="*64)
    print(f"  MODULE: {BW:.0f} x {BH:.0f} mm  ({BW*BH:.0f} mm²)  {TOTAL_H*1e3:.2f} mm thick")
    print("="*64)
    for name,s21,vl,pref in [
        ("HPF fc=2.5GHz",s21h,[2.5,5.5],"hpf"),
        ("BPF1 2.5-3.75GHz",s21b1,[2.5,3.75],"bpf1"),
        ("BPF2 3.75-5.0GHz",s21b2,[3.75,5.0],"bpf2"),
    ]:
        s21db=db(s21); pb=(fg>=vl[0])&(fg<=vl[-1])
        if np.any(pb):
            il_lo=float(-np.max(s21db[pb])); il_hi=float(-np.min(s21db[pb]))
            print(f"  {name:20s}  IL: {il_lo:.1f} - {il_hi:.1f} dB")
    print(f"\n  Castellated edge pads for SMT reflow soldering")
    print(f"  Ground via fence: {VIA_FENCE_PITCH:.1f} mm pitch around perimeter")
    print(f"  50Ω microstrip: {W50_MM:.3f} mm  |  50Ω stripline: {SL_W50_MM:.3f} mm")

    # ── Output ───────────────────────────────────────────────────────
    print("\n"+"="*64)
    print("  OUTPUT IMAGES")
    print("="*64)
    for img in generated:
        sz=os.path.getsize(img)/1024
        print(f"  {img:40s}  {sz:6.0f} KB  OK")
    print("="*64)
    print("  Complete.")
    print("="*64)

if __name__=="__main__":
    main()
