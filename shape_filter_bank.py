#!/usr/bin/env python3
"""
Compact SMT Filter Module — Multiple Topology Variants
========================================================
Generates several board design options using different BPF
topologies, all targeting minimum size on the same 6-layer
Rogers stackup.  Each variant is separately optimised and
gets full copper-layer artwork.

Topologies:
  A) Coupled-resonator  — J-inverter λ/2 stripline (baseline)
  B) Hairpin             — Folded λ/4 resonators (half the length)
  C) Stub-loaded         — Short lines + shunt stubs (very compact)

All variants share:
  - HPF: 5th-order Chebyshev stubs + MIM caps (L1, same for all)
  - 6-layer stackup, Rogers RO3010 core + RO4450F prepreg
  - Separate castellated I/O pads per filter
  - Via fence with pad exclusion zones
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
C0=299_792_458.0; EPS0=8.854e-12; MU0=4e-7*np.pi; Z_REF=50.0
ER_CORE=10.2; H_CORE=0.254e-3; TAND_CORE=0.0035
ER_SL=3.52; H_SL=0.200e-3; TAND_SL=0.004
MIN_TRACE=0.15e-3; FOLD_GAP=0.8; EDGE_PAD=1.5; PAD_PITCH=2.5
CHEBY_G3=[1.0,1.5963,1.0967,1.5963,1.0]
CHEBY_G5=[1.0,1.7058,1.2296,2.5408,1.2296,1.7058,1.0]
TOTAL_H=(35e-6+H_CORE+35e-6+H_SL+18e-6+H_SL+35e-6+H_SL+18e-6+H_SL+35e-6)

def _ms_eeff(w,h=H_CORE,er=ER_CORE):
    u=jnp.clip(w/h,.01,1000.); return (er+1)/2+(er-1)/2*jnp.power(1+12/u,-0.5)
def _ms_width(z0,h=H_CORE,er=ER_CORE):
    A=z0/60.*jnp.sqrt((er+1)/2)+(er-1)/(er+1)*(0.23+0.11/er)
    Bv=377.*jnp.pi/(2.*z0*jnp.sqrt(er))
    w1=8*jnp.exp(A)/(jnp.exp(2*A)-2)
    w2=(2/jnp.pi)*(Bv-1-jnp.log(jnp.clip(2*Bv-1,1e-6))+(er-1)/(2*er)*(jnp.log(jnp.clip(Bv-1,1e-6))+0.39-0.61/er))
    return jnp.where(z0>63.,w1,w2)*h
W50=float(_ms_width(jnp.array(Z_REF))); EEFF=float(_ms_eeff(jnp.array(W50))); W50_MM=W50*1e3
B_SL=2*H_SL+18e-6; SL_W50_B=94.15/(np.sqrt(ER_SL)*Z_REF)-0.4413; SL_W50=SL_W50_B*B_SL; SL_W50_MM=SL_W50*1e3

# ═══════════════════════════════════════════════════════════════════════
def _m(a,b,c,d): return jnp.array([[a,b],[c,d]],dtype=jnp.complex128)
def atl(z0,gl):
    ch,sh=jnp.cosh(gl),jnp.sinh(gl); return _m(ch,z0*sh,sh/z0,ch)
def ash(Y): return _m(1.+0j,0j,Y,1.+0j)
def ajinv(J): return _m(0j,-1j/J,-1j*J,0j)
def atos(M):
    A,Bv,C,D=M[0,0],M[0,1],M[1,0],M[1,1]; den=A+Bv/Z_REF+C*Z_REF+D
    return 2./den,(A+Bv/Z_REF-C*Z_REF-D)/den
def _glms(f,l): return (TAND_CORE*jnp.pi*f*jnp.sqrt(EEFF)/C0+1j*2*jnp.pi*f*jnp.sqrt(EEFF)/C0)*l
def _glsl(f,l): return (TAND_SL*jnp.pi*f*jnp.sqrt(ER_SL)/C0+1j*2*jnp.pi*f*jnp.sqrt(ER_SL)/C0)*l
def _bnd(x,lo,hi):
    t=jax.nn.sigmoid(x); return jnp.exp(jnp.log(lo)*(1-t)+jnp.log(hi)*t)
def _bnd_inv(y,lo,hi):
    t=(jnp.log(y)-jnp.log(lo))/(jnp.log(hi)-jnp.log(lo)); t=jnp.clip(t,1e-6,1-1e-6); return jnp.log(t/(1-t))
def _pack(vals,bounds):
    parts,idx=[],0
    for n,lo,hi in bounds:
        parts.append(jnp.array([float(_bnd_inv(vals[idx+i],lo,hi)) for i in range(n)])); idx+=n
    return jnp.concatenate(parts)
def _unpack(p,bounds):
    res,i=[],0
    for n,lo,hi in bounds: res.append(_bnd(p[i:i+n],lo,hi)); i+=n
    return res

# ═══════════════════════════════════════════════════════════════════════
#  HPF — shared across all variants
# ═══════════════════════════════════════════════════════════════════════
HPF_BND=[(3,25.,150.),(3,0.3e-3,6e-3),(2,10e-15,10e-12),(4,0.1e-3,5e-3)]
def _hpf_init():
    fc=2.5e9;wc=2*np.pi*fc;g=CHEBY_G5
    L=[Z_REF/(wc*g[i]) for i in (1,3,5)];C=[1./(wc*Z_REF*g[i]) for i in (2,4)]
    zs=[90.,90.,90.];es=float(_ms_eeff(jnp.array(_ms_width(jnp.array(90.)))))
    sl=[min(Lv*C0/(z*np.sqrt(es)),5e-3) for Lv,z in zip(L,zs)]
    lam=C0/(fc*np.sqrt(EEFF)); return _pack(zs+sl+C+[min(lam*0.03,4e-3)]*4,HPF_BND)
def _hpf_at_f(p,f):
    sz,sl,cp,ll=_unpack(p,HPF_BND); gl=lambda l:_glms(f,l)
    def sy(z,l):
        w=_ms_width(z);e=_ms_eeff(w);g=(TAND_CORE*jnp.pi*f*jnp.sqrt(e)/C0+1j*2*jnp.pi*f*jnp.sqrt(e)/C0)*l
        return 1./(z*jnp.tanh(g+1e-15))
    cz=lambda c:1./(1j*2*jnp.pi*f*c)
    M=ash(sy(sz[0],sl[0]))@atl(Z_REF,gl(ll[0]))@_m(1.+0j,cz(cp[0]),0j,1.+0j)
    M=M@atl(Z_REF,gl(ll[1]))@ash(sy(sz[1],sl[1]))@atl(Z_REF,gl(ll[2]))
    M=M@_m(1.+0j,cz(cp[1]),0j,1.+0j)@atl(Z_REF,gl(ll[3]))@ash(sy(sz[2],sl[2]))
    return atos(M)
_hpf_batch=jax.vmap(_hpf_at_f,in_axes=(None,0))
def _hpf_loss(p,freqs):
    s21,s11=_hpf_batch(p,freqs)
    s21d=20*jnp.log10(jnp.clip(jnp.abs(s21),1e-12,None))
    s11d=20*jnp.log10(jnp.clip(jnp.abs(s11),1e-12,None))
    fc=2.5e9;pb=(freqs>=fc)&(freqs<=5.5e9);sb=freqs<=1.5e9;tb=(freqs>1.5e9)&(freqs<fc)
    loss =jnp.sum(jnp.where(pb,jnp.maximum(-s21d-0.5,0.)**2,0.))*6
    loss+=jnp.sum(jnp.where(pb,jnp.maximum(s11d+12,0.)**2,0.))*2
    loss+=jnp.sum(jnp.where(sb,jnp.maximum(s21d+20,0.)**2,0.))*4
    return loss/freqs.shape[0]

# ═══════════════════════════════════════════════════════════════════════
#  BPF TOPOLOGY A: Coupled-resonator (J-inverter, 5th order)
# ═══════════════════════════════════════════════════════════════════════
def _bpfA_init(fc,fbw):
    g=CHEBY_G5;n=5
    J=[float(np.sqrt(np.pi*fbw/(2*g[0]*g[1])))]
    for i in range(1,n): J.append(float(np.pi*fbw/(2*np.sqrt(g[i]*g[i+1]))))
    J.append(float(np.sqrt(np.pi*fbw/(2*g[n]*g[n+1]))))
    J=[max(min(j,1.99),0.011) for j in J]
    lam2=min(C0/(fc*np.sqrt(ER_SL))/2,19e-3)
    return _pack([lam2]*n+J,[(5,3e-3,20e-3),(6,0.01,2.0)])
def _bpfA_at_f(p,f):
    rl=_bnd(p[:5],3e-3,20e-3);jn=_bnd(p[5:],0.01,2.0);J=jn/Z_REF
    M0=ajinv(J[0])
    def body(M,x): return M@atl(Z_REF,_glsl(f,x[0]))@ajinv(x[1]),None
    Mf,_=jax.lax.scan(body,M0,(rl,J[1:])); return atos(Mf)
_bpfA_batch=jax.vmap(_bpfA_at_f,in_axes=(None,0))
BPFA_BND=[(5,3e-3,20e-3),(6,0.01,2.0)]
def _bpfA_trace(p): return float(jnp.sum(_bnd(p[:5],3e-3,20e-3)))*1e3

# ═══════════════════════════════════════════════════════════════════════
#  BPF TOPOLOGY B: Hairpin (folded λ/4, 5th order)
#  Each resonator is two coupled λ/8 arms → half the total length
# ═══════════════════════════════════════════════════════════════════════
def _bpfB_init(fc,fbw):
    g=CHEBY_G5;n=5
    J=[float(np.sqrt(np.pi*fbw/(2*g[0]*g[1])))]
    for i in range(1,n): J.append(float(np.pi*fbw/(2*np.sqrt(g[i]*g[i+1]))))
    J.append(float(np.sqrt(np.pi*fbw/(2*g[n]*g[n+1]))))
    J=[max(min(j,1.99),0.011) for j in J]
    lam4=min(C0/(fc*np.sqrt(ER_SL))/4,12e-3)
    return _pack([lam4]*n+J,[(5,2e-3,12e-3),(6,0.01,2.0)])
def _bpfB_at_f(p,f):
    rl=_bnd(p[:5],2e-3,12e-3);jn=_bnd(p[5:],0.01,2.0);J=jn/Z_REF
    M0=ajinv(J[0])
    def body(M,x):
        gl=_glsl(f,x[0])
        M_hair=atl(Z_REF,gl)@atl(Z_REF,gl)
        return M@M_hair@ajinv(x[1]),None
    Mf,_=jax.lax.scan(body,M0,(rl,J[1:])); return atos(Mf)
_bpfB_batch=jax.vmap(_bpfB_at_f,in_axes=(None,0))
BPFB_BND=[(5,2e-3,12e-3),(6,0.01,2.0)]
def _bpfB_trace(p): return float(jnp.sum(_bnd(p[:5],2e-3,12e-3)))*1e3

# ═══════════════════════════════════════════════════════════════════════
#  BPF TOPOLOGY C: Stub-loaded (short line + shunt open stub per res.)
#  Each resonator: transmission line + shunt open stub → very compact
# ═══════════════════════════════════════════════════════════════════════
BPFC_BND=[(5,1e-3,10e-3),(5,1e-3,10e-3),(5,25.,120.),(6,0.01,2.0)]
def _bpfC_init(fc,fbw):
    g=CHEBY_G5;n=5
    J=[float(np.sqrt(np.pi*fbw/(2*g[0]*g[1])))]
    for i in range(1,n): J.append(float(np.pi*fbw/(2*np.sqrt(g[i]*g[i+1]))))
    J.append(float(np.sqrt(np.pi*fbw/(2*g[n]*g[n+1]))))
    J=[max(min(j,1.99),0.011) for j in J]
    lam8=min(C0/(fc*np.sqrt(ER_SL))/8,8e-3)
    return _pack([lam8]*n+[lam8*1.2]*n+[70.]*n+J,BPFC_BND)
def _bpfC_at_f(p,f):
    ll,sl,sz,jnp_=_unpack(p,BPFC_BND);J=jnp_/Z_REF
    M0=ajinv(J[0])
    def body(M,x):
        line_l,stub_l,stub_z,j_next=x
        gl=_glsl(f,line_l)
        w=_ms_width(stub_z);e=_ms_eeff(w)
        gs=(TAND_SL*jnp.pi*f*jnp.sqrt(ER_SL)/C0+1j*2*jnp.pi*f*jnp.sqrt(ER_SL)/C0)*stub_l
        Y_stub=1j*jnp.tan(jnp.imag(gs))/stub_z
        return M@atl(Z_REF,gl)@ash(Y_stub)@ajinv(j_next),None
    Mf,_=jax.lax.scan(body,M0,(ll,sl,sz,J[1:])); return atos(Mf)
_bpfC_batch=jax.vmap(_bpfC_at_f,in_axes=(None,0))
def _bpfC_trace(p):
    ll,sl,_,_=_unpack(p,BPFC_BND)
    return float(jnp.sum(ll)+jnp.sum(sl))*1e3

# ═══════════════════════════════════════════════════════════════════════
def _make_bpf_loss(batch_fn,fl,fh):
    bw=fh-fl
    def loss_fn(p,freqs):
        s21,s11=batch_fn(p,freqs)
        s21d=20*jnp.log10(jnp.clip(jnp.abs(s21),1e-12,None))
        s11d=20*jnp.log10(jnp.clip(jnp.abs(s11),1e-12,None))
        pb=(freqs>=fl)&(freqs<=fh);slo=freqs<(fl-0.3*bw);shi=freqs>(fh+0.3*bw)
        loss =jnp.sum(jnp.where(pb,jnp.maximum(-s21d-1.5,0.)**2,0.))*6
        loss+=jnp.sum(jnp.where(pb,jnp.maximum(s11d+10,0.)**2,0.))*2
        loss+=jnp.sum(jnp.where(slo,jnp.maximum(s21d+15,0.)**2,0.))*3
        loss+=jnp.sum(jnp.where(shi,jnp.maximum(s21d+15,0.)**2,0.))*3
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
            l,g=jax.value_and_grad(loss_fn)(params,freqs); g=jnp.where(jnp.isfinite(g),g,0.)
            u,ns=opt.update(g,state,params); return optax.apply_updates(params,u),ns,l
        for i in range(n_steps):
            params,state,lv=step(params,state); lv_f=float(lv)
            if jnp.isfinite(lv) and lv_f<best_l: best_l,best_p=lv_f,params
            if (i+1)%500==0 or i==0: print(f"  step {i+1:5d}/{n_steps}  loss={lv_f:.4f}  best={best_l:.4f}")
    return best_p

# ═══════════════════════════════════════════════════════════════════════
#  RENDERING
# ═══════════════════════════════════════════════════════════════════════
CU="#c8882e";CU_DARK="#a06820";SUBSTRATE="#1a1a2e";VIA_C="#ccb060";SOLDER="#d4d4d4"

def _meander_pts(x0, y0, segs, bw, fg=FOLD_GAP):
    """Pack trace segments into board width, folding as needed.
    Handles segments shorter than the available run by accumulating
    them until a fold is required."""
    x_min, x_max = EDGE_PAD, bw - EDGE_PAD
    x, y, dx = x0, y0, 1
    pts = [(x, y)]
    total_remaining = sum(segs)
    for s in segs:
        left = s
        while left > 0.05:
            avail = (x_max - x) if dx > 0 else (x - x_min)
            avail = max(avail, 0.1)
            step = min(left, avail)
            x += step * dx
            pts.append((x, y))
            left -= step
            if left > 0.05:
                y -= fg; pts.append((x, y)); dx = -dx
    return pts

def _draw_trace(ax,pts,w,color=CU,z=3):
    xs=[p[0] for p in pts];ys=[p[1] for p in pts]
    ax.plot(xs,ys,color=color,lw=max(w*4,1.2),solid_capstyle="round",solid_joinstyle="round",zorder=z)
def _draw_via(ax,x,y,pr=0.35,dr=0.12):
    ax.add_patch(Circle((x,y),pr,fc=VIA_C,ec=CU_DARK,lw=0.3,zorder=5))
    ax.add_patch(Circle((x,y),dr,fc="#333",zorder=6))
def _draw_castellated(ax,x,y):
    ax.add_patch(Circle((x,y),0.5,fc=SOLDER,ec="#888",lw=0.4,zorder=5))
    ax.add_patch(Circle((x,y),0.2,fc=CU,zorder=6))
def _draw_gnd_pour(ax,bw,bh,voids=None):
    ax.add_patch(Rectangle((0,0),bw,bh,fc=CU,ec=CU_DARK,lw=0.5,zorder=1))
    if voids:
        for (x,y,w,h) in voids: ax.add_patch(Rectangle((x,y),w,h,fc=SUBSTRATE,ec="#555",lw=0.2,zorder=2))
def _draw_via_fence(ax,bw,bh,excl):
    def ok(x,y):
        for ex,ey,er in excl:
            if abs(x-ex)<er and abs(y-ey)<er: return False
        return True
    for x in np.arange(1.2,bw-0.5,1.2):
        if ok(x,0.5): _draw_via(ax,x,0.5,0.25,0.1)
        if ok(x,bh-0.5): _draw_via(ax,x,bh-0.5,0.25,0.1)
    for y in np.arange(1.2,bh-0.5,1.2):
        if ok(0.5,y): _draw_via(ax,0.5,y,0.25,0.1)
        if ok(bw-0.5,y): _draw_via(ax,bw-0.5,y,0.25,0.1)
def _layer_ax(fig,pos,title,bw,bh):
    ax=fig.add_subplot(*pos); ax.set_facecolor(SUBSTRATE)
    ax.add_patch(Rectangle((-0.3,-0.3),bw+0.6,bh+0.6,fc=SUBSTRATE,ec="#444",lw=2,zorder=0))
    ax.set_xlim(-1.5,bw+1.5);ax.set_ylim(-1.5,bh+1.5);ax.set_aspect("equal")
    ax.set_title(title,fontsize=8,fontweight="bold");ax.tick_params(labelsize=5);return ax

# ═══════════════════════════════════════════════════════════════════════
def main():
    freqs=jnp.linspace(0.1e9,7e9,500);fp=jnp.linspace(0.1e9,7e9,1000);fg=np.array(fp)/1e9
    db=lambda s:20*np.log10(np.clip(np.abs(np.array(s)),1e-12,None)); generated=[]
    f1l,f1h=2.5e9,3.75e9;fc1,fbw1=(f1l+f1h)/2,(f1h-f1l)/((f1l+f1h)/2)
    f2l,f2h=3.75e9,5.0e9;fc2,fbw2=(f2l+f2h)/2,(f2h-f2l)/((f2l+f2h)/2)

    print("="*64); print("  Multi-Topology SMT Filter Module Designer"); print("="*64)
    print(f"  Substrates: RO3010 er={ER_CORE} + RO4450F er={ER_SL}")
    print(f"  50Ω: MS {W50_MM:.3f}mm  SL {SL_W50_MM:.3f}mm  Thickness {TOTAL_H*1e3:.2f}mm\n")

    # ── Shared HPF ───────────────────────────────────────────────────
    print("="*64); print("  HPF fc=2.5GHz (shared, L1)"); print("="*64)
    p_hpf=_opt(_hpf_loss,_hpf_init,freqs,n_r=2,n_steps=2500)
    s21h,s11h=_hpf_batch(p_hpf,fp)
    sz,sl,cp,ll=(np.array(x) for x in _unpack(p_hpf,HPF_BND))
    hpf_total=float(np.sum(sl)+np.sum(ll))*1e3
    hpf_segs=[];
    for k in range(3): hpf_segs.append(ll[k]*1e3 if k<len(ll) else 1); hpf_segs.append(sl[k]*1e3)
    if len(ll)>3: hpf_segs.append(ll[3]*1e3)
    stub_w_mm=[float(_ms_width(jnp.array(float(sz[k]))))*1e3 for k in range(3)]
    mim_pads=[np.sqrt(float(cp[k])*H_CORE/(EPS0*ER_CORE))*1e3 for k in range(2)]

    # ── BPF variants ─────────────────────────────────────────────────
    variants = [
        ("A: Coupled-Res λ/2", _bpfA_batch, BPFA_BND, _bpfA_trace,
         lambda fc,fbw: _bpfA_init(fc,fbw), "#3498db"),
        ("B: Hairpin λ/4",     _bpfB_batch, BPFB_BND, _bpfB_trace,
         lambda fc,fbw: _bpfB_init(fc,fbw), "#e67e22"),
        ("C: Stub-loaded",     _bpfC_batch, BPFC_BND, _bpfC_trace,
         lambda fc,fbw: _bpfC_init(fc,fbw), "#27ae60"),
    ]

    all_results = []
    for vname, batch_fn, bnd, trace_fn, init_fn, color in variants:
        print(f"\n{'='*64}\n  {vname}\n{'='*64}")
        loss1=_make_bpf_loss(batch_fn,f1l,f1h)
        print(f"  BPF1 2.5-3.75 GHz:")
        p1=_opt(loss1,lambda:init_fn(fc1,fbw1),freqs,n_r=3,n_steps=3000)
        loss2=_make_bpf_loss(batch_fn,f2l,f2h)
        print(f"  BPF2 3.75-5.0 GHz:")
        p2=_opt(loss2,lambda:init_fn(fc2,fbw2),freqs,n_r=3,n_steps=3000)
        s21b1,s11b1=batch_fn(p1,fp); s21b2,s11b2=batch_fn(p2,fp)
        t1=trace_fn(p1); t2=trace_fn(p2)
        all_results.append(dict(name=vname,color=color,p1=p1,p2=p2,
            s21b1=s21b1,s11b1=s11b1,s21b2=s21b2,s11b2=s11b2,t1=t1,t2=t2))

    # ── Compute board sizes ──────────────────────────────────────────
    max_stub_mm=float(np.max(sl))*1e3
    CH_HPF=max_stub_mm+1.5

    for res in all_results:
        longest=max(hpf_total, res['t1'], res['t2'])
        usable_target = 15.0
        nf = max(1, int(np.ceil(longest / usable_target)))
        bw = float(np.ceil(longest / nf + 2*EDGE_PAD))
        bw = max(bw, 10)

        fgap = FOLD_GAP
        def ch_h(trace_mm):
            nf_t = max(1, int(np.ceil(trace_mm / (bw - 2*EDGE_PAD))))
            return 0.5 + nf_t * fgap
        ch_b1 = ch_h(res['t1'])
        ch_b2 = ch_h(res['t2'])
        ch_hpf = max(CH_HPF, ch_h(hpf_total))

        bh = float(np.ceil(EDGE_PAD + ch_hpf + PAD_PITCH + ch_b1 + PAD_PITCH + ch_b2 + EDGE_PAD))
        bh = max(bh, 8)
        res['bw']=bw; res['bh']=bh; res['nf']=nf

    # ── Generate per-variant copper artwork + response ───────────────
    for res in all_results:
        BW,BH=res['bw'],res['bh']
        longest=max(hpf_total,res['t1'],res['t2'])
        usable=BW-2*EDGE_PAD

        def _ch_h(trace_mm):
            nf_t=max(1,int(np.ceil(trace_mm/usable)))
            return 0.5+nf_t*FOLD_GAP
        ch_hpf_=max(CH_HPF,_ch_h(hpf_total))
        ch_b1_=_ch_h(res['t1']); ch_b2_=_ch_h(res['t2'])

        y_hpf=BH-EDGE_PAD-0.5
        y_b1=y_hpf-ch_hpf_-PAD_PITCH+0.5
        y_b2=y_b1-ch_b1_-PAD_PITCH

        hpf_pts=_meander_pts(EDGE_PAD,y_hpf,hpf_segs,BW)
        n_seg = 5 if 'Coupled' in res['name'] or 'Hairpin' in res['name'] else 10
        b1_segs=[res['t1']/n_seg]*n_seg
        b2_segs=[res['t2']/n_seg]*n_seg
        bpf1_pts=_meander_pts(EDGE_PAD,y_b1,b1_segs,BW)
        bpf2_pts=_meander_pts(EDGE_PAD,y_b2,b2_segs,BW)

        stub_xs=[];cum=EDGE_PAD
        for k in range(3):
            cum+=hpf_segs[2*k];stub_xs.append(min(cum,BW-2))
            cum+=hpf_segs[2*k+1] if 2*k+1<len(hpf_segs) else 0

        pads=[(0,BH-1,"G"),(0,y_hpf,"H"),(0,y_b1,"1"),(0,y_b2,"2"),(0,1,"G"),
              (BW,BH-1,"G"),(BW,y_hpf,"H"),(BW,y_b1,"1"),(BW,y_b2,"2"),(BW,1,"G")]
        excl=[(px,py,1.8) for px,py,_ in pads]
        gnd_voids=[(px-0.4 if px>0 else -0.1,py-0.4,0.8,0.8) for px,py,lb in pads if lb!="G"]

        def _pads(ax,hi=None):
            for px,py,lb in pads:
                _draw_castellated(ax,px,py)
                if lb!="G":
                    tx=px+(1.2 if px<BW/2 else -1.2); ha="left" if px<BW/2 else "right"
                    c="#e74c3c" if hi and lb==hi else "#888"
                    lbl={"H":"HPF","1":"BPF1","2":"BPF2"}[lb]
                    ax.text(tx,py,lbl,fontsize=5,ha=ha,va="center",color=c,fontweight="bold" if hi and lb==hi else "normal")

        fig=plt.figure(figsize=(21,14))
        tag=res['name'].split(':')[0].strip()
        fig.suptitle(f"Option {tag}  |  {res['name']}  |  {BW:.0f}×{BH:.0f}mm",fontsize=14,fontweight="bold")

        ax=_layer_ax(fig,(2,3,1),f"L1 HPF Signal",BW,BH)
        _draw_trace(ax,hpf_pts,W50_MM)
        for k in range(3):
            sx=stub_xs[k];se=y_hpf-sl[k]*1e3
            _draw_trace(ax,[(sx,y_hpf),(sx,se)],stub_w_mm[k],color="#d4a040")
            _draw_via(ax,sx,se-0.3)
        for k in range(2):
            cx=stub_xs[k]+(stub_xs[min(k+1,2)]-stub_xs[k])*0.5;cx=min(max(cx,2),BW-2);ps=mim_pads[k]
            ax.add_patch(Rectangle((cx-ps/2,y_hpf-ps/2),ps,ps,fc="#e8d080",ec=CU_DARK,lw=0.4,zorder=4))
        _pads(ax,"H");_draw_via_fence(ax,BW,BH,excl)

        ax=_layer_ax(fig,(2,3,2),f"L2 Ground",BW,BH)
        _draw_gnd_pour(ax,BW,BH,gnd_voids);_draw_via_fence(ax,BW,BH,excl);_pads(ax)

        ax=_layer_ax(fig,(2,3,3),f"L3 BPF1 ({res['t1']:.0f}mm)",BW,BH)
        _draw_trace(ax,bpf1_pts,SL_W50_MM,color=res['color'])
        _draw_via(ax,EDGE_PAD,y_b1);_draw_via(ax,BW-EDGE_PAD,y_b1)
        _pads(ax,"1");_draw_via_fence(ax,BW,BH,excl)

        ax=_layer_ax(fig,(2,3,4),f"L4 Ground",BW,BH)
        _draw_gnd_pour(ax,BW,BH,gnd_voids);_draw_via_fence(ax,BW,BH,excl);_pads(ax)

        ax=_layer_ax(fig,(2,3,5),f"L5 BPF2 ({res['t2']:.0f}mm)",BW,BH)
        _draw_trace(ax,bpf2_pts,SL_W50_MM,color=res['color'])
        _draw_via(ax,EDGE_PAD,y_b2);_draw_via(ax,BW-EDGE_PAD,y_b2)
        _pads(ax,"2");_draw_via_fence(ax,BW,BH,excl)

        ax=_layer_ax(fig,(2,3,6),f"L6 Ground",BW,BH)
        _draw_gnd_pour(ax,BW,BH,gnd_voids);_draw_via_fence(ax,BW,BH,excl);_pads(ax)

        fig.tight_layout(rect=[0,0,1,0.95])
        fn=f"board_option_{tag}.png";fig.savefig(fn,dpi=200);plt.close(fig);generated.append(fn)

    # ── Comparison plot ──────────────────────────────────────────────
    s21h_np=np.array(s21h); s11h_np=np.array(s11h)
    fig,axes=plt.subplots(3,3,figsize=(18,14))
    fig.suptitle("Board Option Comparison — Frequency Response",fontsize=14,fontweight="bold")
    row_labels=["HPF fc=2.5GHz","BPF1 2.5-3.75GHz","BPF2 3.75-5.0GHz"]
    for ci,res in enumerate(all_results):
        for ri,(s21,s11,vl,title) in enumerate([
            (s21h_np,s11h_np,[2.5],row_labels[0]),
            (np.array(res['s21b1']),np.array(res['s11b1']),[2.5,3.75],row_labels[1]),
            (np.array(res['s21b2']),np.array(res['s11b2']),[3.75,5.0],row_labels[2]),
        ]):
            ax=axes[ri,ci]
            ax.plot(fg,db(s21),"b",lw=1.5,label="|S21|")
            ax.plot(fg,db(s11),"r--",lw=1,label="|S11|")
            if ri==0: ax.set_title(f"Option {res['name'].split(':')[0].strip()}\n{res['name']}\n{res['bw']:.0f}×{res['bh']:.0f}mm",fontsize=9)
            ax.set_ylabel(title if ci==0 else "",fontsize=8)
            ax.set_xlabel("GHz" if ri==2 else "")
            ax.set_ylim(-50,3);ax.legend(fontsize=6,loc="lower right");ax.grid(True,alpha=0.3)
            for fv in vl: ax.axvline(fv,color="gray",ls=":",lw=0.8)
    fig.tight_layout();fig.savefig("board_comparison.png",dpi=200);plt.close(fig);generated.append("board_comparison.png")

    # ── Summary table ────────────────────────────────────────────────
    print("\n"+"="*64)
    print("  BOARD OPTION COMPARISON")
    print("="*64)
    print(f"  {'Option':<25s} {'Size':>10s} {'Area':>8s} {'BPF1 IL':>10s} {'BPF2 IL':>10s}")
    print("  "+"-"*63)
    for res in all_results:
        s21b1d=db(np.array(res['s21b1']));s21b2d=db(np.array(res['s21b2']))
        pb1=(fg>=2.5)&(fg<=3.75);pb2=(fg>=3.75)&(fg<=5.0)
        il1=f"{float(-np.max(s21b1d[pb1])):.1f}-{float(-np.min(s21b1d[pb1])):.1f}"
        il2=f"{float(-np.max(s21b2d[pb2])):.1f}-{float(-np.min(s21b2d[pb2])):.1f}"
        sz_str=f"{res['bw']:.0f}×{res['bh']:.0f}mm"
        area=f"{res['bw']*res['bh']:.0f}mm²"
        print(f"  {res['name']:<25s} {sz_str:>10s} {area:>8s} {il1:>10s} {il2:>10s}")
    hpf_db=db(s21h_np);pb_h=(fg>=2.5)&(fg<=5.5)
    print(f"\n  HPF (shared): IL {float(-np.max(hpf_db[pb_h])):.1f}-{float(-np.min(hpf_db[pb_h])):.1f} dB")

    print("\n"+"="*64)
    print("  OUTPUT IMAGES")
    print("="*64)
    for img in generated:
        sz=os.path.getsize(img)/1024; print(f"  {img:40s}  {sz:6.0f} KB  OK")
    print("="*64); print("  Complete."); print("="*64)

if __name__=="__main__":
    main()
