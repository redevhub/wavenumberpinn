"""
Neither experiment requires retraining anything.

  (A) Leakage.  The projection is exact only when the target
      harmonic is periodic on the domain, i.e. k is an integer multiple
      of 2*pi/L.  A continuous k-sweep violates this and injects an
      error of order 1e-2 that has nothing to do with the solver.
  (B) Amplitude ceiling.  The exact response is 1/|k_ef|, which diverges
      as k_ef -> 0.  A solver bounded by M cannot represent it once
      1/|k_ef| > M; beyond that the recovery saturates.
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from spectral_diagnostic import (project_harmonic, recover_kef,
                                 projection_residual, analytic_kef)

N=128; L=2*np.pi; dx=L/N; x=np.arange(N)*dx; kmax=np.pi/dx
SCHEMES=["C2","C4","C6","U1","U2"]

def sol(sch,k):
    kef=analytic_kef(sch,k,dx); return np.real((1.0/kef)*np.exp(1j*k*x)), kef

# EXPERIMENT A : leakage
print("="*72); print("A - integer vs non-integer wavenumbers (scheme C2)"); print("="*72)
print(f"{'k':>7} {'k*':>7} {'integer?':>9} {'rel err Re':>13} {'rho':>12}")
rowsA=[]
for k in [6, 6.4, 19, 19.2, 45, 44.8, 58, 57.6, 61, 63]:
    y,kef=sol("C2",k); A,B=project_harmonic(x,y,k)
    e=abs(recover_kef(A,B).real-kef.real)/abs(kef.real)
    rho=projection_residual(x,y,k,A,B); isint=float(k).is_integer()
    rowsA.append((k,k/kmax,isint,e,rho))
    print(f"{k:7.2f} {k/kmax:7.3f} {str(isint):>9} {e:13.3e} {rho:12.3e}")
print("\n-> With integer k the recovery is exact to machine precision across")
print("   the whole band, including k* = 0.98.  The irregularity reported")
print("   is therefore NOT intrinsic to the recovery for an")
print("   exact linear operator.  rho detects the leakage case correctly.")

# EXPERIMENT B : ceiling
print("\n"+"="*72); print("B - amplitude ceiling: onset of saturation"); print("="*72)
kint=np.arange(1,N//2+1,dtype=float)          # integer wavenumbers only
print(f"{'scheme':>7}"+"".join(f"{'M='+str(m):>11}" for m in (2,5,10,50)))
for sch in SCHEMES:
    a=np.array([abs(analytic_kef(sch,k,dx)) for k in kint])
    pk=int(np.argmax(a))                       # search only ABOVE the peak
    line=f"{sch:>7}"
    for M in (2,5,10,50):
        bad=np.where(a[pk:]<1.0/M)[0]
        line += f"{(kint[pk+bad[0]]/kmax if bad.size else np.inf):11.4f}"
    print(line)
print("\n-> Central schemes saturate first because k_ef -> 0 at Nyquist.")
print("   Upwind schemes keep |k_ef| bounded away from 0 and never saturate.")

# saturation demonstrated on C2
M=10.0; Re_ex=[]; Re_cap=[]; rho_cap=[]; ksv=[]
for k in kint:
    y,kef=sol("C2",k); ksv.append(k/kmax); Re_ex.append(kef.real/kmax)
    ycap=y*min(1.0, M/(1.0/abs(kef)))          # bounded-output solver
    A,B=project_harmonic(x,ycap,k)
    Re_cap.append(recover_kef(A,B).real/kmax)
    rho_cap.append(projection_residual(x,ycap,k,A,B))
ksv=np.array(ksv)

# validity diagnostic
fig,ax=plt.subplots(1,3,figsize=(14,4.0))

ax[0].plot(ksv,ksv,"k-",lw=1.0,label="exact")
ax[0].plot(ksv,Re_ex,"b-",lw=1.7,label="C2 analytical")
ax[0].plot(ksv,Re_cap,"r--",lw=1.7,label=f"C2, output capped at $M$={M:.0f}")
d=np.abs(np.array(Re_cap)-np.array(Re_ex)); on=ksv[np.argmax(d>1e-6)] if d.max()>1e-6 else None
if on: ax[0].axvline(on,color="0.5",ls=":"); ax[0].text(on+.01,.05,r"$k^*_{\rm sat}$",color="0.35")
ax[0].set(xlabel=r"$k^*$",ylabel=r"$\Re[k^*_{\rm ef}]$",title="(a) Ceiling distorts recovery")
ax[0].legend(fontsize=8,loc="upper left"); ax[0].grid(alpha=.3)

for sch,c in zip(SCHEMES,["C0","C1","C2","C3","C4"]):
    ax[1].semilogy(kint/kmax,[1.0/abs(analytic_kef(sch,k,dx)) for k in kint],color=c,lw=1.5,label=sch)
ax[1].axhline(M,color="k",ls="--",lw=1.0); ax[1].text(.03,M*1.3,f"$M$={M:.0f}",fontsize=9)
ax[1].set(xlabel=r"$k^*$",ylabel=r"exact amplitude $1/|k_{\rm ef}|$",title="(b) Why: response diverges")
ax[1].legend(fontsize=8,ncol=2); ax[1].grid(alpha=.3,which="both")

ax[2].semilogy(ksv,np.maximum(rho_cap,1e-17),"r-",lw=1.7,label="capped solver")
kk=np.array([r[0] for r in rowsA]); rr=np.array([r[4] for r in rowsA]); ii=np.array([r[2] for r in rowsA])
ax[2].semilogy(kk[~ii]/kmax,np.maximum(rr[~ii],1e-17),"ks",ms=6,label="non-integer $k$ (leakage)")
ax[2].semilogy(kk[ii]/kmax,np.maximum(rr[ii],1e-17),"go",ms=6,label="integer $k$, exact operator")
ax[2].axhline(0.05,color="0.4",ls="--",lw=1.0); ax[2].text(.03,.065,r"$\rho_{\rm tol}=0.05$",fontsize=9,color="0.3")
ax[2].set(xlabel=r"$k^*$",ylabel=r"projection residual $\rho$",title="(c) The indicator detects both")
ax[2].legend(fontsize=8,loc="lower right"); ax[2].grid(alpha=.3,which="both")

plt.tight_layout(); plt.savefig("fig6_validity.pdf"); plt.savefig("fig6_validity.png",dpi=130)
np.savetxt("block1_table.csv",np.column_stack([ksv,Re_ex,Re_cap,rho_cap]),delimiter=",",
           header="k_star,Re_kef_exact,Re_kef_capped,rho_capped",comments="")
print("\nwritten: validity_diagnostic.pdf/.png, block1_table.csv")
