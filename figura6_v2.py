import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':15,'axes.titlesize':18,'axes.labelsize':17,
                     'legend.fontsize':13,'xtick.labelsize':13,'ytick.labelsize':13})
d = json.load(open('block2_results_v2.json'))
show = [('act_tanh','k--','tanh 4x128'), ('act_tg','r-','tanh-Gaussian'),
        ('F2','b-.','FF F=2'), ('F4','g:','FF F=4')]
def rows(c): return sorted([r for r in d if r['config']==c], key=lambda r:r['k_star'])
fig, ax = plt.subplots(1,2, figsize=(12,4.6))
for cfg,st,lab in show:
    rr = rows(cfg)
    ks=np.array([r['k_star'] for r in rr]); Re=np.array([r['Re_kef_star'] for r in rr])
    rho=np.array([r['rho'] for r in rr]); m=rho<0.05
    ax[0].plot(ks,Re,st,lw=1.4,alpha=0.35)
    ax[0].plot(ks,np.where(m,Re,np.nan),st,lw=3.0,label=lab)
    ax[1].semilogy(ks,np.maximum(rho,1e-4),st,lw=2.8,label=lab)
kr=np.array([r['k_star'] for r in rows('F4')])
ax[0].plot(kr,kr,'k-',lw=2.0,label='Exact')
ax[0].set(xlim=[0,1],ylim=[-0.05,1.0],xlabel=r'$k^*$',ylabel=r'$\Re[k_{ef}^*]$',title='Dispersion (advection)')
ax[0].grid(True,alpha=0.3); ax[0].legend(loc='upper left')
ax[1].axhline(0.05,color='k',ls='--',lw=2.0)
ax[1].text(0.97,0.065,r'$\rho=0.05$',fontsize=12,ha='right')
ax[1].set(xlim=[0,1],xlabel=r'$k^*$',ylabel=r'$\rho$',title='Projection residual (advection)')
ax[1].grid(True,alpha=0.3,which='both'); ax[1].legend(loc='lower right')
fig.tight_layout(); fig.savefig('figura6_v2.pdf'); fig.savefig('figura6_v2.png',dpi=220)
print('ok')
