"""
  - Exact:      u(x) = cos(x)
  - Dispersed:  u(x) = cos(0.8 x)     (deficit in Re[kef] -> phase lag)
  - Dissipated: u(x) = 0.65 cos(x)    (Im[kef] < 0 -> amplitude decay)
"""
import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(0.0, 2*np.pi, 600)
exact      = np.cos(x)
dispersed  = np.cos(0.8*x)
dissipated = 0.65*np.cos(x)
 
fig, ax = plt.subplots(figsize=(13, 6.36))
 
ax.plot(x, exact,      color='green', linestyle='-',  linewidth=4.0, label='Exact ($k$)')
ax.plot(x, dispersed,  color='blue',  linestyle='--', linewidth=2.8, label='Dispersed (phase lag)')
ax.plot(x, dissipated, color='red',   linestyle='-.', linewidth=2.8, label='Dissipated (damped)')
 
ax.set_title('Modified Wavenumber Effects', fontsize=26, pad=14)
ax.set_xlabel('$x$', fontsize=24)
ax.set_ylabel('$u(x)$', fontsize=24)
ax.tick_params(axis='both', labelsize=19)
ax.grid(True, color='0.85', linewidth=1.0)
ax.legend(loc='lower left', fontsize=19, framealpha=1.0)
 
fig.tight_layout()
fig.savefig('modified_wavenumber_effects.png', dpi=300)
print("saved")
 
