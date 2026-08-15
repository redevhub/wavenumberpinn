import numpy as np
import matplotlib.pyplot 
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})  # Y
def beta_from_kstar(kstar):
    """
    Map normalized wavenumber k* in [0, 1] to beta = k* * pi (i.e., beta in [0, pi]).
    Here beta = k * dx is the usual nondimensional wavenumber.
    """
    return np.pi * kstar

# First-derivative schemes: Fourier symbols
#Symbols G(β) for the discrete derivative operator D are defined,
# such that in Fourier space D ~ G(β), and the effective wavenumber satisfies
#   G(β) = i * k_eff(β)
# For centered schemes, G(β) is purely imaginary (no dissipation).
# For upwind schemes, G(β) is complex (dissipation + dispersion).

def symbol_c2(beta):
    """Second-order central: (u_{j+1} - u_{j-1}) / (2 dx) -> G(β) = i sin(β) / dx.
       Here we return the nondimensional symbol times dx: G(β)*dx = i sin(β)."""
    return 1j * np.sin(beta)

def symbol_c4(beta):
    """Fourth-order central: ( -u_{j+2} + 8 u_{j+1} - 8 u_{j-1} + u_{j-2} ) / (12 dx)
       => G(β)*dx = i * (8 sin β - sin 2β) / 6
    """
    return 1j * (8*np.sin(beta) - np.sin(2*beta)) / 6.0

def symbol_c6(beta):
    """Sixth-order central:
       ( 1/60 u_{j-3} - 3/20 u_{j-2} + 3/4 u_{j-1} - 3/4 u_{j+1} + 3/20 u_{j+2} - 1/60 u_{j+3} ) / dx
       Symbol: G(β)*dx = i * (45 sin β - 9 sin 2β + sin 3β) / 60
    """
    return 1j * (45*np.sin(beta) - 9*np.sin(2*beta) + np.sin(3*beta)) / 30.0

def symbol_u1(beta):
    """First-order upwind (backward difference): (u_j - u_{j-1}) / dx
       G(β)*dx = 1 - e^{-iβ} = (1 - cos β) + i sin β
       This has dissipation (real part) and dispersion (imag part).
    """
    return (1 - np.cos(beta)) + 1j*np.sin(beta)

def symbol_u2(beta):
    """Second-order upwind (backward, biased):
       D u ≈ (3 u_j - 4 u_{j-1} + u_{j-2}) / (2 dx)
       G(β)*dx = (3 - 4 e^{-iβ} + e^{-2iβ}) / 2
    """
    return (3 - 4*np.exp(-1j*beta) + np.exp(-2j*beta)) / 2.0

# Effective (nondimensional) wavenumber k_eff*
# Define k_eff* := (k_eff * dx) / pi = Imag(G(β)*dx) / pi
# (For first derivative, the relevant dispersive content is Im(G).)
# For upwind schemes, the real part corresponds to dissipation.

def k_eff_star_from_symbol(Gdx):
    """Compute nondimensional effective wavenumber k_eff* from symbol times dx."""
    return np.imag(Gdx) / np.pi

def dissipation_star_from_symbol(Gdx):
    """Nondimensional dissipation measure (real part) for upwind schemes."""
    return -np.real(Gdx) / np.pi

# Second-derivative (Laplacian) modified wavenumber
# For the second derivative, the standard 2nd-order central Laplacian has symbol:
#   L(β) = (e^{iβ} - 2 + e^{-iβ}) / dx^2 = - 4 sin^2(β/2) / dx^2
# Often an effective squared wavenumber k_eff^2 = 4 sin^2(β/2) / dx^2 is defined
# Nondimensional form: (k_eff* )^2 = (k_eff^2 * dx^2) / π^2 = 4 sin^2(β/2) / π^2

def laplacian_keff_star(beta):
    """Nondimensional sqrt of modified wavenumber for 2nd-derivative (2nd-order central)."""
    k2_eff_star = 4 * (np.sin(beta/2)**2) / (np.pi**2)
    return np.sqrt(k2_eff_star)

# Plotting
def plot_first_derivative_modified_wavenumber():
    kstar = np.linspace(0, 1, 1001)
    beta = beta_from_kstar(kstar)

    # Symbols (times dx)
    G_c2 = symbol_c2(beta)
    G_c4 = symbol_c4(beta)
    G_c6 = symbol_c6(beta)
    G_u1 = symbol_u1(beta)
    G_u2 = symbol_u2(beta)

    # Dispersive parts (nondimensional effective wavenumber)
    kstar_eff_c2 = k_eff_star_from_symbol(G_c2)
    kstar_eff_c4 = k_eff_star_from_symbol(G_c4)
    kstar_eff_c6 = k_eff_star_from_symbol(G_c6)
    kstar_eff_u1 = k_eff_star_from_symbol(G_u1)
    kstar_eff_u2 = k_eff_star_from_symbol(G_u2)

    # Dissipation (real parts) for upwind schemes
    diss_u1 = dissipation_star_from_symbol(G_u1)
    diss_u2 = dissipation_star_from_symbol(G_u2)

    plt.figure(2)
    # reference exact line: k*_eff = k*
    plt.plot(kstar, kstar, 'k-', lw=1.5, label='Exact')

    # centered schemes (purely dispersive)
    plt.plot(kstar, kstar_eff_c2, 'k--', lw=2, label='C2')
    plt.plot(kstar, kstar_eff_c4, 'b-.', lw=2, label='C4')
    plt.plot(kstar, kstar_eff_c6, 'g:', lw=2.5, label='C6')

    # upwind schemes (show dispersive part)
    plt.plot(kstar, kstar_eff_u1, color='crimson', lw=1.8, label='U1')
    plt.plot(kstar, kstar_eff_u2, color='orange', lw=1.8, label='U2')

    plt.xlabel(r'$k*$')
    plt.ylabel(r'$\Re [k_{ef}^*]$')
    plt.title('Dispersion')
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Optional: plot dissipation (real part) for upwind schemes
    plt.figure(1)
    plt.plot(kstar, diss_u1, color='crimson', lw=1.8, label='U1')
    plt.plot(kstar, diss_u2, color='orange', lw=1.8, label='U2')
    plt.axhline(0.0, color='k', lw=1)
    plt.xlabel(r'$k^*$')
    plt.ylabel(r'$\Im [k_{ef}^*]$')
    plt.title('Dissipation')
    plt.xlim(0.0, 1.0)
    plt.ylim(-1.4, 0.1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

def plot_laplacian_modified_wavenumber():
    kstar = np.linspace(0, 1, 1001)
    beta = beta_from_kstar(kstar)
    kstar_eff_lap = laplacian_keff_star(beta)  # sqrt of nondim k_eff^2

    plt.figure(figsize=(7, 5))
    plt.plot(kstar, kstar, 'k-', lw=1.5, label='Exact (sqrt)')
    plt.plot(kstar, kstar_eff_lap, 'k--', lw=2, label='2nd-order Laplacian')
    plt.xlabel(r'$k* = \beta/\pi$')
    plt.ylabel(r'$\sqrt{(k*_{\mathrm{eff}})^2}$')
    # plt.title('Modified Wavenumber: Second Derivative (Laplacian)')
    plt.ylim(0, 1.05)
    plt.xlim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_first_derivative_modified_wavenumber()
