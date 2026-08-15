"""
Recovery of the modified wavenumber from solver output, plus the two
validity indicators.

Depends only on numpy.

Conventions:
    y_hat        = A - i B                     complex response amplitude
    k_ef         = 1 / y_hat = (A + iB)/(A^2+B^2)
    Re[k_ef]     = cos(theta) / r
    Im[k_ef]     = sin(theta) / r
    r            = sqrt(A^2 + B^2)
    theta        = atan2(B, A)

    A dissipative scheme therefore has Im[k_ef] < 0.
    U1:  Im[k_ef] = -(1 - cos(k dx)) / dx
    U2:  Im[k*_ef] -> -4/pi = -1.2732  at the Nyquist limit
"""

import numpy as np

__all__ = [
    "project_harmonic",
    "recover_kef",
    "projection_residual",
    "saturation_threshold",
    "analytic_kef",
    "diagnose",
]


# Projection onto the target harmonic

def project_harmonic(x, y, k, period=2.0 * np.pi):
    """Fourier coefficients of y along cos(kx) and sin(kx).

    Uses the trapezoidal rule on the supplied grid.  If the grid is
    uniform and periodic (last point not duplicating the first), this is
    spectrally accurate for band-limited y.

    Parameters:
    
    x, y : (N,) arrays - collocation points and solver output
    k    : float       - target wavenumber
    period : float     - domain length L (default 2*pi)

    Returns:
    
    A, B : floats
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # close the period so the trapezoidal rule sees a full cycle
    xe = np.append(x, x[0] + period)
    ye = np.append(y, y[0])

    A = (2.0 / period) * np.trapezoid(ye * np.cos(k * xe), xe)
    B = (2.0 / period) * np.trapezoid(ye * np.sin(k * xe), xe)
    return A, B


# Effective wavenumber  

def recover_kef(A, B, k_max=None):
    """Effective wavenumber from the projection coefficients.

    Returns the complex k_ef.  If k_max is given, returns k_ef / k_max.
    """
    r2 = A * A + B * B
    if r2 == 0.0:
        return np.nan + 1j * np.nan
    kef = (A + 1j * B) / r2
    return kef / k_max if k_max is not None else kef


# Indicator 1: harmonic dominance

def projection_residual(x, y, k, A=None, B=None, period=2.0 * np.pi):
    """Relative L2 residual of the single-harmonic projection.

    rho = || y - (A cos kx + B sin kx) ||_2 / || y ||_2

    rho << 1  ->  k_ef is a modified wavenumber in the classical sense.
    rho ~  1  ->  k_ef is a fitted descriptor only; report it as such.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if A is None or B is None:
        A, B = project_harmonic(x, y, k, period)
    fit = A * np.cos(k * x) + B * np.sin(k * x)
    denom = np.linalg.norm(y)
    if denom == 0.0:
        return np.nan
    return float(np.linalg.norm(y - fit) / denom)


# Indicator 2: conditioning / representational ceiling

def saturation_threshold(kef_exact_fn, k_grid, M):
    """Largest wavenumber at which the test problem is representable.

    The exact response amplitude is |y_hat| = 1/|k_ef|.  A solver whose
    output amplitude is bounded by M cannot represent the solution once
    1/|k_ef| > M, i.e. once |k_ef| < 1/M.  Beyond that point the
    recovered amplitude saturates near M and the inferred wavenumber
    saturates near 1/M instead of tending to zero.

    Parameters:
    
    kef_exact_fn : callable k -> complex k_ef  (analytical, for the scheme)
    k_grid       : array of wavenumbers, increasing
    M            : float, amplitude ceiling of the solver

    Returns:
    
    k_sat : float or np.inf  -- first k at which |k_ef| < 1/M
    """
    kef = np.array([abs(kef_exact_fn(k)) for k in k_grid])
    bad = np.where(kef < 1.0 / M)[0]
    return float(k_grid[bad[0]]) if bad.size else np.inf


# Analytical modified wavenumbers (verification reference)

def analytic_kef(scheme, k, dx):
    z = k * dx
    e = np.exp
    if scheme == "C2":
        ik = (e(1j * z) - e(-1j * z)) / (2 * dx)
    elif scheme == "C4":
        ik = (-e(2j * z) + 8 * e(1j * z) - 8 * e(-1j * z) + e(-2j * z)) / (12 * dx)
    elif scheme == "C6":
        ik = (e(3j * z) - 9 * e(2j * z) + 45 * e(1j * z)
              - 45 * e(-1j * z) + 9 * e(-2j * z) - e(-3j * z)) / (60 * dx)
    elif scheme == "U1":
        ik = (1 - e(-1j * z)) / dx
    elif scheme == "U2":
        ik = (3 - 4 * e(-1j * z) + e(-2j * z)) / (2 * dx)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return ik / 1j


# One-call convenience wrapper

def diagnose(x, y, k, k_max, rho_tol=0.05, period=2.0 * np.pi):

    A, B = project_harmonic(x, y, k, period)
    kef = recover_kef(A, B)
    rho = projection_residual(x, y, k, A, B, period)
    r = np.hypot(A, B)
    return {
        "k": k,
        "k_star": k / k_max,
        "A": A,
        "B": B,
        "r": r,
        "theta": np.arctan2(B, A),
        "Re_kef_star": kef.real / k_max,
        "Im_kef_star": kef.imag / k_max,
        "rho": rho,
        "valid": bool(rho < rho_tol),
    }
