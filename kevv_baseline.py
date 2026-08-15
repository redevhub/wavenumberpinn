import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from plotBaseline import *





def solve_finite_differences(k_target, num_points=128, scheme='central'):
    """
    Resuelve du/dx = -sin(k*x) usando diferencias finitas
    
    Condición de contorno: u(0) = 1/k, u(2π) = 1/k (periódica)
    
    Parameters
    ----------
    k_target : float
        Número de onda objetivo
    num_points : int
        Número de puntos en la malla
    scheme : str
        'central', 'forward', o 'backward'
    
    Returns
    -------
    dict con solución
    """
    
    # Crear malla
    x = np.linspace(0, 2*np.pi, num_points)
    dx = x[1] - x[0]
    
    # Vector RHS
    f = -np.sin(k_target * x)  # -sin(k*x)
    
    # Matriz sistema: Diferencias finitas central
    # u_x ≈ (u_{i+1} - u_{i-1}) / (2*dx) = f(x_i)
    
    if scheme == 'central':
        # Matriz para diferencias centrales
        # (u_{i+1} - u_{i-1}) / (2*dx) = f(x_i)
        # => u_{i+1} - u_{i-1} = 2*dx*f(x_i)
        
        A = np.zeros((num_points, num_points))
        b = 2 * dx * f
        
        for i in range(num_points):
            i_plus = (i + 1) % num_points   # Periódica
            i_minus = (i - 1) % num_points
            
            A[i, i_plus] = 1
            A[i, i_minus] = -1

    elif scheme == 'central4':
        # 4th-order central:
        # (-u_{i+2} + 8 u_{i+1} - 8 u_{i-1} + u_{i-2}) / (12*dx) = f_i
        # => -u_{i+2} + 8 u_{i+1} - 8 u_{i-1} + u_{i-2} = 12*dx * f_i
        A = np.zeros((num_points, num_points))
        b = 12.0 * dx * f
    
        for i in range(num_points):
            im2 = (i - 2) % num_points
            im1 = (i - 1) % num_points
            ip1 = (i + 1) % num_points
            ip2 = (i + 2) % num_points
    
            A[i, ip2] = -1.0
            A[i, ip1] =  8.0
            A[i, im1] = -8.0
            A[i, im2] =  1.0
    
    elif scheme == 'central6':
        # 6th-order central:
        # (  1/60 u_{i-3} - 3/20 u_{i-2} + 3/4 u_{i-1}
        #  - 3/4  u_{i+1} + 3/20 u_{i+2} - 1/60 u_{i+3} ) / dx = f_i
        # => (1/60)u_{i-3} - (3/20)u_{i-2} + (3/4)u_{i-1}
        #    - (3/4)u_{i+1} + (3/20)u_{i+2} - (1/60)u_{i+3} = dx * f_i
        A = np.zeros((num_points, num_points))
        b = dx * f
    
        for i in range(num_points):
            im3 = (i - 3) % num_points
            im2 = (i - 2) % num_points
            im1 = (i - 1) % num_points
            ip1 = (i + 1) % num_points
            ip2 = (i + 2) % num_points
            ip3 = (i + 3) % num_points
    
            A[i, im3] =  1.0/60.0
            A[i, im2] = -3.0/20.0
            A[i, im1] =  3.0/4.0
            A[i, ip1] = -3.0/4.0
            A[i, ip2] =  3.0/20.0
            A[i, ip3] = -1.0/60.0
   
    elif scheme == 'forward':
        # Diferencias hacia adelante
        # (u_{i+1} - u_i) / dx = f(x_i)
        
        A = np.zeros((num_points, num_points))
        b = dx * f
        
        for i in range(num_points):
            i_plus = (i + 1) % num_points
            A[i, i] = -1
            A[i, i_plus] = 1
    
    elif scheme == 'backward':
        # Diferencias hacia atrás
        # (u_i - u_{i-1}) / dx = f(x_i)
        
        A = np.zeros((num_points, num_points))
        b = dx * f
        
        for i in range(num_points):
            i_minus = (i - 1) % num_points
            A[i, i] = 1
            A[i, i_minus] = -1
    elif scheme == 'upwind2':
        # Esquema upwind de segundo orden (flujo positivo)
        # (3u_i - 4u_{i-1} + u_{i-2}) / (2*dx) = f(x_i)
        
        A = np.zeros((num_points, num_points))
        b = 2 * dx * f
        
        for i in range(num_points):
            i_minus1 = (i - 1) % num_points
            i_minus2 = (i - 2) % num_points
            
            A[i, i] = 3
            A[i, i_minus1] = -4
            A[i, i_minus2] = 1
    else:
        raise ValueError(f"Scheme '{scheme}' not recognized")
    
    # Resolver sistema lineal

    u_fd = np.linalg.solve(A[1:-1,1:-1], b[1:-1])

    u_fd = np.r_[0,u_fd]
    # Normalizar a u(0) = 1/k
    u_fd = u_fd - u_fd[0] + 1/k_target
    u_fd = np.r_[u_fd,u_fd[0]]
    return {
        'x': x,
        'u': u_fd,
        'dx': dx,
        'k_max': np.pi / dx,
        'num_points': num_points,
        'scheme': scheme
    }






def amplitude_on_cosine_basis(x, y, k):
    """
    Least-squares amplitude of y onto basis cos(k*x).
    a_hat = <c, y> / <c, c>, where c = cos(k*x).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.cos(k * x)
    num = float(np.dot(c, y))
    den = float(np.dot(c, c)) + 1e-20
    A = num / den
    
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.sin(k * x)
    num = float(np.dot(c, y))
    den = float(np.dot(c, c)) + 1e-20
    B = num / den   
    
    
    cc = np.hypot(A, B)          # sqrt(A^2 + B^2)
    dd = np.pi/2-np.arctan2(-B, A) 
    # print(cc, dd)
    
    fit = A*np.cos(k*x) + B*np.sin(k*x)
    rho = float(np.linalg.norm(y - fit) / max(np.linalg.norm(y), 1e-300))
    
    return cc, dd, rho

def k_ratio_metric(x, y, k_true, clamp_to_unit=False, squash=False):
    """
    Returns M = (1/k_true) / a_y, where a_y is the LS amplitude of y on cos(k_true*x).
    - If y = (1/k1)*cos(k_true*x), then M = k1/k_true (exact).
    
    Options for [0,1] mapping:
      * clamp_to_unit=True  =>   min(M, 1.0)
      * squash=True         =>   M / (1 + M)   (smoothly maps (0,∞) -> (0,1))
    """
    cc, dd, rho = amplitude_on_cosine_basis(x, y, k_true)  # ~ 1/k1 if y is pure cosine at k_true
    # if np.isclose(a_y, 0.0):
    #     return 0.0  # y has no cosine(k_true x) component
    # M = (1.0 / k_true) / cc  # desired: k1 / k_true if y = (1/k1) cos(k_true x)
    # M *= np.cos(dd)  #"""Nondimensional dissipative measure (real part) for upwind schemes."""
    
    M1 = (1.0 / k_true) / cc  # desired: k1 / k_true if y = (1/k1) cos(k_true x)
    M = M1*np.sin(dd)  #"""Nondimensional dispersion measure (real part) for upwind schemes."""
    G = -M1*np.cos(dd) 
    # Optional range conditioning
    if clamp_to_unit:
        return float(np.clip(M, 0.0, 1.0))
    if squash:
        return float(M / (1.0 + M))  # smooth [0,∞) -> [0,1)
    return float(M), float(G), rho



def analyze_keff(model, k_true, x_test_points=128,scheme='central'):  
    u_pred = solve_finite_differences(k_true,num_points=128,scheme=scheme)['u']
    y = np.asarray(u_pred, float)
    x =np.linspace(0, 2*np.pi, len(y))
    gg =k_ratio_metric(x, y, k_true)
    
    return gg




# MAIN

if __name__ == '__main__':
    plot_first_derivative_modified_wavenumber()
    num_interior = 128
    
    # Calcular k_max de la malla de entrenamiento
    x_interior_dummy = np.linspace(0, 2*np.pi, num_interior)
    dx = x_interior_dummy[1] - x_interior_dummy[0]
    k_max = np.pi / dx
    
    k_max_safe = 1 * k_max
    k_valores = np.r_[np.arange(1, k_max_safe, 3),k_max]  # 10 valores uniformes
    
    results = []
    
    for k in k_valores:
        print(f"\nAnalizando k = {k:.4f}:")
        result = analyze_keff(
            k, 
            k_true=k, 
            x_test_points=128,
            scheme='central'
        )
        results.append(result)
        
    k_n = k_valores/k_max
    results_n = np.array(results)[:,0]*k_n
    plt.figure(2)
    plt.plot(k_n[:-1], results_n[:-1], 'ok')

    results = []
    
    for k in k_valores:
        print(f"\nAnalizando k = {k:.4f}:")
        result = analyze_keff(
            k, 
            k_true=k, 
            x_test_points=128,
            scheme='central4'
        )
        results.append(result)
        
    k_n = k_valores/k_max
    results_n = np.array(results)[:,0]*k_n
    plt.figure(2)
    plt.plot(k_n[:-1], results_n[:-1], 'ob')
 
    results = []
    
    for k in k_valores:
        print(f"\nAnalizando k = {k:.4f}:")
        result = analyze_keff(
            k, 
            k_true=k, 
            x_test_points=128,
            scheme='central6'
        )
        results.append(result)
        
    k_n = k_valores/k_max
    results_n = -np.array(results)[:,0]*k_n
    plt.figure(2)
    plt.plot(k_n[:-1], results_n[:-1], 'og')   
 
    
 
    results = []
    
    for k in k_valores:
        print(f"\nAnalizando k = {k:.4f}:")
        result = analyze_keff(
            k, 
            k_true=k, 
            x_test_points=128,
            scheme='upwind2'
        )
        results.append(result)
        
    k_n = k_valores/k_max
    results_n = np.array(results)[:,1]*k_n
    plt.figure(1)
    plt.plot(k_n[:-1], results_n[:-1], 'o', color='orange')  
 
    
    results_n = np.array(results)[:,0]*k_n
    plt.figure(2)
    plt.plot(k_n[:-1], results_n[:-1], 'o', color='orange')  
    plt.tight_layout()
 
    results = []
    
    for k in k_valores:
        print(f"\nAnalizando k = {k:.4f}:")
        result = analyze_keff(
            k, 
            k_true=k, 
            x_test_points=128,
            scheme='backward'
        )
        results.append(result)
        
    k_n = k_valores/k_max
    results_n = np.array(results)[:,1]*k_n
    plt.figure(1)
    plt.plot(k_n[:-1], results_n[:-1], 'o', color='crimson')

    plt.figure(1)
    plt.tight_layout()
    plt.savefig("dispersion_FD.pdf")

    plt.figure(2)
    plt.tight_layout()
    plt.savefig("dissipation_FD.pdf")

    plt.figure(1)
    plt.savefig("dissipation_FD.png", dpi=300)
    
    plt.figure(2)
    plt.savefig("dispersion_FD.png", dpi=300)

theta = 4*np.sin(k_n)-np.sin(2*np.sin(k_n))

# sol = solve_finite_differences(k_target=2.0, num_points=256, scheme='upwind2')
# x, u = sol['x'], sol['u']
# u_exact = -np.cos(2.0 * x) / 2.0
# err_inf = np.linalg.norm(u - u_exact, ord=np.inf)
# plt.plot(x,u)
# plt.plot(x,u_exact)
