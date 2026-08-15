"""
Genera los .npz del estudio de Fourier.

"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.gridspec import GridSpec
import scipy.signal

import math

# CONFIGURACIÓN CUDA

class FourierFeatures1D(nn.Module):
    """
    1D Fourier features: x -> [sin(2π B x_norm), cos(2π B x_norm), (optional x_norm)]
    x is expected in radians over [0, 2π]. We normalize: x_norm = x / (2π) in [0,1].
    """
    def __init__(self, 
                 n_frequencies=16,
                 mode='rff',              # 'rff' or 'harmonic'
                 include_input=True,      # append x_norm to features
                 sigma=None,              # std for RFF (in cycles over [0,1])
                 f_max=None,              # max harmonic frequency for 'harmonic'
                 learnable=False, 
                 eps=1e-12):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.mode = mode
        self.include_input = include_input
        self.eps = eps

        if mode == 'rff':
            if sigma is None:
                sigma = 10.0  # default bandwidth ~10 cycles over [0,1]
            B = torch.randn(n_frequencies, 1) * float(sigma)  # cycles on [0,1]
        elif mode == 'harmonic':
            if f_max is None:
                f_max = max(16, n_frequencies)
            # choose evenly spaced frequencies in [1, f_max]
            freqs = torch.linspace(1.0, float(f_max), n_frequencies).unsqueeze(1)  # [F,1]
            # print(freqs)
            B = freqs
        else:
            raise ValueError("mode must be 'rff' or 'harmonic'")

        if learnable:
            self.B = nn.Parameter(B)  # trainable
        else:
            self.register_buffer('B', B)  # frozen

    @property
    def out_features(self):
        base = 2 * self.n_frequencies
        return base + (1 if self.include_input else 0)

    def forward(self, x):
        """
        x: [N,1] in radians, domain [0, 2π]
        returns: [N, 2F (+1 if include_input)]
        """
        # normalize x to [0,1] to make 'cycles' in B meaningful
        x_norm = x 
        # [N,1] @ [1,F] -> broadcast multiply to [N,F]
        # we want sin(2π * (B * x_norm))
        arg = (x_norm @ self.B.t())  # [N,F]
        fea = torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)  # [N,2F]
        if self.include_input:
            fea = torch.cat([fea, x_norm], dim=1)
        return fea


class MLP(nn.Module):
    def __init__(self, in_features, width=64, depth=4, out_features=1, activation=nn.Tanh):
        super().__init__()
        layers = []
        fin = in_features
        for _ in range(depth):
            layers += [nn.Linear(fin, width), activation()]
            fin = width
        layers += [nn.Linear(fin, out_features)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ModelWithFeatures(nn.Module):
    def __init__(self, features: nn.Module, core: nn.Module):
        super().__init__()
        self.features = features
        self.core = core

    def forward(self, x):
        phi = self.features(x)      # [N, D_phi]
        u = self.core(phi)          # [N, 1]
        return u


def get_device():
    """Detecta y retorna el dispositivo disponible (CUDA o CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA no disponible. Usando CPU.")
    return device


# FUNCIONES BÁSICAS DE AJUSTE (CPU - scipy)

def model_cos_over_k(x, k, eps=1e-12):
    """
    Compute f(x; k) = cos(k*x)/k with safe handling near k=0.
    k can be real or complex (numpy supports complex).
    """
    if np.abs(k) < eps:
        k = eps + 0j
    return np.cos(k * x) / k




# ENTRENAMIENTO (OPTIMIZADO PARA CUDA)

def train_pinn(k_target, epochs=20000, lr=1e-4, num_interior=128, device='cpu', verbose=True,
               ff_mode='harmonic', n_frequencies=6, ff_learnable=True,use_lbfgs=True, lbfgs_max_iter=1000):
    """
    PINN with Fourier Features for du/dx + sin(k*x) = 0 on x ∈ [0, 2π].
    """
    import torch, numpy as np, math
    if isinstance(device, str):
        device = torch.device(device)

    def f(x):
        return torch.sin(k_target * x)

    def compute_residual(model, x):
        x.requires_grad = True
        u = model(x)                           # uses features internally
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        return u_x + f(x)

    # Fourier features setup (k-aware)
    if ff_mode == 'harmonic':
        # aim to cover the target frequency; a bit above helps
        f_max = max(n_frequencies, int(1.5 * float(k_target)))
        nF   = min(n_frequencies, f_max)  # cap to requested n_frequencies
        ff = FourierFeatures1D(n_frequencies=nF, mode='harmonic', f_max=f_max, include_input=True,
                               learnable=ff_learnable).to(device)
    elif ff_mode == 'rff':
        # sigma in "cycles" over [0,1]; choose around k_target
        sigma = max(5.0, float(k_target))
        ff = FourierFeatures1D(n_frequencies=n_frequencies, mode='rff', sigma=sigma,
                               include_input=True, learnable=ff_learnable).to(device)
    else:
        raise ValueError("ff_mode must be 'harmonic' or 'rff'")

    # Core MLP that consumes Fourier features
    core = MLP(in_features=ff.out_features, width=64, depth=4, out_features=1, activation=nn.Tanh).to(device)
    model = ModelWithFeatures(ff, core).to(device)

    # Learning rate scaled for high k (optional)
    lr_eff = lr / (1.0 + 0.03 * float(k_target))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_eff, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=600, min_lr=1e-6
    )

    # k-aware sampling density
    num_interior_eff = num_interior #max(num_interior, int(20 * max(1.0, float(k_target))))
    x_interior = (torch.arange(num_interior_eff, device=device, dtype=torch.float32) * (2*math.pi/num_interior_eff)).reshape(-1, 1)
    x_boundary = torch.tensor([[0.0], [2*math.pi]], device=device)
    u_boundary = torch.tensor([[1.0/k_target], [1.0/k_target]], device=device)

    losses = []
    tolerance = 1e-4
    es_patience = 1500 if k_target > 10 else 800
    es_min_delta = 1e-7
    best_loss = float('inf')
    patience_counter = 0

    # (optional) warmup
    warmup_epochs = max(1, epochs // 20)
    base_lr = lr_eff
    start_lr = base_lr * 0.1
    for pg in optimizer.param_groups:
        pg['lr'] = start_lr

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            a = (epoch + 1) / warmup_epochs
            curr_lr = start_lr + a * (base_lr - start_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = curr_lr

        optimizer.zero_grad()
        residual = compute_residual(model, x_interior)
        loss_interior = torch.mean(residual**2)

        u_pred_boundary = model(x_boundary)
        loss_boundary = (
            torch.mean((u_pred_boundary[0] - u_pred_boundary[-1])**2) +
            torch.mean((u_pred_boundary[0] - u_boundary[0])**2)
        )

        loss = loss_interior + loss_boundary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss.item())

        losses.append(loss.item())

        if verbose and epoch % 200 == 0:
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6e} | LR: {curr_lr:.2e} | Nint: {len(x_interior)}")

        if loss.item() < tolerance:
            if verbose:
                print(f"Converged at epoch {epoch}")
            break

        if loss.item() < best_loss - es_min_delta:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= es_patience:
            if verbose:
                print("Early stopping (patience reached)")
            break
        
    # Stage 2: LBFGS refinement
    
    if use_lbfgs:
        if verbose:
            print("[LBFGS] Refinement phase...")

        # Note: do NOT use LR scheduler here; LBFGS manages step size internally.
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,                 # step size for line search; 1.0 works well in practice
            max_iter=lbfgs_max_iter,
            max_eval=lbfgs_max_iter * 2,
            tolerance_grad=1e-7,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn='strong_wolfe'  # more stable than default
        )

        def closure():
            lbfgs.zero_grad()       # IMPORTANT: zero grads inside the closure
            # Recompute the full loss deterministically
            residual = compute_residual(model, x_interior)
            li = torch.mean(residual**2)

            upb = model(x_boundary)
            lb = (
                torch.mean((upb[0] - upb[-1])**2) +
                torch.mean((upb[0] - u_boundary[0])**2)
            )

            total = li + lb
            total.backward()
            return total

        # One or a few LBFGS steps; each step may evaluate the closure many times.
        final_loss = lbfgs.step(closure)

        if verbose:
            print(f"[LBFGS] Done. Final LBFGS loss: {float(final_loss):.6e}")

        # Optionally, append final LBFGS loss to history
        losses.append(float(final_loss))


    return model, losses, x_interior.detach().cpu()



# FUNCIONES AUXILIARES

def amplitude_on_cosine_basis(x, y, k):
    """
    Least-squares amplitude of y onto basis cos(k*x).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.cos(k * x)
    A = float(np.dot(c, y)) / (float(np.dot(c, c)) + 1e-20)
    s = np.sin(k * x)
    B = float(np.dot(s, y)) / (float(np.dot(s, s)) + 1e-20)

    cc = np.hypot(A, B)
    dd = np.pi/2 - np.arctan2(-B, A)

    fit = A*np.cos(k*x) + B*np.sin(k*x)
    rho = float(np.linalg.norm(y - fit) / max(np.linalg.norm(y), 1e-300))

    return cc, dd, rho


def k_ratio_metric(x, y, k_true, clamp_to_unit=False, squash=False):
    """
    Returns M = (1/k_true) / a_y
    """
    cc, dd, rho = amplitude_on_cosine_basis(x, y, k_true)
    M1 = (1.0 / k_true) / cc
    M = M1*np.sin(dd)
    G = -M1*np.cos(dd)

    if clamp_to_unit:
        return float(np.clip(M, 0.0, 1.0))
    if squash:
        return float(M / (1.0 + M))
    return float(M), float(G), rho


# ANÁLISIS DE k_eff (CON TRANSFERENCIA GPU→CPU)

def analyze_keff(model, k_true, x_interior_training, x_test_points=100, device='cpu'):
    """
    Analyze k_eff by fitting model output
    VERSIÓN CUDA: Maneja transferencias GPU ↔ CPU eficientemente
    
    Parameters
    ----------
    model : PINN
        Modelo entrenado (puede estar en GPU)
    k_true : float
        Número de onda verdadero
    x_interior_training : torch.Tensor
        Puntos interiores (en CPU)
    x_test_points : int
        Número de puntos de evaluación
    device : str or torch.device
        Dispositivo del modelo
    
    Returns
    -------
    gg : float
        Métrica k_ratio
    """
    
    if isinstance(device, str):
        device = torch.device(device)
    
    # Calcular k_max del espaciado de puntos interiores
    x_interior_np = x_interior_training.detach().cpu().numpy().flatten()
    dx = x_interior_np[1] - x_interior_np[0]
    k_max = np.pi / dx
    
    # Evaluate PINN (en GPU si está disponible)
    x_test = (torch.arange(x_test_points, device=device, dtype=torch.float32) * (2*np.pi/x_test_points)).reshape(-1, 1)
    
    # Evaluación en GPU
    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy().reshape(-1)
    
    x_test_np = x_test.cpu().numpy().reshape(-1)
    
    # Análisis en CPU (numpy)
    y = np.asarray(u_pred, float)
    x = np.arange(len(y)) * (2*np.pi/len(y))
    gg = k_ratio_metric(x, y, k_true)
    plt.plot(x,y)
    plt.plot(x, 1/k_true*(np.cos(k_true*x)))
    return gg


# MAIN - VERSIÓN CUDA

if __name__ == '__main__':
    
    print("="*80)
    print("NÚMERO DE ONDA MODIFICADA NORMALIZADA PARA PINNs - VERSIÓN CUDA")
    print("="*80)
    
    # DETECTAR DISPOSITIVO
    device = get_device()
    
    # CONFIGURACIÓN
    num_interior = 128
    
    x_interior_dummy = np.arange(num_interior) * (2*np.pi/num_interior)
    dx = x_interior_dummy[1] - x_interior_dummy[0]
    k_max = np.pi / dx
    
    print(f"\n[0] CONFIGURACIÓN DEL SISTEMA")
    print("-" * 80)
    print(f"Dispositivo: {device}")
    print(f"Dominio: [0, 2π]")
    print(f"Número de puntos interiores: {num_interior}")
    print(f"Espaciado: dx = {dx:.6e}")
    print(f"k_max (Nyquist) = π/dx = {k_max:.6f}")
    print(f"Rango resoluble: k ∈ [1, {k_max:.2f}]")
    
    # Crear valores de k
    k_max_safe = 1.0 * k_max
    k_valores = np.arange(1, k_max_safe, 2, dtype=float)
    # k_valores = [10.0]
    print(f"\nValores de k a evaluar:")
    print(f"  {k_valores}")
    
    epochs = 2000000
    lr = 0.0001
    
    print(f"\n[1] Entrenando modelos PINN para k = {list(np.round(k_valores, 2))}")
    print("-" * 80)
    
    # ENTRENAR MÚLTIPLES MODELOS (EN GPU)
    models = {}
    x_interiors = {}
    losses_history = {}
    
    for k in k_valores:
        print(f"\nk = {k:.4f}:")
        
        # Entrenar en GPU
        model, losses, x_interior = train_pinn(
            k_target=k,
            epochs=epochs,
            lr=lr,
            num_interior=num_interior,
            device=device,  # ← USAR GPU
            verbose=True
        )
        
        models[k] = model
        x_interiors[k] = x_interior
        losses_history[k] = losses
        
        plt.plot(x_interior.detach().numpy(), model(x_interior).detach().numpy())
        
        # Liberar memoria GPU si es necesario
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # ANALIZAR k_eff
    print("\n[2] Analizando k_eff")
    print("-" * 80)
    
    results = []
    
    for k in k_valores:
        print(f"\nAnalizando k = {k:.4f}:")
        
        # Análisis (modelo en GPU, resultados en CPU)
        result = analyze_keff(
            models[k],
            k_true=k,
            x_interior_training=x_interiors[k],
            x_test_points=128,
            device=device  # ← MODELO EN GPU
        )
        
        results.append(result)
        print(f"  Re = {result[0]:.6f}  Im = {result[1]:.6f}  rho = {result[2]:.3e}")
    
    # VISUALIZACIÓN
    
    k_n = k_valores / k_max
    results_n = np.array(results)[:, 0] * k_n
    imag_n = np.array(results)[:, 1] * k_n
    rho_n = np.array(results)[:, 2]
    # error = np.abs(results_n - k_n) / k_n * 100
    
    np.savez('prueba_4_64_f6_{}'.format(np.random.randint(0,10**9)) ,results=results_n, k_max=k_max, models=models,epochs=epochs,lr=lr)
    plt.plot(k_n,results_n)
    plt.plot(k_n,k_n)
    plt.ylim([-0.5,1.5])
