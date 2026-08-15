import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from matplotlib.gridspec import GridSpec
import scipy.signal


# CONFIGURACIÓN CUDA

def get_device():
    """Detecta y retorna el dispositivo disponible (CUDA o CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
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

# RED NEURONAL PINN (COMPATIBLE CON CUDA)

class PINN(nn.Module):
    def __init__(self, layers_width=20):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, layers_width),
            nn.Tanh(),
            nn.Linear(layers_width, layers_width),
            nn.Tanh(),
            nn.Linear(layers_width, 1)
        )

    def forward(self, x):
        return self.hidden(x)


# ENTRENAMIENTO (OPTIMIZADO PARA CUDA)

def train_pinn(k_target, epochs=20000, lr=0.001, num_interior=128, device='cpu', verbose=False):
    """
    Train PINN to solve: du/dx + sin(k*x) = 0
    VERSIÓN CUDA: Entrenamiento optimizado en GPU
    
    Parameters
    ----------
    k_target : float
        Número de onda objetivo
    epochs : int
        Épocas de entrenamiento
    lr : float
        Learning rate
    num_interior : int
        Número de puntos interiores
    device : str or torch.device
        Dispositivo ('cpu', 'cuda', o torch.device)
    verbose : bool
        Imprimir durante entrenamiento
    
    Returns
    -------
    model : PINN
        Modelo entrenado
    losses : list
        Historial de pérdidas
    x_interior_training : torch.Tensor
        Puntos interiores (en CPU para compatibilidad)
    """
    
    # Convertir device a torch.device si es string
    if isinstance(device, str):
        device = torch.device(device)
    
    def f(x):
        return torch.sin(k_target * x)

    def compute_residual(model, x):
        x.requires_grad = True
        u = model(x)
        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        residual = u_x + f(x)
        return residual

    # Initialize model y mover a GPU
    model = PINN(layers_width=20).to(device)
    lr_eff = lr / (1.0 + 0.03 * float(k_target))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_eff, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, min_lr=1e-6)
    # Puntos interiores de entrenamiento (en GPU)
    x_interior = torch.linspace(0, 2*np.pi, num_interior, device=device).reshape(-1, 1)
    x_boundary = torch.tensor([[0.0], [2*np.pi]], device=device)
    u_boundary = torch.tensor([[1.0/k_target], [1.0/k_target]], device=device)
    
    losses = []
    tolerance = 0.0001
    # For patience-based
    es_patience = 500        # epochs without improvement
    es_min_delta = 1e-6      # minimum change to count as improvement
    early_stopping_type='patience'
    # For simple-based
    es_tolerance = 0.001     # loss threshold
    best_loss = float('inf')
    patience_counter = 0
    # For residual-based
    residual_tolerance = 0.01  # residual norm threshold
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Interior loss
        residual = compute_residual(model, x_interior)
        loss_interior = torch.mean(residual**2)
        
        # Boundary loss
        u_pred_boundary = model(x_boundary)
        loss_boundary = (
            torch.mean((u_pred_boundary[0] - u_pred_boundary[-1])**2) +
            torch.mean((u_pred_boundary[0] - u_boundary[0])**2)
        )
        
        # Total loss
        loss = loss_interior + loss_boundary
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        # Guardar loss (mover a CPU)
        losses.append(loss.item())
        
        if verbose and epoch % 1000 == 0:
            print(f"  Epoch {epoch:5d} - Loss: {loss.item():.6e}")
        
        if loss.item() < tolerance:
            if verbose:
                print(f"  Converged at epoch {epoch}")
            break
        if early_stopping_type == 'patience':
                if loss.item() < best_loss - es_min_delta:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= es_patience:
                    break
    # Retornar x_interior en CPU para análisis posterior
    x_interior_cpu = x_interior.cpu()
    
    return model, losses, x_interior_cpu


# FUNCIONES AUXILIARES

def amplitude_on_cosine_basis(x, y, k):
    """
    Least-squares amplitude of y onto basis cos(k*x).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.cos(k * x)
    num = float(np.dot(c, y))
    den = float(np.dot(c, c)) + 1e-20
    return num / den


def k_ratio_metric(x, y, k_true, clamp_to_unit=False, squash=False):
    """
    Returns M = (1/k_true) / a_y
    """
    a_y = amplitude_on_cosine_basis(x, y, k_true)
    if np.isclose(a_y, 0.0):
        return 0.0
    M = (1.0 / k_true) / a_y
    
    if clamp_to_unit:
        return float(np.clip(M, 0.0, 1.0))
    if squash:
        return float(M / (1.0 + M))
    return float(M)


# ANÁLISIS DE k_eff (con transferencia GPU→CPU)

def analyze_keff(model, k_true, x_interior_training, x_test_points=100, device='cpu'):
    """
    Analyze k_eff by fitting model output
    VERSIÓN CUDA: Maneja transferencias entre GPU y CPU eficientemente
    
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
    x_test = torch.linspace(0, 2*np.pi, x_test_points, device=device).reshape(-1, 1)
    
    # Evaluación en GPU
    with torch.no_grad():
        u_pred = model(x_test).cpu().numpy().reshape(-1)
    
    x_test_np = x_test.cpu().numpy().reshape(-1)
    
    # Análisis en CPU (numpy)
    y = np.asarray(u_pred, float)
    x = np.linspace(0, 2*np.pi, len(y))
    gg = k_ratio_metric(x, y, k_true)
    
    return gg


# MAIN - VERSIÓN CUDA

if __name__ == '__main__':
    
    print("="*80)
    print("NÚMERO DE ONDA MODIFICADA NORMALIZADA PARA PINNs - VERSIÓN CUDA")
    print("="*80)
    
    # Detectar disositivo
    device = get_device()
    
    # Configuración
    num_interior = 128
    
    x_interior_dummy = np.linspace(0, 2*np.pi, num_interior)
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
    k_valores = np.r_[np.arange(1, k_max_safe, 2), k_max]
    # k_valores = [10.0]
    print(f"\nValores de k a evaluar:")
    print(f"  {k_valores}")
    
    epochs = 200000
    lr = 0.001
    
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
            x_test_points=150,
            device=device  # ← MODELO EN GPU
        )
        
        results.append(result)
        print(f"  k_ratio_metric = {result:.6f}")
    
    # VISUALIZACIÓN
    
    k_n = k_valores / k_max
    results_n = np.array(results) * k_n
    # error = np.abs(results_n - k_n) / k_n * 100
    
    np.savez('prueba_{}'.format(np.random.randint(0,10**9)) ,results=results_n, k_max=k_max, models=models,epochs=epochs,lr=lr)
    
