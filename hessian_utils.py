import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from utils import compute_loss_gradient, load_batch, get_device


#===========================================================================
# Helper utilities
#===========================================================================

def get_params_grad(model):
    """Get model parameters that require gradients."""
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def flatten_tensors(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_tensors(flat_tensor, ref_tensors):
    if isinstance(flat_tensor, np.ndarray):
        flat_tensor = torch.tensor(flat_tensor)
    unflattened = []
    cur_pos = 0
    for t in ref_tensors:
        unflattened.append(flat_tensor[cur_pos: cur_pos + t.numel()].view(t.shape))
        cur_pos += t.numel()
    if cur_pos != flat_tensor.numel():
        raise ValueError("Number of elements mismatch!")
    return unflattened

#===========================================================================
# Hessian vector product and top eigenvectors/eigenvalues
#===========================================================================

def hessian_vector_product(model, criterion, inputs, labels, vector):
    """
    Compute the Hessian-vector product Hv of the loss with respect to model
    parameters, where v = vector.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        inputs: Input batch.
        labels: Label batch.
        vector: List of tensors matching model parameter shapes.

    Returns:
        Tuple of tensors: Hv, one tensor per parameter.
    """
    vector = [v.detach() for v in vector]
    params = get_params_grad(model)[0]
    loss_grad = compute_loss_gradient(model, criterion, inputs, labels,
                                      create_graph=True)
    return torch.autograd.grad(loss_grad, params, grad_outputs=vector,
                               retain_graph=False)


def lanczos(mat_vec_func, params, device, top_dim=1):
    """
    Compute the top_dim largest-magnitude eigenvalues and corresponding
    eigenvectors of the symmetric linear operator defined by mat_vec_func,
    using the Lanczos algorithm (via scipy's eigsh).

    Args:
        mat_vec_func: Callable mapping a list of param-shaped tensors to the
                      same shape (the matrix-vector product).
        params: List of parameter tensors (used for shape reference).
        device: torch.device to place results on.
        top_dim: Number of top eigenpairs to return.

    Returns:
        evals: Float tensor of shape (top_dim,), sorted by decreasing magnitude.
        evecs: List of top_dim eigenvectors, each a list of param-shaped tensors.
    """
    def mv(vec):
        param_shape_vec = [p.to(device) for p in unflatten_tensors(vec, params)]
        return flatten_tensors(mat_vec_func(param_shape_vec)).cpu().numpy()

    dim = sum(p.numel() for p in params)
    operator = LinearOperator((dim, dim), matvec=mv)
    evals_np, evecs_np = eigsh(operator, top_dim)
    sort_idx = np.argsort(np.abs(evals_np))[::-1]
    evals = torch.from_numpy(evals_np[sort_idx].copy()).float()
    evecs = [
        [w.to(device) for w in unflatten_tensors(v, params)]
        for v in evecs_np.T[sort_idx]
    ]
    return evals, evecs


def compute_hessian_top_eigenpairs(model, criterion, dataset, top_dim=1):
    """
    Compute the top_dim largest eigenvalues and eigenvectors of the loss
    Hessian over the full dataset using the Lanczos algorithm.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        dataset: Dataset (loaded in full as a single batch).
        top_dim: Number of top eigenpairs to return.

    Returns:
        evals: Float tensor of shape (top_dim,), sorted by decreasing magnitude.
        evecs: List of top_dim eigenvectors, each a list of param-shaped tensors.
    """
    device = get_device()
    inputs, labels = load_batch(DataLoader(dataset, batch_size=len(dataset),
                                           shuffle=False), device)
    params = get_params_grad(model)[0]
    mat_vec_func = lambda v: hessian_vector_product(model, criterion, inputs,
                                                    labels, v)
    return lanczos(mat_vec_func, params, device, top_dim=top_dim)


def compute_sharpness(model, criterion, dataset):
    """
    Returns the largest eigenvalue of the loss Hessian (sharpness).

    Args:
        model: PyTorch model.
        criterion: Loss function.
        dataset: Dataset (loaded in full as a single batch).

    Returns:
        sharpness: Scalar float — the largest Hessian eigenvalue.
        evec: List of tensors (param-shaped) — the corresponding eigenvector.
    """
    evals, evecs = compute_hessian_top_eigenpairs(model, criterion, dataset, top_dim=1)
    return evals[0].item(), evecs[0]


def compute_sharpness_gradient(model, criterion, dataset):
    """
    Compute the gradient of sharpness (largest Hessian eigenvalue) w.r.t.
    model parameters, using: nabla S(theta) = nabla^3 L(theta)[u, u],
    where u is the top Hessian eigenvector.

    Steps:
      1. Compute top eigenvector u via Lanczos and detach it.
      2. Recompute HVP = nabla^2 L(theta) @ u with create_graph=True so the
         result stays in the autograd graph.
      3. Form S = <HVP, u> with u detached, making S differentiable w.r.t.
         theta only through the HVP.
      4. Differentiate S w.r.t. params to get nabla S(theta).

    Args:
        model: PyTorch model.
        criterion: Loss function.
        dataset: Dataset (loaded in full as a single batch).

    Returns:
        grad_sharpness: Tuple of tensors matching model parameter shapes.
    """
    inputs, labels = load_batch(DataLoader(dataset, batch_size=len(dataset),
                                            shuffle=False))

    # Step 1: top eigenvector u — detached so gradients flow only via params
    _, u = compute_sharpness(model, criterion, dataset)
    u = [v.detach() for v in u]

    # Step 2: HVP = nabla^2 L(theta) @ u, keeping the graph so we can
    # differentiate through it. u is detached so gradients flow only via params.
    params = get_params_grad(model)[0]
    loss_grad = compute_loss_gradient(model, criterion, inputs, labels,
                                      create_graph=True)
    hvp = torch.autograd.grad(loss_grad, params, grad_outputs=u,
                              retain_graph=True, create_graph=True)

    # Step 3: S = <HVP, u>. u is detached, so S is differentiable w.r.t.
    # theta only through hvp.
    sharpness = sum(torch.sum(h * ui) for h, ui in zip(hvp, u))

    # Step 4: nabla S(theta)
    return torch.autograd.grad(sharpness, params)


def compute_beta_delta_sq(alpha, sharp_grad_perp):
    """
    Compute beta = ||∇S_t^⊥||² and delta_sq = 2 * alpha / beta.

    alpha is clamped to zero before division to avoid a negative result.

    Args:
        alpha: PS coefficient α_t (scalar float).
        sharp_grad_perp: List of param-shaped tensors — ∇S_t^⊥.

    Returns:
        beta: Scalar float — ||∇S_t^⊥||².
        delta_sq: Scalar float — 2 * max(alpha, 0) / beta.
    """
    beta = sum(torch.sum(g ** 2) for g in sharp_grad_perp).item()
    delta_sq = 2 * max(alpha, 0) / beta if beta > 1e-30 else 0.0
    return beta, delta_sq


def compute_ps_coefficient(model, criterion, dataset, sharp_grad):
    """
    Compute the progressive sharpening (PS) coefficient:
        -<nabla S(theta), nabla L(theta)>

    A positive value indicates the loss gradient points in a direction that
    decreases sharpness (progressive flattening); negative indicates sharpening.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        dataset: Dataset (loaded in full as a single batch).
        sharp_grad: Pre-computed sharpness gradient (list of param-shaped
                    tensors), as returned by compute_sharpness_gradient.

    Returns:
        ps_coeff: Scalar float.
    """
    inputs, labels = load_batch(DataLoader(dataset, batch_size=len(dataset),
                                            shuffle=False))
    loss_grad = compute_loss_gradient(model, criterion, inputs, labels)
    return -sum(torch.sum(sg * lg) for sg, lg in zip(sharp_grad, loss_grad)).item()
