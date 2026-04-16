import torch
from torch.utils.data import DataLoader, Subset


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_batch(loader, device=None):
    """
    Return the first batch from a DataLoader, moved to device.

    Args:
        loader: DataLoader to iterate.
        device: Target device. Defaults to get_device() if not provided.

    Returns:
        inputs, labels: Tensors on the target device.
    """
    if device is None:
        device = get_device()
    for inputs, labels in loader:
        return inputs.to(device), labels.to(device)


def summarize_stats(stats):
    import numpy as np
    arr = np.array(stats)
    return arr.mean(axis=0), arr.var(axis=0)


def param_inner(a, b):
    """Inner product between two parameter vectors (lists of tensors)."""
    return sum(torch.sum(ai * bi) for ai, bi in zip(a, b))


def compute_loss_gradient(model, criterion, inputs, labels, create_graph=False):
    """
    Compute the loss gradient w.r.t. all trainable model parameters.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        inputs: Input batch (already on the correct device).
        labels: Label batch (already on the correct device).
        create_graph: If True, keep the autograd graph so the returned
                      gradients can be differentiated further (e.g. for
                      computing HVPs or higher-order derivatives). The
                      gradients are then NOT detached. If False (default),
                      return detached tensors.

    Returns:
        grads: List of tensors matching model parameter shapes.
    """
    model.zero_grad()
    params = [p for p in model.parameters() if p.requires_grad]
    loss = criterion(model(inputs), labels)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph)
    return list(grads) if create_graph else [g.detach() for g in grads]


def compute_gradient_noise(model, criterion, dataset, batch_size):
    """
    Compute the gradient noise vector: g_batch - g_full, where g_full is the
    gradient over the full dataset and g_batch is the gradient over a randomly
    sampled mini-batch.

    Args:
        model: PyTorch model.
        criterion: Loss function.
        dataset: Full dataset.
        batch_size: Number of samples in the mini-batch.

    Returns:
        noise: List of tensors matching model parameter shapes, equal to
               g_batch - g_full.
    """
    device = get_device()

    # Noisy gradient on a random mini-batch
    indices = torch.randperm(len(dataset))[:batch_size]
    batch_inputs, batch_labels = load_batch(
        DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False),
        device,
    )
    g_batch = compute_loss_gradient(model, criterion, batch_inputs, batch_labels)

    # Noiseless gradient on the full dataset
    full_inputs, full_labels = load_batch(
        DataLoader(dataset, batch_size=len(dataset), shuffle=False), device
    )
    g_full = compute_loss_gradient(model, criterion, full_inputs, full_labels)

    return [gb - gf for gb, gf in zip(g_batch, g_full)]


def proj_stable_set(model, criterion, dataset, eta, max_iter=100, tol=1e-6):
    """
    Project model parameters onto the stable set
        M = {θ : S(θ) ≤ 2/η  and  <∇L(θ), u(θ)> = 0}
    via alternating linearized projections.

    Each iteration:
      1. If S(θ) > 2/η: subtract [(S - 2/η) / ||∇S||²] ∇S to bring the
         linearized sharpness constraint to equality.
      2. Recompute ∇L and alignment after step 1, then if
         |<∇L, u>| > tol: subtract [<∇L, u> / ||Hu||²] Hu to zero out
         the linearized alignment constraint (u treated as fixed within
         the iteration).
    Parameters are updated in-place; the modified model is returned.

    Args:
        model: PyTorch model (parameters updated in-place).
        criterion: Loss function.
        dataset: Dataset (loaded in full as a single batch).
        eta: Learning rate defining the sharpness threshold 2/eta.
        max_iter: Maximum number of alternating-projection iterations.
        tol: Convergence tolerance for both constraints.

    Returns:
        model: The model with projected parameters.
    """
    # Local import to avoid circular dependency (hessian_utils imports utils)
    from hessian_utils import (compute_sharpness, compute_sharpness_gradient,
                               hessian_vector_product, get_params_grad)

    device = get_device()
    threshold = 2.0 / eta
    inputs, labels = load_batch(
        DataLoader(dataset, batch_size=len(dataset), shuffle=False), device
    )
    params = get_params_grad(model)[0]

    for _ in range(max_iter):
        sharpness, u = compute_sharpness(model, criterion, dataset)
        u = [v.detach() for v in u]
        loss_grad = compute_loss_gradient(model, criterion, inputs, labels)
        alignment = sum(torch.sum(g * ui) for g, ui in zip(loss_grad, u)).item()

        if sharpness <= threshold and abs(alignment) <= tol:
            break

        # Step 1: enforce sharpness constraint
        if sharpness > threshold:
            sharp_grad = compute_sharpness_gradient(model, criterion, dataset)
            sharp_grad_norm_sq = sum(torch.sum(g ** 2) for g in sharp_grad).item()
            step = (sharpness - threshold) / sharp_grad_norm_sq
            with torch.no_grad():
                for p, sg in zip(params, sharp_grad):
                    p.data -= step * sg.detach()
            # Recompute alignment with updated parameters before step 2
            loss_grad = compute_loss_gradient(model, criterion, inputs, labels)
            alignment = sum(torch.sum(g * ui) for g, ui in zip(loss_grad, u)).item()

        # Step 2: enforce alignment constraint
        if abs(alignment) > tol:
            Hu = hessian_vector_product(model, criterion, inputs, labels, u)
            Hu_norm_sq = sum(torch.sum(h ** 2) for h in Hu).item()
            if Hu_norm_sq > tol:
                step = alignment / Hu_norm_sq
                with torch.no_grad():
                    for p, h in zip(params, Hu):
                        p.data -= step * h.detach()

    return model
