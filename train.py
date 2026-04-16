import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import labels_to_onehot
from hessian_utils import (compute_sharpness, compute_sharpness_gradient,
                           compute_ps_coefficient, compute_delta, get_params_grad)
from utils import (proj_stable_set, compute_loss_gradient, compute_gradient_noise,
                   load_batch, get_device, param_inner)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    criterion = nn.MSELoss()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        targets = labels_to_onehot(labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        targets = labels_to_onehot(labels)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def constraint_training(model, criterion, dataset, eta, num_steps, batch_size,
                        init_params):
    """
    Run the constrained trajectory iterate:

        θ†_{t+1} = θ†_t - η P⊥_{u_t, ∇S_t} ∇L(θ†_t)

    where P⊥_{u_t, ∇S_t} projects out the components of ∇L along u_t (top
    Hessian eigenvector) and ∇S_t (sharpness gradient), keeping the update
    in the tangent space of the stable set M.

    The initial parameters are first projected onto M via proj_stable_set.

    Args:
        model: PyTorch model (parameters modified in-place).
        criterion: Loss function.
        dataset: Full dataset (loaded as a single batch each step).
        eta: Learning rate (also defines the sharpness threshold 2/eta in
             proj_stable_set).
        num_steps: Number of gradient steps.
        batch_size: Mini-batch size used by compute_trajectory_stats for
                    gradient noise estimation.
        init_params: Initial state dict loaded into the model before training.

    Returns:
        model: Model with parameters at θ†_{num_steps}.
        stats: List of (sharp_grad_norm_sq, delta_sq, noise_u_sq,
               noise_sharp_sq) recorded at each step before the update.
        loss_history: List of full-dataset loss values recorded at each step.
    """
    model.load_state_dict(init_params)
    proj_stable_set(model, criterion, dataset, eta)

    device = get_device()
    stats = []
    loss_history = []

    for _ in range(num_steps):
        stats.append(compute_trajectory_stats(model, criterion, dataset, batch_size))

        params = get_params_grad(model)[0]

        # Compute u_t and ∇S_t; detach both since they serve as fixed directions
        _, u = compute_sharpness(model, criterion, dataset)
        u = [v.detach() for v in u]
        sharp_grad = [g.detach() for g in
                      compute_sharpness_gradient(model, criterion, dataset)]

        # Compute full-dataset loss gradient ∇L and record loss
        inputs, labels = load_batch(
            DataLoader(dataset, batch_size=len(dataset), shuffle=False), device
        )
        with torch.no_grad():
            loss_history.append(criterion(model(inputs), labels).item())
        loss_grad = compute_loss_gradient(model, criterion, inputs, labels)

        # Gram-Schmidt orthonormalization of {u, ∇S} into {e1, e2}
        e1 = [v / param_inner(u, u).sqrt() for v in u]

        e2 = [sg - param_inner(sharp_grad, e1) * ei for sg, ei in zip(sharp_grad, e1)]
        e2_norm = param_inner(e2, e2).sqrt()

        # P⊥ g = g - <g, e1> e1 - <g, e2> e2
        proj_g = [lg - param_inner(loss_grad, e1) * ei for lg, ei in zip(loss_grad, e1)]
        if e2_norm > 1e-8:
            e2 = [v / e2_norm for v in e2]
            proj_g = [pg - param_inner(proj_g, e2) * ei for pg, ei in zip(proj_g, e2)]

        # θ ← θ - η P⊥ ∇L
        with torch.no_grad():
            for p, pg in zip(params, proj_g):
                p.data -= eta * pg

    return model, stats, loss_history


def compute_trajectory_stats(model, criterion, dataset, batch_size):
    _, u = compute_sharpness(model, criterion, dataset)
    u = [v.detach() for v in u]
    sharp_grad = [g.detach() for g in
                  compute_sharpness_gradient(model, criterion, dataset)]

    sharp_grad_norm_sq = param_inner(sharp_grad, sharp_grad).item()

    gradient_noise = compute_gradient_noise(model, criterion, dataset, batch_size)
    alpha = compute_ps_coefficient(model, criterion, dataset, sharp_grad)
    delta_sq = compute_delta(alpha, gradient_noise) ** 2

    noise_u_sq = param_inner(gradient_noise, u).item() ** 2
    noise_sharp_sq = param_inner(gradient_noise, sharp_grad).item() ** 2

    return sharp_grad_norm_sq, delta_sq, noise_u_sq, noise_sharp_sq
