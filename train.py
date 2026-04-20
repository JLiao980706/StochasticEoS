import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data import labels_to_onehot
from hessian_utils import (compute_sharpness, compute_sharpness_gradient,
                           compute_ps_coefficient, compute_beta_delta_sq,
                           hessian_vector_product, get_params_grad)
from utils import (proj_stable_set, compute_loss_gradient, compute_gradient_noise,
                   load_batch, get_device, param_inner, param_add, log_state_dict,
                   proj_orthogonal)


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


def fullbatch_gd(model, criterion, dataset, eta, num_steps, eval_freq=10):
    """
    Run full-batch gradient descent:

        θ_{t+1} = θ_t - η ∇L(θ_t)

    where ∇L is computed over the entire dataset at each step.

    Args:
        model: PyTorch model (parameters modified in-place).
        criterion: Loss function (takes model outputs and one-hot targets).
        dataset: Dataset object (e.g. from get_cifar10_subset).
        eta: Learning rate.
        num_steps: Number of gradient steps.
        eval_freq: Compute sharpness and accuracy every this many steps, and
                   print a progress line and save a checkpoint. Default: 10.

    Returns:
        model: Model with parameters at θ_{num_steps}.
        loss_history: List of full-dataset loss values recorded before each update.
        acc_history: List of training accuracies recorded every eval_freq steps.
        sharpness_history: List of top Hessian eigenvalues recorded every eval_freq steps.
        checkpoints: List of state_dicts (copied to CPU) saved every eval_freq steps.
    """
    device = get_device()
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)
    loss_history, acc_history, sharpness_history, checkpoints = [], [], [], []

    for _ in range(num_steps):
        inputs, labels = load_batch(loader, device)
        targets = labels_to_onehot(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        step = len(loss_history) + 1
        loss_history.append(loss.item())

        if step % eval_freq == 0:
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            acc_history.append(acc)
            sharpness, _ = compute_sharpness(model, criterion, dataset)
            sharpness_history.append(sharpness)
            print(f'Step {step:>{len(str(num_steps))}d}/{num_steps} | '
                  f'Loss {loss_history[-1]:.4f} | '
                  f'Acc {acc:.3f} | '
                  f'Sharpness {sharpness:.4f}')
            checkpoints.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        loss.backward()
        optimizer.step()

    return model, loss_history, acc_history, sharpness_history, checkpoints


def constraint_training(model, criterion, dataset, eta, num_steps, batch_size,
                        init_params, destination_folder=None):
    """
    Run the constrained trajectory iterate:

        θ†_{t+1} = θ†_t - η P⊥_{u_t, ∇S_t} ∇L(θ†_t)

    where P⊥_{u_t, ∇S_t} projects out the components of ∇L along u_t (top
    Hessian eigenvector) and ∇S_t (sharpness gradient), keeping the update
    in the tangent space of the stable set M.

    The initial parameters are first projected onto M via proj_stable_set.
    All gradients are computed over the full dataset (full-batch GD).

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
        destination_folder: If provided, saves a checkpoint as ckpt_{step}.pt
                            into this folder after every step.

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

    for step_idx in range(num_steps):
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
            loss_history.append(criterion(model(inputs), labels_to_onehot(labels)).item())
            print('Step {}/{} | Loss {:.4f} | Sharp Grad Norm^2 {:.4f}'.format(
                step_idx + 1, num_steps, loss_history[-1], stats[-1][0]))
        loss_grad = compute_loss_gradient(model, criterion, inputs, labels)

        # Gram-Schmidt orthonormalization of {u, ∇S} into {e1, e2}
        e1 = [v / param_inner(u, u).sqrt() for v in u]

        e2 = param_add(sharp_grad, e1, -param_inner(sharp_grad, e1))
        e2_norm = param_inner(e2, e2).sqrt()

        # P⊥ g = g - <g, e1> e1 - <g, e2> e2
        proj_g = param_add(loss_grad, e1, -param_inner(loss_grad, e1))
        if e2_norm > 1e-8:
            e2 = [v / e2_norm for v in e2]
            proj_g = param_add(proj_g, e2, -param_inner(proj_g, e2))

        # θ ← θ - η P⊥ ∇L
        with torch.no_grad():
            for p, pg in zip(params, proj_g):
                p.data -= eta * pg

        if destination_folder is not None:
            log_state_dict(destination_folder, step_idx + 1, model)

    return model, stats, loss_history


def collect_gradient_noise_trajectory(model, criterion, dataset, eta,
                                      num_steps, batch_size, init_state_dict):
    """
    Run T steps of mini-batch SGD from init_state_dict, recording the gradient
    noise xi_t = g_batch_t - g_full_t at each step before the parameter update.
    The same mini-batch is used for both the noise record and the SGD step.

    Args:
        model: PyTorch model (parameters modified in-place).
        criterion: Loss function.
        dataset: Full dataset.
        eta: Learning rate.
        num_steps: Number of SGD steps T.
        batch_size: Mini-batch size.
        init_state_dict: Starting state dict (theta_0).

    Returns:
        noises: List of T gradient-noise vectors, each a list of param-shaped
                tensors equal to g_batch_t - g_full_t.
    """
    model.load_state_dict(init_state_dict)
    device = get_device()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)

    full_inputs, full_labels = load_batch(
        DataLoader(dataset, batch_size=len(dataset), shuffle=False), device
    )

    noises = []
    for _ in range(num_steps):
        indices = torch.randperm(len(dataset))[:batch_size]
        batch_inputs, batch_labels = load_batch(
            DataLoader(Subset(dataset, indices), batch_size=batch_size,
                       shuffle=False), device
        )

        g_full = compute_loss_gradient(model, criterion, full_inputs, full_labels)

        optimizer.zero_grad()
        criterion(model(batch_inputs), batch_labels).backward()
        g_batch = [p.grad.detach().clone()
                   for p in model.parameters() if p.requires_grad]
        optimizer.step()

        noises.append(param_add(g_batch, g_full, -1))

    return noises


def generate_predicted_dynamic(model, criterion, dataset, eta, num_steps,
                               batch_size, ckpt_dir, init_state_dict):
    """
    Generate predicted dynamics (x_hat_t, y_hat_t) by propagating
    v_hat_t = theta_t - theta_dagger_t through the linearized flow around
    the projected trajectory {theta_dagger_t}.

    v_hat_{t+1} = P_{u_{t+1}}^perp [
        (I - eta H_t) P_{u_t}^perp v_hat_t^perp
        + (eta/2)(delta_t^2 - x_hat_t^2) nabla_S_t^perp
    ] - ((1 + eta*y_hat_t)*x_hat_t + (eta/2)*kappa_t*x_hat_t^2) u_{t+1}
      - eta * xi_t

    kappa_t = <nabla_S_t, partial u_t / partial theta * u_t> is approximated
    by a forward finite difference in the u_t direction.

    Args:
        model: PyTorch model (parameters overwritten during computation).
        criterion: Loss function.
        dataset: Full dataset.
        eta: Learning rate.
        num_steps: Number of steps T.
        batch_size: Mini-batch size for gradient-noise collection.
        ckpt_dir: Directory with 'ckpt_{t}.pt' for t = 0, ..., T.
        init_state_dict: Initial model state dict (theta_0).

    Returns:
        x_hat_list: List of T floats — x_hat_t = <v_hat_t, u_t>.
        y_hat_list: List of T floats — y_hat_t = <nabla_S_t^perp, v_hat_t>.
    """
    device = get_device()

    # Step 1: collect {xi_t} along T steps of mini-batch SGD from theta_0
    noises = collect_gradient_noise_trajectory(
        model, criterion, dataset, eta, num_steps, batch_size, init_state_dict
    )

    # v_hat_0 = theta_0 - theta_dagger_0
    state_dict_0 = torch.load(os.path.join(ckpt_dir, 'ckpt_0.pt'),
                              map_location=device)
    model.load_state_dict(state_dict_0)
    proj_params_0 = [p.data.clone() for p in model.parameters() if p.requires_grad]

    model.load_state_dict(init_state_dict)
    theta_0 = [p.data.clone() for p in model.parameters() if p.requires_grad]
    v_hat = param_add(theta_0, proj_params_0, -1)

    # Load u_0 while model is at theta_dagger_0
    model.load_state_dict(state_dict_0)
    inputs, labels = load_batch(
        DataLoader(dataset, batch_size=len(dataset), shuffle=False), device
    )
    _, u_next = compute_sharpness(model, criterion, dataset)
    u_next = [v.detach() for v in u_next]

    x_hat_list, y_hat_list = [], []

    for t in range(num_steps):
        u = u_next  # u_t, cached from end of previous iteration

        ckpt_path_t = os.path.join(ckpt_dir, f'ckpt_{t}.pt')
        state_dict_t = torch.load(ckpt_path_t, map_location=device)
        model.load_state_dict(state_dict_t)

        inputs, labels = load_batch(
            DataLoader(dataset, batch_size=len(dataset), shuffle=False), device
        )
        loss_grad = compute_loss_gradient(model, criterion, inputs, labels)
        sharp_grad = [g.detach() for g in
                      compute_sharpness_gradient(model, criterion, dataset)]

        # nabla_S_t^perp = nabla_S_t - <nabla_S_t, u_t> u_t
        sharp_grad_perp = proj_orthogonal(sharp_grad, u)

        x_hat_t = param_inner(v_hat, u).item()
        y_hat_t = param_inner(sharp_grad_perp, v_hat).item()
        x_hat_list.append(x_hat_t)
        y_hat_list.append(y_hat_t)

        alpha_t = -param_inner(sharp_grad, loss_grad).item()
        _, delta_sq_t = compute_beta_delta_sq(alpha_t, sharp_grad_perp)

        kappa_t = param_inner(sharp_grad, u).item()

        # h_t = (I - eta H_t) v_hat_t^perp
        v_hat_perp = proj_orthogonal(v_hat, u)
        Hv = hessian_vector_product(model, criterion, inputs, labels, v_hat_perp)
        h_t = param_add(v_hat_perp, list(Hv), -eta)

        # g_t = h_t + (eta/2)(delta_t^2 - x_hat_t^2) nabla_S_t^perp
        scale = eta / 2 * (delta_sq_t - x_hat_t ** 2)
        g_t = param_add(h_t, sharp_grad_perp, scale)

        # Load u_{t+1} from theta_dagger_{t+1}
        model.load_state_dict(
            torch.load(os.path.join(ckpt_dir, f'ckpt_{t + 1}.pt'),
                       map_location=device)
        )
        _, u_next = compute_sharpness(model, criterion, dataset)
        u_next = [v.detach() for v in u_next]

        # v_hat_{t+1}
        g_proj = param_inner(g_t, u_next).item()
        scalar_u = ((1 + eta * y_hat_t) * x_hat_t
                    + (eta / 2) * kappa_t * x_hat_t ** 2)
        xi_t = noises[t]
        v_hat = param_add(param_add(g_t, u_next, -(g_proj + scalar_u)), xi_t, -eta)

    return x_hat_list, y_hat_list


def compute_trajectory_stats(model, criterion, dataset, batch_size):
    # _, u = compute_sharpness(model, criterion, dataset)
    # u = [v.detach() for v in u]
    # sharp_grad = [g.detach() for g in
    #               compute_sharpness_gradient(model, criterion, dataset)]

    # sharp_grad_norm_sq = param_inner(sharp_grad, sharp_grad).item()

    # gradient_noise = compute_gradient_noise(model, criterion, dataset, batch_size)
    # alpha = compute_ps_coefficient(model, criterion, dataset, sharp_grad)
    # delta_sq = compute_delta(alpha, gradient_noise) ** 2

    # noise_u_sq = param_inner(gradient_noise, u).item() ** 2
    # noise_sharp_sq = param_inner(gradient_noise, sharp_grad).item() ** 2
    

    # return sharp_grad_norm_sq, delta_sq, noise_u_sq, noise_sharp_sq
    return 0.0, 0.0, 0.0, 0.0
