import math
from collections import defaultdict
from typing import Callable, Dict, Union

import numpy as np
import ot
import torch
from torch import Tensor
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torchdeq.solver.stat import SolverStat
from tqdm import tqdm

import wandb
from plotting import plot_errors, plot_point_clouds_over_time
from utils import compute_masked_mean, get_mask, mean_confidence_interval, LoggingDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gaussian_kernel(x, y, sigma=1.0, **kwargs):
    """Compute the Gaussian kernel between a batch of samples `x` and `y`.

    :param x: torch.tensor of shape N*d, where N is the number of samples and d their dimension;
        or of shape B*N*d, where B is an additional batch dimension.
    :param y: torch.tensor of shape M*d or B*M*d, similar to `x`.
    :param sigma: hyperparameter sigma, bandwidth of the kernel.
    :return: torch.tensor of shape N*M or B*N*M.
    """
    beta = 1.0 / (2.0 * sigma**2)
    dist = torch.cdist(x, y) ** 2
    return torch.exp(-beta * dist)


def riesz_kernel(x, y, r=1.0, **kwargs):
    """Compute the Riesz kernel of order `r` between a batch of samples `x` and `y`.

    Args:
        x (torch.Tensor): Shape (N, d) or (B, N, d).
        y (torch.Tensor): Shape (M, d) or (B, M, d).
        r (float, optional): Order of the Riesz kernel (exponent of the distance). Defaults to 1.0.
    """
    dist = torch.cdist(x, y)
    return -(dist**r)


def squared_mmd(x, y, sigma=1.0, kernel=gaussian_kernel, mask_x=False, mask_y=False):
    """Compute the squared MMD between `x` and `y`.

    :param x: torch.tensor of shape N*d, where N is the number of samples and d their dimension;
        or of shape B*N*p for a batch size B.
    :param y: torch.Tensor of shape M*d or B*M*d, similar to `x`.
    :param sigma: hyperparameter sigma, bandwidth of the kernel.
    :param kernel: kernel to be used. Defaults to Gaussian kernel.
    :param mask: if True, apply a mask to the tensor that masks out zero entries.
    :return: squared MMD between `x` and `y` (tensor of dimension 0 or B).
    """
    xx = kernel(x, x, sigma=sigma)
    yy = kernel(y, y, sigma=sigma)
    xy = kernel(x, y, sigma=sigma)
    if mask_x:
        x_mask = get_mask(x).squeeze()
    else:
        x_mask = torch.ones(x.shape[:-1], device=x.device)
    if mask_y:
        y_mask = get_mask(y).squeeze()
    else:
        y_mask = torch.ones(y.shape[:-1], device=y.device)
    xx_mask = x_mask.unsqueeze(-1) * x_mask.unsqueeze(-2)
    yy_mask = y_mask.unsqueeze(-1) * y_mask.unsqueeze(-2)
    xy_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(-2)
    if xx.dim() == 2:  # no batch dimension
        return (
            compute_masked_mean(xx, [0, 1], xx_mask)
            + compute_masked_mean(yy, [0, 1], yy_mask)
            - 2 * compute_masked_mean(xy, [0, 1], xy_mask)
        )
    elif xx.dim() == 3:  # batch dimension
        return (
            compute_masked_mean(xx, [1, 2], xx_mask)
            + compute_masked_mean(yy, [1, 2], yy_mask)
            - 2 * compute_masked_mean(xy, [1, 2], xy_mask)
        )
    else:
        raise ValueError(
            "`x` and `y` must be either of dimension 2 (no batch dimension) or 3 "
            "(including batch dimension)."
        )


def gaussian_witness_function(
    x, y, z, sigma=1.0, mask_x=False, mask_y=False, mask_z=False, kernel=gaussian_kernel
):
    """Compute the witness function f (see "MMD gradient flow" paper) at a point
    z, between the first measure given by points x and the second measure given by points y.

    :param x: torch.tensor of shape N*d, where N is the number of samples and d their dimension;
        or of dimension B*N*d, where B is a batch size.
    :param y: torch.Tensor of shape M*d or B*M*d, similar to `x`.
    :param z: torch.tensor of shape L*d or B*L*d.
    :param sigma: hyperparameter sigma, bandwidth of the kernel.
    :param mask: if True, apply a mask to the tensor that masks out zero entries.
    :return: torch.tensor of shape (L,) or (B,L).
    """
    if mask_x:
        x_mask = get_mask(x).squeeze()
    else:
        x_mask = torch.ones(x.shape[:-1], device=x.device)
    if mask_y:
        y_mask = get_mask(y).squeeze()
    else:
        y_mask = torch.ones(y.shape[:-1], device=y.device)
    if mask_z:
        z_mask = get_mask(z).squeeze()
    else:
        z_mask = torch.ones(z.shape[:-1], device=z.device)
    xz_mask = x_mask.unsqueeze(-1) * z_mask.unsqueeze(-2)
    yz_mask = y_mask.unsqueeze(-1) * z_mask.unsqueeze(-2)

    xz = kernel(x, z, sigma=sigma)
    yz = kernel(y, z, sigma=sigma)
    xz_mean = compute_masked_mean(xz, [-2], xz_mask)
    yz_mean = compute_masked_mean(yz, [-2], yz_mask)
    out = xz_mean - yz_mean
    out = out * z_mask
    return out


def mmd_callback(
    errors_dict, input_points, x, target, sigma, initial_sigma, kernel, **kwargs
):
    """A callback function for the MMD flow computing as error measures the online MMD
    (using the current sigma), the MMD for sigma=1 and sigma=10 and the initial sigma
    of the flow, the sliced Wasserstein
    distance to the labels and to the input, and the Wasserstein-2 distance to the labels.

    :param errors_dict: dictionary coming from the gradient flow function to which errors
        are written.
    :param input_points: input points of the gradient flow.
    :param x: current position of the flow.
    :param target: labels of the flow.
    :param sigma: (current) bandwidth of the flow.
    :param initial_sigma: initial bandwidth of the flow.
    :param kernel: kernel to be used for MMD.
    :return: None.
    """
    x = x.detach().clone().to(device)
    target = target.detach().clone().to(device)
    N = x.shape[-2]
    a, b = np.ones((N,)) / N, np.ones((N,)) / N

    errors_dict["online MMD to labels"].append(
        0.5 * squared_mmd(x, target, sigma=sigma, kernel=kernel).item()
    )
    errors_dict["MMD-1 to labels"].append(
        0.5 * squared_mmd(x, target, sigma=1.0, kernel=kernel).item()
    )
    errors_dict["MMD-10 to labels"].append(
        0.5 * squared_mmd(x, target, sigma=10.0, kernel=kernel).item()
    )
    errors_dict[f"Initial MMD-{int(initial_sigma)} to labels"].append(
        0.5 * squared_mmd(x, target, sigma=initial_sigma, kernel=kernel).item()
    )
    errors_dict["SWD to labels"].append(
        ot.sliced_wasserstein_distance(x.cpu().numpy(), target.cpu().numpy(), a, b, 50)
    )
    errors_dict["SWD to input"].append(
        ot.sliced_wasserstein_distance(
            x.cpu().numpy(), input_points.detach().cpu().numpy(), a, b, 50
        )
    )
    errors_dict["W2 to labels"].append(
        math.sqrt(ot.emd2(a, b, ot.dist(x.cpu().numpy(), target.cpu().numpy())))
    )
    errors_dict["L2 to labels"].append(torch.norm(x - target).item())
    # return size of the plots
    return {"nrows": 2, "ncols": 4}


def mmd_batch_callback(
    errors_dict,
    input_points,
    x,
    target,
    sigmas,
    initial_sigmas,
    last_sigmas,
    kernel,
    conf=0.95,
    plot_to_wandb=False,
    crit_value=None,
    stop_type=None,
    stop_mode=None,
    step=None,
    plot_kwargs=None,
    mask=False,
    wandb_dict=None,
    **kwargs,
):
    """A callback function for the MMD flow computing as error measures the online MMD
    (using the current sigma), the MMD for sigma=1 and sigma=10, and with the initial sigma
    of the flow. This callback can act on batches of samples and omits computations of
    Wasserstein distances which can not be computed in a batched manner. Adds the average over
    the batch, alongside

    :param errors_dict: dictionary coming from the gradient flow function to which errors
        are written.
    :param x: current position of the flow.
    :param target: targets of the flow.
    :param sigma: (current) bandwidth of the flow.
    :param initial_sigma: initial bandwidth of the flow.
    :param last_sigma: final bandwidth of the flow.
    :param kernel: kernel to be used for MMD.
    :param conf: confidence for the confidence interval of the errors.
    :param plot_to_wandb: if True, plots the errors to wandb.
    :param step: current step of the flow.
    :return: None.
    """
    if wandb_dict is None:
        wandb_dict = LoggingDict()
    x = x.detach().clone()
    target = target.detach().clone()
    sigma = sigmas[0]
    last_sigma = last_sigmas[0]
    initial_sigma = initial_sigmas[0]

    if plot_kwargs is None:
        plot_kwargs = {}
    try:
        batch_str = f" (batch {plot_kwargs['batch_nb']})"
    except KeyError:
        batch_str = ""
    try:
        epoch_str = f" (epoch {plot_kwargs['epoch_nb']})"
    except KeyError:
        epoch_str = ""

    if kernel == gaussian_kernel:
        errors_dict[f"online MMD to targets inner loop ({initial_sigma} to {last_sigma:.2f})"].append(
            mean_confidence_interval(
                0.5
                * squared_mmd(
                    x, target, sigma=sigma, kernel=kernel, mask_x=mask, mask_y=mask
                ),
                conf,
            )
        )
        errors_dict["MMD-1 to targets inner loop"].append(
            mean_confidence_interval(
                0.5
                * squared_mmd(
                    x, target, sigma=1.0, kernel=kernel, mask_x=mask, mask_y=mask
                ),
                conf,
            )
        )
        errors_dict["MMD-10 to targets inner loop"].append(
            mean_confidence_interval(
                0.5
                * squared_mmd(
                    x, target, sigma=10.0, kernel=kernel, mask_x=mask, mask_y=mask
                ),
                conf,
            )
        )
        errors_dict[f"Initial MMD-{int(initial_sigma)} to targets inner loop"].append(
            mean_confidence_interval(
                0.5
                * squared_mmd(
                    x, target, sigma=initial_sigma, kernel=kernel, mask_x=mask, mask_y=mask
                ),
                conf,
            )
        )
    elif kernel == riesz_kernel:
        errors_dict["MMD to targets inner loop"].append(
            mean_confidence_interval(
                0.5
                * squared_mmd(
                    x, target, kernel=kernel, mask_x=mask, mask_y=mask
                ),
                conf,
            )
        )
    if plot_to_wandb:
        if kernel == gaussian_kernel:
            plot_dict = {
                f"online MMD to targets inner loop ({initial_sigma} to {last_sigma:.2f}){batch_str}{epoch_str}": errors_dict[
                    f"online MMD to targets inner loop ({initial_sigma} to {last_sigma:.2f})"
                ][
                    -1
                ][
                    0
                ].item(),
                f"MMD-1 to targets inner loop{batch_str}{epoch_str}": errors_dict["MMD-1 to targets inner loop"][-1][
                    0
                ].item(),
                f"MMD-10 to targets inner loop{batch_str}{epoch_str}": errors_dict["MMD-10 to targets inner loop"][-1][
                    0
                ].item(),
                f"Initial MMD-{int(initial_sigma)} to targets inner loop{batch_str}{epoch_str}": errors_dict[
                    f"Initial MMD-{int(initial_sigma)} to targets inner loop"
                ][-1][0].item(),
                f"Stopping: {stop_mode} {stop_type}{batch_str}{epoch_str}": crit_value,
            }
        elif kernel == riesz_kernel:
            plot_dict = {
                f"MMD to targets inner loop{batch_str}{epoch_str}": errors_dict["MMD to targets inner loop"][-1][0].item(),
                f"Stopping: {stop_mode} {stop_type}{batch_str}{epoch_str}": crit_value,
            }
        wandb.log(plot_dict)
        wandb_dict.append(plot_dict)
    # return size of the plots
    return {"nrows": 1, "ncols": 4, "conf": conf}


def _update_crit_value(
    x0,
    target,
    sigmas,
    kernel,
    mask,
    grad,
    stop_type,
    stop_mode,
    rescale_losses,
    fcts=None,
    mean_grad=None,
    initial_points=None,
    initial_target=None,
):
    with torch.no_grad():
        if stop_type == "mmd":
            initial_mmds = [
                squared_mmd(
                    initial_points,
                    initial_target,
                    sigma=sigma,
                    kernel=kernel,
                    mask_x=mask,
                    mask_y=mask,
                )
                .mean()
                .item()
                for sigma in sigmas
            ]
            mmds = [
                squared_mmd(
                    x0, target, sigma=sigma, kernel=kernel, mask_x=mask, mask_y=mask
                )
                .mean()
                .item()
                for sigma in sigmas
            ]
            if rescale_losses and fcts is None:
                fcts = [
                    initial_mmd / max(initial_mmds[0], 1e-8)
                    for initial_mmd in initial_mmds
                ]
            if rescale_losses:
                initial_mmds = [
                    initial_mmd / max(fct, 1e-8)
                    for initial_mmd, fct in zip(initial_mmds, fcts)
                ]
                mmds = [mmd / max(fct, 1e-8) for mmd, fct in zip(mmds, fcts)]
            initial_mean_mmd = sum(initial_mmds) / len(initial_mmds)
            mean_mmd = sum(mmds) / len(mmds)
            if stop_mode == "abs":
                crit_value = mean_mmd
            elif stop_mode == "rel":
                crit_value = mean_mmd / max(initial_mean_mmd, 1e-8)

        elif stop_type == "grad":
            fcts = None
            if mean_grad is None:
                if stop_mode == "rel":
                    crit_value = 1.0
                else:
                    crit_value = math.inf
            else:
                grad_norm = torch.norm(grad).item()
                if stop_mode == "abs":
                    crit_value = grad_norm
                elif stop_mode == "rel":
                    crit_value = grad_norm / max(mean_grad, 1e-8)
        if isinstance(crit_value, Tensor):
            crit_value = crit_value.item()
        return crit_value, fcts


def mmd_gf(
    func: Callable = None,
    x0: Tensor = None,
    max_iter: int = 2000,
    tol: float = 1e-3,
    stop_mode: str = "rel",
    indexing: Union[list, int] = None,
    stop_frequency: int = 100,
    stop_type: str = "grad",
    target_tensor: Tensor = None,
    beta: Union[list, float] = 0.01,
    gamma_beta: Union[list, float] = 1.0,
    witness_fn: Callable = gaussian_witness_function,
    kernel: Callable = gaussian_kernel,
    sigma: Union[list, float] = 1.0,
    gamma_sigma: Union[list, float] = 1.0,
    min_sigma: Union[list, float] = None,
    optimizer: str = "sgd",
    optim_kwargs: Dict = None,
    scheduler: str = "steplr",
    scheduler_kwargs: Dict = None,
    reg: float = 0.0,
    reg_p: float = 2.0,
    callback: Callable = mmd_batch_callback,
    timestamps_callback: Union[list, int] = 0,
    plot: bool = False,
    plot_to_wandb=False,
    plot_kwargs: Dict = None,
    progress_bar=False,
    callback_kwargs=None,
    mask=False,
    rescale_losses=False,
    autodiff=False,
    rescale_autodiff=False,
    grad_mask=None,
    wandb_dict=None,
    **kwargs,
) -> tuple:
    """Simulate a MMD gradient flow on an input distribution `x`, and a labels distribution
    given either.

    by a function `func` (which computes the labels dynamically by applying `func` to `x`),
    or by a fixed labels tensor `target_tensor`. Objective function is 0.5 * MMD^2.
    :param func: labels function; applying it to the current iterate computes the labels. NOTE:
        pass either `func` OR `target_tensor`.
    :param x0: input distribution, tensor of shape N*d, where N is the number of data points and d their
        dimension; or of shape B*N*d, where B is a batch size.
    :param max_iter: Number of iterations.
    :param tol: Tolerance for stopping criterion.
    :param stop_mode: stopping criterion mode. Can be 'abs' for absolute error, or 'rel' for relative error.
    :param indexing: Either number of timestamps at which intermediate states of the flow are collected,
        or a list containing timestamps at which to save the state of the flow.
        If set to None or 0, no intermediate states are collected.
    :param stop_frequency: Frequency at which the stopping criterion is checked. Default is 50.
    :param stop_type: Type of stopping criterion. Can be 'mmd' for MMD, or 'grad' for gradient norm.
    :param target_tensor: labels tensor of the same shape as `x`. NOTE: Either pass `func` OR
        `target_tensor`.
    :param beta: Noise level to be added to iterates (cf. MMD gradient flow paper). `beta`=0 corresponds
        to no noise. Can also be a list containing noise levels for each iteration.
    :param gamma_beta: parameter of decay of beta. If `beta` is a float, then `beta` is multiplied by
        `gamma_beta` from one iteration to the next.
    :param witness_fn: witness function corresponding to the MMD kernel.
    :param kernel: kernel used for MMD.
    :param sigma: bandwidth of the Gaussian kernel. Either a float, in which case it decays with rate
        `gamma_sigma`, or a list with one sigma for each iteration.
    :param gamma_sigma: parameter of decay of `sigma`. Only used if `sigma` is a float.
    :param min_sigma: minimum value of sigma. If `sigma` is a list, then this parameter is ignored.
    :param optimizer: optimizer used for the flow. Currently 'adam' and 'sgd' are supported.
    :param optim_kwargs: dictionary containing additional arguments for the optimizer, such as `lr`.
    :param scheduler: learning rate scheduler. Currently 'steplr' is supported.
    :param scheduler_kwargs: dictionary containing additional arguments for the scheduler, such as `step_size`.
    :param reg: regularization parameter for the gradient flow.
    :param reg_p: regularizing by the p-norm of the points.
    :param callback: Callback used for computing errors during the flow. Can be set to None to not use a callback.
    :param timestamps_callback: Either number of timestamps at which callback is called, or list containing
        such timestamps.
    :param plot: if True, plots the flow at intermediate time steps after completion.
    :param plot_to_wandb: if True, plots the flow to wandb.
    :param plot_kwargs: dictionary containing plotting arguments, such as the wandb project name.
    :param progress_bar: if True, shows a progress bar during the flow.
    :param callback_kwargs: dictionary containing additional arguments for the callback function.
    :param mask: if True, apply a mask to the tensor that masks out zero entries.
    :param rescale_losses: if True, rescale the losses such that are all of the same magnitude.
        Only needed for multiple sigmas.
    :param autodiff: if True, use autodiff for computing the gradient directly from the loss. Defaults to False,
        in which case the gradient is computed as the gradient on the witness function.
    :param rescale_autodiff: if True, scales the gradient by the number of particles for autodiff.
    :param grad_mask: mask for the gradient. Uses the inverse of the mask passed to the function. If None, no mask is applied.
    :param wandb_dict: dictionary for logging to wandb.
    :return: dict containing the following keys:
                'intermediates': a collection of intermediate steps of the flow, e.g. for plotting.
                'iter_intermediates': contains the numbers of iterations corresponding to the entries in `intermediates`.
                'labels': the labels `target_tensor`, if given.
                'errors': Dict containing different error measures, defined by the `callback` function.
    """
    if grad_mask is not None:
        grad_mask = ~grad_mask.to(bool)
    if wandb_dict is None:
        wandb_dict = LoggingDict()
    x0 = x0.detach().clone()
    if x0 is None:
        raise ValueError("`x0` needs to be passed!")
    if plot and not indexing:
        raise ValueError("If `plot` is set to True, then `indexing` cannot be None.")
    if plot and not callback:
        raise ValueError(
            "If `plot` is set to True, then `callback` needs to be passed."
        )
    assert stop_mode in ["abs", "rel"], "`stop_mode` needs to be 'abs' or 'rel'!"
    if callback_kwargs is None:
        callback_kwargs = {}
    callback_kwargs['wandb_dict'] = wandb_dict
    initial_points = x0.detach().clone()
    errors_dict = defaultdict(list)
    log = {
        "labels": target_tensor,
        "intermediates": [],
        "iter_intermediates": [],
        "errors": errors_dict,
        "final_state": x0.detach().clone(),
        "iter_stop": max_iter,
    }
    if isinstance(beta, int) or isinstance(beta, float):
        beta = [beta]
    if isinstance(sigma, int) or isinstance(sigma, float):
        sigma = [sigma]
    if isinstance(min_sigma, int) or isinstance(min_sigma, float):
        min_sigma = [min_sigma]
    if isinstance(gamma_sigma, int) or isinstance(gamma_sigma, float):
        gamma_sigma = [gamma_sigma]
    if isinstance(gamma_beta, int) or isinstance(gamma_beta, float):
        gamma_beta = [gamma_beta]
    betas = [[b * g**i for b, g in zip(beta, gamma_beta)] for i in range(max_iter)]
    sigmas = [[s * g**i for s, g in zip(sigma, gamma_sigma)] for i in range(max_iter)]

    if min_sigma is not None:
        sigmas = [[max(z) for z in zip(s, min_sigma)] for s in sigmas]
    initial_sigmas = sigmas[0]
    last_sigmas = sigmas[-1]
    if func is not None:
        assert target_tensor is None, "Either `func` or `target_tensor` should be None!"
        with torch.no_grad():
            target = func(x0.detach().clone())
            target = target.detach().clone().requires_grad_(False)
    if target_tensor is not None:
        assert func is None, "Either `func` or `target_tensor` should be None!"
        target = target_tensor.detach().clone().requires_grad_(False)
    initial_target = target.detach().clone()

    crit_value, mmd_fcts = _update_crit_value(
        x0.detach().clone(),
        target,
        sigmas[0],
        kernel,
        mask,
        None,
        stop_type,
        stop_mode,
        rescale_losses,
        None,
        None,
        initial_points,
        initial_target,
    )
    log["stop_crit_value"] = crit_value

    if callback:
        callback_dict = callback(
            errors_dict,
            initial_points,
            x0.detach().clone(),
            target,
            sigmas=sigmas[0],
            initial_sigmas=initial_sigmas,
            last_sigmas=last_sigmas,
            kernel=kernel,
            plot_to_wandb=plot_to_wandb,
            crit_values=log["stop_crit_value"],
            stop_type=stop_type,
            stop_mode=stop_mode,
            step=0,
            plot_kwargs=plot_kwargs,
            mask=mask,
            **callback_kwargs,
        )
    if isinstance(indexing, int):
        indexing = np.linspace(0, max_iter - 1, indexing, dtype=int)
    elif indexing is None:
        indexing = []
    else:
        indexing = [i - 1 for i in indexing]
    if isinstance(timestamps_callback, int):
        timestamps_callback = np.linspace(
            0, max_iter - 1, timestamps_callback, dtype=int
        )
    else:
        timestamps_callback = [i - 1 for i in timestamps_callback]
        if timestamps_callback[0] != 0:
            timestamps_callback = [0] + timestamps_callback
    log["iter_errors"] = timestamps_callback + 1
    if len(log["iter_errors"] >= 1):
        log["iter_errors"][0] = 0
    if progress_bar:
        iter_flow = tqdm(range(max_iter))
    else:
        iter_flow = range(max_iter)

    if mask:
        x0_mask = get_mask(x0)
        if rescale_autodiff:
            n_particles_mask = x0_mask.sum(dim=-2).unsqueeze(-1)
    else:
        if rescale_autodiff:
            n_particles_mask = x0.shape[-2]

    with torch.enable_grad():
        x0.requires_grad = True

        if optimizer == "adam":
            if optim_kwargs is None:
                optim_kwargs = {"lr": 0.001}
            optimizer = Adam([x0], **optim_kwargs)
        elif optimizer == "sgd":
            if optim_kwargs is None:
                optim_kwargs = {"lr": 1}
            optimizer = SGD([x0], **optim_kwargs)
        else:
            raise ValueError("Only 'adam' and 'sgd' are supported as optimizers.")

        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        scheduler_kwargs["gamma"] = scheduler_kwargs.get("gamma", 1.0)
        scheduler_kwargs["step_size"] = scheduler_kwargs.get("step_size", 1)
        if scheduler == "steplr":
            scheduler = StepLR(optimizer, **scheduler_kwargs)
        else:
            raise ValueError("Only 'steplr' is supported as scheduler.")

        for i in iter_flow:
            optimizer.zero_grad()
            u = torch.randn_like(x0)
            x_regularized = [x0 + beta * u for beta in betas[i]]
            if mask:
                x_regularized = [x_r * x0_mask for x_r in x_regularized]
            if func is not None:
                if not autodiff:
                    losses = [
                        witness_fn(
                            x0.detach().clone(),
                            target,
                            x_r,
                            sigma=sigma,
                            mask_x=mask,
                            mask_y=mask,
                            mask_z=mask,
                            kernel=kernel,
                        )
                        - witness_fn(
                            x0.detach().clone(),
                            target,
                            func(x_r),
                            sigma=sigma,
                            mask_x=mask,
                            mask_y=mask,
                            mask_z=mask,
                            kernel=kernel,
                        )
                        for x_r, sigma in zip(x_regularized, sigmas[i])
                    ]
                else:
                    t_diff = [func(x_r) for x_r in x_regularized]
                    losses = [
                        0.5
                        * squared_mmd(
                            x_r, t, sigma=sigma, kernel=kernel, mask_x=mask, mask_y=mask
                        )
                        for x_r, t, sigma in zip(x_regularized, t_diff, sigmas[i])
                    ]
            else:
                if not autodiff:
                    losses = [
                        witness_fn(
                            x0.detach().clone(),
                            target,
                            x_r,
                            sigma=sigma,
                            mask_x=mask,
                            mask_y=mask,
                            mask_z=mask,
                            kernel=kernel,
                        )
                        for x_r, sigma in zip(x_regularized, sigmas[i])
                    ]
                else:
                    losses = [
                        0.5
                        * squared_mmd(
                            x_r,
                            target,
                            sigma=sigma,
                            kernel=kernel,
                            mask_x=mask,
                            mask_y=mask,
                        )
                        for x_r, sigma in zip(x_regularized, sigmas[i])
                    ]

            # if rescaling, compute fixed rescaling factors
            if rescale_losses and i == 0:
                with torch.no_grad():
                    fcts = [torch.mean(loss).item() for loss in losses]
                    fcts = [fct / max(fcts[0], 1e-8) for fct in fcts]
            if rescale_losses:
                losses = [loss / max(fct, 1e-8) for loss, fct in zip(losses, fcts)]
            loss = sum(losses) / len(losses)
            if reg > 0:
                loss += reg * sum([torch.norm(x_r, p=reg_p, dim=-1).mean(dim=-1) for x_r in x_regularized])
            grad = torch.autograd.grad(
                loss, x_regularized, grad_outputs=torch.ones_like(loss)
            )[0]
            if autodiff and rescale_autodiff:
                grad = grad * n_particles_mask
            if grad_mask is not None:
                grad = grad * grad_mask
            with torch.no_grad():
                x0.grad = grad
            optimizer.step()
            scheduler.step()

            if i == 0 and stop_type == "grad":
                mean_grad = torch.norm(grad).item()
                if stop_mode == "abs":
                    log["stop_crit_value"] = mean_grad
                else:
                    log["stop_crit_value"] = 1.0
            if i == 0 and stop_type != "grad":
                mean_grad = None

            log["final_state"] = x0.detach().clone()
            if func is not None:
                with torch.no_grad():
                    target = func(x0.detach().clone())
                    target = target.detach().clone().requires_grad_(False)
            if i in indexing:
                log["intermediates"].append(x0.detach().clone())
                log["iter_intermediates"].append(i + 1)

            if (i + 1) % stop_frequency == 0 and i > 0:
                # compute stopping criterion value and write it to log["stop_crit_value"]
                crit_value, _ = _update_crit_value(
                    x0.detach().clone(),
                    target,
                    sigmas[i],
                    kernel,
                    mask,
                    grad,
                    stop_type,
                    stop_mode,
                    rescale_losses,
                    mmd_fcts,
                    mean_grad,
                    initial_points,
                    initial_target,
                )

            if callback and i in timestamps_callback and i > 0:
                callback(
                    errors_dict,
                    initial_points,
                    x0.detach().clone(),
                    target,
                    sigmas=sigmas[i],
                    initial_sigmas=initial_sigmas,
                    last_sigmas=last_sigmas,
                    kernel=kernel,
                    plot_to_wandb=plot_to_wandb,
                    crit_value=crit_value,
                    stop_type=stop_type,
                    stop_mode=stop_mode,
                    step=i + 1,
                    plot_kwargs=plot_kwargs,
                    mask=mask,
                    **callback_kwargs,
                )

            if crit_value < tol:
                log["intermediates"].append(x0.detach().clone())
                log["iter_intermediates"].append(i + 1)
                log["iter_stop"] = i + 1
                log["stop_crit_value"] = crit_value
                break

            if i == max_iter - 1:
                crit_value, _ = _update_crit_value(
                    x0.detach().clone(),
                    target,
                    sigmas[i],
                    kernel,
                    mask,
                    grad,
                    stop_type,
                    stop_mode,
                    rescale_losses,
                    mmd_fcts,
                    mean_grad,
                    initial_points,
                    initial_target,
                )
                log["stop_crit_value"] = crit_value

        callback(
            errors_dict,
            initial_points,
            x0.detach().clone(),
            target,
            sigmas=sigmas[i],
            initial_sigmas=initial_sigmas,
            last_sigmas=last_sigmas,
            kernel=kernel,
            plot_to_wandb=plot_to_wandb,
            crit_value=crit_value,
            stop_type=stop_type,
            stop_mode=stop_mode,
            step=i + 1,
            plot_kwargs=plot_kwargs,
            mask=mask,
            **callback_kwargs,
        )

    if wandb.run is not None:
        log_dict = {
                f"MMD n_steps": log["iter_stop"],
        }
        wandb.log(log_dict)
        wandb_dict.append(log_dict)

    if func is not None and len(log["intermediates"]) > 0:
        # create a moving label from the last saved iterate of the flow
        with torch.no_grad():
            log["labels"] = func(log["intermediates"][-1])
            log["labels"] = log["labels"].detach().clone().requires_grad_(False)
    if plot:
        if plot_kwargs is None:
            plot_kwargs = {}

        plot_point_clouds_over_time(
            [initial_points] + log["intermediates"],
            log["labels"],
            titles=[f"Iteration {i}" for i in [0] + log["iter_intermediates"]],
            plot_to_wandb=plot_to_wandb,
            mask=mask,
            **plot_kwargs,
        )
        if (
            not plot_to_wandb
        ):  # if plot_to_wandb, the errors are plotted in the callback
            plot_errors(
                log["iter_errors"],
                log["errors"],
                info_dict=callback_dict,
                plot_to_wandb=False,
                **plot_kwargs,
            )
    # convert to torchdeq format
    log["nstep"] = torch.tensor(log["iter_stop"])  # torchdeq format
    log[f"{stop_mode}_lowest"] = torch.tensor(log["stop_crit_value"])  # torchdeq format
    try:
        log["initial_mmd"] = log["errors"]["MMD to targets inner loop"][0][0].item() if kernel == riesz_kernel else log["errors"]["MMD-1 to targets inner loop"][0][0].item()
        log["final_mmd"] = log["errors"]["MMD to targets inner loop"][-1][0].item() if kernel == riesz_kernel else log["errors"]["MMD-1 to targets inner loop"][-1][0].item()
    except:
        log["initial_mmd"] = None
        log["final_mmd"] = None
    solverstats = SolverStat(**log)
    # solverstats should be of the torchdeq format:
    # - nstep: int, number of steps until the stopping criterion is met, or until the best estimate
    # - abs_lowest: best iterate according to the absolute stopping criterion (only this or rel_lowest)
    # - rel_lowest: best iterate according to the relative stopping criterion (only this or abs_lowest)
    # - optionally could include `abs_trace` or `rel_trace`
    
    # return in same format as dorchdeq solvers
    return log["final_state"], log["intermediates"], solverstats
