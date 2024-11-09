import logging
from typing import Union
import traceback

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import wandb
from utils import get_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.WARNING,  # Set the logging level to WARNING
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a logger for warnings
logger = logging.getLogger(__name__)


def get_axis_limits(array_list, d=2):
    """
    Get the global axis limits for a list of 2D point clouds.

    Args:
        array_list (list): List of 2D tensors (each of shape N*2) or None.

    Returns:
        xmin, xmax, ymin, ymax (float): The minimum and maximum x and y values among all tensors.
    """
    array_list = [array for array in array_list if array is not None]
    array_list = [array.cpu().numpy() if isinstance(array, torch.Tensor) else array for array in array_list]
    if len(array_list) == 0:
        return np.array([-30 for _ in range(d)]), np.array([30 for _ in range(d)])
    array_list = np.concat(array_list)
    finite_mask = np.isfinite(array_list)
    mins = np.where(finite_mask, array_list, np.inf).min(axis=0)
    maxs = np.where(finite_mask, array_list, -np.inf).max(axis=0)

    # ensure no point is laying right on the border of the plot
    for i in range(len(mins)):
        if mins[i] < 0:
            mins[i] *= 1.1
        else:
            mins[i] *= 0.9
        if maxs[i] < 0:
            maxs[i] *= 0.9
        else:
            maxs[i] *= 1.1

    return mins, maxs


def plot_point_clouds_over_time(
    point_clouds: Union[list, torch.Tensor],
    target_distribution: torch.Tensor = None,
    titles: list = None,
    plot_to_wandb=False,
    init_wandb=False,
    init_wandb_kwargs=None,
    title="Point Clouds over Time",
    mask=False,
    max_samples=8,
    model_nb=None,
    exp_nb=None,
    epoch_nb=None,
    batch_nb=None,
    d_plot=2,
    methods="umap",
    method_kwargs=None,
    embedding_model=None,
    **kwargs,
):
    """Plots a sequence of point clouds over multiple time steps.

    :param point_clouds: A tensor of shape (k, N, 2) containing the point clouds, where k is
        the number of time steps and N is the number of points per cloud;
        can also be a tensor of shape (k, B, N, 2), where B is a batch dimension.
    :type point_clouds: torch.Tensor
    :param target_distribution: A tensor of shape (N, 2) containing the labels distribution.
        If provided, it will be plotted as an additional subplot with red dots.
        Can also be a tensor of shape (B, N, 2) for a batch size B.
    :type target_distribution: torch.Tensor, optional
    :param titles: list of title names for each plot. If None, then plots are enumerated.
    :param xmin: Minimum x-axis value for the plot.
    :param xmax: Maximum x-axis value for the plot.
    :param ymin: Minimum y-axis value for the plot.
    :param ymax: Maximum y-axis value for the plot.
    :param plot_to_wand: If True, the plot is logged to WandB. Default is False.
    :param title: Title of the plot. Default is "Point Clouds over Time".
    :return: None. The function creates k subplots arranged horizontally, one for each time
        step in the point clouds. Each subplot shows the point cloud at a given time step
        with blue dots. If a labels distribution is provided, it is plotted in an additional
        subplot to the right of the other plots with red dots and labeled "Target".
    """
    if type(methods) == str:
        methods = [methods]
    if method_kwargs is None:
        method_kwargs = {}
    if "tsne" in methods:
        perplexity = method_kwargs.get("perplexity", [2, 30])
        if isinstance(perplexity, int):
            perplexity = [perplexity]
    if "umap" in methods:
        n_neighbors = method_kwargs.get("n_neighbors", [2, 30])
        if isinstance(n_neighbors, int):
            n_neighbors = [n_neighbors]
    if isinstance(point_clouds, list):
        if embedding_model is not None:
            point_clouds = [embedding_model(cloud, reverse=True).detach() for cloud in point_clouds]
        point_clouds = torch.stack(point_clouds).squeeze(1)
    else:
        if embedding_model is not None:
            point_clouds = embedding_model(point_clouds, reverse=True).detach()
    if point_clouds.dim() == 3:
        k, N, d_z = point_clouds.shape
        B = 1
        point_clouds = point_clouds[None, :]
        target_distribution = (
            target_distribution[None, :] if target_distribution is not None else None
        )
    else:
        point_clouds = point_clouds.permute(1, 0, 2, 3)
        _, k, N, d_z = point_clouds.shape
        B = min(point_clouds.shape[0], max_samples)
        point_clouds = point_clouds.detach().clone()[:max_samples]
        if target_distribution is not None:
            target_distribution = target_distribution[:max_samples]
    if embedding_model is not None and target_distribution is not None:
        target_distribution = embedding_model(target_distribution, reverse=True).detach()

    if batch_nb is not None:
        batch_nb = f" (batch {batch_nb})"
    else:
        batch_nb = ""
    if epoch_nb is not None:
        epoch_nb = f" (epoch {epoch_nb})"
    else:
        epoch_nb = ""

    if plot_to_wandb:
        try:
            wandb_title = f"{exp_nb} - {title}"
            title = f"{model_nb}_{exp_nb} - {title}"
        except:
            raise ValueError(
                "model_nb and exp_nb must be provided for logging to WandB."
            )
        wandb_title += batch_nb
        wandb_title += epoch_nb
    
    if init_wandb:
        if init_wandb_kwargs is None:
            init_wandb_kwargs = {
                "project": "test_exp",
                "name": wandb_title,
                "id": wandb_title,
                "resume": "allow",
            }
        wandb.init(**init_wandb_kwargs)

    title += batch_nb
    title += epoch_nb
    title += f" - d_z={d_z}"

    # Determine the figure size
    rows = B
    cols = k + (1 if target_distribution is not None else 0)

    if d_z <= d_plot:
        methods = [None]

    for method in methods:
        if method == "tsne":
            values = perplexity
        elif method == "umap":
            values = n_neighbors
        else:
            values = [None]
        for value in values:
            try:
                fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), subplot_kw={'projection': '3d'} if d_plot == 3 else None)

                # Flatten axes array if only one row
                if B == 1:
                    axes = [axes]

                # Plot each time step
                for b in range(B):
                    current_clouds = [point_clouds[b, i] for i in range(k)]
                    current_mask = get_mask(point_clouds[b]).squeeze() if mask else None
                    N = current_mask[0].sum().item() if mask else N
                    current_target = (
                        target_distribution[b] if target_distribution is not None else None
                    )
                    target_mask = get_mask(current_target).squeeze() if mask else None
                    if mask:
                        current_clouds = [
                            cloud[current_mask[i]] for i, cloud in enumerate(current_clouds)
                        ]
                        current_target = (
                            current_target[target_mask] if target_distribution is not None else None
                        )
                    if target_distribution is not None:
                        current_clouds = torch.cat(current_clouds + [current_target], dim=0).cpu().numpy()
                    else:
                        current_clouds = torch.cat(current_clouds, dim=0).cpu().numpy()
                    if d_z > d_plot:
                        if method == "umap":
                            current_clouds = umap.UMAP(n_components=d_plot, n_neighbors=value, random_state=42).fit_transform(
                                current_clouds
                            )
                        elif method == "pca":
                            current_clouds = PCA(n_components=d_plot, random_state=42).fit_transform(
                                current_clouds
                            )
                        elif method == "tsne":
                            current_clouds = TSNE(n_components=d_plot, perplexity=value, random_state=42).fit_transform(
                                current_clouds
                            )
                    mins, maxs = get_axis_limits([current_clouds])
                    for i in range(k):
                        if d_plot == 2:
                            axes[b][i].scatter(
                                current_clouds[i * N : (i + 1) * N, 0],
                                current_clouds[i * N : (i + 1) * N, 1],
                                color="blue",
                            )
                            if b == 0:
                                try:
                                    axes[b][i].set_title(titles[i])
                                except:
                                    axes[b][i].set_title(f"Time step {i + 1}")
                            axes[b][i].set_xlim(mins[0], maxs[0])
                            axes[b][i].set_ylim(mins[1], maxs[1])
                        elif d_plot == 3:
                            axes[b][i].scatter(
                                current_clouds[i * N : (i + 1) * N, 0],
                                current_clouds[i * N : (i + 1) * N, 1],
                                current_clouds[i * N : (i + 1) * N, 2],
                                color="blue",
                            )
                            axes[b][i].set_xlabel("X")
                            axes[b][i].set_ylabel("Y")
                            axes[b][i].set_zlabel("Z")
                            if b == 0:
                                try:
                                    axes[b][i].set_title(titles[i])
                                except:
                                    axes[b][i].set_title(f"Time step {i + 1}")
                            axes[b][i].set_xlim(mins[0], maxs[0])
                            axes[b][i].set_ylim(mins[1], maxs[1])
                            axes[b][i].set_zlim(mins[2], maxs[2])
                    if target_distribution is not None:
                        if d_plot == 2:
                            axes[b][-1].scatter(
                                current_clouds[-N:, 0], current_clouds[-N:, 1], color="red"
                            )
                            axes[b][-1].set_xlim(mins[0], maxs[0])
                            axes[b][-1].set_ylim(mins[1], maxs[1])
                        elif d_plot == 3:
                            axes[b][-1].scatter(
                                current_clouds[-N:, 0],
                                current_clouds[-N:, 1],
                                current_clouds[-N:, 2],
                                color="red",
                            )
                            axes[b][-1].set_xlabel("X")
                            axes[b][-1].set_ylabel("Y")
                            axes[b][-1].set_zlabel("Z")
                            axes[b][-1].set_xlim(mins[0], maxs[0])
                            axes[b][-1].set_ylim(mins[1], maxs[1])
                            axes[b][-1].set_zlim(mins[2], maxs[2])
                        if b == 0:
                            axes[b][-1].set_title("F(last iterate)")

                
                if method == "tsne":
                        value_string = f", perplexity={value}"
                elif method == "umap":
                    value_string = f", n_neighbors={value}"
                else:
                    value_string = ""
                method_string = f", {method}" if d_z > d_plot else ""
                fig.suptitle(title + f"{method_string}{value_string}", fontsize="x-large")
                plt.tight_layout()

                if plot_to_wandb:
                    wandb.log({wandb_title: wandb.Image(fig)})
                    plt.close(fig)
                else:
                    plt.show()
            except Exception as e:
                print("An error occurred:")
                traceback.print_exc()
                logger.warning(
                    f"Failed to plot {method} in plot_point_clouds_over_time with error: {e}"
                )
    if init_wandb:
        wandb.finish()


def plot_point_clouds(
    point_clouds: Union[list, torch.Tensor],
    targets: Union[list, torch.Tensor] = None,
    title: str = "3D Point Clouds",
    plot_to_wandb: bool = False,
    init_wandb: bool = False,
    init_wandb_kwargs: dict = None,
    mask: bool = False,
    max_samples: int = 9,
    model_nb: str = "",
    exp_nb: str = "",
    epoch_nb: int = None,
    batch_nb: int = None,
    fade_factor: float = 0.5,
    point_size: int = 20,
    **kwargs
):
    # If point_clouds is a list, stack them into a tensor
    if isinstance(point_clouds, list):
        point_clouds = torch.stack([torch.tensor(cloud, device=device) if isinstance(cloud, np.ndarray) else cloud for cloud in point_clouds])
    elif isinstance(point_clouds, np.ndarray):
        point_clouds = torch.tensor(point_clouds, device=device)

    # Handle 2D (N*2 or N*3) and 3D (B*N*2 or B*N*3) point cloud shapes
    if point_clouds.dim() == 2:
        point_clouds = point_clouds.unsqueeze(0)  # Treat a single point cloud as a batch of 1

    B = min(point_clouds.shape[0], max_samples)  # Limit to max_samples

    # If targets is provided, process it similarly to point_clouds
    if targets is not None:
        if isinstance(targets, list):
            targets = torch.stack(targets)
        if targets.dim() == 2:  # Single point cloud case
            targets = targets.unsqueeze(0)

    # Handle batch and epoch numbering
    if batch_nb is not None:
        batch_nb = f" (batch {batch_nb})"
    else:
        batch_nb = ""
    if epoch_nb is not None:
        epoch_nb = f" (epoch {epoch_nb})"
    else:
        epoch_nb = ""

    # WandB title handling
    if plot_to_wandb:
        try:
            wandb_title = f"{exp_nb} - {title}"
            title = f"{model_nb}_{exp_nb} - {title}"
        except:
            raise ValueError("model_nb and exp_nb must be provided for logging to WandB.")
        wandb_title += batch_nb
        wandb_title += epoch_nb

    if init_wandb:
        if init_wandb_kwargs is None:
            init_wandb_kwargs = {
                "project": "test_exp",
                "name": wandb_title,
                "id": wandb_title,
                "resume": "allow",
            }
        wandb.init(**init_wandb_kwargs)

    title += batch_nb
    title += epoch_nb

    # Setup for plotting in a square grid
    grid_size = int(np.ceil(np.sqrt(B)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10), subplot_kw={'projection': '3d'} if point_clouds.shape[2] == 3 else None)
    if grid_size == 1:
        axs = [axs]

    for b in range(B):
        # Apply mask if needed
        cloud = point_clouds[b]
        if mask:
            current_mask = get_mask(cloud).squeeze()
            cloud = cloud[current_mask]
            if targets is not None:
                target = targets[b][current_mask]
            else:
                target = None
        else:
            if targets is not None:
                target = targets[b]
            else:
                target = None

        ax = axs[b // grid_size, b % grid_size]

        # Convert to CPU for plotting
        cloud = cloud.cpu().numpy()
        if target is not None:
            target = target.cpu().numpy().squeeze()

        # Calculate alpha based on Y-coordinate for each point
        y_coords = cloud[:, 1]  # Y coordinates (the second dimension)
        alpha_values = np.clip(1 - fade_factor * (y_coords - np.min(y_coords)) / (np.ptp(y_coords) + 1e-5), 0, 1)

        # 2D or 3D plotting
        if cloud.shape[1] == 3:  # 3D point cloud
            if target is not None:
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=target, cmap='tab20', s=point_size, alpha=alpha_values)
            else:
                ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c='b', s=point_size, alpha=alpha_values)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        elif cloud.shape[1] == 2:  # 2D point cloud
            if target is not None:
                ax.scatter(cloud[:, 0], cloud[:, 1], c=target, cmap='tab20', s=point_size, alpha=alpha_values)
            else:
                ax.scatter(cloud[:, 0], cloud[:, 1], c='b', s=point_size, alpha=alpha_values)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        ax.set_title(f"Sample {b + 1}")

    # Hide any extra subplots
    for i in range(B, grid_size * grid_size):
        axs[i // grid_size, i % grid_size].axis('off')

    fig.suptitle(title)
    plt.tight_layout()

    # Log to WandB or display the plot
    if plot_to_wandb:
        wandb.log({wandb_title: wandb.Image(fig)})
        plt.close(fig)
    else:
        plt.show()
    if init_wandb:
        wandb.finish()


def plot_errors(
    timestamps,
    errors,
    info_dict,
    title="Evolution of Errors over Time for MMD Flow",
    plot_to_wandb=False,
    **kwargs,
):
    """Creates plots for varying errors coming from an `errors` dict.

    :param timestamps: list containing x-values.
    :param errors: dict containing lists as values.
    :param info_dict: dictionary containing information such as `nrows`, `ncols`, `conf`.
    :param title: Title of the plot.
    :param plot_to_wandb: If True, the plot is logged to WandB. Default is False.
    :return: None.
    """
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.cpu().numpy()
    num_plots = len(errors)
    if num_plots == 0:
        print("No errors to plot.")
        return

    if plot_to_wandb:
        try:
            model_nb = kwargs["model_nb"]
            exp_nb = kwargs["exp_nb"]
            title = f"{model_nb}_{exp_nb} - {title}"
            wandb_title = f"{exp_nb} - {title}"
        except:
            raise ValueError(
                "model_nb and exp_nb must be provided for logging to WandB."
            )

    # Determine the layout of the plots
    if "nrows" not in info_dict.keys():
        if num_plots <= 3:
            cols = num_plots
            rows = 1
        else:
            cols = int(np.ceil(np.sqrt(num_plots)))
            rows = int(np.ceil(num_plots / cols))
    else:
        cols = info_dict["ncols"]
        rows = info_dict["nrows"]

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for idx, (key, values) in enumerate(errors.items()):
        if isinstance(values, torch.Tensor):
            values = values.cpu().numpy()
        ax = axes[idx]
        if "conf" not in info_dict.keys():
            ax.plot(timestamps, values)
        else:
            errors = [value[0] for value in values]
            lower = [value[1] for value in values]
            upper = [value[2] for value in values]
            ax.plot(timestamps, errors, color="blue")
            ax.fill_between(
                timestamps,
                lower,
                upper,
                color="blue",
                alpha=0.2,
                label=f"{info_dict['conf']} Conf. Int.",
            )
        ax.set_title(f"Evolution of {key}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel(key)
        ax.set_ylim(0, max(max(v) for v in values) * 1.1)

    # Remove any empty subplots
    for i in range(len(errors), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if plot_to_wandb:
        wandb.log({wandb_title: wandb.Image(fig)})
        plt.close(fig)
    else:
        plt.show()
