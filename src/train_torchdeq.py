import argparse
import logging
import json
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchdeq.solver as solver

import networks
import wandb
from gradientflow import mmd_gf, squared_mmd, riesz_kernel
from mnist import load_mnist
from networks import TorchDEQModel
from utils import get_mask, mask_batch, LoggingDict, target_completion, set_seed
from modelnet import load_modelnet_saved
from get_hyperparams import get_hyperparams_classifier_mnist, get_hyperparams_classifier_modelnet, get_hyperparams_completion_mnist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(
    level=logging.WARNING,  # Set the logging level to WARNING
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a logger for warnings
logger = logging.getLogger(__name__)

# Register the custom solver
solver.register_solver("mmd", mmd_gf)


def chamfer_distance(pred, target, **kwargs):
    """
    Computes the Chamfer Distance between the predicted point cloud and the target point cloud, accounting for padding.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, N, d).
        target (torch.Tensor): Target tensor of shape (B, M, d).
        pred_mask (torch.Tensor): Mask for valid points in pred, shape (B, N), where 1 indicates a valid point and 0 indicates padding.
        target_mask (torch.Tensor): Mask for valid points in target, shape (B, M), where 1 indicates a valid point and 0 indicates padding.

    Returns:
        torch.Tensor: Chamfer distance loss (scalar).
    """
    B, N, d = pred.shape
    M = target.shape[1]  # Number of points in target
    pred_mask = get_mask(pred).squeeze()
    target_mask = get_mask(target).squeeze()

    # Expand pred and target to (B, N, M, d) for pairwise distance computation
    pred_expanded = pred.unsqueeze(2).expand(B, N, M, d)
    target_expanded = target.unsqueeze(1).expand(B, N, M, d)

    # Compute pairwise distances between pred and target points (B, N, M)
    dist_matrix = torch.norm(pred_expanded - target_expanded, dim=-1, p=2)

    # Mask out padded points in pred and target
    pred_mask_expanded = pred_mask.unsqueeze(2).expand(B, N, M)
    target_mask_expanded = target_mask.unsqueeze(1).expand(B, N, M)

    # Forward loss: for each point in pred, find the min distance to any point in target (masked)
    forward_loss = torch.min(dist_matrix + (~target_mask_expanded) * 1e6, dim=2)[0]  # Mask invalid points in target
    forward_loss = (forward_loss * pred_mask).sum(dim=1) / pred_mask.sum(dim=1)  # Mean over valid points in pred

    # Backward loss: for each point in target, find the min distance to any point in pred (masked)
    backward_loss = torch.min(dist_matrix + (~pred_mask_expanded) * 1e6, dim=1)[0]  # Mask invalid points in pred
    backward_loss = (backward_loss * target_mask).sum(dim=1) / target_mask.sum(dim=1)  # Mean over valid points in target

    # Total Chamfer distance: average over the batch
    chamfer_loss = (forward_loss + backward_loss).mean()

    return chamfer_loss, None


def loss_mmd(y, target, sigma=1.0, kernel=riesz_kernel, mask=False, rescale=False, fcts=None, **kwargs):
    """Compute the MMD loss between the point cloud z and the target point cloud."""
    if isinstance(sigma, list):
        mmds = []
        for s in sigma:
            mmds.append(
                squared_mmd(y, target, sigma=s, kernel=kernel, mask_x=mask, mask_y=True).mean()
            )
        if rescale:
            if fcts is None:
                fcts = [mmd.item() for mmd in mmds]
                fcts = [fct / max(fcts[0], 1e-8) for fct in fcts]
            mmds = [mmd / max(fct, 1e-8) for mmd, fct in zip(mmds, fcts)]
        loss = sum(mmds) / len(mmds)

    else:
        loss = squared_mmd(y, target, sigma=sigma, kernel=kernel, mask_x=mask, mask_y=True).mean()
        fcts = None

    return loss, fcts


def compute_entropy_kde_gaussian(data, bandwidth=1.0, mask=None):
    """
    Compute the entropy of each sample in a batch of data using Kernel Density Estimation (KDE).
    This function is fully vectorized over the batch dimension and supports backpropagation.

    Parameters:
    ----------
    data : torch.Tensor
        Input data of shape (B, N, d), where B is the number of samples,
        N is the number of points in each sample, and d is the dimensionality.
    bandwidth : float or torch.Tensor
        The bandwidth parameter for KDE.

    Returns:
    -------
    entropies : torch.Tensor
        A tensor of shape (B,) containing the entropy of each sample.
    """
    B, N, d = data.shape
    h = bandwidth

    # Compute pairwise differences in a vectorized manner
    delta = data.unsqueeze(2) - data.unsqueeze(1)  # Shape: (B, N, N, d)
    squared_distances = (delta ** 2).sum(-1)  # Shape: (B, N, N)
    if mask is not None:
        mask = mask.squeeze()
        mask = mask.unsqueeze(2)&mask.unsqueeze(1)
        squared_distances = squared_distances.masked_fill(~mask.bool(), 1e8)
    # Compute the Gaussian kernel values
    # Constants are omitted as they cancel out in entropy calculation
    K = torch.exp(-squared_distances / (2 * h ** 2))  # Shape: (B, N, N)
    diag_matrix = torch.eye(N).bool().unsqueeze(0).expand(B, N, N).to(device)
    K = K.masked_fill(diag_matrix, 0)  # Zero out diagonal elements
    # Estimate the density at each point
    p = K.sum(-1) / N  # Shape: (B, N)

    # Compute the log probability
    log_p = torch.log(p + 1e-10)  # Shape: (B, N), adding epsilon to prevent log(0)

    # Compute the entropy for each sample
    entropies = -log_p.mean(-1)  # Shape: (B,)

    return entropies.mean()


def loss_nll(y, labels, **kwargs):
    return F.nll_loss(y, labels), None


def loss_l2(y, target, **kwargs):
    return F.mse_loss(y, target), None


def loss_cross_entropy(y, labels, **kwargs):
    return F.cross_entropy(y, labels), None


def compute_accuracy(logits, labels):
    """
    Compute the accuracy given raw logits and labels.

    Args:
        logits (torch.Tensor): The raw logits of shape (B, k), where B is the batch size
                               and k is the number of classes.
        labels (torch.Tensor): Ground-truth labels of shape (B,), with class indices.

    Returns:
        float: The accuracy of the predictions as a percentage.
    """
    # Step 1: Get predicted class indices (the class with the maximum logit value)
    predictions = torch.argmax(logits, dim=1)

    # Step 2: Compare predictions with labels
    correct_predictions = (predictions == labels).sum().item()

    # Step 3: Compute accuracy as a percentage
    total_samples = labels.size(0)
    accuracy = correct_predictions / total_samples * 100

    return accuracy


def train(
    model,
    train_loader,
    test_loader,
    optimizer,
    target_fn,
    target_kwargs,
    loss_fn,
    dim_x,
    n_epochs,
    max_batches,
    log_interval,
    use_wandb=True,
    stop_mode="rel",
    config=None,
    loss_kwargs=None,
    track_accuracy=False,
    outf=None,
    scheduler_kwargs=None,
    wandb_dict=None,
    fix_encoders=False,
):
    regularizing_fn = config.get("regularizing_fn", None)
    regularizing_const = config.get("regularizing_const", 0)
    adaptive_max_iter = config.get("adaptive_max_iter", False)
    if wandb_dict is None:
        wandb_dict = LoggingDict()
    if scheduler_kwargs is None:
        if len(train_loader) > 500:
            scheduler_kwargs = {"step_size": 1, "gamma": 0.3}
        else:
            scheduler_kwargs = {"step_size": 2, "gamma": 0.5}
    scheduler = optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)
    if loss_kwargs is None:
        loss_kwargs = {}
    fcts = None
    mask = config.get("mask", False)
    n_batches = min(len(train_loader), max_batches)

    for epoch in range(n_epochs):
        if epoch == 15 and adaptive_max_iter and config["dataset"] == "modelnet":
            model.solver_kwargs["optim_kwargs"]["lr"] = 1
            model.solver_kwargs["f_max_iter"] = 1000
        elif epoch == 4 and adaptive_max_iter and config["dataset"] == "mnist":
            model.solver_kwargs["optim_kwargs"]["lr"] = 1
            model.solver_kwargs["f_max_iter"] = 1000

        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= max_batches:
                break

            model.train()  # Switch back to training mode

            points, labels = mask_batch(batch, mask=True, dim=dim_x)
            with torch.no_grad():
                target = target_fn(points, labels, mask=True, **target_kwargs)

            if target_fn == target_completion:
                target = points.clone()
                points = target_completion(points, labels, return_input=False, **target_kwargs)
                grad_mask = get_mask(points)
                model.solver_kwargs["grad_mask"] = grad_mask
            else:
                grad_mask = None
            B = points.size(0)

            optimizer.zero_grad()
            _, info, y = model(points, fix_mask=grad_mask, fix_encoders=fix_encoders)
            
            loss, fcts = loss_fn(
                y, target=target, labels=labels, mask=mask, fcts=fcts, **loss_kwargs
            )

            if regularizing_fn is not None:
                reg_mask = get_mask(y) & ~grad_mask
                loss -= regularizing_const * regularizing_fn(y, mask=reg_mask)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log the training loss every log_interval batches and log losses
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"Batch {batch_idx + 1}/{max_batches} - Training Loss: {loss.item():.4f}"
                )
                if use_wandb:
                    stop_iter = info["nstep"].float().mean().item()
                    stop_value = info[f"{stop_mode}_lowest"].float().mean().item()
                    if stop_iter < torch.inf:
                        log_dict = {
                            "Training Loss": loss.item(),
                            "Batch": epoch * n_batches + batch_idx + 1,
                            "Training Iter Stop": stop_iter,
                            "Training Stop Crit Value": stop_value,
                        }
                    else:
                        log_dict = {
                            "Training Loss": loss.item(),
                            "Batch": epoch * n_batches + batch_idx + 1,
                        }
                    if info.get("initial_mmd", None) is not None:
                        log_dict["Initial MMD outer loop"] = info["initial_mmd"]
                        log_dict["Final MMD outer loop"] = info["final_mmd"]
                    if track_accuracy:
                        accuracy = compute_accuracy(y, labels)
                        log_dict["Training Accuracy"] = accuracy
                    wandb.log(log_dict)
                    wandb_dict.append(log_dict)

                # Compute the test loss on a single batch
                model.eval()
                with torch.no_grad():
                    test_batch = next(iter(test_loader))
                    test_points, test_labels = mask_batch(
                        test_batch, 
                        mask=True, 
                        dim=dim_x,
                    )
                    target = target_fn(test_points, test_labels, mask=True, **target_kwargs)

                    if target_fn == target_completion:
                        target = test_points.clone()
                        test_points = target_completion(test_points, test_labels, return_input=False, **target_kwargs)
                        grad_mask = get_mask(test_points)
                        model.solver_kwargs["grad_mask"] = grad_mask
                    else:
                        grad_mask = None

                    _, test_info, y_test = model(test_points, fix_mask=grad_mask, fix_encoders=fix_encoders) 

                    test_loss, _ = loss_fn(
                        y_test, target, mask=mask, fcts=fcts, **loss_kwargs
                    )

                    print(f"Test Loss on a Single Batch: {test_loss.item():.4f}")
                    if use_wandb:
                        if test_info["nstep"].float().mean() < torch.inf:
                            log_dict = {
                                "Test Loss": test_loss.item(),
                                "Batch": epoch * n_batches + batch_idx + 1,
                                "Test Iter Stop": test_info["nstep"]
                                .float()
                                .mean()
                                .item(),
                                "Test Stop Crit Value": test_info[f"{stop_mode}_lowest"]
                                .float()
                                .mean()
                                .item(),
                            }
                        else:
                            log_dict = {
                                "Test Loss": test_loss.item(),
                                "Batch": epoch * n_batches + batch_idx + 1,
                            }
                        if track_accuracy:
                            accuracy = compute_accuracy(y_test, test_labels)
                            print(f"Epoch {epoch+1} / {n_epochs}, batch {batch_idx+1} / {n_batches} - Test Accuracy: {accuracy:.4f}")
                            log_dict["Test Accuracy"] = accuracy
                        wandb.log(log_dict)
                        wandb_dict.append(log_dict)
        scheduler.step()
        torch.save(model.state_dict(), f"{outf}_epoch_{epoch+1}.pth")

    # return average loss in last epoch
    return total_loss / min(len(train_loader), max_batches)


def evaluate(
    model,
    test_loader,
    max_batches,
    target_fn,
    target_kwargs,
    loss_fn,
    dim_x,
    use_wandb=True,
    stop_mode="rel",
    config=None,
    loss_kwargs=None,
    track_accuracy=False,
    wandb_dict=None,
    fix_encoders=False,
):
    if wandb_dict is None:
        wandb_dict = LoggingDict()
    fcts_train = None
    fcts_eval = None
    if loss_kwargs is None:
        loss_kwargs = {}
    if config is None:
        config = {}

    mask = config.get("mask", False)
    total_loss_train = 0.0
    total_accuracy_train = 0.0
    total_loss_eval = 0.0
    total_accuracy_eval = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= max_batches:
                break

            # Extract and pad the points from the test batch
            points, labels = mask_batch(batch, mask=True, dim=dim_x)

            target = target_fn(points, labels, mask=True, **target_kwargs)

            if target_fn == target_completion:
                target = points.clone()
                points = target_completion(points, labels, return_input=False, **target_kwargs)
                grad_mask = get_mask(points)
                model.solver_kwargs["grad_mask"] = grad_mask
            else:
                grad_mask = None

            with torch.no_grad():
                model.train()
                _, info_train, y_train = model(points, fix_mask=grad_mask, fix_encoders=fix_encoders)
                model.eval()
                _, info_eval, y_eval = model(points, fix_mask=grad_mask, fix_encoders=fix_encoders)

                loss_train, fcts_train = loss_fn(y_train, target=target, labels=labels, mask=mask, fcts=fcts_train, **loss_kwargs)
                loss_eval, fcts_eval = loss_fn(y_eval, target=target, labels=labels, mask=mask, fcts=fcts_eval, **loss_kwargs)

                total_loss_train += loss_train.item()
                total_loss_eval += loss_eval.item()

            if use_wandb:
                log_dict = {}
                if info_train["nstep"].float().mean() < torch.inf:
                    log_dict = {
                        "Eval Iter Stop (train mode)": info_train["nstep"].float().mean().item(),
                        "Eval Stop Crit Value (train mode)": info_train[f"{stop_mode}_lowest"]
                        .float()
                        .mean()
                        .item(),
                    }
                if info_eval["nstep"].float().mean() < torch.inf:
                    log_dict.update(
                        {
                            "Eval Iter Stop (eval mode)": info_eval["nstep"].float().mean().item(),
                            "Eval Stop Crit Value (eval mode)": info_eval[f"{stop_mode}_lowest"]
                            .float()
                            .mean()
                            .item(),
                        }
                    )
                if track_accuracy:
                    accuracy_train = compute_accuracy(y_train, labels)
                    total_accuracy_train += accuracy_train
                    log_dict["Eval Accuracy (train mode)"] = accuracy_train
                    accuracy_eval = compute_accuracy(y_eval, labels)
                    total_accuracy_eval += accuracy_eval
                    log_dict["Eval Accuracy (eval mode)"] = accuracy_eval

                wandb.log(log_dict)
                wandb_dict.append(log_dict)
    total_accuracy_train /= min(len(test_loader), max_batches)
    total_accuracy_eval /= min(len(test_loader), max_batches)
    if track_accuracy and use_wandb:
        log_dict = {"Eval Average Accuracy (train mode)": total_accuracy_train, "Eval Average Accuracy (eval mode)": total_accuracy_eval}
        wandb.log(log_dict)
        wandb_dict.append(log_dict)

    return total_loss_train / min(len(test_loader), max_batches), total_loss_eval / min(len(test_loader), max_batches)


def main(args):
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, config=args)

    # Load the dataset
    train_loader, test_loader = load_mnist(batch_size=args.batch_size)

    # Initialize the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dynamically retrieve the DEQ class
    deq_class = getattr(networks, args.deq_model_class)

    loss_fn = globals()[args.loss_fn]
    target_fn = globals()[args.target_fn]

    # Instantiate the model
    model = TorchDEQModel(
        deq_model_class=deq_class,
        deq_model_kwargs=args.deq_model_kwargs,
        deq_kwargs=args.deq_kwargs,
        init_kwargs=args.init_kwargs,
        solver_kwargs=args.solver_kwargs,
    ).to(device)

    # Load the model if a path is provided
    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
        print(f"Model loaded from {args.load_model_path}")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    if not args.no_train:
        try:
            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch+1}/{args.epochs}...")
                train_loss = train(
                    model,
                    train_loader,
                    test_loader,
                    optimizer,
                    target_fn,
                    loss_fn,
                    device,
                    args.max_batches_per_epoch,
                    args.log_interval,
                    args.plot_interval,
                    not args.no_wandb,
                    args.shift_fct,
                )

                test_loss = evaluate(
                    model,
                    test_loader,
                    device,
                    args.max_batches_per_epoch,
                    target_fn,
                    loss_fn,
                    not args.no_wandb,
                    args.shift_fct,
                )
                if not args.no_wandb:
                    wandb.log(
                        {
                            "Epoch": epoch + 1,
                            "Average Train Loss": train_loss,
                            "Average Test Loss": test_loss,
                        }
                    )

                print(
                    f"Epoch {epoch+1}/{args.epochs} - Avg Train Loss: {train_loss:.4f} - Avg Test Loss: {test_loss:.4f}"
                )

        except KeyboardInterrupt:
            print("Training interrupted. Returning the current model...")

    # Save the model if a path is provided
    if args.save_model_path:
        torch.save(model.state_dict(), args.save_model_path)
        print(f"Model saved to {args.save_model_path}")

    return model, train_loader, test_loader, optimizer, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a DDEQ on MNIST or ModelNet40."
    )
    parser.add_argument("--job_id", type=str, default="test_run", help="job ID")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="path to model. Set to None for randomly initialized model",
    )
    parser.add_argument(
        "--batch_per_epoch",
        type=int,
        default=938,
        help="maximum number of batches per epoch",
    )
    parser.add_argument(
        "--batch_per_eval",
        type=int,
        default=100,
        help="number of batches per evaluation",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=30,
        help="interval for logging",
    )
    parser.add_argument(
        "--outf",
        type=str,
        default="/n/holyscratch01/dam_lab/Users/jgeuter/DEQ-GFs/",
        help="name of output folder",
    )
    parser.add_argument(
        "--projectname", type=str, default="test_exp", help="name of project"
    )
    parser.add_argument(
        "--stop_mode", type=str, default="rel", help="stop mode for DEQ, 'rel' or 'abs'"
    )
    parser.add_argument(
        "--hyperparam_fn",
        type=str,
        default="get_hyperparams_classifier_mnist",
        help="name of hyperparameter function",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="name of dataset, 'mnist', 'modelnet', 'modelnet-s', or 'modelnet-l'",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="only evaluate model",
    )

    args = parser.parse_args()
    hyperparam_fn = getattr(sys.modules[__name__], args.hyperparam_fn)

    print(args)

    set_seed(42)

    model, target_fn, loss_fn, config = hyperparam_fn(args)
    
    if loss_fn == "cross_entropy":
        loss_fn = loss_cross_entropy
    elif loss_fn == "mmd":
        loss_fn = loss_mmd
    if config.get("regularizing_fn", None) == "entropy_kde_gaussian":
        config["regularizing_fn"] = compute_entropy_kde_gaussian
    
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    
    config["hyperparam_fn"] = hyperparam_fn.__name__

    scheduler_kwargs = config.get("scheduler_kwargs", None)

    loss_kwargs = config.get("loss_kwargs", {})
    track_accuracy = config.get("track_accuracy", False)
    target_kwargs = config.get("target_kwargs", {})
    fix_encoders = config.get("fix_encoders", False)

    lr_adam = config.get("lr_adam", 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=lr_adam)
    run_id = args.job_id
    wandb.init(
        project=args.projectname,  # Specify the project (experiment) name
        name=run_id,  # Specify the run name
        id=run_id,  # allows for resuming the run
        config=config,  # Save the hyperparameters
        resume="allow",
    )

    wandbDict = LoggingDict()

    outf = (
        args.outf
        + args.projectname
        + "/logs/"
        + args.job_id
    )

    try:
        os.makedirs(outf)
    except OSError:
        pass

    batch_size = config.get("batch_size", 64)
    batch_per_epoch = args.batch_per_epoch

    if args.dataset is None:
        if hyperparam_fn == get_hyperparams_classifier_mnist:
            args.dataset = "mnist"
        elif hyperparam_fn == get_hyperparams_classifier_modelnet:
            args.dataset = "modelnet-s"
        elif hyperparam_fn == get_hyperparams_completion_mnist:
            args.dataset = "mnist"
    if args.dataset == "modelnet-s":
        train_loader, test_loader, label_to_class = load_modelnet_saved(size="s", batch_size=batch_size)
        p = 3
        n_classes = len(label_to_class)
    elif args.dataset == "modelnet":
        train_loader, test_loader, label_to_class = load_modelnet_saved(size="m", batch_size=batch_size)
        p = 3
        n_classes = len(label_to_class)
    elif args.dataset == "modelnet-l":
        train_loader, test_loader, label_to_class = load_modelnet_saved(size="l", batch_size=batch_size)
        p = 3
        n_classes = len(label_to_class)
    elif args.dataset == "mnist":
        train_loader, test_loader = load_mnist(batch_size=batch_size)
        p = 2
        n_classes = 10
        label_to_class = None

    if not args.eval_only:
        train_loss = train(
            model,
            train_loader,
            test_loader,
            optimizer,
            target_fn,
            target_kwargs,
            loss_fn,
            p,
            args.n_epochs,
            batch_per_epoch,
            args.log_interval,
            True,
            stop_mode=args.stop_mode,
            config=config,
            loss_kwargs=loss_kwargs,
            track_accuracy=track_accuracy,
            outf=outf,
            scheduler_kwargs=scheduler_kwargs,
            wandb_dict=wandbDict,
            fix_encoders=fix_encoders,
        )

    test_losses = evaluate(
        model,
        test_loader,
        args.batch_per_eval,
        target_fn,
        target_kwargs,
        loss_fn,
        p,
        True,
        stop_mode=args.stop_mode,
        config=config,
        loss_kwargs=loss_kwargs,
        track_accuracy=track_accuracy,
        wandb_dict=wandbDict,
        fix_encoders=fix_encoders,
    )

    log_dict = {
            "Average Train Loss": train_loss,
            "Average Test Loss (train mode)": test_losses[0],
            "Average Test Loss (eval mode)": test_losses[1],
    }
    wandb.log(log_dict)
    wandbDict.append(log_dict)

    with open(outf + "_wandbDict.json", "w") as f:
        json.dump(wandbDict, f)

    wandb.finish()
