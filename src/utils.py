import random

import numpy as np
import scipy.stats
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LoggingDict(defaultdict):
    def __init__(self):
        super(LoggingDict, self).__init__(list)

    def __str__(self):
        return str(dict(self))
    
    def __repr__(self):
        return str(dict(self))
    
    def append(self, dct):
        for key, value in dct.items():
            if type(value) == list:
                self[key].extend(value)
            else:
                self[key].append(value)


def scale_mask(mask, tau=1.19):
    """
    Scale the number of True entries in the mask by multiplying by tau, while ensuring that all original True
    entries remain True. Additional True entries are filled from the start of the mask (cyclic filling).

    Args:
        mask (torch.Tensor): Tensor of shape (B, N, 1) containing boolean values.
        tau (float): A scaling factor (tau >= 1).

    Returns:
        torch.Tensor: A new mask of the same shape (B, N, 1), where the number of True entries is scaled by tau.
    """
    B, N, _ = mask.shape  # Get dimensions of the input mask
    
    # Flatten the mask to (B, N) for easier processing
    mask_flat = mask.squeeze(-1)
    
    # Count the number of True values for each sample in the batch
    true_counts = mask_flat.sum(dim=1)  # Shape: (B,)
    
    # Determine the new number of True values based on tau
    scaled_true_counts = (true_counts * tau).long().clamp(max=N)  # Ensure the count does not exceed N
    
    # Create the output mask initialized with the original mask
    new_mask = mask_flat.clone()
    
    # Iterate over each sample in the batch and update the mask
    for i in range(B):
        current_true_count = true_counts[i].item()  # Number of original True values
        new_true_count = scaled_true_counts[i].item()  # Scaled number of True values

        # If the new count is greater than the current count, we need to add more True entries
        if new_true_count > current_true_count:
            # Get the indices of the False entries in the mask
            false_indices = torch.nonzero(~new_mask[i], as_tuple=False).squeeze()
            
            # Fill additional True values starting from the beginning (cyclic filling)
            additional_true_count = new_true_count - current_true_count
            new_true_indices = false_indices[:additional_true_count]
            new_mask[i, new_true_indices] = True

    # Return the reshaped new mask to match (B, N, 1)
    return new_mask.unsqueeze(-1)


def get_mask(x, dims=-1):
    """
    Create a binary mask where rows with non-zero values across the last dimension are marked as 1.

    Args:
        x (np.ndarray): Input array of shape (B, N, d).

    Returns:
        np.ndarray: A mask of shape (B, N, 1) where non-zero rows across the last dimension are 1.
    """
    if isinstance(x, torch.Tensor):
        return (x.abs().sum(dim=dims) > 0).unsqueeze(dims)
    # Compute the absolute sum along the last dimension and check if it's greater than 0
    mask = (np.abs(x).sum(axis=dims) > 0)
    
    # Expand the mask to have the shape (B, N, 1)
    return np.expand_dims(mask, axis=dims).astype(float)


def masked_std(tensor, dims, t_mask=None, keepdim=True):
    """
    Compute the standard deviation of the tensor along specified dimensions,
    ignoring zero tokens if t_mask is passed.

    Args:
        tensor (torch.Tensor): The input tensor of shape (B, N, d).
        dim (tuple): Dimensions along which to compute the standard deviation.
        t_mask (torch.Tensor): (optional) mask tensor.

    Returns:
        torch.Tensor: Standard deviation of the tensor along the specified dimensions.
    """
    if isinstance(tensor, np.ndarray):
        return _masked_std_numpy(tensor, dims, t_mask, keepdim)
    if t_mask is not None:
        if t_mask is True:
            t_mask = get_mask(tensor)
        masked_mean = compute_masked_mean(tensor, dims, t_mask).unsqueeze(dims)
        var = compute_masked_mean((tensor - masked_mean) ** 2, dims, t_mask)
        var = torch.sqrt(var)
    else:
        var = tensor.std(dim=dims)
    if keepdim:
        return var.unsqueeze(dims)
    return var

def _masked_std_numpy(tensor, dims, t_mask=None, keepdim=True):
    """
    Compute the standard deviation of the numpy array along specified dimensions,
    ignoring zero tokens if t_mask is passed.

    Args:
        tensor (np.ndarray): The input array of shape (B, N, d).
        dims (tuple): Dimensions along which to compute the standard deviation.
        t_mask (np.ndarray): (optional) mask array.

    Returns:
        np.ndarray: Standard deviation of the array along the specified dimensions.
    """
    if t_mask is not None:
        masked_mean = compute_masked_mean(tensor, dims, t_mask, keepdim=True)
        var = compute_masked_mean((tensor - masked_mean) ** 2, dims, t_mask)
        std = np.sqrt(var)
    else:
        std = np.std(tensor, axis=dims, keepdims=keepdim)
    
    if keepdim:
        return np.expand_dims(std, axis=dims)
    
    return std


def compute_masked_max(tensor, dims, t_mask=None, keepdim=False):
    """
    Compute the maximum of the tensor along specified dimensions,
    ignoring zero tokens if t_mask is passed.

    Args:
        tensor (torch.Tensor): The input tensor of shape (B, N, d).
        dim (tuple): Dimensions along which to compute the maximum.
        t_mask (torch.Tensor): (optional) mask tensor.

    Returns:
        torch.Tensor: Maximum of the tensor along the specified dimensions.
    """
    tensor = tensor.clone()
    if t_mask is True:
        t_mask = get_mask(tensor)
    if isinstance(tensor, np.ndarray):
        return _compute_masked_max_numpy(tensor, dims, t_mask, keepdim)
    if t_mask is not None and t_mask is not False:
        tensor[t_mask.expand_as(tensor) == 0] = float("-inf")
    maximum = tensor.max(dim=dims)[0]
    if keepdim:
        return maximum.unsqueeze(dims)
    return maximum


def _compute_masked_max_numpy(tensor, dims, t_mask=None, keepdim=False):
    """
    Compute the maximum of the numpy array along specified dimensions,
    ignoring zero tokens if t_mask is passed.

    Args:
        tensor (np.ndarray): The input array of shape (B, N, d).
        dims (tuple): Dimensions along which to compute the maximum.
        t_mask (np.ndarray): (optional) mask array.

    Returns:
        np.ndarray: Maximum of the array along the specified dimensions.
    """
    if t_mask is not None:
        mask = t_mask.astype(bool)
        mask = np.broadcast_to(mask, tensor.shape)
        tensor[~mask] = float("-inf")
    maximum = np.max(tensor, axis=dims)
    if keepdim:
        return np.expand_dims(maximum, axis=dims)
    return maximum



def compute_masked_mean(t, dims, t_mask, keepdim=False):
    """Compute the mean of a tensor `t` along dimensions `dims` with a mask `mask`.

    :param t: torch.tensor of shape B*N*d.
    :param dims: dimensions along which to compute the mean.
    :param mask: mask to apply to the tensor.
    :return: torch.tensor of shape B*d.
    """
    if isinstance(t, np.ndarray):
        return _compute_masked_mean_numpy(t, dims, t_mask, keepdim)
    if t_mask is True:
        t_mask = get_mask(t)
    if t_mask is not None and t_mask is not False:
        mask_sum = torch.max(
            t_mask.sum(dim=dims), torch.tensor(1e-10).to(t_mask.device)
        )
        mean = (t * t_mask).sum(dim=dims) / mask_sum
    else:
        mean = t.mean(dim=dims)
    if keepdim:
        return mean.unsqueeze(dims)
    return mean


def _compute_masked_mean_numpy(t, dims, t_mask, keepdim=False):
    """
    Compute the mean of a numpy array `t` along dimensions `dims` with a mask `mask`.

    Args:
        t (np.ndarray): Input array of shape (B, N, d).
        dims (tuple): Dimensions along which to compute the mean.
        t_mask (np.ndarray): Mask to apply to the array.

    Returns:
        np.ndarray: Mean of the array along the specified dimensions.
    """
    if t_mask is True:
        t_mask = get_mask(t)
    if t_mask is not None and t_mask is not False:
        mask_sum = np.maximum(np.sum(t_mask, axis=dims), 1e-10)
        mean = np.sum(t * t_mask, axis=dims) / mask_sum
    else:
        mean = np.mean(t, axis=dims)

    if keepdim:
        return np.expand_dims(mean, axis=dims)
    
    return mean


def pad_batch(batch, pad_length=None):
    """
    Pads a `batch` such that all sequences in the batch have the same length.
    :param batch: Iterable containing sequences of shape (L, d), where the
        sequence length `L` is variable and the token dimension `d` is fixed.
    :param pad_length: if `None`, pads all sequences to the maximum sequence
        length in the batch. If set to an int, then all sequences are padded to
        this length. Needs to be at least as large as the longest sequence in
        the batch.
    :return: Tensor of shape (B, M, d), where `B` is the batch size and `M` the
        sequence length.
    """
    batch = [torch.tensor(seq) if not isinstance(seq, torch.Tensor) else seq for seq in batch]
    padded = pad_sequence(batch, batch_first=True, padding_value=0)
    if pad_length is None:
        return padded
    B, L, d = padded.shape
    assert L <= pad_length, (
        "`pad_length` is smaller than the longest" "sequence in the batch!"
    )
    zeros = torch.zeros((B, pad_length - L, d))
    return torch.cat((padded, zeros), dim=1)


def mask_batch(batch, mask=False, dim=2, shift_fct=1.0, return_mask=False, n_samples=None, unit_var=True):
    """
    Masks a batch of sequences with a binary mask.
    :param batch: Iterable containing sequences of shape (L, d), where the
        sequence length `L` is variable and the token dimension `d` is fixed.
    :param mask: Binary mask of shape (L, 1) or (B, L, 1) to apply to the batch.
    :param shift_fct: Shift factor for the input towards 0 mean.
    :param return_mask: If `True`, returns the mask along with the masked batch.
    :param n_samples: Number of samples to return. If `None`, returns all samples.
    :return: Tensor of shape (B, L, d), where `B` is the batch size and `L` the
        sequence length.
    """
    points, labels = zip(*batch)
    points = pad_batch(points)[:, :, :dim].to(
        device
    )  # Take the first two dimensions of the padded points
    labels = torch.stack(labels).to(device)
    if n_samples is not None:
        points = points[:n_samples]
        labels = labels[:n_samples]
    points_mask = get_mask(points) if mask else None
    points = points - shift_fct * compute_masked_mean(
        points, dims=1, t_mask=points_mask
    ).unsqueeze(1)
    if mask:
        points = points * points_mask
    if unit_var:
        points = points / torch.max(masked_std(points, dims=-2, t_mask=points_mask), torch.tensor(1e-10).to(points.device))
        if mask:
            points = points * points_mask
    if return_mask:
        return points, labels, points_mask
    return points, labels


def normalize_points(points, mask=True, dims=-2):
    """
    Normalizes the points to have zero mean and unit variance.
    :param points: Tensor of shape (B, N, d) containing the points.
    :param mask: Binary mask of shape (B, N, 1) to apply to the points.
    :return: Normalized points.
    """
    if isinstance(points, np.ndarray):
        return _normalize_points_numpy(points, mask, dims=dims)
    initial_dim = points.dim()
    if initial_dim == 2:
        points = points.unsqueeze(0)
    p_mask = get_mask(points) if mask else None
    mean = compute_masked_mean(points, dims=dims, t_mask=p_mask, keepdim=True)
    std = masked_std(points, dims=dims, t_mask=p_mask, keepdim=True)
    points = p_mask * (points - mean) if mask else points - mean
    points = points / torch.max(std, torch.tensor(1e-10).to(points.device))
    points = p_mask * points if mask else points
    if initial_dim == 2:
        return points.squeeze(0)
    return points


def _normalize_points_numpy(points, mask=True, dims=-2):
    """
    Normalizes the points to have zero mean and unit variance.

    Args:
        points (np.ndarray): Array of shape (B, N, d) containing the points.
        mask (np.ndarray): Binary mask of shape (B, N, 1) to apply to the points.
    
    Returns:
        np.ndarray: Normalized points.
    """
    initial_dim = points.ndim
    if initial_dim == 2:
        points = np.expand_dims(points, axis=0)
    p_mask = get_mask(points) if mask else None
    mean = compute_masked_mean(points, dims=dims, t_mask=p_mask, keepdim=True)
    std = masked_std(points, dims=dims, t_mask=p_mask, keepdim=True)
    
    if p_mask is not None:
        points = p_mask * (points - mean)
    else:
        points = points - mean

    points = points / np.maximum(std, 1e-10)

    if p_mask is not None:
        points = p_mask * points
    if initial_dim == 2:
        points = points.squeeze(0)
    return points


def target_identity(points, labels, **kwargs):
    return labels


def target_completion(points, labels, mode="cluster", n_clusters=2, d_clusters=0.6, perc=None, return_input=True, **kwargs):
    """
    Modifies a batch of point clouds based on the specified mode ('cluster' or 'global').

    Args:
        points (torch.Tensor): Tensor of shape (B, N, d), representing B samples of point clouds.
        labels (torch.Tensor): Tensor of labels (not used in the function but included as per the signature).
        mode (str): The mode of modification ('cluster' or 'global').
        n_clusters (int): Number of clusters to select randomly (only for 'cluster' mode).
        d_clusters (float): Euclidean distance threshold for selecting points in cluster mode.
        perc (float): Percentage of points to set to zero in 'global' mode (value between 0 and 1).
        **kwargs: Additional arguments (not used but provided for future flexibility).

    Returns:
        torch.Tensor: Modified points tensor with certain points set to zero.
    """
    if return_input:
        return points
    if isinstance(points, np.ndarray):
        out_points = torch.tensor(points).to(device)
    else:
        out_points = points.detach().clone()
    input_dim = out_points.dim()
    if input_dim == 2:
        out_points = out_points.unsqueeze(0)
    B, N, d = out_points.shape  # Get the dimensions of the points tensor

    # Get the mask of non-zero points (B, N, 1), and collapse it to (B, N) for easier indexing
    mask = get_mask(out_points).squeeze(-1)  # (B, N)

    if mode == "cluster":
        # Cluster mode: randomly select n_clusters points from non-zero points, and set nearby points to 0
        for i in range(B):
            # Get indices of non-zero points for this sample
            non_zero_indices = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if len(non_zero_indices) == 0:
                continue  # Skip if no non-zero points

            # Randomly select n_clusters points from non-zero points
            selected_indices = non_zero_indices[torch.randint(0, len(non_zero_indices), (n_clusters,))]
            selected_points = out_points[i, selected_indices, :]  # (n_clusters, d)

            # Calculate distances from all points to the selected cluster points
            distances = torch.norm(out_points[i] - selected_points[:, None, :], dim=2)  # (n_clusters, N)

            # Find points within the distance threshold and set them to zero
            within_threshold = (distances <= d_clusters).any(dim=0)  # (N,)
            out_points[i, within_threshold] = 0  # Set selected points to zero

    elif mode == "global":
        # Global mode: randomly set perc percentage of non-zero points to zero
        assert perc is not None and 0 <= perc <= 1, "perc must be between 0 and 1 in global mode"
        for i in range(B):
            # Get indices of non-zero points for this sample
            non_zero_indices = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
            if len(non_zero_indices) == 0:
                continue  # Skip if no non-zero points
            
            # Determine how many non-zero points to set to zero
            n_zeros = int(perc * len(non_zero_indices))
            zero_indices = non_zero_indices[torch.randperm(len(non_zero_indices))[:n_zeros]]  # Random indices
            out_points[i, zero_indices] = 0  # Set those points to zero

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'cluster' or 'global'.")
    if isinstance(points, np.ndarray):
        out_points = out_points.cpu().numpy()
    if input_dim == 2:
        return out_points.squeeze(0)
    return out_points


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data.cpu())
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    return chamfer_loss
