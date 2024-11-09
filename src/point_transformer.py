import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def farthest_point_sampling(xyz, npoint, mask=None):
    """
    Input:
        xyz: point cloud data, [B, N, C]
        npoint: number of samples
        mask: mask indicating valid points, [B, N]
    Return:
        centroids: sampled point cloud indices, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.ones(B, N, device=xyz.device) * 1e10  # Large initial distance

    if mask is not None:
        # Set distances of invalid (masked) points to infinity
        distance[~mask] = float('inf')
        valid_indices = mask.nonzero(as_tuple=False)  # [TotalValidPoints, 2]
        # Split batch indices and point indices
        batch_indices_valid, point_indices_valid = valid_indices[:, 0], valid_indices[:, 1]
        # Group point indices by batch
        valid_indices_per_batch = [point_indices_valid[batch_indices_valid == b] for b in range(B)]
        # Randomly select a valid point for each batch
        farthest = torch.stack([
            indices[torch.randint(len(indices), (1,)).item()] if len(indices) > 0 else torch.tensor(0, device=xyz.device)
            for indices in valid_indices_per_batch
        ])
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)

    batch_indices = torch.arange(B, dtype=torch.long, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask_dist = dist < distance
        distance[mask_dist] = dist[mask_dist]
        farthest = torch.max(distance, -1)[1]
    
    return centroids  # Shape [B, npoint]


def gathering(points, idx):
    """
    Gather points based on index.
    
    Args:
        points (torch.Tensor): [B, N, C] points in the point cloud.
        idx (torch.Tensor): [B, M] indices of points to sample.
    
    Returns:
        new_points (torch.Tensor): [B, M, C] sampled points.
    """
    B, N, C = points.shape
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1)
    new_points = points[batch_indices, idx, :]  # [B, M, C]
    
    return new_points


def knn_point(k, xyz, new_xyz, mask=None):
    """
    Find k-nearest neighbors of new_xyz in xyz, considering a mask for valid points.
    
    Args:
        k (int): Number of neighbors.
        xyz (torch.Tensor): [B, N, C] point cloud.
        new_xyz (torch.Tensor): [B, M, C] query points.
        mask (torch.Tensor, optional): [B, N] mask indicating valid points.
    
    Returns:
        idx (torch.Tensor): [B, M, K] indices of the k-nearest neighbors.
        dist (torch.Tensor): [B, M, K] squared distances to the k-nearest neighbors.
    """
    B, N, C = xyz.shape
    M = new_xyz.shape[1]

    # Compute pairwise distances between new_xyz and xyz
    dist = torch.cdist(new_xyz, xyz)  # [B, M, N]

    # Internal zero-vector masking: set distances of zero-vectors to infinity
    zero_mask = torch.all(xyz == 0, dim=-1)  # [B, N], True where the point is a zero-vector
    if mask is not None:
        # Combine the mask and the zero-mask
        mask = mask & ~zero_mask  # Keep the mask provided but also exclude zero vectors
    else:
        # If no mask provided, use the zero_mask to exclude zero vectors
        mask = ~zero_mask  # [B, N]

    # Apply the mask by setting distances of invalid points to infinity
    mask_expanded = mask.unsqueeze(1).expand_as(dist)  # [B, M, N]
    dist[~mask_expanded] = float('inf')  # Set distances of invalid points to infinity

    # Check if the number of available points N is smaller than k (num_neighbors)
    if N < k:
        # If fewer points than neighbors, return all points as neighbors
        dist, idx = torch.topk(dist, k=N, largest=False, dim=-1)  # Return all points
        idx = torch.cat([idx, idx[:, :, :k - N]], dim=-1)  # Duplicate indices to match size K
    else:
        # Standard case: find the top-k nearest neighbors
        dist, idx = torch.topk(dist, k=k, largest=False, dim=-1)  # [B, M, K]

    return idx, dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, k]
    Return:
        new_points: indexed points data, [B, S, C] or [B, S, k, C]
    """
    B = points.shape[0]
    batch_indices = torch.arange(B, dtype=torch.long, device=points.device).view(B, 1, 1)
    new_points = points[batch_indices, idx, :]
    return new_points


class PointTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_neighbors=16, coord_dims=3):
        super(PointTransformerLayer, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        self.num_neighbors = num_neighbors
        self.coord_dims = coord_dims

        self.to_query = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_key = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_value = nn.Conv1d(in_channels, self.out_channels, kernel_size=1)
        self.to_pos_enc = nn.Sequential(
            nn.Conv2d(self.coord_dims, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        )
        self.to_attn = nn.Sequential(
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        )
        self.softmax = nn.Softmax(dim=-1)
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, px, mask=None):
        p, x = px  # p: [B, N, 3], x: [B, C_in, N]

        # Compute query, key, value
        q = self.to_query(x)  # [B, C_out, N]
        k = self.to_key(x)    # [B, C_out, N]
        v = self.to_value(x)  # [B, C_out, N]

        # Find k-nearest neighbors
        
        idx, _ = knn_point(self.num_neighbors, p, p)  # [B, N, K]

        # Index points
        n_k = index_points(k.transpose(1, 2), idx)  # [B, N, K, C_out]
        n_v = index_points(v.transpose(1, 2), idx)  # [B, N, K, C_out]
        n_p = index_points(p, idx)  # [B, N, K, 3]
        p = p.unsqueeze(2)  # [B, N, 1, 3]
        relative_pos = p - n_p  # [B, N, K, 3]

        # Positional encoding
        n_r = self.to_pos_enc(relative_pos.permute(0, 3, 1, 2))  # [B, C_out, N, K]

        # Compute attention
        q = q.unsqueeze(-1)  # [B, C_out, N, 1]
        a = self.to_attn(q - n_k.permute(0, 3, 1, 2) + n_r)  # [B, C_out, N, K]

        if mask is not None:
            actual_num_neighbors = n_k.shape[2]
            attn_mask = mask.unsqueeze(2).unsqueeze(1).expand(-1, a.shape[1], -1, actual_num_neighbors)
            a = a.masked_fill(~attn_mask, -1e10)  # Set masked attention to a very low value
            
        a = self.softmax(a)
        a = a.masked_fill(torch.isnan(a), 0)  # Handle NaN values

        # Aggregate features
        y = torch.sum(a * (n_v.permute(0, 3, 1, 2) + n_r), dim=-1)  # [B, C_out, N]
        return [p.squeeze(2), y]
    

class PointTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_neighbors=16, coord_dims=3):
        super(PointTransformerBlock, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        self.coord_dims = coord_dims
        self.linear1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.transformer = PointTransformerLayer(self.out_channels, num_neighbors=num_neighbors, coord_dims=coord_dims)
        self.bn = nn.BatchNorm1d(self.out_channels)
        self.linear2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.out_channels)
        # self.relu = nn.ReLU()
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, px, mask=None):
        p, x = px  # p: [B, N, 3], x: [B, C_in, N]

        y = F.relu(self.bn1(self.linear1(x)))  # [B, C_out, N]
        _, y = self.transformer([p, y], mask=mask)  # [B, C_out, N]
        y = F.relu(self.bn(self.linear2(y)))  # [B, C_out, N]
        y = y + x  # Residual connection
        y = F.relu(y)
        return [p, y], mask
    

class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=4, num_neighbors=16, coord_dims=3):
        super(TransitionDown, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        self.stride = stride
        self.num_neighbors = num_neighbors

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + coord_dims, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, p1x, mask=None):
        """
        Forward pass for TransitionDown module.
        p1x: Tuple of (p1, x1), where p1 is the point cloud and x1 are the features.
        mask: Optional mask indicating valid points in p1.
        """
        p1, x1 = p1x  # p1: [B, N, 3], x1: [B, C_in, N]
        B, N, _ = p1.shape

        # Number of points to sample (M)
        M = max(1, N // self.stride)

        if mask is not None:
            # Count valid points in each sample
            num_valid_points = mask.sum(dim=1)  # [B]
            
            # Handle cases where number of valid points < M
            if torch.any(num_valid_points < M):
                p1_masked = p1.clone()
                p1_masked[~mask] = float('inf')  # Mark invalid points with 'inf' for FPS
                
                # Perform farthest point sampling
                idx = farthest_point_sampling(p1_masked, M)  # [B, M]
                
                # Check if some selected points are padded (inf)
                p1_selected = gathering(p1, idx)
                selected_is_valid = ~torch.isinf(p1_selected).any(dim=-1)  # [B, M]
                
                # Update mask: True for valid selected points, False for invalid ones
                new_mask = selected_is_valid
            else:
                idx = farthest_point_sampling(p1, M)
                new_mask = torch.ones(B, M, dtype=torch.bool, device=p1.device)  # All valid

        else:
            # No mask provided, perform regular FPS
            idx = farthest_point_sampling(p1, M)
            new_mask = None  # No mask used
        
        # Gathering the downsampled points
        p2 = gathering(p1, idx)  # [B, M, 3]

        # k-Nearest Neighbors for feature pooling
        idx_knn, _ = knn_point(self.num_neighbors, p1, p2, mask=mask)
        grouped_p1 = index_points(p1, idx_knn)
        grouped_x1 = index_points(x1.transpose(1, 2), idx_knn).permute(0, 3, 1, 2)

        # Relative positional encoding
        relative_p = (grouped_p1 - p2.unsqueeze(2)).permute(0, 3, 1, 2)
        grouped_features = torch.cat([relative_p, grouped_x1], dim=1)

        # Apply MLP and local max pooling
        
        new_features = self.mlp(grouped_features)
        kernel_size = min(self.num_neighbors, new_features.shape[-1])
        new_features = F.max_pool2d(new_features, kernel_size=[1, kernel_size]).squeeze(-1)

        return [p2, new_features], new_mask
    

class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(TransitionUp, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels

        # MLP to reduce the dimensionality of the upsampled features
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU()
        )
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x1, x2, mask=None):
        """
        Forward pass for TransitionUp module.
        
        x1: [B, C_in, N], features corresponding to the lower-resolution point cloud.
        x2: [B, C_in, M], features corresponding to the higher-resolution point cloud (from skip connection).
        mask: [B, M] mask for the higher-resolution points (optional).
        """
        B, C_in, N = x1.shape  # Lower-resolution features
        M = x2.shape[2]        # Number of points in the higher-resolution feature set

        # Upsample x1 to match the number of points in x2
        x1_upsampled = F.interpolate(x1, size=M, mode='linear', align_corners=True)  # [B, C_in, M]
        
        # Apply the MLP to the upsampled features to reduce dimensionality
        x1_upsampled = self.mlp(x1_upsampled)  # [B, out_channels, M]

        # Add the skip connection features (x2)
        x = x1_upsampled + x2  # Add the upsampled features to the skip connection features

        # Apply mask (if provided) to the final features
        if mask is not None:
            x = x * mask.unsqueeze(1).float()

        return x
    

class PointTransformerCls(nn.Module):
    def __init__(self, dim=3, num_classes=10, num_neighbors=16, masking=True, strides=None, channels=None):
        super(PointTransformerCls, self).__init__()

        self.num_neighbors = num_neighbors
        self.strides = strides if strides is not None else [1, 4, 4, 4, 4]
        self.channels = channels if channels is not None else [32, 64, 128, 256, 512]
        self.dim = dim
        self.masking = masking
        # Input MLP
        self.in_mlp = nn.Sequential(
            nn.Conv1d(dim, self.channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU()
        )

        # Encoder Layers
        self.encoder_layers = nn.ModuleList()
        self.transition_downs = nn.ModuleList()
        in_channels = self.channels[0]
        for i in range(len(self.channels)):
            self.encoder_layers.append(PointTransformerBlock(in_channels, num_neighbors=self.num_neighbors, coord_dims=dim))
            if i < len(self.channels) - 1:
                out_channels = self.channels[i+1]
                self.transition_downs.append(TransitionDown(in_channels, out_channels,
                                                            stride=self.strides[i+1],
                                                            num_neighbors=self.num_neighbors,
                                                            coord_dims=dim))
                in_channels = out_channels

        # Global pooling and MLP for classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.channels[-1], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # x: [B, N, C_in]
        p = x[:, :, :self.dim]  # [B, N, dim]
        if x.shape[-1] > self.dim:
            features = x[:, :, self.dim:].transpose(1, 2).contiguous()  # [B, C_in-dim, N]
        else:
            features = p.transpose(1, 2).contiguous()  # Use coordinates as features
        mask = create_mask(x) if self.masking else None
        # Input MLP
        x = self.in_mlp(features)  # [B, C, N]

        # Encoder
        for i in range(5):
            (p, x), mask = self.encoder_layers[i]([p, x], mask=mask)
            if i < 4:
                (p, x), mask = self.transition_downs[i]([p, x], mask=mask)

        x = x * mask.unsqueeze(1).float() if mask is not None else x
        # Global pooling
        
        x = self.global_pool(x)  # [B, C, 1]
        x = x.squeeze(-1)  # [B, C]
        # Classification
        x = self.classifier(x)  # [B, num_classes]
        return x, None, None # match the return signature of PointNetCls
    

def create_mask(x):
    """
    Creates a mask for valid points (non-zero points).
    x: [B, N, C_in]
    Returns:
        mask: [B, N], dtype=torch.bool
    """
    mask = torch.any(x != 0, dim=-1)  # True for valid points
    return mask


class PointTransformerSeg(nn.Module):
    
    def __init__(self, dim=3, num_classes=10, num_neighbors=16, masking=True, strides=None, channels=None):
        super(PointTransformerSeg, self).__init__()

        self.num_neighbors = num_neighbors
        self.strides = strides if strides is not None else [1, 4, 4, 4, 4]
        self.channels = channels if channels is not None else [32, 64, 128, 256, 512]
        self.dim = dim
        self.masking = masking
        
        # Input MLP
        self.in_mlp = nn.Sequential(
            nn.Conv1d(dim, self.channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(),
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU()
        )

        # Encoder Layers
        self.encoder_layers = nn.ModuleList()
        self.transition_downs = nn.ModuleList()
        in_channels = self.channels[0]
        for i in range(len(self.channels)):
            self.encoder_layers.append(PointTransformerBlock(in_channels, num_neighbors=self.num_neighbors, coord_dims=dim))
            if i < len(self.channels) - 1:
                out_channels = self.channels[i+1]
                self.transition_downs.append(TransitionDown(in_channels, out_channels,
                                                            stride=self.strides[i+1],
                                                            num_neighbors=self.num_neighbors,
                                                            coord_dims=dim))
                in_channels = out_channels

        # Decoder Layers (Upsampling)
        self.transition_ups = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # First decoder layer maps from channels[-1] to channels[-1] (no skip connection)
        self.decoder_layers.append(PointTransformerBlock(self.channels[-1], num_neighbors=self.num_neighbors, coord_dims=dim))

        # Initialize the remaining TransitionUp and decoder layers
        for i in reversed(range(1, len(self.channels))):
            in_channels = self.channels[i]
            out_channels = self.channels[i-1]
            
            # TransitionUp should map from in_channels to out_channels
            self.transition_ups.append(TransitionUp(in_channels, out_channels))
            
            # Decoder layers should map from out_channels to out_channels
            self.decoder_layers.append(PointTransformerBlock(out_channels, num_neighbors=self.num_neighbors, coord_dims=dim))
        
        # Output MLP
        self.out_mlp = nn.Sequential(
            nn.Conv1d(self.channels[0], self.channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.channels[0]),
            nn.ReLU(),
            nn.Conv1d(self.channels[0], num_classes, kernel_size=1)  # Final output with class predictions per point
        )
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass for PointTransformerSeg.
        x: [B, N, C_in], input point cloud with N points and C_in input channels (including coordinates and features).
        """
        p = x[:, :, :self.dim]  # [B, N, dim]
        if x.shape[-1] > self.dim:
            features = x[:, :, self.dim:].transpose(1, 2).contiguous()  # [B, C_in-dim, N]
        else:
            features = p.transpose(1, 2).contiguous()  # Use coordinates as features if no extra features present

        # Create mask if masking is enabled
        mask = create_mask(x) if self.masking else None

        # Encoder
        skip_connections = []  # To store encoder outputs for skip connections
        x = self.in_mlp(features)  # [B, C, N]

        for i in range(len(self.channels)):
            (p, x), mask = self.encoder_layers[i]([p, x], mask=mask)
            skip_connections.append((p, x, mask))  # Store for decoder skip connections
            if i < len(self.channels) - 1:
                (p, x), mask = self.transition_downs[i]([p, x], mask=mask)

        # Decoder (Upsampling with skip connections)
        for i in range(len(self.channels)):
            if i == 0:
                # First decoder layer, no skip connection, no transition up
                (p, x), mask = self.decoder_layers[i]([p, x], mask=mask)  # Simply process the encoder output directly
            else:
                # Apply TransitionUp and skip connection for the remaining decoder layers
                p_skip, x_skip, mask_skip = skip_connections[-(i + 1)]
                x = self.transition_ups[i-1](x, x_skip, mask=mask_skip)
                (p, x), mask = self.decoder_layers[i]([p, x], mask=mask)

        # Apply mask before final output if masking is enabled
        x = x * mask.unsqueeze(1).float() if mask is not None else x
        
        # Output point-wise classification
        x = self.out_mlp(x)  # [B, num_classes, N]
        x = x.transpose(1, 2).contiguous()  # [B, N, num_classes]
        return x


if __name__ == '__main__':
    # Test PointTransformerCls
    model = PointTransformerSeg(dim=2)
    x = torch.randn(8, 14, 2)
    y = model(x)[0]
    print(y.shape)