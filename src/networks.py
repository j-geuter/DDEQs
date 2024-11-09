import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdeq import get_deq

from utils import compute_masked_mean, get_mask, masked_std, scale_mask, compute_masked_max

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.WARNING,  # Set the logging level to WARNING
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Create a logger for warnings
logger = logging.getLogger(__name__)


class EquivariantBilinearLayer(nn.Module):
    def __init__(
        self,
        p: int,
        d: int,
        h: int = None,
        push_forward: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
    ):
        """
        Defines a bilinear layer that is equivariant in its first argument and invariant
        in its second argument, w.r.t. permutations of the rows.
        :param p: hidden dimension of the first input.
        :param d: hidden dimension of the second input.
        :param h: hidden dimension of the output. Default to `p`.
        :param push_forward: if True, the function acts as a push-forward operator on the
            measure. This means there are no interactions between particles in the first
            argument.
        :param activation: Activation function. Currently, supports `None`, 'none' or 'relu'.
        :param dropout_rate: Dropout rate used for training. Dropout is applied in a
            row-invariant way.
        """
        super(EquivariantBilinearLayer, self).__init__()
        self.p = p
        self.d = d
        self.h = h if h is not None else p
        self.push_forward = push_forward

        # Parameters
        self.alpha = nn.Parameter(torch.randn(p, self.h, d))
        if not push_forward:
            self.beta = nn.Parameter(torch.randn(p, self.h, d))

        self.bias = nn.Parameter(torch.randn(self.h))

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "none" or activation is None:
            self.activation = None
        else:
            raise ValueError("Activation needs to be `None` or `relu`.")

        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.h)  # Normalizes across the features (p)
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, X, Z, mask=None, fix_mask=None):
        # Note: here X and Z are switched from usual notation
        if X.dim() == 2:
            X = X.unsqueeze(0)
        if Z.dim() == 2:
            Z = Z.unsqueeze(0)
        if fix_mask is not None:
            fix_mask = fix_mask.expand_as(X)
            X_start = X.clone()
        if mask is not None:
            X = X * mask
        # X shape: (B, N, p)
        # Z shape: (B, M, d)

        # Compute Z^T * 1_M
        Z_sum = torch.sum(Z, dim=1)  # shape: (B,d)

        # First term: X @ alpha @ Z_sum with scaling
        X_alpha = torch.einsum("bnp,phd,bd->bnh", X, self.alpha, Z_sum)

        if not self.push_forward:
            # Second term: (1_N^T X) @ beta @ Z_sum with scaling
            X_sum = torch.sum(X, dim=1)  # shape: (B,p)
            X_beta = (
                torch.einsum(
                    "bnp,phd,bd->bnh", X_sum.unsqueeze(1).expand_as(X), self.beta, Z_sum
                )
                / X.shape[1]
            )
        else:
            X_beta = 0

        # Add bias (same for all rows)
        bias = self.bias.unsqueeze(0).unsqueeze(0).expand_as(X_alpha)

        # Final output before normalization
        X_star = X_alpha + X_beta + bias

        if mask is not None:
            X_star = X_star * mask
        if fix_mask is not None:
            X_star[fix_mask] = X_start[fix_mask]

        # Apply layer normalization
        X_star = self.layer_norm(X_star)

        if mask is not None:
            X_star = X_star * mask
        if fix_mask is not None:
            X_star[fix_mask] = X_start[fix_mask]

        # Apply activation function if specified
        if self.activation is not None:
            X_star = self.activation(X_star)

        # Apply dropout if specified
        if self.training and self.dropout is not None:
            # Row-invariant dropout
            dropout_mask = (
                self.dropout(torch.ones(X_star.shape[0], X_star.shape[2]).to(device))
                .unsqueeze(1)
                .expand_as(X_star)
            )
            X_star = X_star * dropout_mask
        
        if mask is not None:
            X_star = X_star * mask
        if fix_mask is not None:
            X_star[fix_mask] = X_start[fix_mask]

        return X_star
    

class CrossVectorAttentionEncoder(nn.Module):
    """Cross Vector attention layer from the paper "Point Transformer".
    """
    def __init__(
        self,
        d_model=512,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        k_neighbors=16,
        theta=None,
        gamma=None,
        phi=None,
        psi=None,
        alpha=None,
        dim_src_pos=2,
        dim_tgt_pos=512,
    ):
        super(CrossVectorAttentionEncoder, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.k_neighbors = k_neighbors

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize phi, psi, alpha
        self.phi = phi or nn.Linear(self.head_dim, self.head_dim)
        self.psi = psi or nn.Linear(self.head_dim, self.head_dim)
        self.alpha = alpha or nn.Linear(self.head_dim, self.head_dim)

        # Initialize theta and gamma
        self.theta = theta or torch.nn.Sequential(
            nn.Linear(self.head_dim, 4 * self.head_dim),
            nn.ReLU(),
            nn.Linear(4 * self.head_dim, self.head_dim),
        )
        self.gamma = gamma or torch.nn.Sequential(
            nn.Linear(self.head_dim, 4 * self.head_dim),
            nn.ReLU(),
            nn.Linear(4 * self.head_dim, self.head_dim),
        )
        
        self.dim_src_pos = dim_src_pos
        if self.dim_src_pos != d_model:
            self.up_src_pos = nn.Linear(dim_src_pos, d_model)
        else:
            self.up_src_pos = lambda x: x
        self.dim_tgt_pos = dim_tgt_pos
        if self.dim_tgt_pos != d_model:
            self.up_tgt_pos = nn.Linear(dim_tgt_pos, d_model)
        else:
            self.up_tgt_pos = lambda x: x
        

        self.ffn = FFN(d_model, d_model, dim_feedforward, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, tgt, src, tgt_pos, src_pos, tgt_mask=None, src_mask=None, fix_mask=None):
        B, N_tgt, _ = tgt.size()
        N_src = src.size(1)
        H = self.num_heads
        D = self.head_dim

        if fix_mask is not None:
            fix_mask_expanded = fix_mask.expand_as(tgt)
            tgt_start = tgt.clone()

        # Prepare attention masks
        if tgt_mask is not None:
            initial_tgt_mask = tgt_mask.clone() # this is True where there are non-zero tokens, i.e. where the target should be non-zero
            tgt_mask = tgt_mask.squeeze(-1).eq(0)  # (B, N_tgt); this is now True where not to attend
        else:
            initial_tgt_mask = None

        # Compute distances
        distances = torch.cdist(tgt, src) # (B, N_tgt, N_src)

        # Mask invalid positions in distances if masks are provided
        if src_mask is not None:
            src_mask = src_mask.squeeze(-1).eq(0)  # (B, N_src)
            distances = distances.masked_fill(src_mask.unsqueeze(1), float('inf'))

        # For each target token, find the indices of k nearest source tokens
        k = min(self.k_neighbors, N_src)  # Adjust k if N_src is smaller
        _, indices = torch.topk(-distances, k=k, dim=2)  # Negative distances for smallest values

        # Now, for each target token, we have indices of k nearest source tokens
        # Gather source tokens and related variables using these indices

        # Expand indices for batch and heads
        indices_expanded = indices.unsqueeze(-1).expand(B, N_tgt, k, self.d_model)  # For gathering src embeddings
        indices_expanded_src_pos = indices.unsqueeze(-1).expand(B, N_tgt, k, self.dim_src_pos)  # For gathering src pos

        # Gather src embeddings
        src_selected = src.unsqueeze(1).expand(B, N_tgt, N_src, self.d_model)
        src_selected = torch.gather(src_selected, 2, indices_expanded)  # (B, N_tgt, k, d_model)

        # src_pos of shape (B, N_src, dim_src_pos) -> (B, N_tgt, k, dim_src_pos)
        src_pos_selected = src_pos.unsqueeze(1).expand(B, N_tgt, N_src, self.dim_src_pos)
        src_pos_selected = torch.gather(src_pos_selected, 2, indices_expanded_src_pos)

        # Reshape tgt and src_selected for multi-head attention
        tgt = tgt.view(B, N_tgt, H, D)  # (B, N_tgt, H, D)
        src_selected = src_selected.view(B, N_tgt, k, H, D)  # (B, N_tgt, k, H, D)

        # Transpose to bring heads to the second dimension
        tgt = tgt.permute(0, 2, 1, 3)  # (B, H, N_tgt, D)
        src_selected = src_selected.permute(0, 3, 1, 2, 4)  # (B, H, N_tgt, k, D)

        # Compute phi(p_i), psi(p_j), alpha(p_j)
        phi_p_i = self.phi(tgt.reshape(B, H * N_tgt, D)).view(B, H, N_tgt, D)  # (B, H, N_tgt, D)
        psi_p_j = self.psi(src_selected.reshape(B, H * N_tgt * k, D)).view(B, H, N_tgt, k, D)  # (B, H, N_tgt, k, D)
        alpha_p_j = self.alpha(src_selected.reshape(B, H * N_tgt * k, D)).view(B, H, N_tgt, k, D)  # Same shape
        
        # Compute delta_ij = theta(p_i - p_j)
        src_pos_selected = self.up_src_pos(src_pos_selected).view(B, N_tgt, k, H, D)  # (B, N_tgt, k, H, D)
        src_pos_selected = src_pos_selected.permute(0, 3, 1, 2, 4)  # (B, H, N_tgt, k, D)
        tgt_pos_expanded = self.up_tgt_pos(tgt_pos).unsqueeze(2).view(B, N_tgt, 1, H, D).permute(0, 3, 1, 2, 4)  # (B, H, N_tgt, 1, D)
        delta_ij = self.theta(tgt_pos_expanded - src_pos_selected) # (B, H, N_tgt, k, D)

        # Compute attn_input = phi(p_i) - psi(p_j) + delta_ij
        attn_vectors = self.gamma(phi_p_i.unsqueeze(3) - psi_p_j + delta_ij) # (B, H, N_tgt, k, D)
        attn_vectors = F.softmax(attn_vectors, dim=-2)  # Softmax over k neighbors
        if tgt_mask is not None:
            attn_vectors = attn_vectors.masked_fill(tgt_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1), 0)

        # Compute source_vectors = alpha(p_j) + delta_ij
        source_vectors = alpha_p_j + delta_ij  # (B, H, N_tgt, k, D)

        # Element-wise multiply and sum over k neighbors
        attn_output = (attn_vectors * source_vectors).sum(dim=3)  # (B, H, N_tgt, D)

        # Reshape and permute back to (B, N_tgt, d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, N_tgt, self.d_model)

        # Apply fix_mask and initial_tgt_mask if provided
        if initial_tgt_mask is not None:
            attn_output = attn_output * initial_tgt_mask

        if fix_mask is not None:
            attn_output = attn_output.masked_scatter(fix_mask_expanded, tgt_start)

        # Add & Norm
        tgt_reshaped = tgt.permute(0, 2, 1, 3).reshape(B, N_tgt, self.d_model)
        tgt = self.norm(tgt_reshaped + attn_output)

        if initial_tgt_mask is not None:
            tgt = tgt * initial_tgt_mask
        if fix_mask is not None:
            tgt = tgt.masked_scatter(fix_mask_expanded, tgt_start)

        # Feedforward network
        output = self.ffn(tgt, mask=initial_tgt_mask, fix_mask=fix_mask)

        return output, None


class CrossAttentionEncoder(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dim_feedforward=2048, dropout=0.1, **kwargs):
        super(CrossAttentionEncoder, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.num_heads = num_heads
        self.ffn = FFN(d_model, d_model, dim_feedforward, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, tgt, src, tgt_mask=None, src_mask=None, fix_mask=None, **kwargs):
        if fix_mask is not None:
            fix_mask_expanded = fix_mask.expand_as(tgt)
            tgt_start = tgt.clone()
        if tgt_mask is not None:
            initial_tgt_mask = tgt_mask.clone()
            initial_src_mask = src_mask.clone()
            tgt_mask = tgt_mask.squeeze(-1)
            src_mask = src_mask.squeeze(-1)
            tgt_mask = tgt_mask.eq(0)
            src_mask = src_mask.eq(0)
            attn_mask = torch.logical_or(tgt_mask.unsqueeze(2), src_mask.unsqueeze(1))
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_mask = attn_mask.reshape(-1, tgt.size(1), src.size(1))
        elif src_mask is not None:
            initial_tgt_mask = None
            src_mask = src_mask.squeeze(-1)
            src_mask = src_mask.eq(0)
            tgt_mask = torch.zeros(
                tgt.shape[:-1], dtype=torch.bool, device=src_mask.device
            )
            attn_mask = torch.logical_or(tgt_mask.unsqueeze(2), src_mask.unsqueeze(1))
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_mask = attn_mask.reshape(-1, tgt.size(1), src.size(1))
        else:
            attn_mask = None
            initial_tgt_mask = None

        # Apply cross-attention, without attention mask because it seems buggy
        attn_output, attn_weights = self.multihead_attn(
            tgt, src, src,
        )
        if initial_tgt_mask is not None:
            attn_output = attn_output.masked_fill(~initial_tgt_mask, 0)
        
        #attn_output[attn_output.isnan()] = 0
        if fix_mask is not None:
            attn_output[fix_mask_expanded] = tgt_start[fix_mask_expanded]

        # Add & Norm
        tgt = self.norm(tgt + attn_output)

        if initial_tgt_mask is not None:
            tgt = tgt * initial_tgt_mask
        if fix_mask is not None:
            tgt[fix_mask_expanded] = tgt_start[fix_mask_expanded]
        #tgt[tgt.isnan()] = 0
        # Feedforward network
        output = self.ffn(tgt, mask=initial_tgt_mask, fix_mask=fix_mask)
        #output[output.isnan()] = 0
        return output, attn_weights


class StackedEncoder(nn.Module):
    def __init__(
        self, encoder=CrossAttentionEncoder, d_model=512, num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1, **kwargs
    ):
        super(StackedEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                encoder(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    **kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, tgt, src, tgt_mask=None, src_mask=None, fix_mask=None, **kwargs):
        attn_weights_list = []
        for layer in self.layers:
            tgt, attn_weights = layer(tgt, src, tgt_mask=tgt_mask, src_mask=src_mask, fix_mask=fix_mask, **kwargs)
            attn_weights_list.append(attn_weights)
        return tgt, attn_weights_list


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x, mask=None, fix_mask=None,):
        x_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if fix_mask is not None:
            fix_mask = fix_mask.expand_as(x)
            x_start = x.clone()
        out = F.relu(self.linear1(x))
        if self.training and self.dropout is not None:
            # Row-invariant dropout
            dropout_mask = (
                self.dropout(torch.ones(out.shape[0], out.shape[2]).to(device))
                .unsqueeze(1)
                .expand_as(out)
            )
            out = out * dropout_mask
        out = self.linear2(out)

        # Add & Norm
        out = self.norm(x + out)
        if mask is not None:
            out = out * mask
        if fix_mask is not None:
            out[fix_mask] = x_start[fix_mask]
        if x_shape != out.shape:
            out = out.squeeze(0)
        return out


class EquivariantLinearNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, max_pool=False):
        super(EquivariantLinearNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.max_pool = max_pool
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x, mask=None, fix_mask=None):
        if fix_mask is not None:
            fix_mask = fix_mask.expand_as(x)
            x_start = x.clone()

        x = self.linear1(x)
        x = F.relu(x)
        if mask is not None:
            x = x * mask
        if fix_mask is not None and x.shape == x_start.shape:
            x[fix_mask] = x_start[fix_mask]

        x = self.linear2(x)
        if mask is not None:
            x = x * mask
        if fix_mask is not None and x.shape == x_start.shape:
            x[fix_mask] = x_start[fix_mask]

        if self.max_pool:
            x = torch.max(x, dim=1, keepdim=True)[0]
            if mask is not None:
                x = x * mask
            if fix_mask is not None and x.shape == x_start.shape:
                x[fix_mask] = x_start[fix_mask]
        return x


class EquivariantConvNetwork(nn.Module):
    def __init__(self, d, h_2, h_1, max_pool=False):
        super(EquivariantConvNetwork, self).__init__()
        self.conv1 = nn.Conv1d(d, h_1, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(h_1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(h_1, h_2, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(h_2)
        self.max_pool = max_pool
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x, mask=None, fix_mask=None):
        # x has shape (B, N, d)
        if fix_mask is not None:
            fix_mask = fix_mask.expand_as(x)
            x_start = x.clone()
            fix_mask_start = fix_mask.clone()
            fix_mask = fix_mask.permute(0, 2, 1)
        x = x.permute(0, 2, 1)  # (B, d, N)

        x = self.conv1(x)  # (B, h_1, N)
        if mask is not None:
            x = x * mask.permute(0, 2, 1)
        if fix_mask is not None and x.shape == x_start.permute(0, 2, 1).shape:
            x[fix_mask] = x_start.permute(0, 2, 1)[fix_mask]

        x = self.bn1(x)
        if mask is not None:
            x = x * mask.permute(0, 2, 1)
        if fix_mask is not None and x.shape == x_start.permute(0, 2, 1).shape:
            x[fix_mask] = x_start.permute(0, 2, 1)[fix_mask]

        x = self.relu(x)

        x = self.conv2(x)  # (B, h_2, N)
        if mask is not None:
            x = x * mask.permute(0, 2, 1)
        if fix_mask is not None and x.shape == x_start.permute(0, 2, 1).shape:
            x[fix_mask] = x_start.permute(0, 2, 1)[fix_mask]

        x = self.bn2(x)
        if mask is not None:
            x = x * mask.permute(0, 2, 1)
        if fix_mask is not None and x.shape == x_start.permute(0, 2, 1).shape:
            x[fix_mask] = x_start.permute(0, 2, 1)[fix_mask]

        if self.max_pool:
            x, _ = torch.max(x, dim=2, keepdim=True)  # Max pool across N dimension

        x = x.permute(0, 2, 1)  # (B, N or 1, h_2)
        if mask is not None:
            x = x * mask
        if fix_mask is not None and x.shape == x_start.shape:
            x[fix_mask_start] = x_start[fix_mask_start]

        return x


class DEQ(nn.Module):
    """A DEQ model that does not correspond to a push-forward measure, i.e.,
    points can interact with one another. This is achieved by another attention
    layer that computes self-attention, in addition to the cross-attention layer.
    """

    def __init__(
        self,
        d_z=2,
        d_x=2,
        hidden_dim_equiv_1=32,
        hidden_dim_equiv_2=64,
        hidden_dim_equiv_3=None,
        d_encoder=64,
        encoder=CrossAttentionEncoder,
        num_heads=2,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        num_heads_self_encoders=None,
        num_layers_self_encoders=None,
        dim_feedforward_self_encoders=None,
        ffn_hidden=None,
        bilinear=True,
        global_skip=False,
        x_encoder=True,
        z_encoder=True,
        bilinear_pushforward=False,
        EquivariantNetwork=EquivariantLinearNetwork,
        FFNNetwork=None,
    ):
        super(DEQ, self).__init__()
        self.d_z = d_z
        self.d_x = d_x
        self.d_encoder = d_encoder
        if bilinear:
            if hidden_dim_equiv_3 is None:
                hidden_dim_equiv_3 = hidden_dim_equiv_1 // 2
            self.equivariant_z = EquivariantNetwork(d_z, hidden_dim_equiv_1, hidden_dim_equiv_3)
            self.ln1 = nn.LayerNorm(hidden_dim_equiv_1)
            self.equivariant_x = EquivariantNetwork(d_x, hidden_dim_equiv_1, hidden_dim_equiv_3)
            self.ln2 = nn.LayerNorm(hidden_dim_equiv_1)
            self.bilinear = EquivariantBilinearLayer(hidden_dim_equiv_1, hidden_dim_equiv_1, h=hidden_dim_equiv_1, push_forward=bilinear_pushforward)
            self.ln3 = nn.LayerNorm(hidden_dim_equiv_1)
            self.equivariant_z2 = EquivariantNetwork(hidden_dim_equiv_1, d_encoder, hidden_dim_equiv_2)
            self.ln4 = nn.LayerNorm(d_encoder)
            self.equivariant_x2 = EquivariantNetwork(hidden_dim_equiv_1, d_encoder, hidden_dim_equiv_2)
            self.ln5 = nn.LayerNorm(d_encoder)
            self.ln6 = nn.LayerNorm(d_encoder)
        else:
            self.bilinear = None
            self.equivariant_z = EquivariantNetwork(d_z, d_encoder, hidden_dim_equiv_2)
            self.ln1 = nn.LayerNorm(d_encoder)
            self.equivariant_x = EquivariantNetwork(d_x, d_encoder, hidden_dim_equiv_2)
            self.ln2 = nn.LayerNorm(d_encoder)

        if num_heads_self_encoders is None:
            num_heads_self_encoders = num_heads
        if num_layers_self_encoders is None:
            num_layers_self_encoders = num_layers
        if dim_feedforward_self_encoders is None:
            dim_feedforward_self_encoders = dim_feedforward

        self.cross_encoder = StackedEncoder(
            encoder=encoder,
            d_model=d_encoder,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            dim_src_pos=d_x,
            dim_tgt_pos=d_z,
        )
        if z_encoder:
            self.self_encoder = StackedEncoder(
                encoder=encoder,
                d_model=d_encoder,
                num_heads=num_heads_self_encoders,
                num_layers=num_layers_self_encoders,
                dim_feedforward=dim_feedforward_self_encoders,
                dropout=dropout,
                dim_src_pos=d_z,
                dim_tgt_pos=d_z,
            )

        if x_encoder:
            self.x_self_encoder = StackedEncoder(
                encoder=encoder,
                d_model=d_encoder,
                num_heads=num_heads_self_encoders,
                num_layers=num_layers_self_encoders,
                dim_feedforward=dim_feedforward_self_encoders,
                dropout=dropout,
                dim_src_pos=d_x,
                dim_tgt_pos=d_x,
            )
        self.x_encoder = x_encoder
        self.z_encoder = z_encoder
        if FFNNetwork is None:
            FFNNetwork = EquivariantNetwork
        if ffn_hidden is None:
            ffn_hidden = 4 * d_encoder
        self.ffn = FFNNetwork(d_encoder, d_z, ffn_hidden)
        self.global_skip = global_skip
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, z, x, z_mask=None, x_mask=None, fix_mask=None, fix_all=False, fix_encoders=False):
        if fix_mask is not None:
            fix_mask_expanded = fix_mask.expand_as(z)
        else:
            fix_mask_expanded = None
        # z is the iterable variable, x the input
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if z.dim() == 2:
            z = z.unsqueeze(0)
        z_start = z
        x_start = x
        z = self.ln1(self.equivariant_z(z, mask=z_mask))
        if fix_mask_expanded is not None and fix_all and z.shape == z_start.shape:
            z[fix_mask_expanded] = z_start[fix_mask_expanded]
        x = self.ln2(self.equivariant_x(x, mask=x_mask))
        if self.bilinear is not None:
            z_in = z  # skip connection
            z = self.ln3(z_in + self.bilinear(z, x, mask=z_mask, fix_mask=fix_mask if fix_all else None))
            if fix_mask is not None and fix_all:
                z[fix_mask.expand_as(z)] = z_in[fix_mask.expand_as(z)]
            z_in2 = z
            z = self.ln4(self.equivariant_z2(z, mask=z_mask))
            if fix_mask is not None and fix_all and z.shape == z_in2.shape:
                z[fix_mask.expand_as(z)] = z_in2[fix_mask.expand_as(z)]
            x = self.ln5(self.equivariant_x2(x, mask=x_mask))
            if self.d_z == self.d_encoder:
                z = self.ln6(z + z_start)
                if fix_mask is not None and fix_all:
                    z[fix_mask.expand_as(z)] = z_start[fix_mask.expand_as(z)]
        if self.x_encoder:
            x, _ = self.x_self_encoder(x, x, tgt_mask=x_mask, src_mask=x_mask, tgt_pos=x_start, src_pos=x_start)
        if self.z_encoder:
            z, self_attn = self.self_encoder(z, z, tgt_mask=z_mask, src_mask=z_mask, fix_mask=fix_mask if fix_encoders else None, tgt_pos=z_start, src_pos=z_start)
        else:
            self_attn = None
        out, cross_attn = self.cross_encoder(
            z, x, tgt_mask=z_mask, src_mask=x_mask, fix_mask=fix_mask,
            tgt_pos=z_start, src_pos=x_start
        )
        out = self.ffn(out, mask=z_mask, fix_mask=fix_mask).squeeze()
        if self.global_skip:  # adds a global skip connection
            out += z_start.squeeze()
        # out of same shape as z
        out = out.view(z_start.shape)
        if fix_mask_expanded is not None:
            out[fix_mask_expanded] = z_start[fix_mask_expanded]
        return out, cross_attn, self_attn

    def __str__(self):
        return f"{self.__class__.__name__} ({self.num_params} params)"
    

def initialization_func(
    x, mode="gaussian", p=None, len_z="mean", mask=False, adaptive_gaussian=True, scale=1.275,
):
    """Function for initializing z in DEQ models.

    Args:
        x (torch.tensor): Input tensor.
        mode (str, optional): Mode of initialization; must be 'points', or 'gaussian', or 'zeros'. Defaults to 'points'.
        p (int, optional): Point dimension of z. Defaults to the dimension of x.
        len_z (Union[str, int], optional): Sequence length of 'z'. Can be an integer or 'mean', in which case
            it is set to the mean sequence length in x (modulo padding). Can also be 'max' in which case it is
            set to the maximum sequence length in x. Defaults to 'mean'.
        mask (bool, optional): Whether to return a mask where x is zero. Defaults to False.

    Returns:
        _type_: _description_
    """
    B, M, d = x.shape
    init_x = x.clone()
    if p is None:
        p = d

    if len_z == "mean":
        # Calculate mean sequence length
        seq_lengths = (
            (x.abs().sum(dim=-1) > 0).sum(dim=1).float()
        )  # Sum along d and count non-zero along M
        len_z = seq_lengths.mean().int().item()
    elif isinstance(len_z, int):
        len_z = len_z
    elif len_z == "max":
        len_z = M
    else:
        raise ValueError(f"Unsupported len_z value: {len_z}")

    if mask is True:
        z_mask = get_mask(x)
    else:
        z_mask = None

    if mode == "points":
        if d != p:
            raise ValueError("For 'points' mode, d must be equal to p.")
        z = x[:, :len_z, :]
    elif mode == "gaussian":
        if adaptive_gaussian:
            if d != p:
                raise ValueError("For 'adaptive_gaussian' mode, d must be equal to p.")
            mean = compute_masked_mean(x, dims=1, t_mask=z_mask).unsqueeze(1)
            std = masked_std(x, dims=1, t_mask=z_mask)
            z = torch.randn(B, len_z, p).to(device) * std + mean
        else:
            z = torch.randn(B, len_z, p).to(device)
    elif mode == "zeros":
        z = torch.zeros(B, len_z, p).to(device)
    elif mode == "completion":
        mask = True
        len_z = M
        if adaptive_gaussian:
            if d != p:
                raise ValueError("For 'adaptive_gaussian' mode, d must be equal to p.")
            mean = compute_masked_mean(x, dims=1, t_mask=z_mask).unsqueeze(1)
            std = masked_std(x, dims=1, t_mask=z_mask)
            z = torch.randn(B, len_z, p).to(device) * std + mean
        else:
            z = torch.randn(B, len_z, p).to(device)
        x_mask = get_mask(init_x).expand_as(init_x)
        z[x_mask] = init_x[x_mask]
        z_mask = scale_mask(z_mask, scale)
    else:
        raise ValueError(f"Unsupported initialization mode: {mode}")

    if mask:
        z = z * z_mask

    return z, z_mask


class TorchDEQModel(nn.Module):
    def __init__(
        self,
        deq_model_class,
        deq_model_kwargs=None,
        deq_kwargs=None,
        d_z=2,
        d_x=2,
        n_classes=10,
        classifier=None,
        classifier_kwargs=None,
        embedding_model=None,
        embedding_kwargs=None,
        fix_embedding_out=False,
        init_func=initialization_func,
        init_kwargs=None,
        solver_kwargs=None,
        n_warmup=0,
        mask=False,
        **kwargs,
    ):
        super(TorchDEQModel, self).__init__()
        if deq_model_kwargs is None:
            deq_model_kwargs = {}
        self.embedding_kwargs = embedding_kwargs if embedding_kwargs is not None else {}
        self.fix_embedding_out = fix_embedding_out
        self.embedding_model = embedding_model(**self.embedding_kwargs) if embedding_model is not None else None
        if deq_kwargs is None:
            deq_kwargs = {}
        deq_model_kwargs["d_z"] = deq_model_kwargs.get("d_z", d_z)
        deq_model_kwargs["d_x"] = deq_model_kwargs.get("d_x", d_x)
        self.d_x = d_x
        self.d_z = d_z
        self.deq_model = deq_model_class(**deq_model_kwargs)
        self.deq = get_deq(**deq_kwargs)
        self.classifier_kwargs = classifier_kwargs if classifier_kwargs is not None else {}
        if "d" not in self.classifier_kwargs:
            self.classifier_kwargs["d"] = d_z
        if "n_classes" not in self.classifier_kwargs:
            self.classifier_kwargs["n_classes"] = n_classes
        self.classifier = (
            classifier(**self.classifier_kwargs) if classifier is not None else None
        )
        self.init_func = init_func
        self.init_kwargs = init_kwargs if init_kwargs is not None else {}
        self.init_kwargs["p"] = self.init_kwargs.get("p", d_z)
        if not solver_kwargs:
            self.solver_kwargs = {
                "sigma": 15.0,
                "gamma_sigma": 1.0,
                "beta": 1.0,
                "gamma_beta": 0.999,
                "lr": 10.0,
                "gamma_lr": 0.999,
            }
        else:
            self.solver_kwargs = solver_kwargs
        self.n_warmup = n_warmup
        self.mask = mask
        self.solver_kwargs["mask"] = mask

        # if using `mask`=True, set `len_z` to 'max' to enable masking
        if mask is True:
            if self.init_kwargs.get("len_z") != "max":
                logger.warning("Setting `len_z` to 'max' to enable masking.")
                self.init_kwargs["len_z"] = "max"
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x, use_classifier=True, use_embedding=True, fix_mask=None, fix_all=False, fix_encoders=False):
        z_init, z_mask = self.init_func(x, mask=self.mask, **self.init_kwargs)
        z_init_start = z_init
        x_mask = get_mask(x)

        # Define the DEQ function to be used in the DEQ solver
        def deq_func(z):
            return self.deq_model(z, x, z_mask=z_mask, x_mask=x_mask, fix_mask=fix_mask, fix_all=fix_all, fix_encoders=fix_encoders)[0]
        
        if self.embedding_model is not None:
            z_init = self.embedding_model(z_init, reverse=False)
            if z_mask is not None:
                z_init = z_init * z_mask

        with torch.no_grad():  # because of the DEQ backprop formula, we cannot backprop trough this
            for _ in range(self.n_warmup):
                z_init = deq_func(z_init)

        # Apply the DEQ model
        z_star, info = self.deq(deq_func, z_init, solver_kwargs=self.solver_kwargs)

        if self.embedding_model is not None and use_embedding:
            z_star[0] = self.embedding_model(z_star[0], reverse=True)
            if self.fix_embedding_out and fix_mask is not None:
                z_star[0][fix_mask.expand_as(z_star[0])] = z_init_start[fix_mask.expand_as(z_star[0])]
            if z_mask is not None:
                z_star[0] = z_star[0] * z_mask
            if fix_mask is not None:
                max_diff = (z_star[0] - z_init_start).abs()[fix_mask.expand_as(z_star[0])].max()
                if max_diff > 3e-5:
                    logger.warning(
                        f"z_star is not equal to z_init at the fix_mask entries! Max diff: {max_diff}. Setting it to z_init manually."
                    )
                    z_star[0][fix_mask.expand_as(z_star[0])] = z_init_start[fix_mask.expand_as(z_star[0])]
        
        if z_mask is not None and use_embedding:
            abs_diff = torch.abs(z_star[0] - z_init_start)
            max_diff = (abs_diff * ~z_mask).max()
            if max_diff > 0:
                logger.warning(
                    f"z_star is not zero everywhere it should be! Max diff to z_init at zeros of z_init: {max_diff}."
                    " Setting it zero manually."
                )
                z_star = [z_star[0] * z_mask]

        z_star = z_star[0]
        if self.classifier is not None and use_classifier:
            y = self.classifier(z_star)
            return z_star, info, y
        return z_star, info, z_star

    def apply_model(self, x):
        z_init, z_mask = self.init_func(x, mask=self.mask, **self.init_kwargs)
        x_mask = get_mask(x)
        return self.deq_model(z_init, x, z_mask=z_mask, x_mask=x_mask, fix_mask=self.fix_mask)[0]


class DEQWrapper(nn.Module):
    def __init__(self, model_class, model_kwargs=None, init_func=None, init_kwargs=None, return_torchdeq_format=True):
        if model_kwargs is None:
            model_kwargs = {}
        if init_func is None:
            init_func = initialization_func
        super(DEQWrapper, self).__init__()
        self.deq_model = model_class(**model_kwargs)
        self.num_params = sum([p.numel() for p in self.parameters()])
        self.init_func = init_func
        self.init_kwargs = {} if init_kwargs is None else init_kwargs
        self.deq = None
        self.solver_kwargs = None
        self.return_torchdeq_format = return_torchdeq_format

    def forward(self, x):
        x_mask = get_mask(x)
        z_init, z_mask = self.init_func(x, **self.init_kwargs)
        z_star = self.deq_model(z_init, x, z_mask=z_mask, x_mask=x_mask)[0]
        info = {}

        # Dummy info
        info["nstep"] = torch.tensor(torch.inf)
        info["rel_lowest"] = torch.tensor(torch.inf)
        info["abs_lowest"] = torch.tensor(torch.inf)

        if self.return_torchdeq_format:
            return [z_star], info  # in list form to match TorchDEQModel
        return z_star
    

class DEQWrapper2(nn.Module):
    def __init__(self, model_class, model_kwargs=None, classifier=None, classifier_kwargs=None, init_func=None, init_kwargs=None, n_iters=1, transform_points=False):
        if model_kwargs is None:
            model_kwargs = {}
        if init_func is None:
            init_func = initialization_func
        super(DEQWrapper2, self).__init__()
        self.deq_model = model_class(**model_kwargs)
        self.classifier_kwargs = classifier_kwargs if classifier_kwargs is not None else {}
        self.classifier = classifier(**self.classifier_kwargs) if classifier is not None else None
        
        self.init_func = init_func
        self.init_kwargs = {} if init_kwargs is None else init_kwargs
        self.n_iters = n_iters
        self.transform_points = transform_points

        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        x_mask = get_mask(x)
        z, z_mask = self.init_func(x, **self.init_kwargs)
        for _ in range(self.n_iters):
            z = self.deq_model(z, x, z_mask=z_mask, x_mask=x_mask)[0]
        if self.classifier is not None:
            if self.transform_points:
                z = z.transpose(2, 1)
            y = self.classifier(z)
            if type(y) == tuple:
                y = y[0]
            return y, z, None
        return None, z, None


class MaxClassifier(nn.Module):
    def __init__(self, n_classes=10, d=2):
        super(MaxClassifier, self).__init__()
        self.linear = nn.Linear(d, n_classes)
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, points):
        if points.dim() == 3:
            points = compute_masked_max(points, dims=-2, t_mask=True)
        return self.linear(points)


class AffineCouplingLayer(nn.Module):
    def __init__(self, dim):
        super(AffineCouplingLayer, self).__init__()
        self.dim = dim

        # Define the transformation networks for scale and shift (only on half of the input)
        self.scale_net = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 2)
        )
        self.shift_net = nn.Sequential(
            nn.Linear(dim // 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 2)
        )
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, x, reverse=False):
        # x is of shape (B, N, dim)
        B, N, dim = x.shape
        assert dim == self.dim, "Input dimension does not match layer dimension."

        # Reshape to (B*N, dim) for processing
        x = x.view(B * N, dim)

        # Split the input into two halves
        x1, x2 = x.chunk(2, dim=1)  # Each of shape (B*N, dim/2)
        s = self.scale_net(x1)
        t = self.shift_net(x1)

        if not reverse:
            # Forward transformation: transform x2 based on x1
            x2 = x2 * torch.exp(s) + t
        else:
            # Inverse transformation: revert x2 based on x1
            x2 = (x2 - t) * torch.exp(-s)

        # Concatenate the two halves and return
        transformed_x = torch.cat([x1, x2], dim=1)
        # Reshape back to (B, N, dim)
        transformed_x = transformed_x.view(B, N, dim)
        return transformed_x


class MultiLayerNormalizingFlow(nn.Module):
    def __init__(self, dim_sequence=None, **kwargs):
        """
        Multi-layer Normalizing Flow model with progressive dimensional expansion.

        Args:
            dim_sequence (list): List of dimensions, e.g., [d, 8, 16, 32, h].
        """
        super(MultiLayerNormalizingFlow, self).__init__()
        if dim_sequence is None:
            dim_sequence = [2, 8, 16, 32, 64]
        self.dim_sequence = dim_sequence
        self.layers = nn.ModuleList()
        for dim in dim_sequence[1:]:
            self.layers.append(AffineCouplingLayer(dim))
        self.num_params = sum([p.numel() for p in self.parameters()])

    def forward(self, z, reverse=False):
        if not reverse:
            # Forward pass
            for i, layer in enumerate(self.layers):
                # Pad input to the current dimension if necessary
                dim = self.dim_sequence[i + 1]
                z = self.pad_input(z, dim)
                z = layer(z)
            return z
        else:
            # Reverse pass
            for i in reversed(range(len(self.layers))):
                layer = self.layers[i]
                z = layer(z, reverse=True)
                # Unpad to the previous dimension if necessary
                dim = self.dim_sequence[i]
                z = self.unpad_input(z, dim)
            return z

    def pad_input(self, x, dim):
        """Pad input x to dimension dim."""
        B, N, current_dim = x.shape
        if current_dim < dim:
            padding = torch.zeros(B, N, dim - current_dim, device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=2)
        return x

    def unpad_input(self, x, dim):
        """Unpad input x to dimension dim."""
        B, N, current_dim = x.shape
        if current_dim > dim:
            x = x[:, :, :dim]
        return x
