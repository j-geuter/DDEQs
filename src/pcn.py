# adapted from:
# https://github.com/qinglew/PCN-PyTorch/blob/master/models/pcn.py

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, dim=3, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(dim, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, dim * self.num_coarse)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + dim + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, dim, 1)
        )
        
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)
        self.dim = dim
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, self.dim
                                                  )                    # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)             # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, self.dim).transpose(2, 1)               # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)             # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)                                           # (B, 2, num_fine)

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)          # (B, 1024, num_fine)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)                          # (B, 1024+2+3, num_fine)
        
        fine = self.final_conv(feat) + point_feat                                            # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()