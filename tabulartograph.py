"""
Core Neural Architecture and Denoising Dynamics for Tabular-to-Graph Diffusion (TGD)
Corresponds to Figure 1:
- Module 2: Transposed Feature Graph with Category Masking
- Module 3: Warm-Start Flow Diffusion via Deterministic Probability Flow ODE
- Module 4: Decoupled Projection (Moment Alignment, Value Clamping, Argmax Competition)
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal temporal embedding vector e_t in R^{d_h}."""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.emb_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class ResidualGCNBackbone(nn.Module):
    """
    Module 3: Topological Denoising Network.
    Two residual graph convolutional blocks with broadcast temporal injection and SiLU activations.
    """
    def __init__(self, node_feat_dim: int, hidden_dim: int = 64, use_gcn: bool = True):
        super().__init__()
        self.use_gcn = use_gcn
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Linear(node_feat_dim, hidden_dim)

        if self.use_gcn:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            # Ablation baseline: independent linear projections (w/o Graph)
            self.mlp1 = nn.Linear(hidden_dim, hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, node_feat_dim)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, edge_weight, t, batch):
        t_per_node = self.time_mlp(t)[batch]
        h = self.act(self.in_proj(x) + t_per_node)
        res1 = h

        if self.use_gcn:
            h = self.act(self.conv1(h, edge_index, edge_weight=edge_weight) + t_per_node) + res1
            res2 = h
            h = self.act(self.conv2(h, edge_index, edge_weight=edge_weight) + t_per_node) + res2
        else:
            h = self.act(self.mlp1(h) + t_per_node) + res1
            res2 = h
            h = self.act(self.mlp2(h) + t_per_node) + res2

        return self.out_proj(h)


class ProbabilityFlowDiffusionManager:
    """
    Module 3: Forward perturbation and deterministic probability flow ODE reverse integration.
    """
    def __init__(self, num_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cpu"):
        self.num_steps = num_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

    @torch.no_grad()
    def p_sample_ode_step(self, model, x_t, edge_index, edge_weight, t_val: int, eta: float = 0.0) -> torch.Tensor:
        """
        Discrete parameterization of the deterministic probability flow ODE trajectory (eta = 0.0).
        """
        t_tensor = torch.tensor([t_val], device=self.device, dtype=torch.long)
        batch = torch.zeros(x_t.size(0), dtype=torch.long, device=self.device)

        predicted_noise = model(x_t, edge_index, edge_weight, t_tensor, batch)
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t_val]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_val]
        alpha_bar_prev = self.alphas_cumprod_prev[t_val]

        # Closed-form reconstruction of clean state H^(0)
        pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        if t_val == 0:
            return pred_x0

        # Deterministic ODE drift without Brownian variance
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev, min=0.0)) * predicted_noise
        x_prev = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        if eta > 0.0:  # Stochastic SDE mode for ablation
            sigma_t = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - self.alphas_cumprod[t_val])) * \
                      torch.sqrt(1.0 - self.alphas_cumprod[t_val] / alpha_bar_prev)
            x_prev += sigma_t * torch.randn_like(x_t)

        return x_prev


def train_diffusion(
    graph_list: list,
    window_size: int,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    use_gcn: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Unified score matching training pipeline."""
    model = ResidualGCNBackbone(node_feat_dim=window_size, hidden_dim=64, use_gcn=use_gcn).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    diff_manager = ProbabilityFlowDiffusionManager(num_steps=100, device=device)
    loader = DataLoader(graph_list, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch_data in loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            t = torch.randint(0, diff_manager.num_steps, (batch_data.num_graphs,), device=device).long()
            t_per_node = t[batch_data.batch]

            noise = torch.randn_like(batch_data.x)
            x_noisy = diff_manager.q_sample(batch_data.x, t_per_node, noise)
            edge_weight = batch_data.edge_attr.squeeze(-1) if batch_data.edge_attr is not None else None

            pred_noise = model(x_noisy, batch_data.edge_index, edge_weight, t, batch_data.batch)
            loss = nn.functional.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()

    return model, diff_manager


@torch.no_grad()
def sample_balanced_minority(
    model: nn.Module,
    diff_manager: ProbabilityFlowDiffusionManager,
    graph_list: list,
    feature_names: list,
    data: pd.DataFrame,
    minority_class: int,
    label_col: str,
    t_warm: int = 30,
    eta: float = 0.0,
    ratio: float = 0.8,
    use_moment_align: bool = True,
) -> pd.DataFrame:
    """
    Warm-start reverse integration and continuous moment alignment.
    """
    model.eval()
    device = diff_manager.device

    class_counts = data[label_col].value_counts()
    num_majority = class_counts.max()
    num_minority = class_counts[minority_class]
    num_needed = max(0, int(num_majority * ratio) - num_minority)

    if num_needed == 0:
        return pd.DataFrame(columns=feature_names + [label_col])

    generated_samples = []
    graph_idx = 0
    num_graphs = len(graph_list)

    while len(generated_samples) < num_needed:
        base_graph = graph_list[graph_idx % num_graphs].to(device)
        graph_idx += 1
        x_0 = base_graph.x
        edge_index = base_graph.edge_index
        edge_weight = base_graph.edge_attr.squeeze(-1) if base_graph.edge_attr is not None else None

        # Warm-start perturbation prior: H_{t_warm}
        t_start = torch.tensor([t_warm] * x_0.size(0), device=device, dtype=torch.long)
        x_t = diff_manager.q_sample(x_0, t_start)

        # Deterministic ODE integration: t_warm -> 0
        for step in range(t_warm - 1, -1, -1):
            x_t = diff_manager.p_sample_ode_step(model, x_t, edge_index, edge_weight, step, eta=eta)

        # Matrix transposition back to tabular row format: (H^(0))^T in R^{N* x D}
        batch_samples = x_t.detach().cpu().numpy().T
        generated_samples.extend(batch_samples)

    raw_samples = np.array(generated_samples[:num_needed])

    # First- and second-order moment alignment
    if use_moment_align:
        real_min_df = data[data[label_col] == minority_class][feature_names]
        mean_real = real_min_df.mean().values
        std_real = real_min_df.std().values

        mean_gen = raw_samples.mean(axis=0)
        std_gen = raw_samples.std(axis=0) + 1e-8

        aligned_feats = (raw_samples - mean_gen) / std_gen * std_real + mean_real
        aligned_feats = np.clip(
            aligned_feats, real_min_df.min().values, real_min_df.max().values
        )
    else:
        aligned_feats = raw_samples

    virtual_df = pd.DataFrame(aligned_feats, columns=feature_names)
    virtual_df[label_col] = minority_class
    return virtual_df