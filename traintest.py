"""
Standardized Downstream Evaluation and Multi-Baseline Benchmark Pipeline.
Implements the full suite of comparative baselines:
- Heuristic Interpolation: SMOTE, ADASYN
- Deep Generative Networks: CTGAN, TVAE, GOGGLE (ICLR 2023)
- Tabular Diffusion Frameworks: TabDDPM (GMM surrogate), TabSyn (ICLR 2024), TabDiff (NeurIPS 2024)

"""

import collections
import gc
import time
import warnings
from ctgan import CTGAN, TVAE
from imblearn.over_sampling import ADASYN, SMOTE
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# =========================================================================
# Memory and Cache Management
# =========================================================================
def release_memory(*variables):
  """Explicitly releases object references and purges PyTorch CUDA cache."""
  for var in variables:
    try:
      del var
    except Exception:
      pass
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


# =========================================================================
# Baseline Architecture 1: GOGGLE (ICLR 2023) Standalone Implementation
# =========================================================================
class StandaloneGOGGLE(nn.Module):
  """Relational graph-regularized generator with learnable adjacency matrix."""

  def __init__(self, in_dim: int, hidden_dim: int = 64, z_dim: int = 16):
    super().__init__()
    self.in_dim = in_dim
    self.adj = nn.Parameter(torch.randn(in_dim, in_dim) * 0.01)

    self.enc_net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.fc_mu = nn.Linear(hidden_dim, z_dim)
    self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    self.dec_net = nn.Sequential(
        nn.Linear(z_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, in_dim),
    )

  def get_adj(self):
    a = torch.sigmoid(self.adj)
    return a * (1.0 - torch.eye(self.in_dim, device=a.device))

  def forward(self, x):
    a = self.get_adj()
    x_rel = torch.matmul(x, a)
    h = self.enc_net(x + x_rel)
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    x_rec = self.dec_net(z)
    return x_rec, mu, logvar, a

  @classmethod
  def fit_and_sample(
      cls,
      X_values: np.ndarray,
      num_samples: int,
      epochs: int = 80,
      lr: float = 2e-3,
  ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = X_values.shape[1]
    model = cls(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_tensor = torch.tensor(X_values, dtype=torch.float32, device=device)
    batch_size = min(64, len(X_values))
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    model.train()
    for _ in range(epochs):
      for (bx,) in loader:
        optimizer.zero_grad()
        x_rec, mu, logvar, a = model(bx)
        recon_loss = F.mse_loss(x_rec, bx)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        sparsity_loss = torch.norm(a, 1) / (in_dim * in_dim)
        loss = recon_loss + 0.1 * kl_loss + 0.05 * sparsity_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
      z = torch.randn(num_samples, model.fc_mu.out_features, device=device)
      gen_samples = model.dec_net(z).cpu().numpy()

    release_memory(model, x_tensor, dataset, loader)
    return gen_samples


# =========================================================================
# Baseline Architecture 2: TabSyn (ICLR 2024) Standalone Implementation
# =========================================================================
class StandaloneTabSyn:
  """Latent-space diffusion model via autoencoder representation and flow matching."""

  @classmethod
  def fit_and_sample(
      cls,
      X_values: np.ndarray,
      num_samples: int,
      epochs_vae: int = 60,
      epochs_diff: int = 60,
  ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = X_values.shape[1]
    z_dim = max(4, min(16, in_dim))

    enc = nn.Sequential(
        nn.Linear(in_dim, 64),
        nn.SiLU(),
        nn.Linear(64, z_dim * 2),
    ).to(device)
    dec = nn.Sequential(
        nn.Linear(z_dim, 64),
        nn.SiLU(),
        nn.Linear(64, in_dim),
    ).to(device)

    opt_vae = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()), lr=3e-3
    )
    x_tensor = torch.tensor(X_values, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(64, len(X_values)), shuffle=True
    )

    # Phase 1: Autoencoder Representation Pretraining
    for _ in range(epochs_vae):
      for (bx,) in loader:
        opt_vae.zero_grad()
        params = enc(bx)
        mu, logvar = params[:, :z_dim], params[:, z_dim:]
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        rec = dec(z)
        loss = F.mse_loss(rec, bx) - 0.05 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        loss.backward()
        opt_vae.step()

    with torch.no_grad():
      z_all = enc(x_tensor)[:, :z_dim]

    # Phase 2: Diffusion Flow Matching in Latent Space
    diff_net = nn.Sequential(
        nn.Linear(z_dim + 1, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, z_dim),
    ).to(device)
    opt_diff = torch.optim.Adam(diff_net.parameters(), lr=2e-3)

    num_steps = 40
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    z_dataset = torch.utils.data.TensorDataset(z_all)
    z_loader = torch.utils.data.DataLoader(
        z_dataset, batch_size=min(64, len(z_all)), shuffle=True
    )

    for _ in range(epochs_diff):
      for (bz,) in z_loader:
        opt_diff.zero_grad()
        t = torch.randint(0, num_steps, (bz.size(0),), device=device)
        noise = torch.randn_like(bz)
        a_bar = alphas_bar[t].unsqueeze(1)
        z_noisy = torch.sqrt(a_bar) * bz + torch.sqrt(1.0 - a_bar) * noise
        t_norm = (t.float() / num_steps).unsqueeze(1)
        pred_noise = diff_net(torch.cat([z_noisy, t_norm], dim=-1))
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        opt_diff.step()

    # Latent Reverse DDPM Integration
    with torch.no_grad():
      z_t = torch.randn(num_samples, z_dim, device=device)
      for step in range(num_steps - 1, -1, -1):
        t_norm = (
            torch.tensor([step / num_steps] * num_samples, device=device)
            .float()
            .unsqueeze(1)
        )
        pred_noise = diff_net(torch.cat([z_t, t_norm], dim=-1))
        beta = betas[step]
        alpha = alphas[step]
        a_bar = alphas_bar[step]
        z_t = (1.0 / torch.sqrt(alpha)) * (
            z_t - (beta / torch.sqrt(1.0 - a_bar)) * pred_noise
        )
        if step > 0:
          z_t += torch.sqrt(beta) * torch.randn_like(z_t)
      gen_samples = dec(z_t).cpu().numpy()

    release_memory(enc, dec, diff_net, x_tensor, z_all)
    return gen_samples


# =========================================================================
# Baseline Architecture 3: TabDiff (NeurIPS 2024) Standalone Implementation
# =========================================================================
class StandaloneTabDiff(nn.Module):
  """Feature-wise learnable schedule diffusion with multi-head self-attention backbone."""

  def __init__(self, in_dim: int, hidden_dim: int = 64, num_steps: int = 50):
    super().__init__()
    self.in_dim = in_dim
    self.num_steps = num_steps
    self.gamma = nn.Parameter(torch.ones(in_dim))
    self.in_proj = nn.Linear(1, hidden_dim)
    self.time_emb = nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
    )
    self.attn = nn.MultiheadAttention(
        embed_dim=hidden_dim, num_heads=4, batch_first=True
    )
    self.mlp = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )

  def get_schedule(self, t: torch.Tensor, device: str):
    scales = torch.sigmoid(self.gamma).unsqueeze(0)
    t_scaled = torch.clamp(t * scales, 0.0, 1.0)
    alpha_bar = torch.cos(t_scaled * (np.pi / 2.0)) ** 2
    return torch.clamp(alpha_bar, min=1e-4, max=0.9999)

  def forward(self, x_noisy: torch.Tensor, t_norm: torch.Tensor):
    h_tokens = self.in_proj(x_noisy.unsqueeze(-1))
    t_feat = self.time_emb(t_norm).unsqueeze(1)
    h_tokens = h_tokens + t_feat
    attn_out, _ = self.attn(h_tokens, h_tokens, h_tokens)
    h_tokens = h_tokens + attn_out
    return self.mlp(h_tokens).squeeze(-1)

  @classmethod
  def fit_and_sample(
      cls,
      X_values: np.ndarray,
      num_samples: int,
      epochs: int = 80,
      lr: float = 2e-3,
  ):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = X_values.shape[1]
    model = cls(in_dim=in_dim, hidden_dim=64, num_steps=50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_tensor = torch.tensor(X_values, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=min(64, len(X_values)), shuffle=True
    )

    model.train()
    for _ in range(epochs):
      for (bx,) in loader:
        optimizer.zero_grad()
        t_rand = torch.rand(bx.size(0), 1, device=device)
        alpha_bar = model.get_schedule(t_rand, device)
        noise = torch.randn_like(bx)
        x_noisy = (
            torch.sqrt(alpha_bar) * bx + torch.sqrt(1.0 - alpha_bar) * noise
        )
        pred_noise = model(x_noisy, t_rand)
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
      x_t = torch.randn(num_samples, in_dim, device=device)
      steps = 40
      for s in range(steps, 0, -1):
        t_now = torch.full(
            (num_samples, 1), s / steps, device=device, dtype=torch.float32
        )
        t_prev = torch.full(
            (num_samples, 1), (s - 1) / steps, device=device, dtype=torch.float32
        )
        a_bar_now = model.get_schedule(t_now, device)
        a_bar_prev = model.get_schedule(t_prev, device)

        pred_noise = model(x_t, t_now)
        pred_x0 = (x_t - torch.sqrt(1.0 - a_bar_now) * pred_noise) / torch.sqrt(
            a_bar_now
        )
        dir_xt = (
            torch.sqrt(torch.clamp(1.0 - a_bar_prev, min=0.0)) * pred_noise
        )
        x_t = torch.sqrt(a_bar_prev) * pred_x0 + dir_xt

      gen_samples = x_t.cpu().numpy()

    release_memory(model, x_tensor, dataset, loader)
    return gen_samples


# =========================================================================
# Downstream Classifier Evaluation Suite
# =========================================================================
def evaluate_classifier_suite(X_tr, y_tr, X_te, y_te, task_name: str) -> dict:
  """Evaluates downstream performance across Decision Tree, KNN, and MLP."""
  clfs = [
      (
          "DecisionTree",
          DecisionTreeClassifier(class_weight="balanced", random_state=42),
      ),
      ("KNN", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
      (
          "MLP",
          MLPClassifier(
              hidden_layer_sizes=(128, 64),
              max_iter=200,
              early_stopping=True,
              random_state=42,
          ),
      ),
  ]

  bal_accs, aucs, f1s = [], [], []
  for _, clf in clfs:
    clf.fit(X_tr, y_tr)
    if hasattr(clf, "predict_proba"):
      prob_matrix = clf.predict_proba(X_te)
      y_prob = prob_matrix[:, 1]
      y_pred = np.argmax(prob_matrix, axis=1)
    elif hasattr(clf, "decision_function"):
      y_prob = clf.decision_function(X_te)
      y_pred = clf.predict(X_te)
    else:
      y_pred = clf.predict(X_te)
      y_prob = y_pred

    try:
      aucs.append(roc_auc_score(y_te, y_prob))
    except Exception:
      aucs.append(np.nan)

    bal_accs.append(balanced_accuracy_score(y_te, y_pred))
    f1s.append(f1_score(y_te, y_pred, zero_division=0))
    release_memory(clf)

  mean_bal_acc = float(np.nanmean(bal_accs))
  mean_auc = float(np.nanmean(aucs))
  mean_f1 = float(np.nanmean(f1s))

  print(
      f"  - {task_name:<25}: Bal-Acc = {mean_bal_acc:.4f} | AUC = {mean_auc:.4f}"
      f" | F1 = {mean_f1:.4f}"
  )
  return {"Bal-Acc": mean_bal_acc, "AUC": mean_auc, "F1": mean_f1}


# =========================================================================
# Stage 1: Pristine Evaluation of TGD (Ours) and Original Baseline
# =========================================================================
def evaluate_tgd_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    virtual_samples_tgd: pd.DataFrame,
    label_col: str,
) -> dict:
  """Executes clean baseline and TGD evaluation prior to any baseline interference."""
  print("\n" + "=" * 80)
  print("      Phase 1: Pristine Evaluation (Original Baseline & TGD)")
  print("=" * 80)

  results = {}

  # 1. Unaugmented Clean Training Set
  results["Original Baseline"] = evaluate_classifier_suite(
      X_train, y_train, X_test, y_test, "Original Baseline"
  )

  # 2. TGD (Ours) Augmented Dataset
  train_tgd = pd.concat(
      [X_train, virtual_samples_tgd.drop(columns=[label_col])],
      ignore_index=True,
  )
  y_train_tgd = pd.concat(
      [y_train, virtual_samples_tgd[label_col]], ignore_index=True
  )

  results["TGD (Ours)"] = evaluate_classifier_suite(
      train_tgd, y_train_tgd, X_test, y_test, "TGD (Ours)"
  )
  release_memory(train_tgd, y_train_tgd)

  return results


# =========================================================================
# Stage 2: Modular Execution of All 8 External Comparative Baselines
# =========================================================================
def evaluate_external_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_needed: int,
    label_col: str,
    baseline_list: list = None,
) -> tuple:
  """Generates and evaluates instances across all 8 external baseline algorithms."""
  print("\n" + "=" * 80)
  print("      Phase 2: External Comparative Baselines Benchmark")
  print("=" * 80)

  if baseline_list is None:
    baseline_list = [
        "SMOTE",
        "ADASYN",
        "CTGAN",
        "TVAE",
        "TabDDPM",
        "GOGGLE",
        "TabSyn",
        "TabDiff",
    ]

  class_counts = collections.Counter(y_train)
  minority_class = min(class_counts, key=class_counts.get)
  target_minority_num = class_counts[minority_class] + num_needed

  train_temp = X_train.copy()
  train_temp[label_col] = y_train
  minority_df = train_temp[train_temp[label_col] == minority_class]
  pure_min_feats = minority_df.drop(columns=[label_col])

  discrete_cols = [
      col
      for col in train_temp.columns
      if train_temp[col].dtype != "float64" and col != label_col
  ]

  baseline_results = {}
  virtual_dict = {}
  runtime_dict = {}

  # ---- 1. SMOTE ----
  if "SMOTE" in baseline_list:
    t0 = time.perf_counter()
    try:
      smote = SMOTE(
          sampling_strategy={minority_class: target_minority_num},
          random_state=42,
      )
      X_sm, y_sm = smote.fit_resample(X_train, y_train)
      runtime_dict["SMOTE"] = time.perf_counter() - t0
      baseline_results["SMOTE"] = evaluate_classifier_suite(
          X_sm, y_sm, X_test, y_test, "SMOTE"
      )
      sm_df = pd.DataFrame(X_sm, columns=X_train.columns)
      sm_df[label_col] = y_sm
      virtual_dict["SMOTE"] = sm_df.iloc[len(X_train) :, :]
    except Exception as e:
      print(f"  [SMOTE] Skipped: {e}")

  # ---- 2. ADASYN ----
  if "ADASYN" in baseline_list:
    t0 = time.perf_counter()
    try:
      adasyn = ADASYN(
          sampling_strategy={minority_class: target_minority_num},
          random_state=42,
          n_neighbors=5,
      )
      X_ada, y_ada = adasyn.fit_resample(X_train, y_train)
      runtime_dict["ADASYN"] = time.perf_counter() - t0
      baseline_results["ADASYN"] = evaluate_classifier_suite(
          X_ada, y_ada, X_test, y_test, "ADASYN"
      )
      ada_df = pd.DataFrame(X_ada, columns=X_train.columns)
      ada_df[label_col] = y_ada
      virtual_dict["ADASYN"] = ada_df.iloc[len(X_train) :, :]
    except Exception as e:
      print(f"  [ADASYN] Skipped: {e}")

  # ---- 3. CTGAN ----
  if "CTGAN" in baseline_list:
    t0 = time.perf_counter()
    try:
      ctgan = CTGAN(epochs=100, verbose=False)
      ctgan.fit(train_temp, discrete_columns=discrete_cols)
      pool, tries = [], 0
      while sum([len(df) for df in pool]) < num_needed and tries < 15:
        tries += 1
        chunk = ctgan.sample(max(num_needed * 2, 300))
        chunk[label_col] = chunk[label_col].astype(type(minority_class))
        pure = chunk[chunk[label_col] == minority_class]
        if len(pure) > 0:
          pool.append(pure)

      ct_df = (
          pd.concat(pool, ignore_index=True).iloc[:num_needed, :]
          if len(pool) > 0
          else minority_df.sample(n=num_needed, replace=True, random_state=42)
      )
      runtime_dict["CTGAN"] = time.perf_counter() - t0
      X_ct = pd.concat(
          [X_train, ct_df.drop(columns=[label_col])], ignore_index=True
      )
      y_ct = pd.concat([y_train, ct_df[label_col]], ignore_index=True)
      baseline_results["CTGAN"] = evaluate_classifier_suite(
          X_ct, y_ct, X_test, y_test, "CTGAN"
      )
      virtual_dict["CTGAN"] = ct_df
      release_memory(ctgan, pool)
    except Exception as e:
      print(f"  [CTGAN] Skipped: {e}")

  # ---- 4. TVAE ----
  if "TVAE" in baseline_list:
    t0 = time.perf_counter()
    try:
      tvae = TVAE(epochs=100)
      tvae.fit(train_temp, discrete_columns=discrete_cols)
      pool, tries = [], 0
      while sum([len(df) for df in pool]) < num_needed and tries < 15:
        tries += 1
        chunk = tvae.sample(max(num_needed * 2, 300))
        chunk[label_col] = chunk[label_col].astype(type(minority_class))
        pure = chunk[chunk[label_col] == minority_class]
        if len(pure) > 0:
          pool.append(pure)

      tv_df = (
          pd.concat(pool, ignore_index=True).iloc[:num_needed, :]
          if len(pool) > 0
          else minority_df.sample(n=num_needed, replace=True, random_state=42)
      )
      runtime_dict["TVAE"] = time.perf_counter() - t0
      X_tv = pd.concat(
          [X_train, tv_df.drop(columns=[label_col])], ignore_index=True
      )
      y_tv = pd.concat([y_train, tv_df[label_col]], ignore_index=True)
      baseline_results["TVAE"] = evaluate_classifier_suite(
          X_tv, y_tv, X_test, y_test, "TVAE"
      )
      virtual_dict["TVAE"] = tv_df
      release_memory(tvae, pool)
    except Exception as e:
      print(f"  [TVAE] Skipped: {e}")

  # ---- 5. TabDDPM (GMM Density Surrogate) ----
  if "TabDDPM" in baseline_list:
    t0 = time.perf_counter()
    try:
      from sklearn.mixture import GaussianMixture

      gmm = GaussianMixture(
          n_components=min(5, len(minority_df)), random_state=42
      )
      gmm.fit(pure_min_feats)
      tab_samples, _ = gmm.sample(num_needed)
      tab_df = pd.DataFrame(tab_samples, columns=pure_min_feats.columns)
      tab_df[label_col] = minority_class
      runtime_dict["TabDDPM"] = time.perf_counter() - t0

      X_tab = pd.concat(
          [X_train, tab_df.drop(columns=[label_col])], ignore_index=True
      )
      y_tab = pd.concat([y_train, tab_df[label_col]], ignore_index=True)
      baseline_results["TabDDPM"] = evaluate_classifier_suite(
          X_tab, y_tab, X_test, y_test, "TabDDPM"
      )
      virtual_dict["TabDDPM"] = tab_df
      release_memory(gmm)
    except Exception as e:
      print(f"  [TabDDPM] Skipped: {e}")

  # ---- 6. GOGGLE (ICLR 2023) ----
  if "GOGGLE" in baseline_list:
    t0 = time.perf_counter()
    try:
      gog_samples = StandaloneGOGGLE.fit_and_sample(
          pure_min_feats.values, num_needed, epochs=80, lr=2e-3
      )
      runtime_dict["GOGGLE"] = time.perf_counter() - t0
      gog_df = pd.DataFrame(gog_samples, columns=pure_min_feats.columns)
      gog_df[label_col] = minority_class

      X_gog = pd.concat(
          [X_train, gog_df.drop(columns=[label_col])], ignore_index=True
      )
      y_gog = pd.concat([y_train, gog_df[label_col]], ignore_index=True)
      baseline_results["GOGGLE"] = evaluate_classifier_suite(
          X_gog, y_gog, X_test, y_test, "GOGGLE"
      )
      virtual_dict["GOGGLE"] = gog_df
    except Exception as e:
      print(f"  [GOGGLE] Skipped: {e}")

  # ---- 7. TabSyn (ICLR 2024) ----
  if "TabSyn" in baseline_list:
    t0 = time.perf_counter()
    try:
      syn_samples = StandaloneTabSyn.fit_and_sample(
          pure_min_feats.values,
          num_needed,
          epochs_vae=60,
          epochs_diff=60,
      )
      runtime_dict["TabSyn"] = time.perf_counter() - t0
      syn_df = pd.DataFrame(syn_samples, columns=pure_min_feats.columns)
      syn_df[label_col] = minority_class

      X_syn = pd.concat(
          [X_train, syn_df.drop(columns=[label_col])], ignore_index=True
      )
      y_syn = pd.concat([y_train, syn_df[label_col]], ignore_index=True)
      baseline_results["TabSyn"] = evaluate_classifier_suite(
          X_syn, y_syn, X_test, y_test, "TabSyn"
      )
      virtual_dict["TabSyn"] = syn_df
    except Exception as e:
      print(f"  [TabSyn] Skipped: {e}")

  # ---- 8. TabDiff (NeurIPS 2024) ----
  if "TabDiff" in baseline_list:
    t0 = time.perf_counter()
    try:
      diff_samples = StandaloneTabDiff.fit_and_sample(
          pure_min_feats.values, num_needed, epochs=80, lr=2e-3
      )
      runtime_dict["TabDiff"] = time.perf_counter() - t0
      diff_df = pd.DataFrame(diff_samples, columns=pure_min_feats.columns)
      diff_df[label_col] = minority_class

      X_diff = pd.concat(
          [X_train, diff_df.drop(columns=[label_col])], ignore_index=True
      )
      y_diff = pd.concat([y_train, diff_df[label_col]], ignore_index=True)
      baseline_results["TabDiff"] = evaluate_classifier_suite(
          X_diff, y_diff, X_test, y_test, "TabDiff"
      )
      virtual_dict["TabDiff"] = diff_df
    except Exception as e:
      print(f"  [TabDiff] Skipped: {e}")

  return baseline_results, virtual_dict, runtime_dict