"""
Tabular-to-Graph Diffusion (TGD) Main Execution Pipeline.
Systematic implementation strictly corresponding to the paper's four modules:
1. TwoNN Adaptive Manifold Extraction (Section 3.2, Steps 1.1 & 1.2)
2. Transposed Feature Graph Construction with Category Masking (Section 3.3, Steps 2.1 & 2.2)
3. Warm-Start Flow Diffusion via Deterministic Probability Flow ODE (Section 3.4 & 3.5, Module 3)
4. Dual-Stream Decoupled Projection (Section 3.5, Steps 4.1 & 4.2)
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data

from data_preprocessing import load_and_preprocess_dataset
from tabulartograph import train_diffusion, sample_balanced_minority
from traintest import evaluate_tgd_pipeline, evaluate_external_baselines, release_memory


def estimate_twonn_tangent_window(X_minority: np.ndarray, alpha: float = 1.8, min_bound: int = 5, max_bound: int = 60) -> int:
    """
    Module 1 (Step 1.2): Adaptive Local Tangent Space Estimation via TwoNN.
    Closed-form non-parametric intrinsic dimension fitting: d = -ln(1 - F(mu)) / ln(mu).
    """
    M, _ = X_minority.shape
    if M <= min_bound:
        return max(3, M)

    nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(X_minority)
    distances, _ = nbrs.kneighbors(X_minority)
    r1, r2 = distances[:, 1], distances[:, 2]

    valid_mask = (r1 > 1e-8) & (r2 > 1e-8)
    if valid_mask.sum() < min_bound:
        return min(15, M)

    mu = r2[valid_mask] / r1[valid_mask]
    mu_sorted = np.sort(mu)
    n_pts = len(mu_sorted)
    y_cdf = np.arange(1, n_pts + 1) / n_pts

    # Truncate extremes [0.05, 0.95] for stable slope regression
    idx_start, idx_end = int(n_pts * 0.05), int(n_pts * 0.95)
    x_log = np.log(mu_sorted[idx_start:idx_end])
    y_log = -np.log(1.0 - y_cdf[idx_start:idx_end])

    d_intrinsic = float(np.linalg.lstsq(x_log[:, None], y_log, rcond=None)[0][0])
    d_intrinsic = max(2.0, d_intrinsic)

    # Tangent space window expansion: N* = clip(ceil(alpha * d), 5, 60)
    N_star = int(np.ceil(alpha * d_intrinsic))
    return max(min_bound, min(N_star, max_bound, M))


def dual_stream_decoupled_projection(
    gen_df: pd.DataFrame,
    orig_train_df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list,
    cont_cols: list,
    cat_groups: dict,
    label_col: str
) -> pd.DataFrame:
    """
    Module 4 (Steps 4.1 & 4.2): Dual-Stream Decoupled Projection.
    Enforces physical value clamping on continuous features and argmax mutual exclusivity on one-hot groups.
    """
    features_only = gen_df[feature_cols].copy()
    unscaled = pd.DataFrame(scaler.inverse_transform(features_only), columns=feature_cols, index=gen_df.index)

    # Step 4.1: Continuous Value Clamping: clip(x, x_min, x_max)
    for c in cont_cols:
        c_min, c_max = orig_train_df[c].min(), orig_train_df[c].max()
        unscaled[c] = unscaled[c].clip(c_min, c_max)

    # Step 4.2: Categorical Argmax Competition: Argmax(z)
    for _, cols in cat_groups.items():
        if len(cols) > 1:
            logits = unscaled[cols].values
            max_idx = np.argmax(logits, axis=1)
            orthogonal_onehot = np.zeros_like(logits)
            orthogonal_onehot[np.arange(len(orthogonal_onehot)), max_idx] = 1.0
            unscaled[cols] = orthogonal_onehot
        else:
            c_name = cols[0]
            unscaled[c_name] = (unscaled[c_name] >= 0.5).astype(float)

    # Project back into scaled evaluation space
    rescaled = pd.DataFrame(scaler.transform(unscaled[feature_cols]), columns=feature_cols, index=gen_df.index)
    rescaled[label_col] = gen_df[label_col].values
    return rescaled


def main():
    # -------------------------------------------------------------------------
    # Unified Hyperparameter Configuration
    # -------------------------------------------------------------------------
    DATASET_NAME = "churn"         # Benchmark target (CLIMB / OpenML)
    ALPHA = 1.8                    # Tangent expansion scaling factor
    CORR_THRESHOLD = 0.8           # Empirical Pearson correlation cutoff (tau)
    DIFF_STEPS = 100               # Total diffusion steps (T)
    WARM_START_STEP = 30           # Shallow manifold warm-start perturbation (t_warm)
    ETA = 0.0                      # Deterministic probability flow ODE parameter
    EPOCHS = 100                   # Denoising network training epochs
    BATCH_SIZE = 32                # Unified batch size for stable graph score matching
    LEARNING_RATE = 1e-3           # AdamW learning rate
    WEIGHT_DECAY = 1e-4            # AdamW weight decay

    # Ingestion & Standardization
    data = load_and_preprocess_dataset(dataset_name_or_id=DATASET_NAME, is_climb=True)
    label_col = data.columns[-1]
    feature_cols = [c for c in data.columns if c != label_col]
    minority_class = data[label_col].value_counts().idxmin()

    # Stratified 7:3 Partitioning
    X = data.drop(columns=[label_col])
    y = data[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Identify continuous features and categorical coordinate groups
    cat_cols = [c for c in feature_cols if len(X_train[c].unique()) <= 2]
    cont_cols = [c for c in feature_cols if c not in cat_cols]
    cat_groups = {}
    for c in cat_cols:
        prefix = c.rsplit("_", 1)[0] if "_" in c else c
        cat_groups.setdefault(prefix, []).append(c)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    # Step 1.1: Minority Isolation
    minority_scaled_df = X_train_scaled[y_train.values == minority_class]

    # Step 1.2: TwoNN Adaptive Tangent Window Estimation
    N_star = estimate_twonn_tangent_window(minority_scaled_df.values, alpha=ALPHA, min_bound=5, max_bound=60)

    # Construct Categorical Mask Matrix M_cat in {0, 1}^{D x D}
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    M_cat = np.zeros((len(feature_cols), len(feature_cols)), dtype=bool)
    for _, cols in cat_groups.items():
        if len(cols) > 1:
            grp_indices = [col_to_idx[c] for c in cols]
            for u in grp_indices:
                for v in grp_indices:
                    M_cat[u, v] = True

    # -------------------------------------------------------------------------
    # Module 2: Transposed Feature Graph Construction with Category Masking
    # -------------------------------------------------------------------------
    knn = NearestNeighbors(n_neighbors=N_star, metric="euclidean").fit(minority_scaled_df.values)
    _, neighbor_indices = knn.kneighbors(minority_scaled_df.values)

    graph_list = []
    for i in range(len(minority_scaled_df)):
        local_patch = minority_scaled_df.iloc[neighbor_indices[i], :]

        # Matrix transposition: H^(0) = X_loc^T in R^{D x N*}
        x_node_attrs = torch.tensor(local_patch.T.values, dtype=torch.float)

        # Pairwise Pearson correlations: C_uv = |Corr(X_*,u, X_*,v)|
        corr_matrix = np.nan_to_num(local_patch.corr().abs().values)
        np.fill_diagonal(corr_matrix, 0.0)

        # Decouple intra-group message passing: A_mask = A * (1 - M_cat)
        corr_matrix[M_cat] = 0.0

        edge_indices, edge_weights = [], []
        for u in range(len(feature_cols)):
            for v in range(u + 1, len(feature_cols)):
                val = corr_matrix[u, v]
                if val > CORR_THRESHOLD:
                    edge_indices.extend([[u, v], [v, u]])
                    edge_weights.extend([[val], [val]])

        if len(edge_indices) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)

        graph_list.append(Data(x=x_node_attrs, edge_index=edge_index, edge_attr=edge_attr))

    # -------------------------------------------------------------------------
    # Module 3: Training & Warm-Start Probability Flow Reverse Sampling
    # -------------------------------------------------------------------------
    t0 = time.perf_counter()
    model, diff_manager = train_diffusion(
        graph_list=graph_list,
        window_size=N_star,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY
    )

    virtual_samples_raw = sample_balanced_minority(
        model=model,
        diff_manager=diff_manager,
        graph_list=graph_list,
        feature_names=feature_cols,
        data=pd.concat([X_train_scaled, y_train], axis=1),
        minority_class=minority_class,
        label_col=label_col,
        t_warm=WARM_START_STEP,
        eta=ETA,
        use_moment_align=True
    )

    # -------------------------------------------------------------------------
    # Module 4: Dual-Stream Decoupled Projection
    # -------------------------------------------------------------------------
    virtual_samples_final = dual_stream_decoupled_projection(
        gen_df=virtual_samples_raw,
        orig_train_df=X_train,
        scaler=scaler,
        feature_cols=feature_cols,
        cont_cols=cont_cols,
        cat_groups=cat_groups,
        label_col=label_col
    )
    tgd_runtime = time.perf_counter() - t0
    print(f"\n[TGD Execution Complete] Runtime: {tgd_runtime:.2f}s | Synthesized Records: {len(virtual_samples_final)}")

    release_memory(graph_list, model, diff_manager)

    # -------------------------------------------------------------------------
    # Execution: Pristine Phase (Original Baseline & TGD)
    # -------------------------------------------------------------------------
    pristine_results = evaluate_tgd_pipeline(
        X_train=X_train_scaled,
        y_train=y_train,
        X_test=X_test_scaled,
        y_test=y_test,
        virtual_samples_tgd=virtual_samples_final,
        label_col=label_col
    )

    # -------------------------------------------------------------------------
    # Execution: Modular External Baselines Phase (Full 8 Comparative Baselines)
    # -------------------------------------------------------------------------
    run_all_baselines = True
    baseline_results = {}
    if run_all_baselines:
        baseline_results, virtual_dict, runtime_dict = evaluate_external_baselines(
            X_train=X_train_scaled,
            y_train=y_train,
            X_test=X_test_scaled,
            y_test=y_test,
            num_needed=len(virtual_samples_final),
            label_col=label_col,
            baseline_list=[
                "SMOTE",
                "ADASYN",
                "CTGAN",
                "TVAE",
                "TabDDPM",
                "GOGGLE",
                "TabSyn",
                "TabDiff",
            ],
        )

    # Print Global Benchmark Summary
    print("\n" + "=" * 90)
    print("Global Benchmark Evaluation Summary (Bal-Acc / AUC / F1):")
    print("=" * 90)
    all_results = {**pristine_results, **baseline_results}
    for model_name, metrics in all_results.items():
        print(
            f"  - {model_name:<20}: Bal-Acc = {metrics['Bal-Acc']:.4f} | AUC ="
            f" {metrics['AUC']:.4f} | F1 = {metrics['F1']:.4f}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()