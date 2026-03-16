"""
StackFeat-RL: Reinforcement Learning over Iterative Dual-Criterion Feature Selection
=====================================================================================

Implementation of the StackFeat-RL algorithm as described in:

    Yermekov & Herrera Martí, "StackFeat-RL: Reinforcement Learning over
    Iterative Dual-Criterion Feature Selection for Stable Biomarker Discovery",
    ECCB 2026 / OUP Bioinformatics (submitted).

The algorithm learns the feature retention fraction (m_frac) and per-gene
penalty modulation via REINFORCE policy gradients, while using a single
ElasticNetCV call per outer fold for regularisation strength (alpha).

Usage:
    from stackfeat_rl import StackFeatRL
    model = StackFeatRL(episodes=15, inner_folds=5, lr=0.5)
    results = model.fit_nested_cv(X, y, M, outer_folds=10)
"""

import os
import json
import time
import numpy as np
from scipy import sparse
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def compute_m_fraction(theta_5, a=0.25, b=0.65):
    """
    Compute feature retention fraction from theta_5.

    m_frac = a + b * sigmoid(theta_5)

    Default: m_frac in [0.25, 0.90], init = 0.575 (at theta_5=0).

    Parameters
    ----------
    theta_5 : float
        Learned parameter controlling m_frac.
    a : float
        Lower bound of m_frac range.
    b : float
        Width of m_frac range.

    Returns
    -------
    m_frac : float
        Feature retention fraction.
    """
    return a + b * sigmoid(theta_5)


def compute_state(c, w, M, cosel, last_selected, t, k,
                  total_folds_so_far, use_string=True):
    """
    Compute per-gene state representation at iteration t.

    State features (per gene i):
        s[i, 0] = p_hat_i   : normalised selection frequency
        s[i, 1] = |mu_hat_i|: normalised absolute mean coefficient
        s[i, 2] = n_i       : mean STRING interaction with selected set
        s[i, 3] = d_i       : mean co-selection frequency with selected set
        s[i, 4] = 1.0       : constant (bias for theta_5 / m_frac)

    Parameters
    ----------
    c : np.ndarray (p,)
        Cumulative selection counts.
    w : np.ndarray (p,)
        Cumulative signed coefficients.
    M : np.ndarray or sparse (p, p)
        STRING interaction matrix.
    cosel : sparse (p, p)
        Co-selection count matrix.
    last_selected : set
        Gene indices selected in the previous iteration.
    t : int
        Current iteration (1-indexed).
    k : int
        Number of inner CV folds.
    total_folds_so_far : int
        Total folds completed before this iteration.
    use_string : bool
        If False, network features (n_i, d_i) are set to zero.

    Returns
    -------
    state : np.ndarray (p, 5)
        State matrix.
    """
    p = len(c)
    state = np.zeros((p, 5))

    if t > 1:
        total_obs = (t - 1) * k
        state[:, 0] = c / total_obs                # p_hat
        state[:, 1] = np.abs(w) / total_obs         # |mu_hat|

        if len(last_selected) > 0 and use_string:
            sel_idx = sorted(last_selected)
            # n_i: mean STRING interaction with selected set
            state[:, 2] = np.asarray(M[:, sel_idx].mean(axis=1)).ravel()
            # d_i: mean co-selection frequency with selected set
            if total_folds_so_far > 0:
                cosel_sel = np.asarray(cosel[:, sel_idx].mean(axis=1)).ravel()
                state[:, 3] = cosel_sel / total_folds_so_far

    # Constant feature for m_frac learning (theta_5 acts as bias)
    state[:, 4] = 1.0

    return state


def compute_posterior_network(M, cosel, total_folds):
    """
    Compute posterior network M* = M * psi.

    psi_ij = co-selection frequency = cosel_ij / total_folds.
    M*_ij = M_ij * psi_ij: retains a STRING edge only if both genes
    were repeatedly co-selected.

    Parameters
    ----------
    M : np.ndarray or sparse (p, p)
        STRING interaction matrix.
    cosel : sparse (p, p)
        Co-selection count matrix.
    total_folds : int
        Total folds across all iterations.

    Returns
    -------
    M_star : sparse (p, p)
        Posterior interaction matrix.
    psi : sparse (p, p)
        Co-selection frequency matrix.
    """
    psi = cosel / total_folds
    M_sparse = sparse.csr_matrix(M) if not sparse.issparse(M) else M
    M_star = M_sparse.multiply(psi)
    return M_star, psi


# =============================================================================
# CORE ALGORITHM
# =============================================================================

def _dual_criterion_select(w, c, genes_per_fold, m_frac, min_genes=3,
                           use_union=False):
    """
    Apply dual-criterion selection: S* = top-m by |w| ∩ top-m by c.

    Parameters
    ----------
    w : np.ndarray (p,)
        Cumulative signed coefficients.
    c : np.ndarray (p,)
        Cumulative selection counts.
    genes_per_fold : list
        Number of genes selected per fold (for computing m).
    m_frac : float
        Feature retention fraction.
    min_genes : int
        Hard minimum number of genes.
    use_union : bool
        If True, use union instead of intersection.

    Returns
    -------
    S : set
        Selected gene indices.
    """
    m = max(int(np.mean(genes_per_fold) * m_frac), min_genes)
    S_w = set(np.argsort(-np.abs(w))[:m])
    S_c = set(np.argsort(-c)[:m])
    S = S_w | S_c if use_union else S_w & S_c

    # Fallback: widen m if intersection too small
    if len(S) < min_genes:
        m = max(m + min_genes, int(np.mean(genes_per_fold) * 0.75))
        S_w = set(np.argsort(-np.abs(w))[:m])
        S_c = set(np.argsort(-c)[:m])
        S = S_w & S_c

    # Final fallback: top genes by weight
    if len(S) < min_genes:
        S = set(np.argsort(-np.abs(w))[:min_genes])

    return S


def _evaluate_gene_set(X, y, S, skf, seed=0):
    """
    Evaluate a gene set via cross-validated AUC using KNN classifier.

    Parameters
    ----------
    X : np.ndarray (n, p)
        Feature matrix.
    y : np.ndarray (n,)
        Target vector.
    S : set
        Selected gene indices.
    skf : StratifiedKFold
        CV splitter.
    seed : int
        Random seed for classifier.

    Returns
    -------
    auc : float
        Mean AUC across folds.
    """
    if len(S) < 2:
        return 0.5

    S_list = sorted(S)
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        try:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx][:, S_list])
            X_te = scaler.transform(X[test_idx][:, S_list])

            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(X_tr, y[train_idx])
            y_pred = clf.predict(X_te)

            if (len(np.unique(y[test_idx])) >= 2 and
                    len(np.unique(y_pred)) >= 2):
                fold_aucs.append(metrics.roc_auc_score(y[test_idx], y_pred))
            else:
                fold_aucs.append(0.5)
        except Exception:
            fold_aucs.append(0.5)

    return np.mean(fold_aucs) if fold_aucs else 0.5


def run_one_episode(X, y, M, theta, k=5, max_iter=20, eps=0.02,
                    l1_ratio=0.5, episode=0, frozen_alpha=None,
                    sparsity_penalty=0.001, min_genes=3, verbose=False):
    """
    Run one StackFeat-RL episode with fixed theta.

    Each episode:
    1. Warmup (t=1): uniform penalties, populate w, c
    2. Policy loop (t=2..T): per-gene penalties from theta, gradient tracking
    3. Dual-criterion selection at each iteration
    4. Convergence check (2 consecutive AUC diffs < eps)

    Parameters
    ----------
    X : np.ndarray (n, p)
        Feature matrix.
    y : np.ndarray (n,)
        Target vector (binary).
    M : np.ndarray or sparse (p, p)
        STRING interaction matrix.
    theta : np.ndarray (5,)
        Policy parameters.
    k : int
        Number of inner CV folds.
    max_iter : int
        Maximum iterations per episode.
    eps : float
        Convergence tolerance.
    l1_ratio : float
        ElasticNet L1/L2 mixing ratio.
    episode : int
        Episode index (for seed computation).
    frozen_alpha : float
        Regularisation strength from ElasticNetCV.
    sparsity_penalty : float
        Per-gene penalty in reward (lambda_s).
    min_genes : int
        Minimum genes in final panel.
    verbose : bool
        Print progress.

    Returns
    -------
    R : float
        Episode reward (AUC - lambda_s * |S*|).
    grad : np.ndarray (5,)
        Accumulated policy gradient.
    S : set
        Selected gene indices.
    n_iters : int
        Number of iterations completed.
    history : dict
        Episode history (w, c, cosel, total_folds).
    """
    n, p = X.shape
    c = np.zeros(p)
    w = np.zeros(p)
    cosel = sparse.lil_matrix((p, p))
    total_grad = np.zeros(5)
    total_folds = 0
    prev_aucs = []
    genes_per_fold_all = []
    last_selected = set()

    s_0 = 0
    T = max_iter
    use_string = np.sum(M > 0) > 0 if not sparse.issparse(M) else M.nnz > 0

    alpha = frozen_alpha if frozen_alpha is not None else 0.1

    # ================================================================
    # WARMUP (t=1): uniform penalties, no gradient
    # ================================================================
    fold_seed = s_0 + episode * T + 0
    penalty_weights = np.full(p, 0.5)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=fold_seed)

    iter_selected = set()
    iter_genes_per_fold = []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])

        X_tr_pen = X_tr * penalty_weights[np.newaxis, :]
        enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                          max_iter=200, random_state=fold_seed)
        enet.fit(X_tr_pen, y[train_idx])
        beta = enet.coef_ / penalty_weights

        selected = (beta != 0)
        c += selected.astype(float)
        w += beta

        sel_idx = np.where(selected)[0]
        for ii in range(len(sel_idx)):
            for jj in range(ii + 1, len(sel_idx)):
                cosel[sel_idx[ii], sel_idx[jj]] += 1
                cosel[sel_idx[jj], sel_idx[ii]] += 1

        total_folds += 1
        iter_selected.update(sel_idx)
        iter_genes_per_fold.append(int(selected.sum()))

    last_selected = iter_selected
    genes_per_fold_all.extend(iter_genes_per_fold)

    m_frac = compute_m_fraction(theta[4])
    S = _dual_criterion_select(w, c, iter_genes_per_fold, m_frac, min_genes)
    auc = _evaluate_gene_set(X, y, S, skf, seed=fold_seed)
    prev_aucs.append(auc)

    if verbose:
        print(f"  [Warmup] AUC={auc:.4f} (S*={len(S)})")

    # ================================================================
    # POLICY LOOP (t=2..T)
    # ================================================================
    for t in range(2, max_iter + 1):
        fold_seed = s_0 + episode * T + (t - 1)

        state = compute_state(c, w, M, cosel, last_selected, t, k,
                              total_folds, use_string)

        logits = state @ theta
        pi = sigmoid(logits)
        pi = np.clip(pi, 1e-10, 1 - 1e-10)
        penalty_weights = np.where(pi > 1e-10, pi, 1e-10)

        skf = StratifiedKFold(n_splits=k, shuffle=True,
                              random_state=fold_seed)
        iter_selected = set()
        iter_genes_per_fold = []

        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])

            X_tr_pen = X_tr * penalty_weights[np.newaxis, :]
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                              max_iter=200, random_state=fold_seed)
            enet.fit(X_tr_pen, y[train_idx])
            beta = enet.coef_ / penalty_weights

            selected = (beta != 0)
            c += selected.astype(float)
            w += beta

            sel_idx = np.where(selected)[0]
            for ii in range(len(sel_idx)):
                for jj in range(ii + 1, len(sel_idx)):
                    cosel[sel_idx[ii], sel_idx[jj]] += 1
                    cosel[sel_idx[jj], sel_idx[ii]] += 1

            total_folds += 1
            iter_selected.update(sel_idx)
            iter_genes_per_fold.append(int(selected.sum()))

        last_selected = iter_selected
        genes_per_fold_all.extend(iter_genes_per_fold)

        # Gradient: (1[excluded] - pi) * state, normalised by p*(T-1)
        excluded = np.zeros(p)
        for j in range(p):
            if j not in iter_selected:
                excluded[j] = 1.0
        per_gene_signal = excluded - pi
        grad_iter = (per_gene_signal[:, np.newaxis] * state).sum(axis=0)
        total_grad += grad_iter / p

        m_frac = compute_m_fraction(theta[4])
        S = _dual_criterion_select(w, c, iter_genes_per_fold,
                                   m_frac, min_genes)
        auc = _evaluate_gene_set(X, y, S, skf, seed=fold_seed)
        prev_aucs.append(auc)

        if verbose:
            print(f"  Iter {t}: AUC={auc:.4f} (S*={len(S)})")

        # Convergence: 2 consecutive diffs < eps
        if len(prev_aucs) >= 3:
            d1 = abs(prev_aucs[-1] - prev_aucs[-2])
            d2 = abs(prev_aucs[-2] - prev_aucs[-3])
            if d1 < eps and d2 < eps:
                if verbose:
                    print(f"  Converged at iteration {t}")
                break

    # Normalise gradient by (T-1)
    n_policy_iters = max(t - 1, 1)
    total_grad /= n_policy_iters

    # Final selection
    m_frac = compute_m_fraction(theta[4])
    m_final = max(int(np.mean(genes_per_fold_all) * m_frac), min_genes)
    S = _dual_criterion_select(w, c, genes_per_fold_all, m_frac, min_genes)

    R = prev_aucs[-1] - sparsity_penalty * len(S)

    history = {
        'aucs': prev_aucs,
        'c': c.copy(),
        'w': w.copy(),
        'cosel': cosel.tocsr(),
        'total_folds': total_folds,
    }

    return R, total_grad, S, t, history


# =============================================================================
# MAIN CLASS
# =============================================================================

class StackFeatRL:
    """
    StackFeat-RL: REINFORCE-optimised dual-criterion feature selection.

    Parameters
    ----------
    episodes : int
        Number of REINFORCE episodes per outer fold (default 15).
    inner_folds : int
        Number of inner CV folds (default 5).
    max_iter : int
        Maximum iterations per episode (default 20).
    lr : float
        REINFORCE learning rate (default 0.5).
    baseline_decay : float
        EMA decay for reward baseline (default 0.9).
    eps : float
        Convergence tolerance (default 0.02).
    sparsity_penalty : float
        Per-gene penalty in reward (default 0.001).
    l1_ratio : float
        ElasticNet L1/L2 mixing (default 0.5).
    min_genes : int
        Minimum genes in final panel (default 3).
    verbose : bool
        Print progress (default True).
    """

    def __init__(self, episodes=15, inner_folds=5, max_iter=20, lr=0.5,
                 baseline_decay=0.9, eps=0.02, sparsity_penalty=0.001,
                 l1_ratio=0.5, min_genes=3, verbose=True):
        self.episodes = episodes
        self.inner_folds = inner_folds
        self.max_iter = max_iter
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.eps = eps
        self.sparsity_penalty = sparsity_penalty
        self.l1_ratio = l1_ratio
        self.min_genes = min_genes
        self.verbose = verbose

    def fit_nested_cv(self, X, y, M=None, outer_folds=10, gene_names=None,
                      save_dir=None):
        """
        Run StackFeat-RL with nested cross-validation.

        Outer CV provides unbiased AUC estimates. Inner CV is used for
        feature selection and policy learning. The outer test fold is
        never seen during training.

        Parameters
        ----------
        X : np.ndarray (n, p)
            Feature matrix.
        y : np.ndarray (n,)
            Binary target vector.
        M : np.ndarray or sparse (p, p), optional
            STRING interaction matrix. If None, a zero matrix is used
            (network features disabled).
        outer_folds : int
            Number of outer CV folds (default 10).
        gene_names : list of str, optional
            Gene names for output.
        save_dir : str, optional
            Directory to save results.

        Returns
        -------
        results : dict
            Dictionary with keys:
            - outer_aucs: list of per-fold AUCs
            - mean_auc, std_auc, median_auc: summary statistics
            - consensus_genes: gene indices selected in majority of folds
            - consensus_gene_names: gene names (if provided)
            - average_theta: mean learned parameters
            - fold_details: per-fold details
        """
        n, p = X.shape

        if M is None:
            M = np.zeros((p, p))

        outer_skf = StratifiedKFold(n_splits=outer_folds, shuffle=True,
                                    random_state=42)

        outer_aucs = []
        all_genes = {}
        all_thetas = []
        fold_details = []

        if self.verbose:
            print("=" * 70)
            print("StackFeat-RL: Nested Cross-Validation")
            print("=" * 70)
            print(f"Outer: {outer_folds}-fold | Inner: {self.inner_folds}-fold"
                  f" | Episodes: {self.episodes} | lr: {self.lr}")
            print("=" * 70)

        for fold_idx, (train_idx, test_idx) in enumerate(
                outer_skf.split(X, y)):

            if self.verbose:
                print(f"\nOUTER FOLD {fold_idx + 1}/{outer_folds}")

            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]

            # Step 1: ElasticNetCV for alpha
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            ecv_skf = StratifiedKFold(n_splits=5, shuffle=True,
                                      random_state=42)
            try:
                ecv = ElasticNetCV(l1_ratio=self.l1_ratio, cv=ecv_skf,
                                   random_state=42, n_jobs=-1, max_iter=200)
                ecv.fit(X_train_scaled, y_train)
                frozen_alpha = ecv.alpha_
            except Exception:
                frozen_alpha = 0.1

            if self.verbose:
                print(f"  ElasticNetCV alpha = {frozen_alpha:.6f}")

            # Step 2: REINFORCE episodes
            theta = np.zeros(5)
            baseline = None

            for ep in range(self.episodes):
                R, grad, genes, n_iters, hist = run_one_episode(
                    X_train, y_train, M, theta,
                    k=self.inner_folds, max_iter=self.max_iter,
                    eps=self.eps, l1_ratio=self.l1_ratio,
                    episode=fold_idx * self.episodes + ep,
                    frozen_alpha=frozen_alpha,
                    sparsity_penalty=self.sparsity_penalty,
                    min_genes=self.min_genes,
                    verbose=(self.verbose and ep < 2),
                )

                if baseline is None:
                    baseline = R

                R_adj = R - len(genes) * self.sparsity_penalty
                theta = theta + self.lr * (R_adj - baseline) * grad
                theta[4] = np.clip(theta[4], -4, 4)  # clip m_frac param
                baseline = (self.baseline_decay * baseline +
                            (1 - self.baseline_decay) * R)

            # Step 3: Evaluate on outer test fold
            final_genes = genes

            if len(final_genes) >= 2:
                genes_list = sorted(final_genes)
                scaler_eval = StandardScaler()
                X_tr_eval = scaler_eval.fit_transform(X_train[:, genes_list])
                X_te_eval = scaler_eval.transform(X_test[:, genes_list])

                clf = KNeighborsClassifier(n_neighbors=3)
                clf.fit(X_tr_eval, y_train)
                y_pred = clf.predict(X_te_eval)

                if (len(np.unique(y_test)) >= 2 and
                        len(np.unique(y_pred)) >= 2):
                    outer_auc = metrics.roc_auc_score(y_test, y_pred)
                else:
                    outer_auc = 0.5
            else:
                outer_auc = 0.5

            outer_aucs.append(outer_auc)
            all_genes[fold_idx] = final_genes
            all_thetas.append(theta.copy())

            gene_names_fold = ([gene_names[i] for i in sorted(final_genes)]
                               if gene_names else [])

            fold_details.append({
                'fold': fold_idx,
                'outer_auc': outer_auc,
                'n_genes': len(final_genes),
                'gene_names': gene_names_fold,
                'theta': theta.tolist(),
                'm_frac': compute_m_fraction(theta[4]),
            })

            if self.verbose:
                m_frac = compute_m_fraction(theta[4])
                print(f"  Outer AUC: {outer_auc:.4f} | "
                      f"Genes: {len(final_genes)} | m_frac: {m_frac:.3f}")

        # Aggregate
        avg_theta = np.mean(all_thetas, axis=0)
        gene_counts = {}
        for fold_genes in all_genes.values():
            for g in fold_genes:
                gene_counts[g] = gene_counts.get(g, 0) + 1
        consensus = {g for g, cnt in gene_counts.items()
                     if cnt >= outer_folds // 2 + 1}

        results = {
            'outer_aucs': outer_aucs,
            'mean_auc': float(np.mean(outer_aucs)),
            'std_auc': float(np.std(outer_aucs)),
            'median_auc': float(np.median(outer_aucs)),
            'consensus_genes': sorted(consensus),
            'consensus_gene_names': (
                [gene_names[i] for i in sorted(consensus)]
                if gene_names else []
            ),
            'average_theta': avg_theta.tolist(),
            'average_m_frac': float(compute_m_fraction(avg_theta[4])),
            'fold_details': fold_details,
            'all_genes': {k: sorted(v) for k, v in all_genes.items()},
        }

        if self.verbose:
            print(f"\n{'='*70}")
            print("NESTED CV RESULTS")
            print("=" * 70)
            print(f"AUC: {results['mean_auc']:.4f} "
                  f"± {results['std_auc']:.4f}")
            print(f"Consensus genes: {len(consensus)}")
            if results['consensus_gene_names']:
                print(f"  {', '.join(results['consensus_gene_names'])}")
            print("=" * 70)

        # Save results
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            results_path = os.path.join(save_dir, 'nested_cv_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            if self.verbose:
                print(f"Saved results to {results_path}")

        return results

    def fit_full(self, X, y, M=None, gene_names=None, save_dir=None):
        """
        Run StackFeat-RL on full dataset (no outer CV).

        Useful for generating the posterior network M* and final gene panel
        after hyperparameters are validated via nested CV.

        Parameters
        ----------
        X, y, M, gene_names, save_dir : see fit_nested_cv.

        Returns
        -------
        results : dict
            Final theta, genes, M_star, psi matrices.
        """
        n, p = X.shape
        if M is None:
            M = np.zeros((p, p))

        # ElasticNetCV for alpha
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ecv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            ecv = ElasticNetCV(l1_ratio=self.l1_ratio, cv=ecv_skf,
                               random_state=42, n_jobs=-1, max_iter=200)
            ecv.fit(X_scaled, y)
            frozen_alpha = ecv.alpha_
        except Exception:
            frozen_alpha = 0.1

        if self.verbose:
            print(f"ElasticNetCV alpha = {frozen_alpha:.6f}")

        # REINFORCE episodes
        theta = np.zeros(5)
        baseline = None
        all_results = []

        for ep in range(self.episodes):
            R, grad, genes, n_iters, hist = run_one_episode(
                X, y, M, theta,
                k=self.inner_folds, max_iter=self.max_iter,
                eps=self.eps, l1_ratio=self.l1_ratio,
                episode=ep, frozen_alpha=frozen_alpha,
                sparsity_penalty=self.sparsity_penalty,
                min_genes=self.min_genes,
                verbose=(self.verbose and ep < 3),
            )

            if baseline is None:
                baseline = R

            R_adj = R - len(genes) * self.sparsity_penalty
            theta = theta + self.lr * (R_adj - baseline) * grad
            theta[4] = np.clip(theta[4], -4, 4)
            baseline = (self.baseline_decay * baseline +
                        (1 - self.baseline_decay) * R)

            all_results.append({
                'episode': ep, 'R': float(R),
                'n_genes': len(genes), 'n_iters': n_iters,
            })

            if self.verbose:
                m_frac = compute_m_fraction(theta[4])
                print(f"Ep {ep:3d} | AUC={R + len(genes)*self.sparsity_penalty:.4f}"
                      f" ({len(genes)} genes) | m_frac={m_frac:.3f}")

        # Posterior network
        M_star, psi = compute_posterior_network(M, hist['cosel'],
                                                hist['total_folds'])

        # Save
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            sparse.save_npz(os.path.join(save_dir, 'posterior_M_star.npz'),
                            sparse.csr_matrix(M_star))
            sparse.save_npz(os.path.join(save_dir, 'coselection_psi.npz'),
                            sparse.csr_matrix(psi))

            meta = {
                'final_theta': theta.tolist(),
                'final_genes': sorted(genes),
                'final_gene_names': (
                    [gene_names[i] for i in sorted(genes)]
                    if gene_names else []
                ),
                'm_frac': float(compute_m_fraction(theta[4])),
                'episodes': all_results,
            }
            with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
                json.dump(meta, f, indent=2)

            if self.verbose:
                print(f"\nSaved M*, psi, metadata to {save_dir}")

        return {
            'theta': theta,
            'genes': genes,
            'gene_names': ([gene_names[i] for i in sorted(genes)]
                           if gene_names else []),
            'M_star': M_star,
            'psi': psi,
            'episodes': all_results,
        }
