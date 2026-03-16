# StackFeat-RL

**Reinforcement Learning over Iterative Dual-Criterion Feature Selection for Stable Biomarker Discovery**

> Yermekov A., Herrera Martí D.A. — ECCB 2026 / OUP Bioinformatics (submitted)

StackFeat-RL learns the hyperparameters of an iterative dual-criterion feature selection algorithm via REINFORCE policy gradients. It selects compact, stable gene panels from high-dimensional expression data while incorporating protein interaction priors from the STRING database.

## Key features

- **Dual-criterion selection**: intersects genes ranked by coefficient magnitude *and* selection frequency, guarding against sign-inconsistent and infrequent-but-consistent features
- **REINFORCE-optimised**: learns the feature retention fraction (m_frac) from data — no manual threshold tuning
- **Convergence guarantees**: normalised statistics converge to population-level importance measures via the law of large numbers
- **Biological priors**: accepts STRING protein–protein interaction networks through the policy state representation
- **Posterior network**: outputs a disease-filtered interaction network M* = M · ψ via co-selection frequencies
- **10–17× faster** than base StackFeat through single-fit ElasticNetCV-anchored regularisation
- **Fully automatic**: no manual specification of panel size, regularisation strength, or stopping criterion

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

## Quick start

```python
import numpy as np
from stackfeat_rl import StackFeatRL

# Load your data
X = np.load('expression_matrix.npy')   # (n_samples, n_genes)
y = np.load('labels.npy')              # (n_samples,) binary

# Optional: STRING interaction matrix (p x p)
# M = np.load('string_adjacency.npy')
M = None  # runs without network priors

# Run nested CV (unbiased evaluation)
model = StackFeatRL(episodes=15, inner_folds=5, lr=0.5)
results = model.fit_nested_cv(X, y, M, outer_folds=10, gene_names=gene_names)

print(f"AUC: {results['mean_auc']:.4f} ± {results['std_auc']:.4f}")
print(f"Consensus genes: {results['consensus_gene_names']}")
```

## Full dataset run (for posterior network M*)

```python
# After validating via nested CV, run on full data for M* and final panel
results = model.fit_full(X, y, M, gene_names=gene_names, save_dir='outputs/')
# Saves: posterior_M_star.npz, coselection_psi.npz, metadata.json
```

## Algorithm overview

```
For each outer fold:
  1. ElasticNetCV → frozen alpha
  2. For each REINFORCE episode (K=15):
       a. Warmup iteration (t=1): uniform penalties
       b. Policy loop (t=2..T):
          - Compute state: (p_hat, |mu_hat|, n_i, d_i, 1)
          - Per-gene penalty: lambda_i = alpha * sigmoid(theta · s_i)
          - k-fold ElasticNet → update w, c, psi
          - Accumulate REINFORCE gradient
          - Convergence check
       c. Dual-criterion selection: S* = top-m(|w|) ∩ top-m(c)
       d. Reward: R = AUC(S*) - lambda_s * |S*|
       e. Update theta
  3. Evaluate S* on held-out outer test fold
```

## Policy parameters

| Parameter | Feature | Role |
|-----------|---------|------|
| θ₁ | Selection frequency (p̂) | Per-gene penalty modulation |
| θ₂ | Coefficient magnitude (\|μ̂\|) | Per-gene penalty modulation |
| θ₃ | STRING network support (n_i) | Biological prior |
| θ₄ | Co-selection density (d_i) | Data-driven interaction |
| θ₅ | m_frac (feature retention) | Panel size control |

m_frac = 0.25 + 0.65 · σ(θ₅), initialised at 0.575.

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodes` | 15 | REINFORCE episodes per outer fold |
| `inner_folds` | 5 | Inner CV folds |
| `lr` | 0.5 | REINFORCE learning rate |
| `baseline_decay` | 0.9 | EMA decay for reward baseline |
| `eps` | 0.02 | Convergence tolerance |
| `sparsity_penalty` | 0.001 | λ_s: per-gene reward penalty |
| `l1_ratio` | 0.5 | ElasticNet L1/L2 mixing |
| `min_genes` | 3 | Minimum genes in panel |

## Results (from paper)

### COVID-19 miRNA ([GSE240888](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE240888))

| Method | Mean AUC | Std | Median | Avg genes |
|--------|----------|-----|--------|-----------|
| mRMR (k=9) | 0.865 | 0.125 | 0.825 | 9.0 |
| Stab. Sel. | 0.875 | 0.132 | 0.875 | 46.5 |
| StackFeat | 0.880 | 0.151 | 0.975 | 4.6 |
| Boruta | 0.890 | 0.133 | 0.975 | 25.1 |
| **SF-RL** | **0.895** | 0.126 | **0.975** | **8.8** |
| ElasticNet | 0.906 | 0.123 | 0.975 | 20.1 |

No significant pairwise differences (p > 0.19).

### Alzheimer's disease ([GSE84422](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84422))

| Task | Method | Mean AUC | Genes | p vs SF-RL |
|------|--------|----------|-------|------------|
| Normal vs Possible | mRMR (k=48) | 0.796 | 48.0 | <0.001 |
| | ElasticNet | 0.884 | 188.9 | 0.096 |
| | Boruta | 0.880 | 109.9 | 0.090 |
| | **SF-RL** | **0.915** | **47.8** | — |
| Normal vs Probable | mRMR (k=42) | 0.748 | 42.0 | <0.001 |
| | ElasticNet | 0.835 | 172.3 | **0.004** |
| | Boruta | 0.856 | 124.2 | 0.109 |
| | **SF-RL** | **0.882** | **41.8** | — |
| Normal vs Definite | mRMR (k=56) | 0.816 | 56.0 | **0.003** |
| | ElasticNet | 0.903 | 205.8 | 0.096 |
| | Boruta | 0.921 | 140.9 | 0.567 |
| | **SF-RL** | **0.925** | **55.6** | — |

## Citation

```bibtex
@article{yermekov2026stackfeatrl,
  title={StackFeat-RL: Reinforcement Learning over Iterative Dual-Criterion
         Feature Selection for Stable Biomarker Discovery},
  author={Yermekov, A. and Herrera Mart{\'i}, D. A.},
  journal={Submitted to Bioinformatics (ECCB 2026)},
  year={2026}
}
```

## License

MIT — see [LICENSE](LICENSE).

## TODO

- [ ] Add STRING network loader utility (currently expects pre-built adjacency matrix)
- [ ] Add example notebook with synthetic data
- [ ] Add data preprocessing scripts for GEO datasets
- [ ] Parallel episode execution (episodes are independent by design)
- [ ] Support for custom classifiers (currently KNN; paper also uses ensemble)
- [ ] Network visualization utilities (posterior M* plotting)
- [ ] Benchmark scripts for reproducing paper results
