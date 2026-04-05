"""Claude-in-the-loop UMAP optimization for heart MrVI.

Each call runs one UMAP trial with given params and saves:
- 2D scatter plot colored by cell type (stanhue palette)
- Metrics dict
- Coords as .npy

Claude reads the image after each trial and decides next params.
"""

import sys
import os
import json
import time

sys.path.insert(0, '/home/users/chensj16/.claude/skills/stanhue/scripts')

import numpy as np
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import umap

from scatter_colormap import assign_celltype_colors
from paloalto.data import load_and_validate, subsample_stratified
from paloalto.metrics import compute_all

OUT_DIR = '/scratch/users/chensj16/projects/paloalto/trials/heart_claude_loop'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/plots', exist_ok=True)
os.makedirs(f'{OUT_DIR}/coords', exist_ok=True)

# Global state: load once
_STATE = {}


def init():
    """Load adata once."""
    if 'adata' in _STATE:
        return _STATE
    print('Loading heart data...')
    adata = ad.read_h5ad('/scratch/users/chensj16/projects/mrvi/results/heart/heart_mrvi_results.h5ad')
    adata = load_and_validate(adata, 'X_mrvi_u', 'batch', 'cell_type')
    sub_idx = subsample_stratified(adata, 'cell_type', 'batch', n=10000, seed=42)
    labels = adata.obs['cell_type'].values
    # Generate palette from baseline UMAP (stable reference)
    color_map = assign_celltype_colors(adata.obsm['X_umap_u'], labels)
    # Load existing history if present (cumulative across script invocations)
    hist_path = f'{OUT_DIR}/history.json'
    existing = []
    if os.path.exists(hist_path):
        try:
            with open(hist_path) as f:
                existing = json.load(f).get('trials', [])
        except Exception:
            existing = []
    _STATE.update({
        'adata': adata,
        'sub_idx': sub_idx,
        'labels': labels,
        'color_map': color_map,
        'history': existing,
        'baseline_scores': None,
    })
    # Compute baseline scores once
    sub = adata[sub_idx].copy()
    bs = compute_all(sub, 'X_umap_u', 'cell_type', 'batch')
    _STATE['baseline_scores'] = {'scib_overall': bs['scib_overall'],
                                  'scgraph_score': bs['scgraph_corr_weighted']}
    print(f"Baseline X_umap_u: scIB={bs['scib_overall']:.4f} scGraph={bs['scgraph_corr_weighted']:.4f}")
    return _STATE


def run_trial(n_neighbors, min_dist, trial_name=None, seed=42):
    """Fit UMAP with given params, compute metrics, save plot.

    Uses fixed random_state for reproducibility (UMAP is otherwise
    highly stochastic — σ ≈ 0.013 on the objective at n=60k).
    """
    s = init()
    adata = s['adata']
    labels = s['labels']
    sub_idx = s['sub_idx']
    color_map = s['color_map']

    trial_id = len(s['history']) + 1
    if trial_name is None:
        trial_name = f'trial_{trial_id:02d}_n{n_neighbors}_d{min_dist:.3f}_s{seed}'

    t0 = time.time()
    print(f'\n=== Trial {trial_id}: n_neighbors={n_neighbors} min_dist={min_dist:.4f} seed={seed} ===')
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=2, random_state=seed,
    )
    X = adata.obsm['X_mrvi_u'].astype(np.float32)
    coords = reducer.fit_transform(X)
    t_fit = time.time() - t0

    # Metrics on subsample
    adata.obsm['_trial'] = coords
    sub = adata[sub_idx].copy()
    scores = compute_all(sub, '_trial', 'cell_type', 'batch')
    del adata.obsm['_trial']

    scib = float(scores['scib_overall'])
    scgraph = float(scores['scgraph_corr_weighted'])
    obj = 0.5 * scib + 0.5 * scgraph

    # Save coords
    np.save(f'{OUT_DIR}/coords/{trial_name}.npy', coords)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    unique_types = sorted(np.unique(labels))
    for ct in unique_types:
        mask = labels == ct
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color_map.get(ct, '#ccc'), s=0.3, alpha=0.4, rasterized=True)
    ax.set_title(
        f'Trial {trial_id}: n_neighbors={n_neighbors} min_dist={min_dist:.4f}\n'
        f'scIB={scib:.4f} scGraph={scgraph:.4f} obj={obj:.4f}',
        fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=color_map.get(ct, '#ccc'),
                      markersize=7, label=ct) for ct in unique_types]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=7, frameon=False)
    fig.tight_layout()

    # Save full + thumbnail
    fig.savefig(f'{OUT_DIR}/plots/{trial_name}.png', dpi=120,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # Resize to small version for Claude to view
    img = Image.open(f'{OUT_DIR}/plots/{trial_name}.png')
    img.thumbnail((1500, 1500))
    small_path = f'{OUT_DIR}/plots/{trial_name}_small.png'
    img.save(small_path)

    elapsed = time.time() - t0
    result = {
        'trial': trial_id,
        'name': trial_name,
        'params': {'n_neighbors': int(n_neighbors), 'min_dist': float(min_dist)},
        'scores': {'scib_overall': scib, 'scgraph_score': scgraph, 'objective': obj},
        'plot': small_path,
        'elapsed': round(elapsed, 1),
    }
    s['history'].append(result)

    # Persist history
    with open(f'{OUT_DIR}/history.json', 'w') as f:
        json.dump({
            'baseline': s['baseline_scores'],
            'trials': s['history'],
        }, f, indent=2)

    print(f'  scIB={scib:.4f} scGraph={scgraph:.4f} obj={obj:.4f} (fit {t_fit:.0f}s, total {elapsed:.0f}s)')
    print(f'  Plot: {small_path}')
    return result


def summary():
    """Print all trials sorted by objective."""
    s = init()
    print('\n=== Summary ===')
    print(f"Baseline: scIB={s['baseline_scores']['scib_overall']:.4f} "
          f"scGraph={s['baseline_scores']['scgraph_score']:.4f} "
          f"obj={0.5*(s['baseline_scores']['scib_overall']+s['baseline_scores']['scgraph_score']):.4f}")
    print(f"{'#':>3} {'n_nb':>5} {'min_d':>7} {'scIB':>7} {'scGraph':>9} {'obj':>7}")
    for t in sorted(s['history'], key=lambda x: -x['scores']['objective']):
        p = t['params']
        sc = t['scores']
        print(f"{t['trial']:>3} {p['n_neighbors']:>5} {p['min_dist']:>7.4f} "
              f"{sc['scib_overall']:>7.4f} {sc['scgraph_score']:>9.4f} "
              f"{sc['objective']:>7.4f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--d', type=float, required=True)
    p.add_argument('--name', default=None)
    args = p.parse_args()
    run_trial(args.n, args.d, args.name)
    summary()
