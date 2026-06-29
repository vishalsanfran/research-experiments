"""
ZOO query-budget ablation (C6).
Tests whether XGBoost ZOO degeneracy persists at higher query budgets.
ZOO budget = number of random coordinates sampled to estimate the gradient.
Full-feature budget (n_features) = 48 for Phishing, varies for UNSW.
If XGB RI stays flat across budgets → degeneracy is intrinsic to the
  piecewise-constant surface, not a query-budget artifact.
If RF RI drops at higher budgets → ZOO works on smooth surfaces with enough queries.
Run: .venv/bin/python zoo_budget_ablation.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import xgboost as xgb

EPSILONS  = np.linspace(0, 0.3, 10)
EPS_MAX   = EPSILONS[-1]
N_RI      = 100   # smaller for speed; enough to see the trend
SEED      = 42

os.makedirs('figures', exist_ok=True)

def _xent(proba, y):
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-10, 1.0))

def zoo_attack_budget(predict_proba, X, y, epsilon, n_queries, rng, delta=1e-3):
    """ZOO with random coordinate sampling (n_queries coords out of n_features).
    n_queries controls the query budget; full budget = X.shape[1].
    Higher n_queries → better gradient estimate → stronger attack.
    """
    n_features = X.shape[1]
    n_q = min(n_queries, n_features)
    coords = rng.choice(n_features, size=n_q, replace=False)
    base = _xent(predict_proba(X), y)
    grad = np.zeros_like(X)
    for j in coords:
        Xp = X.copy(); Xp[:, j] += delta
        grad[:, j] = (_xent(predict_proba(Xp), y) - base) / delta
    return np.clip(X + epsilon * np.sign(grad), X - epsilon, X + epsilon)

def compute_ri(model, X, y, n_queries):
    rng = np.random.default_rng(SEED)
    accs = []
    for eps in EPSILONS:
        Xadv = X if eps == 0 else zoo_attack_budget(
            model.predict_proba, X, y, eps, n_queries, rng)
        accs.append(model.score(Xadv, y))
    return np.array(accs), round(float(np.trapezoid(accs, EPSILONS) / EPS_MAX), 3)

def load_phishing():
    df = pd.read_csv('Phishing_Legitimate_full.csv')
    y = df['CLASS_LABEL'].values
    X = df.drop(columns=['id', 'CLASS_LABEL'], errors='ignore').values.astype(np.float32)
    return X, y

def load_unsw():
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    df = df.drop(columns=['id', 'attack_cat'], errors='ignore')
    for c in ['proto', 'service', 'state']:
        if c in df.columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    y = df['label'].values.astype(int)
    X = df.drop(columns=['label']).apply(
        pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    return X.values, y

DATASETS = [
    dict(name='Phishing',  key='phishing', loader=load_phishing,
         rf=dict(n_estimators=200, max_depth=12),
         xgb=dict(n_estimators=200, max_depth=6, subsample=0.8, colsample_bytree=0.8)),
    dict(name='UNSW-NB15', key='unsw',     loader=load_unsw,
         rf=dict(n_estimators=200, max_depth=12),
         xgb=dict(n_estimators=200, max_depth=6, subsample=0.8, colsample_bytree=0.8)),
]

all_results = {}

for ds in DATASETS:
    name = ds['name']
    print(f'\n{"="*55}')
    print(f'Dataset: {name}')

    X_all, y_all = ds['loader']()
    n_features = X_all.shape[1]
    # Budget fractions: 12.5%, 25%, 50%, 100% of features
    BUDGETS = sorted(set([
        max(1, n_features // 8),
        max(1, n_features // 4),
        max(1, n_features // 2),
        n_features
    ]))
    print(f'n_features={n_features}  budgets={BUDGETS}')
    print('='*55)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    ri_idx = resample(np.arange(len(X_te)), n_samples=N_RI,
                      stratify=y_te, random_state=SEED)
    X_ri, y_ri = X_te[ri_idx], y_te[ri_idx]

    rf_kw  = dict(**ds['rf'], random_state=SEED, n_jobs=-1)
    xgb_kw = dict(**ds['xgb'], learning_rate=0.05, random_state=SEED,
                  eval_metric='logloss', verbosity=0)

    rf_m  = RandomForestClassifier(**rf_kw).fit(X_tr, y_tr)
    xgb_m = xgb.XGBClassifier(**xgb_kw).fit(X_tr, y_tr)
    print(f'Trained: RF clean={rf_m.score(X_te, y_te):.3f}  '
          f'XGB clean={xgb_m.score(X_te, y_te):.3f}')

    ds_res = {'RF': {}, 'XGB': {}}
    for budget in BUDGETS:
        print(f'\n  budget={budget:>3}q ({budget/n_features*100:.0f}%):', end=' ', flush=True)
        for model, mname in [(rf_m, 'RF'), (xgb_m, 'XGB')]:
            accs, ri = compute_ri(model, X_ri, y_ri, budget)
            ds_res[mname][budget] = (accs, ri)
            print(f'{mname}={ri}', end='  ', flush=True)
    print()
    all_results[name] = (ds_res, BUDGETS, n_features)

    print(f'\n── {name}: ZOO RI vs query budget ───────────────────────────')
    print(f'{"Budget":>10}', end='')
    for b in BUDGETS:
        pct = int(b / n_features * 100)
        print(f'  {b}q({pct}%)', end='')
    print()
    for mname in ['RF', 'XGB']:
        print(f'{mname:>10}', end='')
        for b in BUDGETS:
            print(f'  {ds_res[mname][b][1]:>8.3f}', end='')
        print()

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(BUDGETS)))
    for ax, mname in [(axes[0], 'RF'), (axes[1], 'XGB')]:
        for i, b in enumerate(BUDGETS):
            accs, ri = ds_res[mname][b]
            pct = int(b / n_features * 100)
            ax.plot(EPSILONS, accs, 'o-', color=colors[i],
                    label=f'{b}q/{n_features} ({pct}%), RI={ri}')
        ax.set_title(f'{name} — {mname}', fontsize=11)
        ax.set_xlabel('ε')
        ax.set_ylabel('Accuracy under ZOO')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'ZOO Query-Budget Ablation — {name}', fontsize=12)
    fig.tight_layout()
    out = f'figures/zoo_budget_{ds["key"]}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

# ── Combined summary ───────────────────────────────────────────────────────────
print(f'\n{"="*62}')
print('ZOO Budget Ablation — Summary')
print(f'{"="*62}')
for ds_name in ['Phishing', 'UNSW-NB15']:
    if ds_name not in all_results:
        continue
    ds_res, BUDGETS, n_features = all_results[ds_name]
    print(f'\n{ds_name} (n_features={n_features}):')
    print(f'  {"Budget":>14}', end='')
    for b in BUDGETS:
        pct = int(b / n_features * 100)
        print(f'  {b}q({pct}%)', end='')
    print('  ΔRI(min→max)')
    for mname in ['RF', 'XGB']:
        ris = [ds_res[mname][b][1] for b in BUDGETS]
        print(f'  {mname:>14}', end='')
        for ri in ris:
            print(f'  {ri:>8.3f}', end='')
        print(f'  {ris[-1]-ris[0]:>+8.3f}')
