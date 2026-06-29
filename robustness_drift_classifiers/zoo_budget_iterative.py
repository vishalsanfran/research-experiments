"""
ZOO iterative-step ablation.
Varies the number of coordinate-descent passes (20→200) over the full feature set.
N_RI=50 for speed; n_steps controls iterative refinement budget.
Run: .venv/bin/python zoo_budget_iterative.py
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

EPSILONS    = np.linspace(0, 0.3, 7)   # 7 points for speed
EPS_MAX     = EPSILONS[-1]
N_RI        = 50                        # reduced for speed
SEED        = 42
STEP_COUNTS = [20, 50, 100, 200]

os.makedirs('figures', exist_ok=True)

def _xent(proba, y):
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-10, 1.0))

def zoo_attack_iter(predict_proba, X, y, epsilon, n_steps, delta=1e-3):
    """Full-feature coordinate descent, n_steps iterative passes."""
    Xa = X.copy()
    step = epsilon / n_steps
    for _ in range(n_steps):
        base = _xent(predict_proba(Xa), y)
        grad = np.zeros_like(Xa)
        for j in range(Xa.shape[1]):
            Xp = Xa.copy(); Xp[:, j] += delta
            grad[:, j] = (_xent(predict_proba(Xp), y) - base) / delta
        Xa = np.clip(Xa + step * np.sign(grad), X - epsilon, X + epsilon)
    return Xa

def compute_ri(model, X, y, n_steps):
    accs = []
    for eps in EPSILONS:
        Xadv = X if eps == 0 else zoo_attack_iter(
            model.predict_proba, X, y, eps, n_steps)
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
    print(f'Dataset: {name}  (N_RI={N_RI})')
    print('='*55)

    X_all, y_all = ds['loader']()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    ri_idx = resample(np.arange(len(X_te)), n_samples=N_RI,
                      stratify=y_te, random_state=SEED)
    X_ri, y_ri = X_te[ri_idx], y_te[ri_idx]

    rf_m  = RandomForestClassifier(**ds['rf'], random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    xgb_m = xgb.XGBClassifier(**ds['xgb'], learning_rate=0.05, random_state=SEED,
                               eval_metric='logloss', verbosity=0).fit(X_tr, y_tr)
    print(f'Trained: RF clean={rf_m.score(X_te, y_te):.3f}  '
          f'XGB clean={xgb_m.score(X_te, y_te):.3f}')

    n_features = X_all.shape[1]
    n_calls_per_step = n_features * len(EPSILONS)
    ds_res = {'RF': {}, 'XGB': {}}
    for n_steps in STEP_COUNTS:
        print(f'\n  n_steps={n_steps:>3} '
              f'({n_steps*n_calls_per_step:,} predict_proba calls × {N_RI} samples):',
              end=' ', flush=True)
        for model, mname in [(rf_m, 'RF'), (xgb_m, 'XGB')]:
            accs, ri = compute_ri(model, X_ri, y_ri, n_steps)
            ds_res[mname][n_steps] = (accs, ri)
            print(f'{mname}={ri}', end='  ', flush=True)
    print()
    all_results[name] = ds_res

    print(f'\n── {name}: ZOO RI vs n_steps ────────────────────────────────')
    print(f'{"Steps":>8}', end='')
    for s in STEP_COUNTS:
        print(f'  {s:>6}', end='')
    print('  Δ(20→200)')
    for mname in ['RF', 'XGB']:
        print(f'{mname:>8}', end='')
        for s in STEP_COUNTS:
            print(f'  {ds_res[mname][s][1]:>6.3f}', end='')
        print(f'  {ds_res[mname][200][1]-ds_res[mname][20][1]:>+6.3f}')

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(STEP_COUNTS)))
    for ax, mname in [(axes[0], 'RF'), (axes[1], 'XGB')]:
        for i, s in enumerate(STEP_COUNTS):
            accs, ri = ds_res[mname][s]
            ax.plot(EPSILONS, accs, 'o-', color=colors[i],
                    label=f'{s} steps (RI={ri})')
        ax.set_title(f'{name} — {mname}', fontsize=11)
        ax.set_xlabel('ε'); ax.set_ylabel('Accuracy under ZOO')
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f'ZOO Iterative-Step Ablation — {name}', fontsize=12)
    fig.tight_layout()
    out = f'figures/zoo_iter_{ds["key"]}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

print(f'\n{"="*60}')
print('Iterative ZOO Budget — Summary')
print(f'{"="*60}')
for ds_name in ['Phishing', 'UNSW-NB15']:
    if ds_name not in all_results:
        continue
    ds_res = all_results[ds_name]
    print(f'\n{ds_name}:')
    for mname in ['RF', 'XGB']:
        r = ds_res[mname]
        print(f'  {mname}: ', end='')
        for s in STEP_COUNTS:
            print(f'{s}→{r[s][1]}  ', end='')
        print(f'Δ={r[200][1]-r[20][1]:+.3f}')
