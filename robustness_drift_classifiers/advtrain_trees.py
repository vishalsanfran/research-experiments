"""
Tree ensemble adversarial training via Square Attack data augmentation.
Mirrors the MLP FGSM adversarial training from the base study.
Datasets: Phishing, UNSW-NB15.
Approach: train base model → generate Square Attack examples at ε_max → retrain
          on augmented set → compare RI before/after.
Run: python advtrain_trees.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import os
import time

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import xgboost as xgb

EPSILONS  = np.linspace(0, 0.3, 10)
EPS_MAX   = EPSILONS[-1]
EPS_AUG   = 0.3        # perturbation strength for training augmentation
AUG_RATIO = 0.25       # fraction of training set to augment (25%)
N_RI      = 300
N_HSJ     = 120
SEED      = 42

os.makedirs('figures', exist_ok=True)

# ── Attacks ────────────────────────────────────────────────────────────────────
def _xent(proba, y):
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-10, 1.0))

def zoo_attack(predict_proba, X, y, epsilon, delta=1e-3):
    base = _xent(predict_proba(X), y)
    grad = np.zeros_like(X)
    for j in range(X.shape[1]):
        Xp = X.copy(); Xp[:, j] += delta
        grad[:, j] = (_xent(predict_proba(Xp), y) - base) / delta
    return np.clip(X + epsilon * np.sign(grad), X - epsilon, X + epsilon)

def square_attack(predict_proba, X, y, epsilon, n_queries=150, p_init=0.5, seed=0):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    Xa = np.clip(X + rng.choice([-epsilon, epsilon], size=X.shape), X - epsilon, X + epsilon)
    best = _xent(predict_proba(Xa), y)
    for q in range(1, n_queries + 1):
        k = max(1, int(max(p_init * (1 - q / n_queries) ** 2, 1 / d) * d))
        Xn = Xa.copy()
        for i in range(n):
            idx = rng.choice(d, size=k, replace=False)
            Xn[i, idx] = X[i, idx] + rng.choice([-epsilon, epsilon], size=k)
        Xn = np.clip(Xn, X - epsilon, X + epsilon)
        new_loss = _xent(predict_proba(Xn), y)
        imp = new_loss > best
        Xa[imp] = Xn[imp]; best[imp] = new_loss[imp]
    return Xa

def hopskipjump(model, X, y, epsilon, n_iter=8, n_grad=20, seed=0):
    rng = np.random.RandomState(seed)
    Xa = np.clip(X + rng.uniform(-epsilon, epsilon, X.shape), X - epsilon, X + epsilon)
    for _ in range(n_iter):
        stuck = model.predict(Xa) == y
        if stuck.any():
            Xa[stuck] = np.clip(
                X[stuck] + rng.uniform(-epsilon, epsilon, X[stuck].shape),
                X[stuck] - epsilon, X[stuck] + epsilon)
        grad = np.zeros_like(X)
        for _ in range(n_grad):
            u = rng.randn(*X.shape)
            u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-10
            Xp = np.clip(Xa + 0.01 * u, X - epsilon, X + epsilon)
            grad += u * (model.predict(Xp) != y)[:, None]
        grad /= (n_grad + 1e-10)
        Xa = np.clip(Xa + epsilon * np.sign(grad), X - epsilon, X + epsilon)
    return Xa

# ── RI ─────────────────────────────────────────────────────────────────────────
def compute_ri_proba(model, X, y, attack_fn):
    accs = []
    for eps in EPSILONS:
        Xadv = X if eps == 0 else attack_fn(model.predict_proba, X, y, eps)
        accs.append(model.score(Xadv, y))
        print(f'    ε={eps:.3f}  acc={accs[-1]:.3f}', flush=True)
    return np.array(accs), round(float(np.trapezoid(accs, EPSILONS) / EPS_MAX), 3)

def compute_ri_model(model, X, y, attack_fn):
    accs = []
    for eps in EPSILONS:
        Xadv = X if eps == 0 else attack_fn(model, X, y, eps)
        accs.append(model.score(Xadv, y))
        print(f'    ε={eps:.3f}  acc={accs[-1]:.3f}', flush=True)
    return np.array(accs), round(float(np.trapezoid(accs, EPSILONS) / EPS_MAX), 3)

# ── Dataset loaders ────────────────────────────────────────────────────────────
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
    X = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    return X.values, y

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASETS = [
    dict(name='Phishing',  key='phishing', loader=load_phishing,
         rf=dict(n_estimators=200, max_depth=12),
         xgb=dict(n_estimators=200, max_depth=6, subsample=0.8, colsample_bytree=0.8)),
    dict(name='UNSW-NB15', key='unsw',     loader=load_unsw,
         rf=dict(n_estimators=200, max_depth=12),
         xgb=dict(n_estimators=200, max_depth=6, subsample=0.8, colsample_bytree=0.8)),
]

# ── Per-dataset run ────────────────────────────────────────────────────────────
all_results = {}

for ds in DATASETS:
    name = ds['name']
    print(f'\n{"="*60}')
    print(f'Dataset: {name}')
    print(f'{"="*60}')

    X_all, y_all = ds['loader']()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    ri_idx  = resample(np.arange(len(X_te)), n_samples=N_RI,
                       stratify=y_te, random_state=SEED)
    hsj_idx = resample(np.arange(len(X_te)), n_samples=N_HSJ,
                       stratify=y_te, random_state=SEED)
    X_ri,  y_ri  = X_te[ri_idx],  y_te[ri_idx]
    X_hsj, y_hsj = X_te[hsj_idx], y_te[hsj_idx]

    rf_kw  = dict(**ds['rf'],  random_state=SEED, n_jobs=-1)
    xgb_kw = dict(**ds['xgb'], learning_rate=0.05, random_state=SEED,
                  eval_metric='logloss', verbosity=0)

    ds_res = {}

    for label, make_model in [
        ('baseline',    lambda: None),
        ('adv_trained', lambda: None),
    ]:
        if label == 'baseline':
            rf_m  = RandomForestClassifier(**rf_kw).fit(X_tr, y_tr)
            xgb_m = xgb.XGBClassifier(**xgb_kw).fit(X_tr, y_tr)
            print(f'\n[Baseline] RF={rf_m.score(X_te, y_te):.3f}  '
                  f'XGB={xgb_m.score(X_te, y_te):.3f}')
        else:
            # Generate adversarial training examples using baseline models
            n_aug = int(AUG_RATIO * len(X_tr))
            aug_idx = resample(np.arange(len(X_tr)), n_samples=n_aug,
                               stratify=y_tr, random_state=SEED)
            X_aug_clean = X_tr[aug_idx]
            y_aug       = y_tr[aug_idx]

            print(f'\nGenerating Square Attack augmentation: '
                  f'{n_aug} samples at ε={EPS_AUG}...')

            t0 = time.time()
            X_aug_rf  = square_attack(rf_m.predict_proba,  X_aug_clean, y_aug, EPS_AUG)
            print(f'  RF  aug done ({time.time()-t0:.0f}s)')
            t0 = time.time()
            X_aug_xgb = square_attack(xgb_m.predict_proba, X_aug_clean, y_aug, EPS_AUG)
            print(f'  XGB aug done ({time.time()-t0:.0f}s)')

            X_tr_rf  = np.vstack([X_tr, X_aug_rf])
            X_tr_xgb = np.vstack([X_tr, X_aug_xgb])
            y_tr_aug = np.concatenate([y_tr, y_aug])

            rf_m  = RandomForestClassifier(**rf_kw).fit(X_tr_rf,  y_tr_aug)
            xgb_m = xgb.XGBClassifier(**xgb_kw).fit(X_tr_xgb, y_tr_aug)
            print(f'[Adv-trained] RF={rf_m.score(X_te, y_te):.3f}  '
                  f'XGB={xgb_m.score(X_te, y_te):.3f}')

        label_res = {}
        for model, mname in [(rf_m, 'RF'), (xgb_m, 'XGB')]:
            print(f'\n  {label} {mname} — ZOO RI:')
            accs_zoo, ri_zoo = compute_ri_proba(model, X_ri, y_ri, zoo_attack)
            print(f'  → RI={ri_zoo}')

            print(f'  {label} {mname} — Square RI:')
            accs_sq, ri_sq = compute_ri_proba(model, X_ri, y_ri, square_attack)
            print(f'  → RI={ri_sq}')

            print(f'  {label} {mname} — HSJ RI:')
            accs_hsj, ri_hsj = compute_ri_model(model, X_hsj, y_hsj, hopskipjump)
            print(f'  → RI={ri_hsj}')

            label_res[mname] = dict(
                clean_acc=round(model.score(X_te, y_te), 3),
                accs_zoo=accs_zoo, ri_zoo=ri_zoo,
                accs_sq=accs_sq,   ri_sq=ri_sq,
                accs_hsj=accs_hsj, ri_hsj=ri_hsj,
            )
        ds_res[label] = label_res

    all_results[name] = ds_res

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n── {name} Before/After ──────────────────────────────────────────')
    print(f'{"Model":<6} {"":12} {"Clean":>7} {"ZOO RI":>8} {"Sq RI":>8} {"HSJ RI":>8}')
    for mname in ['RF', 'XGB']:
        for label in ['baseline', 'adv_trained']:
            r = ds_res[label][mname]
            tag = 'base' if label == 'baseline' else 'adv '
            print(f'{mname:<6} {tag:<12} {r["clean_acc"]:>7.3f} '
                  f'{r["ri_zoo"]:>8.3f} {r["ri_sq"]:>8.3f} {r["ri_hsj"]:>8.3f}')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'baseline': '#4878D0', 'adv_trained': '#EE854A'}
    styles = {'baseline': 'o--', 'adv_trained': 's-'}

    for ax, mname in [(axes[0], 'RF'), (axes[1], 'XGB')]:
        for label in ['baseline', 'adv_trained']:
            r = ds_res[label][mname]
            tag = 'Baseline' if label == 'baseline' else 'Adv-trained'
            ax.plot(EPSILONS, r['accs_sq'], styles[label],
                    color=colors[label],
                    label=f'{tag} (RI={r["ri_sq"]})')
        ax.set_title(f'{name} — {mname}', fontsize=11)
        ax.set_xlabel('ε')
        ax.set_ylabel('Accuracy (Square Attack)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Square Attack Adversarial Training — {name}', fontsize=12)
    fig.tight_layout()
    out = f'figures/advtrain_{ds["key"]}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

# ── Combined figure (both datasets, Square Attack only) ───────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for row, ds_name in enumerate(['Phishing', 'UNSW-NB15']):
    if ds_name not in all_results:
        continue
    ds_res = all_results[ds_name]
    for col, mname in enumerate(['RF', 'XGB']):
        ax = axes[row, col]
        for label in ['baseline', 'adv_trained']:
            r = ds_res[label][mname]
            tag = 'Baseline' if label == 'baseline' else 'Adv-trained'
            ax.plot(EPSILONS, r['accs_sq'],
                    'o--' if label == 'baseline' else 's-',
                    color='#4878D0' if label == 'baseline' else '#EE854A',
                    label=f'{tag} (RI={r["ri_sq"]})')
        ax.set_title(f'{ds_name} — {mname}', fontsize=11)
        ax.set_xlabel('ε')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

fig.suptitle('Square Attack Adversarial Training: Before vs After', fontsize=12)
fig.tight_layout()
fig.savefig('figures/advtrain_trees_combined.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('\nSaved: figures/advtrain_trees_combined.png')

# ── Final summary ──────────────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('Adversarial Training Summary — RI gain (adv_trained − baseline)')
print(f'{"="*70}')
print(f'{"Dataset":<12} {"Model":<6} {"ΔClean":>8} {"ΔZOO RI":>9} '
      f'{"ΔSq RI":>9} {"ΔHSJ RI":>9}')
print('-' * 55)
for ds_name in ['Phishing', 'UNSW-NB15']:
    if ds_name not in all_results:
        continue
    for mname in ['RF', 'XGB']:
        b = all_results[ds_name]['baseline'][mname]
        a = all_results[ds_name]['adv_trained'][mname]
        print(f'{ds_name:<12} {mname:<6} '
              f'{a["clean_acc"]-b["clean_acc"]:>+8.3f} '
              f'{a["ri_zoo"]-b["ri_zoo"]:>+9.3f} '
              f'{a["ri_sq"]-b["ri_sq"]:>+9.3f} '
              f'{a["ri_hsj"]-b["ri_hsj"]:>+9.3f}')
