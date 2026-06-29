"""
ESI validation under Square Attack — NF-ToN-IoT, HIKARI-2021, UNSW-NB15
Verifies RF > XGB ESI ordering holds under Square Attack (not just ZOO).
Order: HIKARI → NF-ToN-IoT → UNSW-NB15 (UNSW last; most data).
Run: python esi_square_validation.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import os
import time

warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import xgboost as xgb

# ── Constants ──────────────────────────────────────────────────────────────────
EPSILONS  = np.linspace(0, 0.3, 10)
EPS_MAX   = EPSILONS[-1]
N_ESI     = 256
SEED      = 42

os.makedirs('figures', exist_ok=True)

# ── Attacks ────────────────────────────────────────────────────────────────────
def _xent(proba, y):
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-10, 1.0))

def zoo_attack(predict_proba, X, y, epsilon, delta=1e-3):
    n, d = X.shape
    base = _xent(predict_proba(X), y)
    grad = np.zeros_like(X)
    for j in range(d):
        Xp = X.copy(); Xp[:, j] += delta
        grad[:, j] = (_xent(predict_proba(Xp), y) - base) / delta
    return np.clip(X + epsilon * np.sign(grad), X - epsilon, X + epsilon)

def square_attack(predict_proba, X, y, epsilon, n_queries=150, p_init=0.5, seed=0):
    rng = np.random.RandomState(seed)
    n, d = X.shape
    Xa = np.clip(X + rng.choice([-epsilon, epsilon], size=X.shape), X - epsilon, X + epsilon)
    best_loss = _xent(predict_proba(Xa), y)
    for q in range(1, n_queries + 1):
        k = max(1, int(max(p_init * (1 - q / n_queries) ** 2, 1 / d) * d))
        Xn = Xa.copy()
        for i in range(n):
            idx = rng.choice(d, size=k, replace=False)
            Xn[i, idx] = X[i, idx] + rng.choice([-epsilon, epsilon], size=k)
        Xn = np.clip(Xn, X - epsilon, X + epsilon)
        new_loss = _xent(predict_proba(Xn), y)
        improved = new_loss > best_loss
        Xa[improved] = Xn[improved]
        best_loss[improved] = new_loss[improved]
    return Xa

# ── ESI ────────────────────────────────────────────────────────────────────────
def _extract_shap_class1(sv):
    if isinstance(sv, list): return sv[1]
    if sv.ndim == 3:         return sv[:, :, 1]
    return sv

def compute_esi(model, X_clean, y_clean, attack_fn, label=''):
    explainer = shap.TreeExplainer(model)
    sv_clean  = _extract_shap_class1(explainer.shap_values(X_clean))
    drifts = []
    for eps in EPSILONS:
        if eps == 0:
            drifts.append(0.0)
        else:
            X_adv = attack_fn(model.predict_proba, X_clean, y_clean, eps)
            sv_adv = _extract_shap_class1(explainer.shap_values(X_adv))
            drifts.append(float(np.abs(sv_adv - sv_clean).mean()))
        print(f'    ε={eps:.3f}  drift={drifts[-1]:.4f}', flush=True)
    drifts = np.array(drifts)
    D_max = drifts[-1] if drifts[-1] > 0 else 1.0
    auc = np.trapezoid(drifts / D_max, EPSILONS) / EPS_MAX
    return drifts, round(float(1.0 - auc), 3)

# ── Dataset loaders ────────────────────────────────────────────────────────────
def load_hikari():
    print('  Loading ALLFLOWMETER_HIKARI2021.csv...')
    df = pd.read_csv('ALLFLOWMETER_HIKARI2021.csv', low_memory=False)
    label_candidates = [c for c in df.columns if 'label' in c.lower()]
    LABEL_COL = label_candidates[0]
    BENIGN_VALUES = {'Background', 'Browsing', 'benign', 'background', 'browsing', '0', 0}
    df['y'] = df[LABEL_COL].apply(lambda x: 0 if str(x).strip() in BENIGN_VALUES else 1)

    DROP_SUBSTRINGS = ['ip', 'mac', 'timestamp', 'flow_id', 'flow id']
    DROP_EXACT      = {'uid', 'originh', 'responh', 'traffic_category'}
    drop_cols = set(label_candidates + ['y']) | DROP_EXACT
    for col in df.columns:
        if any(s in col.lower() for s in DROP_SUBSTRINGS):
            drop_cols.add(col)
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X_raw = df[feat_cols].copy()

    for col in X_raw.select_dtypes(include='object').columns:
        X_raw[col] = LabelEncoder().fit_transform(X_raw[col].astype(str))
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y_raw = df['y'].values

    n_per_class = min(10_000, (y_raw == 1).sum())
    rng = np.random.RandomState(SEED)
    idx = np.concatenate([
        rng.choice(np.where(y_raw == 0)[0], n_per_class, replace=False),
        rng.choice(np.where(y_raw == 1)[0], n_per_class, replace=False),
    ])
    print(f'  {X_raw.shape[1]} features, {2*n_per_class:,} balanced samples')
    return X_raw.iloc[idx].values, y_raw[idx]

def load_toniot():
    print('  Loading NF-ToN-IoT.parquet...')
    df = pd.read_parquet('NF-ToN-IoT.parquet')
    feat_cols = [c for c in df.columns if c not in ('Label', 'Attack')]
    X_raw = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y_raw = df['Label'].values.astype(int)

    n_per_class = min(20_000, (y_raw == 1).sum())
    rng = np.random.RandomState(SEED)
    idx = np.concatenate([
        rng.choice(np.where(y_raw == 0)[0], n_per_class, replace=False),
        rng.choice(np.where(y_raw == 1)[0], n_per_class, replace=False),
    ])
    print(f'  {X_raw.shape[1]} features, {2*n_per_class:,} balanced samples')
    return X_raw.iloc[idx].values, y_raw[idx]

def load_unsw():
    print('  Loading UNSW_NB15_training-set.csv...')
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    df = df.drop(columns=['id', 'attack_cat'], errors='ignore')
    for col in ['proto', 'service', 'state']:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    y_raw = df['label'].values.astype(int)
    X_raw = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    print(f'  {X_raw.shape[1]} features, {len(y_raw):,} samples (no rebalancing; using full training set)')
    return X_raw.values, y_raw

# ── Per-dataset run ────────────────────────────────────────────────────────────
# Model hyperparams match the originating notebooks exactly so ZOO ESI reference
# values are reproducible.
#   Phishing/UNSW (robustness_study_extension.ipynb): RF 200/depth-12, XGB 200/depth-6 + sub=0.8
#   NF-ToN-IoT    (robustness_toniot.ipynb):          RF 100/depth-8,  XGB 100/depth-5 + sub=0.8
#   HIKARI        (robustness_hikari.py):              RF 100/depth-8,  XGB 100/depth-5 (no sub)
DATASETS = [
    ('HIKARI-2021',  load_hikari, 'hikari',
     dict(rf_n=100, rf_d=8, xgb_n=100, xgb_d=5, xgb_sub=None)),
    ('NF-ToN-IoT',   load_toniot, 'toniot',
     dict(rf_n=100, rf_d=8, xgb_n=100, xgb_d=5, xgb_sub=0.8)),
    ('UNSW-NB15',    load_unsw,   'unsw',
     dict(rf_n=200, rf_d=12, xgb_n=200, xgb_d=6, xgb_sub=0.8)),
]

# ZOO ESI reference values from the main robustness study
ZOO_ESI_REF = {
    'HIKARI-2021': {'RF': 0.140, 'XGB': 0.056},
    'NF-ToN-IoT':  {'RF': 0.214, 'XGB': 0.159},
    'UNSW-NB15':   {'RF': 0.167, 'XGB': 0.063},
}

all_results = {}

for ds_name, loader, ds_key, hp in DATASETS:
    print(f'\n{"="*60}')
    print(f'Dataset: {ds_name}')
    print(f'{"="*60}')

    X, y = loader()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    esi_idx = resample(np.arange(len(X_te)), n_samples=N_ESI,
                       stratify=y_te, random_state=SEED)
    X_esi = X_te[esi_idx]
    y_esi = y_te[esi_idx]

    print(f'  Training RF (n={hp["rf_n"]}, depth={hp["rf_d"]})...')
    rf = RandomForestClassifier(n_estimators=hp['rf_n'], max_depth=hp['rf_d'],
                                random_state=SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    print(f'  RF  clean acc: {rf.score(X_te, y_te):.3f}')

    xgb_kwargs = dict(n_estimators=hp['xgb_n'], max_depth=hp['xgb_d'],
                      learning_rate=0.05, random_state=SEED,
                      eval_metric='logloss', verbosity=0)
    if hp['xgb_sub']:
        xgb_kwargs.update(subsample=hp['xgb_sub'], colsample_bytree=hp['xgb_sub'])
    print(f'  Training XGB (n={hp["xgb_n"]}, depth={hp["xgb_d"]}'
          + (f', sub={hp["xgb_sub"]}' if hp['xgb_sub'] else '') + ')...')
    xgb_model = xgb.XGBClassifier(**xgb_kwargs)
    xgb_model.fit(X_tr, y_tr)
    print(f'  XGB clean acc: {xgb_model.score(X_te, y_te):.3f}')

    results = {}
    for model, mname in [(rf, 'RF'), (xgb_model, 'XGB')]:
        for attack_fn, aname in [(zoo_attack, 'ZOO'), (square_attack, 'Square')]:
            print(f'\n── ESI ({aname}) — {mname} ──')
            t0 = time.time()
            drift, esi = compute_esi(model, X_esi, y_esi, attack_fn)
            print(f'  {mname} {aname} ESI = {esi}  ({time.time()-t0:.0f}s)')
            results[(mname, aname)] = (drift, esi)

    all_results[ds_name] = results

    # ── Summary ───────────────────────────────────────────────────────────────
    ref = ZOO_ESI_REF[ds_name]
    print(f'\n── {ds_name} ESI Summary ──────────────────────────────────────')
    print(f'{"Model":<6} {"ZOO (ref)":<12} {"ZOO (now)":<12} {"Square":<10} {"RF>XGB (Sq)"}')
    rf_sq  = results[('RF',  'Square')][1]
    xgb_sq = results[('XGB', 'Square')][1]
    rf_zoo  = results[('RF',  'ZOO')][1]
    xgb_zoo = results[('XGB', 'ZOO')][1]
    print(f'{"RF":<6} {ref["RF"]:<12} {rf_zoo:<12} {rf_sq:<10}')
    print(f'{"XGB":<6} {ref["XGB"]:<12} {xgb_zoo:<12} {xgb_sq:<10}')
    print(f'RF > XGB under Square Attack: {rf_sq > xgb_sq}')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mname, color_zoo, color_sq in [
        (axes[0], 'RF',  '#4878D0', '#EE854A'),
        (axes[1], 'XGB', '#4878D0', '#EE854A'),
    ]:
        drift_zoo, esi_zoo = results[(mname, 'ZOO')]
        drift_sq,  esi_sq  = results[(mname, 'Square')]
        ax.plot(EPSILONS, drift_zoo, 'o--', color=color_zoo,
                label=f'ZOO (ESI={esi_zoo})')
        ax.plot(EPSILONS, drift_sq,  's-',  color=color_sq,
                label=f'Square (ESI={esi_sq})')
        ax.set_title(f'{ds_name} — {mname}', fontsize=11)
        ax.set_xlabel('ε')
        ax.set_ylabel('Mean |ΔSHAP|')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'SHAP Attribution Drift: ZOO vs Square Attack — {ds_name}',
                 fontsize=12)
    fig.tight_layout()
    out = f'figures/esi_zoo_vs_square_{ds_key}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

# ── Final combined summary ─────────────────────────────────────────────────────
print(f'\n{"="*70}')
print('ESI Validation Summary: ZOO vs Square Attack')
print(f'{"="*70}')
print(f'{"Dataset":<16} {"Model":<6} {"ZOO ref":<10} {"ZOO now":<10} {"Square":<10} {"RF>XGB Sq"}')
print('-' * 70)
for ds_name, _, _, _ in DATASETS:
    results = all_results[ds_name]
    ref = ZOO_ESI_REF[ds_name]
    rf_sq  = results[('RF',  'Square')][1]
    xgb_sq = results[('XGB', 'Square')][1]
    rf_zoo  = results[('RF',  'ZOO')][1]
    xgb_zoo = results[('XGB', 'ZOO')][1]
    print(f'{ds_name:<16} {"RF":<6} {ref["RF"]:<10} {rf_zoo:<10} {rf_sq:<10} {rf_sq > xgb_sq}')
    print(f'{"":16} {"XGB":<6} {ref["XGB"]:<10} {xgb_zoo:<10} {xgb_sq:<10}')
    print()
