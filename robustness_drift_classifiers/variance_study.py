"""
Variance and statistical significance study.
Runs ZOO RI, Square RI, and ESI (ZOO) for RF and XGB across 5 seeds.
Seed controls train/test split, model init, and evaluation subset selection.
Skips HSJ to keep runtime manageable.
Order: NF-ToN-IoT → Phishing → HIKARI → UNSW-NB15 (fastest to slowest).
Run: python variance_study.py
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shap, warnings, os, time, json
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import xgboost as xgb

SEEDS    = [0, 1, 2, 3, 42]
EPSILONS = np.linspace(0, 0.3, 10)
EPS_MAX  = EPSILONS[-1]
N_RI     = 300
N_ESI    = 256

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

# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_ri(model, X, y, attack_fn):
    accs = []
    for eps in EPSILONS:
        Xadv = X if eps == 0 else attack_fn(model.predict_proba, X, y, eps)
        accs.append(model.score(Xadv, y))
    return round(float(np.trapezoid(accs, EPSILONS) / EPS_MAX), 3)

def _sv1(sv):
    if isinstance(sv, list): return sv[1]
    return sv[:, :, 1] if sv.ndim == 3 else sv

def compute_esi(model, X, y, attack_fn):
    exp  = shap.TreeExplainer(model)
    base = _sv1(exp.shap_values(X))
    drifts = []
    for eps in EPSILONS:
        Xadv = X if eps == 0 else attack_fn(model.predict_proba, X, y, eps)
        drifts.append(0.0 if eps == 0 else float(np.abs(_sv1(exp.shap_values(Xadv)) - base).mean()))
    drifts = np.array(drifts)
    D_max  = drifts[-1] if drifts[-1] > 0 else 1.0
    return round(float(1.0 - np.trapezoid(drifts / D_max, EPSILONS) / EPS_MAX), 3)

# ── Dataset loaders (return raw X, y arrays) ──────────────────────────────────
def load_toniot():
    pf = pq.ParquetFile('NF-ToN-IoT.parquet')
    FEAT = [c for c in pf.schema_arrow.names if c not in ('Label', 'Attack')]
    TARGET = 20_000
    benign, attack = [], []
    n_b = n_a = 0
    for batch in pf.iter_batches(batch_size=50_000):
        df_b = batch.to_pandas()
        if n_b < TARGET:
            b = df_b[df_b['Label'] == 0]; take = min(len(b), TARGET - n_b)
            if take: benign.append(b.sample(take, random_state=42)); n_b += take
        if n_a < TARGET:
            a = df_b[df_b['Label'] == 1]; take = min(len(a), TARGET - n_a)
            if take: attack.append(a.sample(take, random_state=42)); n_a += take
        del df_b
        if n_b >= TARGET and n_a >= TARGET: break
    df_s = pd.concat(benign + attack).reset_index(drop=True)
    X = df_s[FEAT].values.astype(np.float32)
    y = df_s['Label'].values.astype(int)
    return X, y

def load_phishing():
    df = pd.read_csv('Phishing_Legitimate_full.csv')
    y  = df['CLASS_LABEL'].values
    X  = df.drop(columns=['id', 'CLASS_LABEL'], errors='ignore').values.astype(np.float32)
    return X, y

def load_hikari():
    df = pd.read_csv('ALLFLOWMETER_HIKARI2021.csv', low_memory=False)
    label_col = next(c for c in df.columns if 'label' in c.lower())
    BENIGN = {'Background', 'Browsing', 'benign', 'background', 'browsing', '0', 0}
    df['y'] = df[label_col].apply(lambda x: 0 if str(x).strip() in BENIGN else 1)
    DROP_SUB   = ['ip', 'mac', 'timestamp', 'flow_id', 'flow id']
    DROP_EXACT = {'uid', 'originh', 'responh', 'traffic_category'}
    drop = set([label_col, 'y']) | DROP_EXACT
    for c in df.columns:
        if any(s in c.lower() for s in DROP_SUB): drop.add(c)
    X_raw = df[[c for c in df.columns if c not in drop]].copy()
    for c in X_raw.select_dtypes('object').columns:
        X_raw[c] = LabelEncoder().fit_transform(X_raw[c].astype(str))
    X_raw = X_raw.apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    y_raw = df['y'].values
    n = min(10_000, (y_raw == 1).sum())
    rng = np.random.RandomState(42)
    idx = np.concatenate([rng.choice(np.where(y_raw==0)[0], n, replace=False),
                          rng.choice(np.where(y_raw==1)[0], n, replace=False)])
    return X_raw.iloc[idx].values, y_raw[idx]

def load_unsw():
    df = pd.read_csv('UNSW_NB15_training-set.csv')
    df = df.drop(columns=['id', 'attack_cat'], errors='ignore')
    for c in ['proto', 'service', 'state']:
        if c in df.columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    y = df['label'].values.astype(int)
    X = df.drop(columns=['label']).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    return X.values, y

# ── Per-dataset config ─────────────────────────────────────────────────────────
# Hyperparams match originating notebooks for each dataset.
DATASETS = [
    dict(name='NF-ToN-IoT',  key='toniot',  loader=load_toniot,
         rf=dict(n_estimators=100, max_depth=8),
         xgb=dict(n_estimators=100, max_depth=5, subsample=0.8, colsample_bytree=0.8)),
    dict(name='Phishing',    key='phishing', loader=load_phishing,
         rf=dict(n_estimators=200, max_depth=12),
         xgb=dict(n_estimators=200, max_depth=6, subsample=0.8, colsample_bytree=0.8)),
    dict(name='HIKARI-2021', key='hikari',  loader=load_hikari,
         rf=dict(n_estimators=100, max_depth=8),
         xgb=dict(n_estimators=100, max_depth=5)),
    dict(name='UNSW-NB15',   key='unsw',    loader=load_unsw,
         rf=dict(n_estimators=200, max_depth=12),
         xgb=dict(n_estimators=200, max_depth=6, subsample=0.8, colsample_bytree=0.8)),
]

# ── Main loop ──────────────────────────────────────────────────────────────────
all_rows = []

for ds in DATASETS:
    print(f'\n{"="*60}')
    print(f'Dataset: {ds["name"]}')
    print(f'{"="*60}')

    print('  Loading data (once)...')
    X_all, y_all = ds['loader']()
    print(f'  Loaded: {X_all.shape}')

    for seed in SEEDS:
        t0 = time.time()
        print(f'\n  Seed {seed}:', end=' ', flush=True)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=0.2, random_state=seed, stratify=y_all)
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

        ri_idx  = resample(np.arange(len(X_te)), n_samples=N_RI,
                           stratify=y_te, random_state=seed)
        esi_idx = resample(np.arange(len(X_te)), n_samples=N_ESI,
                           stratify=y_te, random_state=seed + 1000)
        X_ri,  y_ri  = X_te[ri_idx],  y_te[ri_idx]
        X_esi, y_esi = X_te[esi_idx], y_te[esi_idx]

        rf_kw = dict(**ds['rf'], random_state=seed, n_jobs=-1)
        xgb_kw = dict(**ds['xgb'], learning_rate=0.05, random_state=seed,
                      eval_metric='logloss', verbosity=0)

        rf_m = RandomForestClassifier(**rf_kw).fit(X_tr, y_tr)
        xg_m = xgb.XGBClassifier(**xgb_kw).fit(X_tr, y_tr)
        print(f'RF={rf_m.score(X_te, y_te):.3f} XGB={xg_m.score(X_te, y_te):.3f}', end='  ')

        row = {'dataset': ds['name'], 'seed': seed}

        for model, mname in [(rf_m, 'RF'), (xg_m, 'XGB')]:
            row[f'{mname}_zoo_ri']  = compute_ri(model, X_ri, y_ri, zoo_attack)
            row[f'{mname}_sq_ri']   = compute_ri(model, X_ri, y_ri, square_attack)
            row[f'{mname}_esi']     = compute_esi(model, X_esi, y_esi, zoo_attack)
            print(f'{mname}:zoo={row[f"{mname}_zoo_ri"]}'
                  f' sq={row[f"{mname}_sq_ri"]}'
                  f' esi={row[f"{mname}_esi"]}', end='  ')

        all_rows.append(row)
        print(f'({time.time()-t0:.0f}s)')

# ── Summary ────────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(all_rows)
df_res.to_csv('variance_study_raw.csv', index=False)
print('\nSaved: variance_study_raw.csv')

METRICS = [
    ('RF_zoo_ri',  'RF ZOO RI'),
    ('XGB_zoo_ri', 'XGB ZOO RI'),
    ('RF_sq_ri',   'RF Square RI'),
    ('XGB_sq_ri',  'XGB Square RI'),
    ('RF_esi',     'RF ESI'),
    ('XGB_esi',    'XGB ESI'),
]

print(f'\n{"="*72}')
print('Variance Study — Mean ± Std across 5 seeds')
print(f'{"="*72}')
print(f'{"Dataset":<14}', end='')
for _, label in METRICS:
    print(f'  {label:<14}', end='')
print()
print('-' * 72)

summary_rows = []
for ds in DATASETS:
    sub = df_res[df_res['dataset'] == ds['name']]
    print(f'{ds["name"]:<14}', end='')
    srow = {'dataset': ds['name']}
    for col, label in METRICS:
        vals = sub[col].values
        mean, std = vals.mean(), vals.std()
        srow[col + '_mean'] = round(mean, 3)
        srow[col + '_std']  = round(std,  3)
        print(f'  {mean:.3f}±{std:.3f}    ', end='')
    summary_rows.append(srow)
    print()

pd.DataFrame(summary_rows).to_csv('variance_study_summary.csv', index=False)
print('\nSaved: variance_study_summary.csv')

# ── Gap table: ZOO degeneracy gap and RF>XGB ESI margin ───────────────────────
print(f'\n{"="*60}')
print('Key comparisons (mean ± std)')
print(f'{"="*60}')
print(f'{"Dataset":<14}  {"XGB gap ZOO-Sq":<18}  {"RF-XGB ESI":<18}')
print('-' * 60)
for ds in DATASETS:
    sub = df_res[df_res['dataset'] == ds['name']]
    gap_vals   = sub['XGB_zoo_ri'] - sub['XGB_sq_ri']
    esi_margin = sub['RF_esi']     - sub['XGB_esi']
    print(f'{ds["name"]:<14}  '
          f'{gap_vals.mean():.3f}±{gap_vals.std():.3f}          '
          f'{esi_margin.mean():.3f}±{esi_margin.std():.3f}')
