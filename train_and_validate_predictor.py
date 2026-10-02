#!/usr/bin/env python3
"""
Extension 6: Train and validate recovery predictors on held-out
configurations.

Loads results/predictor_features.json, then for each of four split
types trains a baseline (logistic regression on n and secret_type)
and a candidate (gradient boosted trees on all pre-solve features),
and reports AUC, accuracy, F1, and calibration error.

Output:
  - results/predictor_validation.json
  - Printed summary table
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score, log_loss,
    )
except ImportError:
    print("[ERROR] scikit-learn not installed.")
    print("        Add 'scikit-learn>=1.3' to requirements.txt and reinstall.")
    sys.exit(1)


BASELINE_FEATURES = ['n', 'secret_type_id']
CANDIDATE_FEATURES = [
    'n', 'm', 'q', 'sample_ratio', 'secret_type_id',
    'nominal_sigma', 'clip_bound',
    'norm_b_l1', 'norm_b_l2', 'norm_b_inf', 'norm_b_mean',
    'A_density', 'A_row_norm_mean', 'A_row_norm_std',
    'A_max', 'A_min',
    'b_uniformity_score',
]

RANDOM_SEEDS = list(range(10))


def to_arrays(rows, feature_names, label_key):
    X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
    y = np.array([r[label_key] for r in rows], dtype=int)
    return X, y


def eval_predictor(name, model, scaler, X_test, y_test):
    if scaler is not None:
        X_test = scaler.transform(X_test)
    if len(np.unique(y_test)) < 2:
        # Cannot compute AUC on single-class test set
        return {
            'name': name,
            'auc': None, 'acc': None, 'f1': None, 'log_loss': None,
            'n_test': len(y_test),
            'note': 'single class in test set',
        }
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None
    y_pred = model.predict(X_test)
    return {
        'name': name,
        'auc': float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
        'acc': float(accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'log_loss': float(log_loss(y_test, y_prob)) if y_prob is not None else None,
        'n_test': len(y_test),
        'n_positive': int(y_test.sum()),
    }


def train_and_eval(train_rows, test_rows, label_key):
    results = []

    # Baseline: logistic regression on (n, secret_type_id)
    X_train, y_train = to_arrays(train_rows, BASELINE_FEATURES, label_key)
    X_test, y_test = to_arrays(test_rows, BASELINE_FEATURES, label_key)

    if len(np.unique(y_train)) >= 2:
        scaler_b = StandardScaler().fit(X_train)
        model_b = LogisticRegression(max_iter=1000, random_state=42)
        model_b.fit(scaler_b.transform(X_train), y_train)
        results.append(eval_predictor('baseline', model_b, scaler_b, X_test, y_test))
    else:
        results.append({'name': 'baseline', 'note': 'single class in train set'})

    # Candidate: gradient boosted trees on all pre-solve features
    X_train_c, _ = to_arrays(train_rows, CANDIDATE_FEATURES, label_key)
    X_test_c, _ = to_arrays(test_rows, CANDIDATE_FEATURES, label_key)

    if len(np.unique(y_train)) >= 2:
        model_c = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42,
        )
        model_c.fit(X_train_c, y_train)
        results.append(eval_predictor('candidate', model_c, None, X_test_c, y_test))
    else:
        results.append({'name': 'candidate', 'note': 'single class in train set'})

    return results


def main():
    src = Path('results/predictor_features.json')
    if not src.exists():
        print(f"[ERROR] {src} not found")
        sys.exit(1)

    with open(src) as f:
        rows = json.load(f)

    print(f"Loaded {len(rows)} feature rows")

    all_results = defaultdict(list)

    for label_key in ['label_highs', 'label_cbc']:
        print(f"\n{'=' * 70}")
        print(f"Label: {label_key}")
        print(f"{'=' * 70}")

        # ------------------------------------------------------------
        # Split 1: Leave-one-dimension-out
        # ------------------------------------------------------------
        dims = sorted(set(r['n'] for r in rows))
        for held_out_n in dims:
            train = [r for r in rows if r['n'] != held_out_n]
            test = [r for r in rows if r['n'] == held_out_n]
            if not train or not test:
                continue
            res = train_and_eval(train, test, label_key)
            for r in res:
                r['split'] = 'leave_dim_out'
                r['held_out'] = held_out_n
                all_results[label_key].append(r)
            print(f"  leave_dim_out n={held_out_n}: "
                  f"{[(x['name'], round(x.get('auc') or 0, 3)) for x in res]}")

        # ------------------------------------------------------------
        # Split 2: Leave-one-modulus-out
        # ------------------------------------------------------------
        moduli = sorted(set(r['q'] for r in rows))
        for held_out_q in moduli:
            train = [r for r in rows if r['q'] != held_out_q]
            test = [r for r in rows if r['q'] == held_out_q]
            if not train or not test:
                continue
            res = train_and_eval(train, test, label_key)
            for r in res:
                r['split'] = 'leave_modulus_out'
                r['held_out'] = held_out_q
                all_results[label_key].append(r)
            print(f"  leave_modulus_out q={held_out_q}: "
                  f"{[(x['name'], round(x.get('auc') or 0, 3)) for x in res]}")

        # ------------------------------------------------------------
        # Split 3: Leave-one-secret-out
        # ------------------------------------------------------------
        for held_out_secret in [0, 1, 2]:  # ternary, sparse, uniform
            train = [r for r in rows if r['secret_type_id'] != held_out_secret]
            test = [r for r in rows if r['secret_type_id'] == held_out_secret]
            if not train or not test:
                continue
            res = train_and_eval(train, test, label_key)
            for r in res:
                r['split'] = 'leave_secret_out'
                r['held_out'] = held_out_secret
                all_results[label_key].append(r)
            print(f"  leave_secret_out id={held_out_secret}: "
                  f"{[(x['name'], round(x.get('auc') or 0, 3)) for x in res]}")

        # ------------------------------------------------------------
        # Split 4: Random 20% cell split, 10 repeats
        # ------------------------------------------------------------
        cells = sorted(set((r['q'], r['n'], r['secret_type_id']) for r in rows))
        for seed in RANDOM_SEEDS:
            rng = np.random.default_rng(seed)
            cell_idx = rng.permutation(len(cells))
            n_test = max(1, int(0.2 * len(cells)))
            test_cells = set(cells[i] for i in cell_idx[:n_test])
            train = [r for r in rows
                     if (r['q'], r['n'], r['secret_type_id']) not in test_cells]
            test = [r for r in rows
                    if (r['q'], r['n'], r['secret_type_id']) in test_cells]
            if not train or not test:
                continue
            res = train_and_eval(train, test, label_key)
            for r in res:
                r['split'] = 'random_cell_split'
                r['seed'] = seed
                all_results[label_key].append(r)

        # Aggregate random splits
        for predictor_name in ['baseline', 'candidate']:
            aucs = [r['auc'] for r in all_results[label_key]
                    if r['split'] == 'random_cell_split'
                    and r['name'] == predictor_name
                    and r.get('auc') is not None]
            if aucs:
                print(f"  random_cell_split (10 seeds) {predictor_name}: "
                      f"AUC mean={np.mean(aucs):.3f} std={np.std(aucs):.3f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out = Path('results/predictor_validation.json')
    with open(out, 'w') as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()