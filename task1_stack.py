
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

DATA = Path('/Users/kmaxx/Desktop/CSE 158/assignment1_2026')
TRAIN = DATA / 'tasks_1_2_train.csv'
TEST = DATA / 'task_1_test.csv'
OUT = 'task_1_predictions.csv'
SEED = 42
N_FOLDS = 5
N_VAL = 20_000

RES_KS = (5, 20, 50, 200)                   # residual-cosine kNN sizes
CO_KS = (5, 10, 20, 50, 100, 200, 100_000)  # co-occurrence kNN sizes (100_000 = all)
SHARP = (2.0, 4.0)                          # co-occurrence weights s ** alpha


def log(msg):
    print(msg, flush=True)


# ============================================================================ data
def load():
    train, test = pd.read_csv(TRAIN), pd.read_csv(TEST)
    users, items = pd.Index(train.user.unique()), pd.Index(train.item.unique())
    for df in (train, test):
        df['u'] = users.get_indexer(df.user)
        df['i'] = items.get_indexer(df.item)
    return train, test, len(users), len(items)


# ============================================================================ biased MF (ALS)
def _batched_ridge(rows, X, y, n_rows, lam):
    """Per entity e: solve (X_e^T X_e + diag(lam)) w_e = X_e^T y_e."""
    d = X.shape[1]
    A, b = np.zeros((n_rows, d, d)), np.zeros((n_rows, d))
    for s in range(0, len(rows), 200_000):
        r, Xc, yc = rows[s:s + 200_000], X[s:s + 200_000], y[s:s + 200_000]
        outer = (Xc[:, :, None] * Xc[:, None, :]).reshape(len(r), -1)
        for j in range(d * d):
            A[:, j // d, j % d] += np.bincount(r, outer[:, j], minlength=n_rows)
        for j in range(d):
            b[:, j] += np.bincount(r, Xc[:, j] * yc, minlength=n_rows)
    A += np.diag(lam)[None]
    return np.linalg.solve(A, b[..., None])[..., 0]


class BiasMF:
    """r ~ mu + b_u + b_i + p_u . q_i fitted by ALS (k=0 gives the bias-only model)."""

    def __init__(self, k=0, lam_pq=30.0, lam_bu=1.0, lam_bi=3.0, iters=10):
        self.k, self.lam_pq, self.lam_bu, self.lam_bi, self.iters = k, lam_pq, lam_bu, lam_bi, iters

    def fit(self, u, i, r, nu, ni):
        k, rng = self.k, np.random.RandomState(SEED)
        self.mu = r.mean()
        y = r - self.mu
        P, Q = rng.normal(0, .05, (nu, k)), rng.normal(0, .05, (ni, k))
        bu, bi = np.zeros(nu), np.zeros(ni)
        ones = np.ones((len(r), 1))
        for _ in range(self.iters):
            W = _batched_ridge(u, np.hstack([Q[i], ones]), y - bi[i], nu, np.r_[np.full(k, self.lam_pq), self.lam_bu])
            P, bu = W[:, :k], W[:, k]
            W = _batched_ridge(i, np.hstack([P[u], ones]), y - bu[u], ni, np.r_[np.full(k, self.lam_pq), self.lam_bi])
            Q, bi = W[:, :k], W[:, k]
        self.P, self.Q, self.bu, self.bi = P, Q, bu, bi
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)


# ============================================================================ neighbour features
def knn_names():
    names = [n for K in RES_KS for n in (f'k{K}', f'k{K}_w')] + ['k_max']
    names += [n for K in CO_KS for n in (f'c{K}_res', f'c{K}_raw', f'c{K}_w')]
    names += [n for a in SHARP for n in (f's{a:g}_res', f's{a:g}_raw')]
    return names + ['c_max', 'c_mean', 'j_max']


def knn_features(U, I, res, raw, nu, ni, qu, qi, side, U_all, I_all, shrink=10.0, chunk=256):
    """Neighbourhood features for query pairs (qu, qi).
    side='item': neighbours are the items the user rated, compared with the query item.
    side='user': neighbours are the users who rated the item, compared with the query user.
    Residual cosine uses rated pairs; co-occurrence cosine / Jaccard use all interactions."""
    R = sp.csr_matrix((res, (U, I)), shape=(nu, ni))
    RAW = sp.csr_matrix((raw, (U, I)), shape=(nu, ni))
    B = sp.csr_matrix((np.ones_like(res), (U, I)), shape=(nu, ni))
    A = sp.csr_matrix((np.ones(len(U_all)), (U_all, I_all)), shape=(nu, ni))
    A.data[:] = 1
    if side == 'user':
        R, RAW, B, A = R.T.tocsr(), RAW.T.tocsr(), B.T.tocsr(), A.T.tocsr()
        anchor, target = qi, qu
    else:
        anchor, target = qu, qi
    RT, BT, AT = R.T.tocsr(), B.T.tocsr(), A.T.tocsr()
    norm = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-9
    cnt = np.asarray(A.sum(0)).ravel()

    out = np.zeros((len(anchor), len(knn_names())))
    order = np.argsort(target, kind='stable')
    targets, starts = np.unique(target[order], return_index=True)
    starts = np.r_[starts, len(order)]
    for c in range(0, len(targets), chunk):
        ents = targets[c:c + chunk]
        dot = (RT[ents] @ R).toarray()
        n_co = (BT[ents] @ B).toarray()
        s_res = dot / (norm[ents][:, None] * norm[None, :]) * (n_co / (n_co + shrink))
        n_all = (AT[ents] @ A).toarray()
        s_co = n_all / np.sqrt(cnt[ents][:, None] * cnt[None, :] + 1e-9)
        jac = n_all / (cnt[ents][:, None] + cnt[None, :] - n_all + 1e-9)
        for e in range(len(ents)):
            for q in order[starts[c + e]:starts[c + e + 1]]:
                lo, hi = R.indptr[anchor[q]], R.indptr[anchor[q] + 1]
                keep = R.indices[lo:hi] != target[q]
                cols = R.indices[lo:hi][keep]
                if len(cols) == 0:
                    continue
                v, rw = R.data[lo:hi][keep], RAW.data[lo:hi][keep]
                o, p = out[q], 0
                s = s_res[e, cols]
                srt = np.argsort(-s)
                s1, v1 = s[srt], v[srt]
                for K in RES_KS:
                    w = np.maximum(s1[:K], 0)
                    den = w.sum()
                    o[p] = (w * v1[:K]).sum() / den if den > 0 else 0.0
                    o[p + 1] = den
                    p += 2
                o[p] = s1[0]
                p += 1
                s2 = s_co[e, cols]
                srt = np.argsort(-s2)
                s2, v2, r2 = s2[srt], v[srt], rw[srt]
                for K in CO_KS:
                    w = s2[:K]
                    den = w.sum()
                    if den > 0:
                        o[p], o[p + 1] = (w * v2[:K]).sum() / den, (w * r2[:K]).sum() / den
                    o[p + 2] = den
                    p += 3
                for alpha in SHARP:
                    w = s2 ** alpha
                    den = w.sum()
                    if den > 0:
                        o[p], o[p + 1] = (w * v2).sum() / den, (w * r2).sum() / den
                    p += 2
                o[p], o[p + 1], o[p + 2] = s2[0], s2.mean(), jac[e, cols].max()
    return out


# ============================================================================ feature table
def entity_stats(idx, r, n, prefix):
    cnt = np.bincount(idx, minlength=n).astype(float)
    mean = np.bincount(idx, r, minlength=n) / np.maximum(cnt, 1)
    var = np.bincount(idx, r * r, minlength=n) / np.maximum(cnt, 1) - mean ** 2
    stats = {f'{prefix}_n': cnt, f'{prefix}_mean': mean, f'{prefix}_std': np.sqrt(np.maximum(var, 0))}
    for v in range(1, 6):
        stats[f'{prefix}_f{v}'] = np.bincount(idx, (r == v).astype(float), minlength=n) / np.maximum(cnt, 1)
    return stats


def build_features(hist, qu, qi, nu, ni):
    """Features for query pairs (qu, qi) from a history of interactions (zeros included)."""
    t0 = time.time()
    rated = hist[hist.rating > 0]
    U, I, R = rated.u.values, rated.i.values, rated.rating.values.astype(float)
    Ua, Ia = hist.u.values, hist.i.values
    F = {}

    bias = BiasMF(k=0).fit(U, I, R, nu, ni)
    mf = BiasMF(k=10).fit(U, I, R, nu, ni)
    F['bias'], F['bu'], F['bi'] = bias.predict(qu, qi), bias.bu[qu], bias.bi[qi]
    F['mf'], F['mf_dot'] = mf.predict(qu, qi), (mf.P[qu] * mf.Q[qi]).sum(1)

    res = R - bias.predict(U, I)
    for side, prefix in (('item', 'i'), ('user', 'u')):
        arr = knn_features(U, I, res, R, nu, ni, qu, qi, side, Ua, Ia)
        for j, name in enumerate(knn_names()):
            F[prefix + name] = arr[:, j]

    for prefix, idx, q, n in (('u', U, qu, nu), ('i', I, qi, ni)):
        for name, v in entity_stats(idx, R, n, prefix).items():
            F[name] = v[q]

    zero = (hist.rating.values == 0).astype(float)
    for prefix, idx, q, n in (('u', Ua, qu, nu), ('i', Ia, qi, ni)):
        c = np.bincount(idx, minlength=n).astype(float)
        F[f'{prefix}_nall'] = c[q]
        F[f'{prefix}_zero'] = (np.bincount(idx, zero, minlength=n) / np.maximum(c, 1))[q]

    ipop = np.bincount(Ia, minlength=ni).astype(float)
    uact = np.bincount(Ua, minlength=nu).astype(float)
    F['u_avgpop'] = (np.bincount(Ua, np.log1p(ipop[Ia]), minlength=nu) / np.maximum(uact, 1))[qu]
    F['i_avgact'] = (np.bincount(Ia, np.log1p(uact[Ua]), minlength=ni) / np.maximum(ipop, 1))[qi]
    F['u_avg_bi'] = (np.bincount(U, bias.bi[I], minlength=nu) / np.maximum(np.bincount(U, minlength=nu), 1))[qu]
    F['i_avg_bu'] = (np.bincount(I, bias.bu[U], minlength=ni) / np.maximum(np.bincount(I, minlength=ni), 1))[qi]
    log(f'  features for {len(qu):,} pairs: {time.time() - t0:.0f}s')
    return pd.DataFrame(F).astype(np.float32)


def out_of_fold_features(data, nu, ni):
    """Features for every rated pair, each built from the other folds' interactions."""
    fold = np.random.RandomState(7).randint(0, N_FOLDS, len(data))
    Xs, ys = [], []
    for f in range(N_FOLDS):
        held = data[(fold == f) & (data.rating.values > 0)]
        Xs.append(build_features(data[fold != f], held.u.values, held.i.values, nu, ni))
        ys.append(held.rating.values)
        log(f'fold {f + 1}/{N_FOLDS} done')
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys)


# ============================================================================ stacker
def fit_predict(X, y, X_query):
    """Average of a regression stacker (clipped) and a 5-class stacker's expected rating."""
    X_query = X_query[X.columns]
    common = dict(learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
    reg = HistGradientBoostingRegressor(max_iter=1500, **common).fit(X, y)
    log('regressor done')
    cls = HistGradientBoostingClassifier(max_iter=400, **common).fit(X, y)
    log('classifier done')
    reg_pred = np.clip(reg.predict(X_query), 1, 5)
    expected = cls.predict_proba(X_query) @ cls.classes_
    return np.clip(0.5 * expected + 0.5 * reg_pred, 1, 5)


# ============================================================================ main
def main(validate=False):
    train, test, nu, ni = load()

    # 20k holdout of rated rows; out-of-fold features for everything else
    rated = np.where(train.rating.values > 0)[0]
    val_idx = np.random.RandomState(SEED).permutation(rated)[:N_VAL]
    mask = np.ones(len(train), bool)
    mask[val_idx] = False
    rest, val = train[mask].reset_index(drop=True), train.iloc[val_idx]
    X, y = out_of_fold_features(rest, nu, ni)
    X_val = build_features(rest, val.u.values, val.i.values, nu, ni)

    if validate:
        pred = fit_predict(X, y, X_val)
        log(f'validation MSE: {np.mean((pred - val.rating.values) ** 2):.4f}')

    # final stacker: out-of-fold rows + holdout rows; test features from the full history
    X_all = pd.concat([X, X_val[X.columns]], ignore_index=True)
    y_all = np.concatenate([y, val.rating.values])
    pred = fit_predict(X_all, y_all, build_features(train, test.u.values, test.i.values, nu, ni))
    out = test[['user', 'item']].copy()
    out['rating'] = pred
    out.to_csv(OUT, index=False)
    log(f'wrote {OUT}')


if __name__ == '__main__':
    main(validate='--validate' in sys.argv)
