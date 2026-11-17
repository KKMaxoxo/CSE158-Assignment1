
import time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path('/Users/kmaxx/Desktop/CSE 158/assignment1_2026')
TRAIN = DATA / 'tasks_3_4_train.csv'
TEST = DATA / 'task_4_test.csv'
OUT = 'task_4_predictions.csv'
SEED = 1
N_SIM, N_VAL = 120_000, 20_000       # simulated pairs; the last N_VAL are a local holdout


def log(*a): print(*a, flush=True)


# ----------------------------------------------------------------------------- co-occurrence
ALPHAS = ((0.5, 0.5), (0.5, 1.0), (0.0, 1.0), (1.0, 1.0))


def cooc_scores(hu, hi, nu, ni, qu, qi, chunk=512):
    """sim(j, k) = N_jk / (cnt_j^a cnt_k^b), j = candidate, k = an entity in the query's history.
    Called with (users, items) for item-side and with (items, users) for user-side features."""
    A = sp.csr_matrix((np.ones(len(hu)), (hu, hi)), shape=(nu, ni)); A.data[:] = 1
    AT = A.T.tocsr()
    cnt = np.asarray(A.sum(0)).ravel(); ulen = np.asarray(A.sum(1)).ravel()
    out = np.zeros((len(qu), 3 * len(ALPHAS) + 3))
    order = np.argsort(qi, kind='stable')
    ub, starts = np.unique(qi[order], return_index=True); starts = np.r_[starts, len(order)]
    for c in range(0, len(ub), chunk):
        ents = ub[c:c + chunk]
        NA = (AT[ents] @ A).toarray()
        for e in range(len(ents)):
            j = ents[e]
            for q in order[starts[c + e]:starts[c + e + 1]]:
                u = qu[q]
                cols = A.indices[A.indptr[u]:A.indptr[u + 1]]
                inh = np.any(cols == j)
                cols = cols[cols != j]
                o = out[q]; p = 0
                n = NA[e, cols]
                for a, b in ALPHAS:
                    if len(cols):
                        s = n / (max(cnt[j], 1) ** a * cnt[cols] ** b)
                        o[p] = s.sum(); o[p + 1] = s.max(); o[p + 2] = np.sort(s)[-10:].mean()
                    p += 3
                o[p] = inh; o[p + 1] = np.log1p(cnt[j]); o[p + 2] = np.log1p(ulen[u])
    return out


def cooc_names(pre):
    n = []
    for a, b in ALPHAS: n += [f'{pre}{a}_{b}_sum', f'{pre}{a}_{b}_max', f'{pre}{a}_{b}_top10']
    return n


# ----------------------------------------------------------------------------- PureSVD
def svd_scores(hu, hi, nu, ni, qu, qi, ks=(32, 128), d=0.5):
    A = sp.csr_matrix((np.ones(len(hu)), (hu, hi)), shape=(nu, ni)); A.data[:] = 1
    cnt = np.asarray(A.sum(0)).ravel()
    U, s, Vt = svds(A @ sp.diags(1 / np.maximum(cnt, 1) ** d), k=max(ks), random_state=0)
    o = np.argsort(-s); U, s, Vt = U[:, o], s[o], Vt[o]
    F = {}
    for k in ks:
        sc = (U[qu, :k] * s[:k] * Vt[:k, qi].T).sum(1)
        F[f'svd{k}'] = sc; F[f'svd{k}_pn'] = sc / np.maximum(cnt[qi], 1) ** 0.5
    return F


# ----------------------------------------------------------------------------- BPR
def train_bpr(hu, hi, nu, ni, k, epochs, lr=0.05, reg=0.002, batch=20000, seed=0):
    """BPR with popularity-sampled negatives (item of a random interaction), like the test."""
    rng = np.random.RandomState(seed)
    P = rng.normal(0, .1, (nu, k)).astype(np.float32)
    Q = rng.normal(0, .1, (ni, k)).astype(np.float32)
    b = np.zeros(ni, np.float32); n = len(hu)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]; u, i = hu[idx], hi[idx]
            j = hi[rng.randint(0, n, len(idx))]
            pu, qi, qj = P[u], Q[i], Q[j]
            g = (1 / (1 + np.exp((pu * (qi - qj)).sum(1) + b[i] - b[j]))).astype(np.float32)[:, None]
            np.add.at(P, u, lr * (g * (qi - qj) - reg * pu))
            np.add.at(Q, i, lr * (g * pu - reg * qi)); np.add.at(Q, j, lr * (-g * pu - reg * qj))
            np.add.at(b, i, lr * (g[:, 0] - reg * b[i])); np.add.at(b, j, lr * (-g[:, 0] - reg * b[j]))
    return P, Q, b


# ----------------------------------------------------------------------------- time
def time_feats(hu, hi, ht, nu, ni, qu, qi, QS=(0.05, 0.5, 0.95)):
    rank = pd.Series(ht).rank(method='average').values / len(ht)   # robust to garbage timestamps
    df = pd.DataFrame({'u': hu, 'i': hi, 't': rank})
    uq = df.groupby('u').t.quantile(list(QS)).unstack().reindex(range(nu)).values
    iq = df.groupby('i').t.quantile(list(QS)).unstack().reindex(range(ni)).values
    F = {}
    for k, q in enumerate(QS): F[f'u_t{q}'] = uq[qu, k]; F[f'i_t{q}'] = iq[qi, k]
    F['first_minus_ulast'] = iq[qi, 0] - uq[qu, 2]
    F['ufirst_minus_ilast'] = uq[qu, 0] - iq[qi, 2]
    F['med_diff'] = np.abs(iq[qi, 1] - uq[qu, 1])
    umin = df.groupby('u').t.min().reindex(range(nu)).values
    umax = df.groupby('u').t.max().reindex(range(nu)).values
    o = np.lexsort((rank, hi)); key = hi[o].astype(np.float64) * 2 + rank[o]
    icnt = np.bincount(hi, minlength=ni)
    for nm, lo, hi_ in (('win90', uq[qu, 0], uq[qu, 2]), ('winall', umin[qu], umax[qu])):
        a = np.searchsorted(key, qi * 2 + lo - 1e-12, 'left')
        b = np.searchsorted(key, qi * 2 + hi_ + 1e-12, 'right')
        F[f'i_in_{nm}'] = (b - a) / np.maximum(icnt[qi], 1)
    o = np.lexsort((rank, hu)); key = hu[o].astype(np.float64) * 2 + rank[o]
    ucnt = np.bincount(hu, minlength=nu)
    a = np.searchsorted(key, qu * 2 + iq[qi, 0] - 1e-12, 'left')
    b = np.searchsorted(key, qu * 2 + iq[qi, 2] + 1e-12, 'right')
    F['u_in_iwin90'] = (b - a) / np.maximum(ucnt[qu], 1)
    return F


# ----------------------------------------------------------------------------- query-time features
DAY = 86400.



def query_time_feats(hu, hi, ht, ni, qi, qt):
    """Where does the query time fall in the candidate item's reading history?"""
    ht = ht.astype(float); qt = qt.astype(float)
    o = np.lexsort((ht, hi)); key_i = hi[o]; key_t = ht[o]
    lo = np.searchsorted(key_i, qi, 'left'); hi_ = np.searchsorted(key_i, qi, 'right')
    cnt = hi_ - lo
    # position of qt inside the item's sorted times: vectorised via composite key
    key = key_i.astype(np.float64) * 1e11 + np.clip(key_t, 0, 9.9e10)
    qk = qi.astype(np.float64) * 1e11 + np.clip(qt, 0, 9.9e10)
    pos = np.searchsorted(key, qk)
    F = {}
    first = np.where(cnt > 0, key_t[np.clip(lo, 0, len(key_t) - 1)], np.nan)
    last = np.where(cnt > 0, key_t[np.clip(hi_ - 1, 0, len(key_t) - 1)], np.nan)
    sl = lambda d: np.sign(d) * np.log1p(np.abs(d) / DAY)
    F['qt_minus_ifirst'] = sl(qt - first); F['qt_minus_ilast'] = sl(qt - last)
    F['qt_before_ifirst'] = (qt < first).astype(float); F['qt_after_ilast'] = (qt > last).astype(float)
    F['i_frac_before_qt'] = (pos - lo) / np.maximum(cnt, 1)
    for nm, w in (('7d', 7), ('30d', 30), ('365d', 365)):
        a = np.searchsorted(key, qk - w * DAY, 'left'); b = np.searchsorted(key, qk + w * DAY, 'right')
        F[f'i_n_{nm}'] = np.log1p(b - a); F[f'i_sh_{nm}'] = (b - a) / np.maximum(cnt, 1)
    # gap from qt to nearest read of the item by anyone
    prev_t = np.where(pos > lo, key_t[np.clip(pos - 1, 0, None)], np.nan)
    next_t = np.where(pos < hi_, key_t[np.clip(pos, 0, len(key_t) - 1)], np.nan)
    F['i_gap_prev'] = np.log1p(np.abs(qt - prev_t) / DAY); F['i_gap_next'] = np.log1p(np.abs(next_t - qt) / DAY)
    return F


def time_weighted_cooc(hu, hi, ht, nu, ni, qu, qi, qt, taus=(30, 365), near=(5, 20), chunk=512):
    """Item co-occurrence similarity of the candidate to the user's history, weighted by how close in
    time each history item was read to the query time."""
    A = sp.csr_matrix((np.ones(len(hu)), (hu, hi)), shape=(nu, ni)); A.data[:] = 1; A.sort_indices()
    # timestamps aligned with A's storage
    T = sp.csr_matrix((ht.astype(float) + 1.0, (hu, hi)), shape=(nu, ni)); T.sort_indices()
    AT = A.T.tocsr(); cnt = np.asarray(A.sum(0)).ravel()
    nf = 2 * len(taus) + len(near)
    out = np.zeros((len(qu), nf))
    order = np.argsort(qi, kind='stable')
    ub, starts = np.unique(qi[order], return_index=True); starts = np.r_[starts, len(order)]
    qt = qt.astype(float)
    for c in range(0, len(ub), chunk):
        ents = ub[c:c + chunk]
        NA = (AT[ents] @ A).toarray()
        for e in range(len(ents)):
            j = ents[e]
            for q in order[starts[c + e]:starts[c + e + 1]]:
                u = qu[q]; lo, hi_ = A.indptr[u], A.indptr[u + 1]
                cols = A.indices[lo:hi_]; tt = T.data[lo:hi_] - 1.0
                k = cols != j; cols, tt = cols[k], tt[k]
                if len(cols) == 0: continue
                s = NA[e, cols] / np.sqrt(max(cnt[j], 1) * cnt[cols])
                d = np.abs(tt - qt[q]) / DAY
                o = out[q]; p = 0
                for tau in taus:
                    w = np.exp(-d / tau)
                    o[p] = (w * s).sum() / (w.sum() + 1e-3); o[p + 1] = (w * s).max(); p += 2
                srt = np.argsort(d)
                for K in near:
                    o[p] = s[srt[:K]].mean(); p += 1
    names = []
    for tau in taus: names += [f'tw{tau}_mean', f'tw{tau}_max']
    names += [f'near{K}_sim' for K in near]
    return out, names


# ----------------------------------------------------------------------------- features
def build(hist, nu, ni, qu, qi, qt):
    """Features of (user, candidate) pairs given a history DataFrame (u, i, timestamp)."""
    t0 = time.time()
    hu, hi = hist.u.values, hist.i.values
    F = {}
    a = cooc_scores(hu, hi, nu, ni, qu, qi)
    for j, n in enumerate(cooc_names('ic') + ['in_hist', 'logpop', 'loglen']): F[n] = a[:, j]
    b = cooc_scores(hi, hu, ni, nu, qi, qu)
    for j, n in enumerate(cooc_names('uc')): F[n] = b[:, j]
    log(f'  co-occurrence {time.time() - t0:.0f}s')
    F.update(svd_scores(hu, hi, nu, ni, qu, qi))
    F.update(time_feats(hu, hi, hist.timestamp.values, nu, ni, qu, qi))
    ht = hist.timestamp.values
    F.update(query_time_feats(hu, hi, ht, ni, qi, qt))
    tw, nm = time_weighted_cooc(hu, hi, ht, nu, ni, qu, qi, qt)
    for j, n in enumerate(nm): F[n] = tw[:, j]
    log(f'  svd + time {time.time() - t0:.0f}s')
    for k, ep, reg in ((64, 20, 0.002), (128, 30, 0.005)):
        P, Q, bb = train_bpr(hu, hi, nu, ni, k, ep, reg=reg)
        F[f'bpr{k}'] = (P[qu] * Q[qi]).sum(1) + bb[qi]
        log(f'  bpr{k} {time.time() - t0:.0f}s')
    return pd.DataFrame(F)


def pair_matrix(F1, F2):
    return np.hstack([F1.values, F2.values, F1.values - F2.values])


def gbm(seed=0):
    return HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05, max_leaf_nodes=63,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=50, random_state=seed)


def main():
    tr = pd.read_csv(TRAIN); te = pd.read_csv(TEST)
    users = pd.Index(tr.user.unique()); items = pd.Index(tr.item.unique())
    nu, ni = len(users), len(items)
    tr['u'] = users.get_indexer(tr.user); tr['i'] = items.get_indexer(tr.item)
    tu = users.get_indexer(te.user); tt = te.timestamp.values; t1 = items.get_indexer(te.item1); t2 = items.get_indexer(te.item2)

    # 1. simulated pairs
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(tr)); hold, keep = idx[:N_SIM], idx[N_SIM:]
    hist = tr.iloc[keep]
    pu, pi, pt = tr.u.values[hold], tr.i.values[hold], tr.timestamp.values[hold]
    ng = hist.i.values[rng.randint(0, len(keep), N_SIM)]
    sw = np.random.RandomState(0).rand(N_SIM) < .5          # random slot for the positive
    c1, c2 = np.where(sw, ng, pi), np.where(sw, pi, ng); y = (~sw).astype(int)

    log('features for simulated pairs')
    F = build(hist, nu, ni, np.r_[pu, pu], np.r_[c1, c2], np.r_[pt, pt])
    X = pair_matrix(F.iloc[:N_SIM], F.iloc[N_SIM:])
    nt = N_SIM - N_VAL
    g = gbm().fit(X[:nt], y[:nt])
    log(f'LOCAL holdout accuracy: {np.mean((g.predict_proba(X[nt:])[:, 1] > .5) == y[nt:]):.4f}')

    # 2. final: GBM on all simulated pairs, test features from all training interactions
    models = [gbm(s).fit(X, y) for s in range(3)]
    log('features for test pairs')
    Ft = build(tr, nu, ni, np.r_[tu, tu], np.r_[t1, t2], np.r_[tt, tt])
    Xt = pair_matrix(Ft.iloc[:len(te)], Ft.iloc[len(te):])
    p1 = np.mean([m.predict_proba(Xt)[:, 1] for m in models], 0)
    # a candidate already in the user's training history is the negative
    ih1, ih2 = Ft.in_hist.values[:len(te)], Ft.in_hist.values[len(te):]
    p1 = np.where(ih1 > ih2, 0., np.where(ih2 > ih1, 1., p1))
    out = te[['user', 'timestamp', 'item1', 'item2']].copy()
    out['which'] = np.where(p1 > .5, te.item1, te.item2)
    out.to_csv(OUT, index=False)
    log(f'wrote {OUT}; item1 chosen {np.mean(p1 > .5):.3f}')


if __name__ == '__main__':
    main()
