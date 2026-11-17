"""CSE 158 Task 2 v2 — single-file feature generation, validation, and predictions.

Combines the uploaded t2.py, newfeats.py, tfeats.py, t2feats.py, gapfeats.py,
ppmi.py, base.py, allfeats.py and final.py without requiring those modules.

Run:
    python task2_v2_onefile.py
    python task2_v2_onefile.py --eval       # extra held-out split-6 evaluation
    python task2_v2_onefile.py --rebuild    # ignore and overwrite saved feature caches

Input files in DATA:
    tasks_1_2_train.csv, task_2_test.csv

Output in DATA:
    task_2_predictions_v2.csv

Important: Complete feature generation is computationally expensive (especially
six additional simulations, iALS, and PPMI). Completed feature sets are cached.
"""
import os
import argparse
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path(os.environ.get("A1_DATA", Path(__file__).resolve().parent))
CACHE = DATA / ".task2_v2_onefile_cache_v1"
OUT = DATA / "task_2_predictions_v2.csv"

def quiet(*args, **kwargs):
    pass

def status(*args):
    print(*args, flush=True)

# ==================== T2 FEATURES ====================
t2_ALPHAS = ((0.5, 0.5), (0.5, 1.0), (0.0, 1.0), (1.0, 1.0))

def t2_cooc_scores(hu, hi, nu, ni, qu, qi, chunk=512):
    """sim(j, k) = N_jk / (cnt_j^a cnt_k^b), j = candidate, k = an entity in the query's history.
    Called with (users, items) for item-side and with (items, users) for user-side features."""
    A = sp.csr_matrix((np.ones(len(hu)), (hu, hi)), shape=(nu, ni))
    A.data[:] = 1
    AT = A.T.tocsr()
    cnt = np.asarray(A.sum(0)).ravel()
    ulen = np.asarray(A.sum(1)).ravel()
    out = np.zeros((len(qu), 3 * len(t2_ALPHAS) + 3))
    order = np.argsort(qi, kind='stable')
    ub, starts = np.unique(qi[order], return_index=True)
    starts = np.r_[starts, len(order)]
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
                o = out[q]
                p = 0
                n = NA[e, cols]
                for a, b in t2_ALPHAS:
                    if len(cols):
                        s = n / (max(cnt[j], 1) ** a * cnt[cols] ** b)
                        o[p] = s.sum()
                        o[p + 1] = s.max()
                        o[p + 2] = np.sort(s)[-10:].mean()
                    p += 3
                o[p] = inh
                o[p + 1] = np.log1p(cnt[j])
                o[p + 2] = np.log1p(ulen[u])
    return out

def t2_cooc_names(pre):
    n = []
    for a, b in t2_ALPHAS:
        n += [f'{pre}{a}_{b}_sum', f'{pre}{a}_{b}_max', f'{pre}{a}_{b}_top10']
    return n

def t2_svd_scores(hu, hi, nu, ni, qu, qi, ks=(32, 128), d=0.5):
    A = sp.csr_matrix((np.ones(len(hu)), (hu, hi)), shape=(nu, ni))
    A.data[:] = 1
    cnt = np.asarray(A.sum(0)).ravel()
    U, s, Vt = svds(A @ sp.diags(1 / np.maximum(cnt, 1) ** d), k=max(ks), random_state=0)
    o = np.argsort(-s)
    U, s, Vt = (U[:, o], s[o], Vt[o])
    F = {}
    for k in ks:
        sc = (U[qu, :k] * s[:k] * Vt[:k, qi].T).sum(1)
        F[f'svd{k}'] = sc
        F[f'svd{k}_pn'] = sc / np.maximum(cnt[qi], 1) ** 0.5
    return F

def t2_train_bpr(hu, hi, nu, ni, k, epochs, lr=0.05, reg=0.002, batch=20000, seed=0):
    """BPR with popularity-sampled negatives (item of a random interaction), like the test."""
    rng = np.random.RandomState(seed)
    P = rng.normal(0, 0.1, (nu, k)).astype(np.float32)
    Q = rng.normal(0, 0.1, (ni, k)).astype(np.float32)
    b = np.zeros(ni, np.float32)
    n = len(hu)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            u, i = (hu[idx], hi[idx])
            j = hi[rng.randint(0, n, len(idx))]
            pu, qi, qj = (P[u], Q[i], Q[j])
            g = (1 / (1 + np.exp((pu * (qi - qj)).sum(1) + b[i] - b[j]))).astype(np.float32)[:, None]
            np.add.at(P, u, lr * (g * (qi - qj) - reg * pu))
            np.add.at(Q, i, lr * (g * pu - reg * qi))
            np.add.at(Q, j, lr * (-g * pu - reg * qj))
            np.add.at(b, i, lr * (g[:, 0] - reg * b[i]))
            np.add.at(b, j, lr * (-g[:, 0] - reg * b[j]))
    return (P, Q, b)

def t2_time_feats(hu, hi, ht, nu, ni, qu, qi, QS=(0.05, 0.5, 0.95)):
    rank = pd.Series(ht).rank(method='average').values / len(ht)
    df = pd.DataFrame({'u': hu, 'i': hi, 't': rank})
    uq = df.groupby('u').t.quantile(list(QS)).unstack().reindex(range(nu)).values
    iq = df.groupby('i').t.quantile(list(QS)).unstack().reindex(range(ni)).values
    F = {}
    for k, q in enumerate(QS):
        F[f'u_t{q}'] = uq[qu, k]
        F[f'i_t{q}'] = iq[qi, k]
    F['first_minus_ulast'] = iq[qi, 0] - uq[qu, 2]
    F['ufirst_minus_ilast'] = uq[qu, 0] - iq[qi, 2]
    F['med_diff'] = np.abs(iq[qi, 1] - uq[qu, 1])
    umin = df.groupby('u').t.min().reindex(range(nu)).values
    umax = df.groupby('u').t.max().reindex(range(nu)).values
    o = np.lexsort((rank, hi))
    key = hi[o].astype(np.float64) * 2 + rank[o]
    icnt = np.bincount(hi, minlength=ni)
    for nm, lo, hi_ in (('win90', uq[qu, 0], uq[qu, 2]), ('winall', umin[qu], umax[qu])):
        a = np.searchsorted(key, qi * 2 + lo - 1e-12, 'left')
        b = np.searchsorted(key, qi * 2 + hi_ + 1e-12, 'right')
        F[f'i_in_{nm}'] = (b - a) / np.maximum(icnt[qi], 1)
    o = np.lexsort((rank, hu))
    key = hu[o].astype(np.float64) * 2 + rank[o]
    ucnt = np.bincount(hu, minlength=nu)
    a = np.searchsorted(key, qu * 2 + iq[qi, 0] - 1e-12, 'left')
    b = np.searchsorted(key, qu * 2 + iq[qi, 2] + 1e-12, 'right')
    F['u_in_iwin90'] = (b - a) / np.maximum(ucnt[qu], 1)
    return F

def t2_build(hist, nu, ni, qu, qi):
    """Features of (user, candidate) pairs given a history DataFrame (u, i, timestamp)."""
    t0 = time.time()
    hu, hi = (hist.u.values, hist.i.values)
    F = {}
    a = t2_cooc_scores(hu, hi, nu, ni, qu, qi)
    for j, n in enumerate(t2_cooc_names('ic') + ['in_hist', 'logpop', 'loglen']):
        F[n] = a[:, j]
    b = t2_cooc_scores(hi, hu, ni, nu, qi, qu)
    for j, n in enumerate(t2_cooc_names('uc')):
        F[n] = b[:, j]
    quiet(f'  co-occurrence {time.time() - t0:.0f}s')
    F.update(t2_svd_scores(hu, hi, nu, ni, qu, qi))
    F.update(t2_time_feats(hu, hi, hist.timestamp.values, nu, ni, qu, qi))
    quiet(f'  svd + time {time.time() - t0:.0f}s')
    for k, ep, reg in ((64, 20, 0.002), (128, 30, 0.005)):
        P, Q, bb = t2_train_bpr(hu, hi, nu, ni, k, ep, reg=reg)
        F[f'bpr{k}'] = (P[qu] * Q[qi]).sum(1) + bb[qi]
        quiet(f'  bpr{k} {time.time() - t0:.0f}s')
    return pd.DataFrame(F)

def t2_pair_matrix(F1, F2):
    return np.hstack([F1.values, F2.values, F1.values - F2.values])

def t2_gbm(seed=0):
    return HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=seed)

# ==================== NEWFEATS FEATURES ====================
def newfeats_graph_feats(X, qu, qi, B=500):
    """RP3-style: paths u -> j -> v -> i with item weight n_j^-b and user weight n_v^-c; + user-kNN cosine."""
    nU, nI = X.shape
    ni = np.asarray(X.sum(0)).ravel()
    nu = np.asarray(X.sum(1)).ravel()
    Xc = X.tocsc()
    wsets = {'b0': np.ones(nI), 'b5': np.maximum(ni, 1) ** (-0.5), 'b1': 1.0 / np.maximum(ni, 1)}
    Xw = {k: (X @ sp.diags(w.astype(np.float32))).tocsr() for k, w in wsets.items()}
    gsets = {'c0': np.ones(nU), 'c5': np.maximum(nu, 1) ** (-0.5), 'c1': 1.0 / np.maximum(nu, 1)}
    names = [f'g_{b}_{c}' for b in wsets for c in gsets] + ['uk_cos', 'uk_cos2', 'uk_max', 'uk_top50', 'uk_top50s']
    out = {n: np.zeros(len(qu)) for n in names}
    pos = pd.Series(np.arange(len(qu))).groupby(qu).apply(np.asarray).to_dict()
    uniq = np.array(sorted(pos))
    sq = np.sqrt(np.maximum(nu, 1))
    for s in range(0, len(uniq), B):
        us = uniq[s:s + B]
        Os = {bk: (Xw[bk][us] @ X.T).toarray() for bk in wsets}
        for bk in wsets:
            Os[bk][np.arange(len(us)), us] = 0
        cos = Os['b0'] / (sq[us][:, None] * sq[None, :])
        thr = -np.partition(-cos, 50, axis=1)[:, 49]
        for r, u in enumerate(us):
            for q in pos[u]:
                i = qi[q]
                vs = Xc.indices[Xc.indptr[i]:Xc.indptr[i + 1]]
                vs = vs[vs != u]
                if not len(vs):
                    continue
                for bk in wsets:
                    o = Os[bk][r, vs]
                    for ck, g in gsets.items():
                        out[f'g_{bk}_{ck}'][q] = (o * g[vs]).sum()
                cv = cos[r, vs]
                out['uk_cos'][q] = cv.sum()
                out['uk_cos2'][q] = (cv ** 2).sum()
                out['uk_max'][q] = cv.max()
                m = cv >= thr[r]
                out['uk_top50'][q] = m.sum()
                out['uk_top50s'][q] = cv[m].sum()
        if s % 5000 == 0:
            quiet('  graph', s, len(uniq))
    F = {n: np.log(out[n] + 1e-06) for n in names if n != 'uk_top50'}
    F['uk_top50'] = out['uk_top50']
    F['g_b0_c0_pn'] = F['g_b0_c0'] - np.log(np.maximum(ni[qi], 1))
    return F

def newfeats_ials(X, k=64, alpha=20.0, reg=5.0, iters=12, seed=0):
    rng = np.random.RandomState(seed)
    nU, nI = X.shape
    U = rng.normal(0, 0.01, (nU, k))
    V = rng.normal(0, 0.01, (nI, k))
    Xc = X.tocsr()
    XT = X.T.tocsr()
    I = np.eye(k)

    def solve(M, Y, out):
        G = Y.T @ Y + reg * I
        for a in range(M.shape[0]):
            idx = M.indices[M.indptr[a]:M.indptr[a + 1]]
            if not len(idx):
                out[a] = 0
                continue
            Yi = Y[idx]
            out[a] = np.linalg.solve(G + alpha * Yi.T @ Yi, (1 + alpha) * Yi.sum(0))
    for it in range(iters):
        solve(Xc, V, U)
        solve(XT, U, V)
    return (U, V)

def newfeats_all_feats(X, qu, qi):
    t = time.time()
    F = newfeats_graph_feats(X, qu, qi)
    quiet('graph', time.time() - t)
    for k, a, r in ((64, 20.0, 5.0), (128, 5.0, 2.0)):
        U, V = newfeats_ials(X, k=k, alpha=a, reg=r)
        F[f'ials{k}'] = (U[qu] * V[qi]).sum(1)
        quiet('ials', k, time.time() - t)
    return pd.DataFrame(F)

# ==================== TFEATS FEATURES ====================
def tfeats_seq_matrix(u, i, t, ni, W, decay):
    """item-item matrix: pairs within W positions in a user's time-ordered timeline, weight decay^(gap-1)."""
    o = np.lexsort((t, u))
    u, i = (u[o], i[o])
    rows, cols, vals = ([], [], [])
    for g in range(1, W + 1):
        m = u[g:] == u[:-g]
        a, b = (i[:-g][m], i[g:][m])
        w = np.full(m.sum(), decay ** (g - 1), np.float32)
        rows += [a, b]
        cols += [b, a]
        vals += [w, w]
    M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ni, ni))
    M.sum_duplicates()
    return M

def tfeats_same_ts_matrix(u, i, t, ni):
    df = pd.DataFrame({'u': u, 'i': i, 't': t})
    df['g'] = df.groupby(['u', 't']).ngroup()
    sz = df.groupby('g').i.transform('size').values
    df = df[(sz > 1) & (sz <= 50)]
    G = sp.csr_matrix((np.ones(len(df), np.float32), (df.g.values, df.i.values)))
    w = 1.0 / np.asarray(G.sum(1)).ravel()
    M = (G.T @ sp.diags(w.astype(np.float32)) @ G).tocsr()
    M.setdiag(0)
    M.eliminate_zeros()
    return M

def tfeats__ch(f, A, B, c=20000):
    return np.concatenate([f(A[s:s + c], B[s:s + c]) for s in range(0, A.shape[0], c)])

def tfeats_rowdot(A, B):
    return tfeats__ch(lambda a, b: np.asarray(a.multiply(b).sum(1)).ravel(), A, B)

def tfeats_rowmax(A, B):
    return tfeats__ch(lambda a, b: np.asarray(a.multiply(b).tocsr().max(1).todense()).ravel(), A, B)

def tfeats_tfeats(u, i, t, r, nu, ni, qu, qi):
    t0 = time.time()
    X = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(nu, ni))
    ni_ = np.maximum(np.asarray(X.sum(0)).ravel(), 1)
    nu_ = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
    Xq = X[qu]
    F = {}
    for W, d in ((1, 1.0), (3, 0.7), (6, 0.85)):
        M = tfeats_seq_matrix(u, i, t, ni, W, d)
        R = M[qi]
        del M
        s = tfeats_rowdot(Xq, R)
        F[f'seq{W}'] = np.log1p(s)
        F[f'seq{W}_pn'] = np.log(s + 0.001) - np.log(ni_[qi])
        F[f'seq{W}_un'] = np.log(s + 0.001) - np.log(nu_[qu])
        F[f'seq{W}_max'] = np.log1p(tfeats_rowmax(Xq, R))
        Rn = R @ sp.diags((1 / ni_).astype(np.float32))
        F[f'seq{W}_jn'] = np.log(tfeats_rowdot(Xq, Rn) + 1e-06)
        F[f'seq{W}_jnmax'] = np.log(tfeats_rowmax(Xq, Rn) + 1e-06)
        quiet(' seq', W, time.time() - t0)
    quiet(' same ts', time.time() - t0)
    lo, hi = np.percentile(t, [0.5, 99.5])
    tb = np.clip(((t - lo) / (hi - lo) * 150).astype(int), 0, 149)
    Hu = sp.csr_matrix((np.ones(len(u)), (u, tb)), shape=(nu, 150)).toarray()
    Hu /= Hu.sum(1, keepdims=True) + 1e-09
    Hi = sp.csr_matrix((np.ones(len(u)), (i, tb)), shape=(ni, 150)).toarray()
    g = np.bincount(tb, minlength=150) / len(tb)
    from scipy.ndimage import gaussian_filter1d
    for sig in (0.0, 2.0):
        hu = gaussian_filter1d(Hu, sig, axis=1) if sig else Hu
        hi = Hi + 0.5
        hi = gaussian_filter1d(hi, sig, axis=1) if sig else hi
        hi = hi / hi.sum(1, keepdims=True)
        F[f'tlift{sig}'] = np.log((hu[qu] * hi[qi] / (g + 1e-06)).sum(1) + 1e-06)
    df = pd.DataFrame({'u': u, 'i': i, 'r': r})
    im = df.groupby('i').r.mean().reindex(range(ni)).values
    um = df.groupby('u').r.mean().reindex(range(nu)).values
    iz = (df.r == 0).groupby(df.i).mean().reindex(range(ni)).values
    uz = (df.r == 0).groupby(df.u).mean().reindex(range(nu)).values
    F['i_rmean'] = im[qi]
    F['u_rmean'] = um[qu]
    F['i_zero'] = iz[qi]
    F['u_zero'] = uz[qu]
    F['zero_diff'] = np.abs(iz[qi] - uz[qu])
    quiet(' done', time.time() - t0)
    return pd.DataFrame(F)

# ==================== T2FEATS FEATURES ====================
def t2feats_tw_matrix(u, i, t, ni, D, G=30):
    o = np.lexsort((t, u))
    u, i, t = (u[o], i[o], t[o])
    R, C = ([], [])
    for g in range(1, G + 1):
        m = (u[g:] == u[:-g]) & (np.abs(t[g:] - t[:-g]) < D)
        R += [i[:-g][m], i[g:][m]]
        C += [i[g:][m], i[:-g][m]]
    R = np.concatenate(R)
    C = np.concatenate(C)
    M = sp.csr_matrix((np.ones(len(R), np.float32), (R, C)), shape=(ni, ni))
    M.sum_duplicates()
    return M

def t2feats_feats(u, i, t, nu, ni, qu, qi):
    X = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(nu, ni))
    n = np.maximum(np.asarray(X.sum(0)).ravel(), 1).astype(np.float32)
    Xq = X[qu]
    F = {}
    mats = [(f'tw{D}', lambda D=D: t2feats_tw_matrix(u, i, t, ni, D)) for D in (86400, 7 * 86400, 30 * 86400)]
    mats += [('sq3', lambda: tfeats_seq_matrix(u, i, t, ni, 3, 0.7)), ('sq12', lambda: tfeats_seq_matrix(u, i, t, ni, 12, 0.9))]
    for nm, fn in mats:
        M = fn()
        Mc = (sp.diags(1 / np.sqrt(n)) @ M @ sp.diags(1 / np.sqrt(n))).tocsr()
        cols = {k: [] for k in ('', '_pn', '_cos', '_cosmax', '_cos2')}
        for c in range(0, len(qu), 10000):
            sl = slice(c, c + 10000)
            xq = X[qu[sl]]
            R = M[qi[sl]]
            Rc = Mc[qi[sl]]
            s_ = np.asarray(xq.multiply(R).sum(1)).ravel()
            cols[''].append(np.log1p(s_))
            cols['_pn'].append(np.log(s_ + 0.001) - np.log(n[qi[sl]]))
            P = xq.multiply(Rc).tocsr()
            cols['_cos'].append(np.log(np.asarray(P.sum(1)).ravel() + 1e-06))
            cols['_cosmax'].append(np.log(np.asarray(P.max(1).todense()).ravel() + 1e-06))
            P.data = P.data ** 2
            cols['_cos2'].append(np.log(np.asarray(P.sum(1)).ravel() + 1e-09))
        for k, v in cols.items():
            F[nm + k] = np.concatenate(v)
        del M, Mc
        print(nm, flush=True)
    return pd.DataFrame(F)

# ==================== GAPFEATS FEATURES ====================
def gapfeats_gap(u, i, t, nu, ni, qu, qi):
    o = np.lexsort((t, u))
    us, it = (u[o], i[o])
    st = np.searchsorted(us, np.arange(nu + 1))
    n = np.maximum(np.bincount(i, minlength=ni), 1).astype(np.float32)
    F = {}
    for nm, M in (('g3', tfeats_seq_matrix(u, i, t, ni, 3, 0.7)), ('g12', tfeats_seq_matrix(u, i, t, ni, 12, 0.9)), ('gtw30', t2feats_tw_matrix(u, i, t, ni, 30 * 86400))):
        M = (sp.diags(1 / np.sqrt(n)) @ M @ sp.diags(1 / np.sqrt(n))).tocsr()
        res = np.zeros((len(qu), 5))
        buf = np.zeros(ni, np.float32)
        for q in range(len(qu)):
            a, b = (M.indptr[qi[q]], M.indptr[qi[q] + 1])
            ix = M.indices[a:b]
            buf[ix] = M.data[a:b]
            tl = it[st[qu[q]]:st[qu[q] + 1]]
            s = buf[tl]
            buf[ix] = 0
            if len(s) > 1:
                pr = np.sqrt(s[:-1] * s[1:])
                mn = np.minimum(s[:-1], s[1:])
                res[q] = (pr.sum(), pr.max(), mn.max(), np.sort(pr)[-3:].sum(), (s > 0).mean())
            elif len(s):
                res[q, 4] = s[0] > 0
        for k, c in enumerate(('sum', 'max', 'minmax', 'top3', 'frac')):
            F[f'{nm}_{c}'] = np.log(res[:, k] + 1e-07) if c != 'frac' else res[:, k]
        print(nm, flush=True)
    return pd.DataFrame(F)

# ==================== PPMI FEATURES ====================
def ppmi_emb_feats(u, i, t, nu, ni, qu, qi):
    X = sp.csr_matrix((np.ones(len(u), np.float32), (u, i)), shape=(nu, ni))
    n = np.asarray(X.sum(0)).ravel()
    F = {}
    for nm, C in (('seq', lambda: tfeats_seq_matrix(u, i, t, ni, 12, 0.9)), ('tw30', lambda: t2feats_tw_matrix(u, i, t, ni, 30 * 86400))):
        C = C().tocoo()
        tot = C.data.sum()
        r = np.asarray(C.tocsr().sum(1)).ravel()
        pc = r ** 0.75
        pc = pc / pc.sum()
        pmi = np.log(C.data / tot / (r[C.row] / tot * pc[C.col]))
        keep = pmi > 0
        P = sp.csr_matrix((pmi[keep].astype(np.float32), (C.row[keep], C.col[keep])), shape=(ni, ni))
        U, s, _ = svds(P, k=128, random_state=0)
        E = U * np.sqrt(s)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-09)
        Ue = X @ En
        Ue_n = Ue / (np.linalg.norm(Ue, axis=1, keepdims=True) + 1e-09)
        own = np.asarray(X[qu, qi]).ravel()[:, None] * En[qi]
        uq = Ue[qu] - own
        F[f'pp_{nm}_dot'] = (uq * En[qi]).sum(1)
        F[f'pp_{nm}_cos'] = (uq * En[qi]).sum(1) / (np.linalg.norm(uq, axis=1) + 1e-09)
        mx = np.zeros(len(qu))
        Xr = X.tocsr()
        for q in range(len(qu)):
            h = Xr.indices[Xr.indptr[qu[q]]:Xr.indptr[qu[q] + 1]]
            h = h[h != qi[q]]
            if len(h):
                v = En[h] @ En[qi[q]]
                mx[q] = np.sort(v)[-5:].mean()
        F[f'pp_{nm}_top5'] = mx
        print(nm, flush=True)
    return pd.DataFrame(F)

# ===================== SINGLE-FILE PIPELINE =====================

def pair_features(D):
    """Pair both candidate orientations for swap-augmented classification."""
    if not D.columns.is_unique:
        duplicated = D.columns[D.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate feature names: {duplicated}")
    n = len(D) // 2
    a, b = D.to_numpy()[:n], D.to_numpy()[n:]
    return (np.hstack([a, b, a-b]), np.hstack([b, a, b-a]),
            D.in_hist.values[:n], D.in_hist.values[n:])


def build_all_features(hist, nu, ni, qu, qi):
    """Replicates original allfeats.py feature blocks, in the same order."""
    hu, hi = hist.u.to_numpy(), hist.i.to_numpy()
    ht = hist.timestamp.to_numpy(dtype=np.float64)
    hr = hist.rating.to_numpy()
    X = sp.csr_matrix((np.ones(len(hu), np.float32), (hu, hi)), shape=(nu, ni))
    X.data[:] = 1
    sections = [
        ("base (cooc, SVD, BPR, time)", lambda: t2_build(hist, nu, ni, qu, qi)),
        ("graph and iALS", lambda: newfeats_all_feats(X, qu, qi)),
        ("sequential/time density", lambda: tfeats_tfeats(hu, hi, ht, hr, nu, ni, qu, qi)),
        ("time-window cooc", lambda: t2feats_feats(hu, hi, ht, nu, ni, qu, qi)),
        ("gap fitting", lambda: gapfeats_gap(hu, hi, ht, nu, ni, qu, qi)),
        ("PPMI embeddings", lambda: ppmi_emb_feats(hu, hi, ht, nu, ni, qu, qi)),
    ]
    parts = []
    for name, make in sections:
        begin = time.monotonic()
        parts.append(make().reset_index(drop=True))
        status(f"  {name}: {time.monotonic()-begin:.0f}s")
    result = pd.concat(parts, axis=1).astype(np.float32)
    if not result.columns.is_unique:
        raise ValueError("Duplicate generated features: " + str(result.columns[result.columns.duplicated()].tolist()))
    return result


def build_cached(name, maker, *, rebuild=False):
    """Cache completed simulation features for restart after interruption."""
    path = CACHE / f"{name}.pkl"
    if path.exists() and not rebuild:
        status(f"Loading cached {name}")
        return pd.read_pickle(path)
    status(f"Computing {name}")
    item = maker()
    pd.to_pickle(item, path)
    return item


def simulate_big(tr, nu, ni):
    """Original base.py: 120k positives, popularity-sampled negatives."""
    rng = np.random.RandomState(1)
    n = 120_000
    idx = rng.permutation(len(tr))
    hold, keep = idx[:n], idx[n:]
    hist = tr.iloc[keep]
    pu, pi = tr.u.values[hold], tr.i.values[hold]
    neg = hist.i.values[rng.randint(0, len(keep), n)]
    swap = np.random.RandomState(0).rand(n) < 0.5
    c1 = np.where(swap, neg, pi)
    c2 = np.where(swap, pi, neg)
    D = build_all_features(hist, nu, ni, np.r_[pu,pu], np.r_[c1,c2])
    return dict(D=D, y=(~swap).astype(np.int8))


def simulate_small(tr, nu, ni, sid):
    """Original allfeats.py: 20k holdout, RNG seed 1000 + sid."""
    rng = np.random.RandomState(1000 + sid)
    n = 20_000
    idx = rng.permutation(len(tr))
    hold, keep = idx[:n], np.sort(idx[n:])
    pu, pi = tr.u.values[hold], tr.i.values[hold]
    neg = tr.i.values[keep][rng.randint(0, len(keep), n)]
    swap = rng.rand(n) < 0.5
    c1, c2 = np.where(swap, neg, pi), np.where(swap, pi, neg)
    D = build_all_features(tr.iloc[keep], nu, ni, np.r_[pu,pu], np.r_[c1,c2])
    return dict(D=D, y=(~swap).astype(np.int8))


def build_test(tr, te, users, items, nu, ni):
    qu = users.get_indexer(te.user)
    c1 = items.get_indexer(te.item1)
    c2 = items.get_indexer(te.item2)
    if np.any(qu < 0) or np.any(c1 < 0) or np.any(c2 < 0):
        raise ValueError("Test has unknown user/item IDs; the original pipeline cannot index them safely")
    return dict(D=build_all_features(tr, nu, ni, np.r_[qu,qu], np.r_[c1,c2]))


def fit_classifier(X1, X2, y, seed, lr=0.05, leaves=63):
    model = HistGradientBoostingClassifier(
        max_iter=3000, learning_rate=lr, max_leaf_nodes=leaves,
        min_samples_leaf=50, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=80, random_state=seed,
    )
    return model.fit(np.vstack([X1, X2]), np.r_[y, 1-y])


def predict_symmetric(model, X1, X2):
    return (model.predict_proba(X1)[:,1] + 1-model.predict_proba(X2)[:,1])/2


def history_rule(p, a, b):
    return np.where(a > b, 0., np.where(b > a, 1., p))


def evaluate(big, splits, cols):
    """Replicates final.py eval: split 6 excluded from learning."""
    bx1, bx2, _, _ = pair_features(big['D'][cols]); by = big['y']
    v1, v2, va, vb = pair_features(splits[6]['D'][cols]); vy = splits[6]['y']
    xs = [pair_features(splits[s]['D'][cols]) for s in range(1,6)]
    m1 = np.vstack([x[0] for x in xs]); m2 = np.vstack([x[1] for x in xs])
    my = np.concatenate([splits[s]['y'] for s in range(1,6)])
    P = {
        'big': predict_symmetric(fit_classifier(bx1,bx2,by,0),v1,v2),
        'matched': predict_symmetric(fit_classifier(m1,m2,my,0),v1,v2),
        'both': predict_symmetric(fit_classifier(np.vstack([bx1,m1]),np.vstack([bx2,m2]),np.r_[by,my],0),v1,v2),
        'matched_s1': predict_symmetric(fit_classifier(m1,m2,my,1,lr=.03,leaves=127),v1,v2),
    }
    for key,p in P.items():
        status(f"  {key}: {np.mean((history_rule(p,va,vb) > .5)==vy):.5f}")
    for combo in [('big','matched'),('matched','both','matched_s1'),tuple(P)]:
        p = np.mean([P[k] for k in combo], axis=0)
        status(f"  average {combo}: {np.mean((history_rule(p,va,vb)>.5)==vy):.5f}")


def run(*, rebuild=False, eval_only=False, skip_eval=True):
    train = DATA / 'tasks_1_2_train.csv'
    test = DATA / 'task_2_test.csv'
    if not train.exists() or not test.exists():
        raise FileNotFoundError(f"Expected both {train} and {test}. Set A1_DATA to the data directory.")
    CACHE.mkdir(parents=True, exist_ok=True)
    tr, te = pd.read_csv(train), pd.read_csv(test)
    users = pd.Index(tr.user.unique()); items = pd.Index(tr.item.unique())
    nu, ni = len(users), len(items)
    tr['u'] = users.get_indexer(tr.user)
    tr['i'] = items.get_indexer(tr.item)
    status(f"Task 2: {len(tr):,} interactions; {len(te):,} test pairs")
    big = build_cached('big', lambda: simulate_big(tr,nu,ni), rebuild=rebuild)
    splits = {}
    for sid in range(1,7):
        splits[sid] = build_cached(f'split_{sid}', lambda sid=sid: simulate_small(tr,nu,ni,sid), rebuild=rebuild)
    cols = list(splits[1]['D'].columns)
    for name, data in [('big',big)] + [(str(s),splits[s]) for s in range(1,7)]:
        missing = set(cols)-set(data['D'].columns)
        if missing:
            raise ValueError(f"Missing feature columns in {name}: {missing}")
    if eval_only or not skip_eval:
        status('Held-out split-6 accuracy:')
        evaluate(big,splits,cols)
    if eval_only:
        return
    test_data = build_cached('test',lambda: build_test(tr,te,users,items,nu,ni),rebuild=rebuild)
    missing = set(cols)-set(test_data['D'].columns)
    if missing:
        raise ValueError(f"Missing test columns: {missing}")
    big_x1,big_x2,_,_=pair_features(big['D'][cols]); ys = [big['y']]
    x1s,x2s = [big_x1],[big_x2]
    for sid in range(1,7):
        a,b,_,_=pair_features(splits[sid]['D'][cols])
        x1s.append(a); x2s.append(b); ys.append(splits[sid]['y'])
    X1,X2 = np.vstack(x1s),np.vstack(x2s)
    y = np.concatenate(ys)
    T1,T2,h1,h2 = pair_features(test_data['D'][cols])
    status(f"Training final 3-model symmetric ensemble on {len(y):,} simulated pairs")
    ps=[]
    for seed in range(3):
        model=fit_classifier(X1,X2,y,seed)
        ps.append(predict_symmetric(model,T1,T2))
        status(f"  seed={seed} boosting_iterations={model.n_iter_}")
    p=history_rule(np.mean(ps,axis=0),h1,h2)
    out=te[['user','item1','item2']].copy()
    out['which']=np.where(p>.5,te.item1,te.item2)
    out.to_csv(OUT,index=False)
    status(f"Saved {OUT} | item1 selected {np.mean(p>.5):.3f}")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CSE 158 Task 2 v2 unified pipeline')
    parser.add_argument('--rebuild',action='store_true',help='Regenerate cached feature matrices')
    parser.add_argument('--eval',action='store_true',help='Report split-6 accuracy before final prediction')
    parser.add_argument('--eval-only',action='store_true',help='Report split-6 accuracy without final prediction')
    args=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        run(rebuild=args.rebuild,eval_only=args.eval_only,skip_eval=not args.eval)
