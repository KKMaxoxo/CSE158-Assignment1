"""
Task 5 - which of (item1, item2) is the user's NEXT interaction?  (CSE 158 Assignment 1)

Test file: one row per user; the positive is the user's next (final) interaction after the whole
training timeline; the negative is popularity-sampled (item of a random interaction).

Training examples
  * task 5 itself: positive = each user's last training interaction, history = everything before it
  * tasks 1-2 / 3-4 training files (other users, full timelines): positive = the user's true FINAL
    interaction - the same kind of event the test asks about (helped ~1pp on held-out final reads)
  Features are always computed inside each file (item ids are hashed differently per file).

Features per (user, candidate)
  * item/user co-occurrence (several normalisations), PureSVD, BPR k=64 / k=128 (popularity negatives)
  * order: transition probabilities from the user's last 1-10 items (gap 1 and gap <= 3), reverse
    transitions, co-occurrence with the 1 / 3 / 10 most recent items
  * time: where the user's last timestamp falls in the candidate's reading history, reads within
    +-7/30/365 days of it; global recency (reads in the last 90/365/730 days of the data)
  * how often the candidate appears among all candidates of the batch (test file / simulated set)
  * in-history flag (a candidate already in the history is the negative)
Model: pairwise HistGradientBoostingClassifier on [f(c1), f(c2), f(c1)-f(c2)], 3 seeds.

Usage: python task5_next.py   (prints local accuracy on held-out task-5 users, writes predictions)
"""
import time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.ensemble import HistGradientBoostingClassifier

DATA = Path('/Users/kmaxx/Desktop/CSE 158/assignment1_2026')
TRAIN = DATA / 'task_5_train.csv'
TEST = DATA / 'task_5_test.csv'
EXTERNAL = [DATA / 'tasks_1_2_train.csv', DATA / 'tasks_3_4_train.csv']
OUT = 'task_5_predictions.csv'
DAY = 86400.


def log(*a): print(*a, flush=True)


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


def gbm(seed=0):
    return HistGradientBoostingClassifier(max_iter=1000, learning_rate=0.05, max_leaf_nodes=63,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=50, random_state=seed)




def last_items(hist, nu, K=10):
    """hist sorted by (u, timestamp). Returns (nu, K) array of the user's last K items (most recent first,
    -1 padded) and the user's last timestamp."""
    u = hist.u.values; i = hist.i.values; t = hist.timestamp.values.astype(float)
    ends = np.searchsorted(u, np.arange(nu), 'right'); starts = np.searchsorted(u, np.arange(nu), 'left')
    L = np.full((nu, K), -1)
    for k in range(K):
        idx = ends - 1 - k
        ok = idx >= starts
        L[ok, k] = i[idx[ok]]
    tl = np.where(ends > starts, t[np.clip(ends - 1, 0, None)], np.nan)
    return L, tl


def transition_feats(hist, ni, L, qu, qi, gaps=(1, 3)):
    """Sequential transitions a -> b counted over consecutive (gap 1) or nearby (<= gap) positions in
    every user's timeline. Features: P(candidate | user's k-th last item) and aggregates."""
    u = hist.u.values; i = hist.i.values
    F = {}
    for g in gaps:
        rows, cols = [], []
        for d in range(1, g + 1):
            same = u[d:] == u[:-d]
            rows.append(i[:-d][same]); cols.append(i[d:][same])
        a = np.concatenate(rows); b = np.concatenate(cols)
        T = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(ni, ni))
        rs = np.asarray(T.sum(1)).ravel()
        cs = np.asarray(T.sum(0)).ravel()
        tot = np.zeros(len(qu)); wtot = np.zeros(len(qu)); mx = np.zeros(len(qu))
        for k in range(L.shape[1]):
            prev = L[qu, k]; ok = prev >= 0
            v = np.zeros(len(qu))
            if ok.any():
                v[ok] = np.asarray(T[prev[ok], qi[ok]]).ravel()
            p = np.where(ok, v / np.maximum(rs[np.maximum(prev, 0)], 1), 0)
            if k < 3:
                F[f'tr{g}_p_last{k+1}'] = p; F[f'tr{g}_n_last{k+1}'] = np.log1p(v)
            tot += p; wtot += p / (k + 1); mx = np.maximum(mx, p)
            # reverse direction (candidate -> previous): series read out of order
        F[f'tr{g}_sum'] = tot; F[f'tr{g}_wsum'] = wtot; F[f'tr{g}_max'] = mx
        F[f'tr{g}_cand_in'] = np.log1p(cs[qi])         # how often the candidate is a "next" item at all
        if g == 1:
            Tt = T.T.tocsr(); rr = np.zeros(len(qu))
            for k in range(3):
                prev = L[qu, k]; ok = prev >= 0
                if ok.any(): rr[ok] += np.asarray(Tt[prev[ok], qi[ok]]).ravel()
            F['tr1_rev3'] = np.log1p(rr)
    return F


def recent_cooc(hist, nu, ni, L, qu, qi, Ks=(1, 3, 10), chunk=64):
    """Co-occurrence cosine of the candidate with the user's K most recent items (order-aware itemCF)."""
    A = sp.csr_matrix((np.ones(len(hist)), (hist.u.values, hist.i.values)), shape=(nu, ni)); A.data[:] = 1
    AT = A.T.tocsr(); cnt = np.asarray(A.sum(0)).ravel()
    out = np.zeros((len(qu), 2 * len(Ks)))
    order = np.argsort(qi, kind='stable')
    ub, starts = np.unique(qi[order], return_index=True); starts = np.r_[starts, len(order)]
    for c in range(0, len(ub), chunk):
        ents = ub[c:c + chunk]
        NA = (AT[ents] @ A).toarray()
        for e in range(len(ents)):
            j = ents[e]
            for q in order[starts[c + e]:starts[c + e + 1]]:
                li = L[qu[q]]; li = li[(li >= 0) & (li != j)]
                if len(li) == 0: continue
                s = NA[e, li] / np.sqrt(max(cnt[j], 1) * cnt[li])
                for p, K in enumerate(Ks):
                    out[q, 2 * p] = s[:K].mean(); out[q, 2 * p + 1] = s[:K].max()
    names = []
    for K in Ks: names += [f'rc{K}_mean', f'rc{K}_max']
    return out, names


def target_set_feats(qi_all, hist_items, ni):
    """How often the item appears among all candidates of the batch (the test file / the simulated set)
    relative to its training popularity: positives are users' latest reads, negatives popularity-drawn."""
    tc = np.bincount(qi_all, minlength=ni).astype(float); hc = np.bincount(hist_items, minlength=ni).astype(float)
    return {'tgt_cnt': np.log1p(tc[qi_all]), 'tgt_ratio': np.log1p(tc[qi_all]) - np.log1p(hc[qi_all])}


def recency_feats(hist, ni, qi, t_end):
    t=hist.timestamp.values.astype(float); i=hist.i.values
    cnt=np.bincount(i,minlength=ni).astype(float); F={}
    for d in (90,365,730):
        c=np.bincount(i[t>t_end-d*DAY],minlength=ni)
        F[f'g_n{d}']=np.log1p(c[qi]); F[f'g_sh{d}']=c[qi]/np.maximum(cnt[qi],1)
    last=pd.Series(t).groupby(i).max().reindex(range(ni)).values
    med=pd.Series(t).groupby(i).median().reindex(range(ni)).values
    F['g_since_last']=np.log1p(np.maximum(t_end-last[qi],0)/DAY); F['g_since_med']=np.log1p(np.maximum(t_end-med[qi],0)/DAY)
    return F


def features(hist, nu, ni, qu1, c1, c2):
    """Pairwise feature blocks (candidate 1, candidate 2) for users qu1 given a history sorted by (u, t)."""
    t0 = time.time()
    qu = np.r_[qu1, qu1]; qi = np.r_[c1, c2]
    hu, hi, ht = hist.u.values, hist.i.values, hist.timestamp.values
    L, tl = last_items(hist, nu); qt = tl[qu]; t_end = np.percentile(ht, 99.9)
    F = {}
    a = cooc_scores(hu, hi, nu, ni, qu, qi, chunk=64)
    for j, nm in enumerate(cooc_names('ic') + ['in_hist', 'logpop', 'loglen']): F[nm] = a[:, j]
    b = cooc_scores(hi, hu, ni, nu, qi, qu, chunk=256)
    for j, nm in enumerate(cooc_names('uc')): F[nm] = b[:, j]
    F.update(svd_scores(hu, hi, nu, ni, qu, qi))
    F.update(query_time_feats(hu, hi, ht, ni, qi, qt))
    F.update(transition_feats(hist, ni, L, qu, qi))
    rc, nm = recent_cooc(hist, nu, ni, L, qu, qi)
    for j, c in enumerate(nm): F[c] = rc[:, j]
    F.update(target_set_feats(qi, hi, ni)); F.update(recency_feats(hist, ni, qi, t_end))
    F['u_to_end'] = np.log1p(np.maximum(t_end - qt, 0) / DAY)
    log(f'  features {time.time() - t0:.0f}s')
    for k, ep, reg in ((64, 20, 0.002), (128, 30, 0.005)):
        P, Q, bb = train_bpr(hu, hi, nu, ni, k, ep, reg=reg)
        F[f'bpr{k}'] = (P[qu] * Q[qi]).sum(1) + bb[qi]
        log(f'  bpr{k} {time.time() - t0:.0f}s')
    X = pd.DataFrame(F); n = len(qu1)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True)


def load(path, extra_items=None):
    tr = pd.read_csv(path)
    users = pd.Index(tr.user.unique())
    it = tr.item if extra_items is None else pd.concat([tr.item, extra_items])
    items = pd.Index(pd.unique(it))
    tr['u'] = users.get_indexer(tr.user); tr['i'] = items.get_indexer(tr.item)
    return tr.sort_values(['u', 'timestamp'], kind='stable').reset_index(drop=True), users, items


def sim_pairs(tr, nu, ni, seed=1):
    """positive = each user's last interaction in the file; negative = item of a random history row."""
    rk = tr.groupby('u').cumcount(ascending=False).values
    pos = tr[rk == 0]; hist = tr[rk > 0].reset_index(drop=True)
    pos = pos[pos.u.isin(hist.u)]
    rng = np.random.RandomState(seed); pu, pi = pos.u.values, pos.i.values
    ng = hist.i.values[rng.randint(0, len(hist), len(pu))]
    sw = rng.rand(len(pu)) < .5; c1, c2 = np.where(sw, ng, pi), np.where(sw, pi, ng)
    A, B = features(hist, nu, ni, pu, c1, c2)
    return pair_matrix(A, B), (~sw).astype(int), pu


def pair_matrix(A, B):
    return np.hstack([A.values, B.values, A.values - B.values])


def main():
    import warnings; warnings.filterwarnings('ignore')
    te = pd.read_csv(TEST)
    tr, users, items = load(TRAIN, pd.concat([te.item1, te.item2]))
    nu, ni = len(users), len(items)
    log('task 5 simulated pairs'); X5, y5, u5 = sim_pairs(tr, nu, ni)
    ext = []
    for p in EXTERNAL:
        log(f'final interactions from {p.name}')
        t2, us2, it2 = load(p); X, y, _ = sim_pairs(t2, len(us2), len(it2)); ext.append((X, y))
    Xe = np.vstack([X for X, _ in ext]); ye = np.concatenate([y for _, y in ext])

    va = np.random.RandomState(0).rand(nu) < .2; v = va[u5]
    g = gbm(0).fit(np.vstack([X5[~v], Xe]), np.r_[y5[~v], ye])
    log(f'LOCAL accuracy (held-out task-5 users): {np.mean((g.predict_proba(X5[v])[:, 1] > .5) == y5[v]):.4f}')

    Xall = np.vstack([X5, Xe]); yall = np.r_[y5, ye]
    models = [gbm(s).fit(Xall, yall) for s in range(3)]
    log('test features (history = all task 5 training interactions)')
    tu = users.get_indexer(te.user); t1 = items.get_indexer(te.item1); t2_ = items.get_indexer(te.item2)
    A, B = features(tr, nu, ni, tu, t1, t2_)
    p1 = np.mean([m.predict_proba(pair_matrix(A, B))[:, 1] for m in models], 0)
    ih1, ih2 = A.in_hist.values, B.in_hist.values
    p1 = np.where(ih1 > ih2, 0., np.where(ih2 > ih1, 1., p1))
    out = te[['user', 'item1', 'item2']].copy()
    out['which'] = np.where(p1 > .5, te.item1, te.item2)
    out.to_csv(OUT, index=False)
    log(f'wrote {OUT}; item1 chosen {np.mean(p1 > .5):.3f}')


if __name__ == '__main__':
    main()
