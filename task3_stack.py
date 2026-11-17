
import sys, time, os
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path(os.environ.get('T3_DATA', '/Users/kmaxx/Desktop/CSE 158/assignment1_2026'))
TRAIN = DATA / 'tasks_3_4_train.csv'
TEST = DATA / 'task_3_test.csv'
SEED = 42
N_STACK = int(os.environ.get('T3_NSTACK', 0)) or None  # stacker rows (out-of-fold); None = all non-zero rows
K_FOLDS = int(os.environ.get('T3_KFOLDS', 6))         # folds over the stacker rows
N_VAL = 20_000                                        # validation holdout (only in --validate mode)
CACHE = os.environ.get('T3_CACHE')                    # optional dir to cache feature blocks


def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)


def mse(a, b): return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


# ----------------------------------------------------------------------------- biased MF (ALS)
def _batched_solve(rows, X, y, n_rows, lam_diag):
    """For every entity e solve (X_e^T X_e + diag(lam)) w_e = X_e^T y_e."""
    d = X.shape[1]
    A = np.zeros((n_rows, d, d)); b = np.zeros((n_rows, d))
    CH = 200_000
    for s in range(0, len(rows), CH):
        r = rows[s:s + CH]; Xc = X[s:s + CH]
        outer = (Xc[:, :, None] * Xc[:, None, :]).reshape(len(r), -1)
        for j in range(d * d):
            A[:, j // d, j % d] += np.bincount(r, outer[:, j], minlength=n_rows)
        for j in range(d):
            b[:, j] += np.bincount(r, Xc[:, j] * y[s:s + CH], minlength=n_rows)
    A += np.diag(lam_diag)[None]
    return np.linalg.solve(A, b[..., None])[..., 0]


class BiasMF:
    """r ~ mu + bu + bi + p_u.q_i, fitted by ALS; lam_* are absolute L2 weights. k=0 -> bias only."""
    def __init__(self, k=5, lam_p=10., lam_q=10., lam_bu=1., lam_bi=3., iters=10, seed=SEED):
        self.k, self.lam_p, self.lam_q, self.lam_bu, self.lam_bi = k, lam_p, lam_q, lam_bu, lam_bi
        self.iters, self.seed = iters, seed

    def fit(self, u, i, r, nu, ni):
        rng = np.random.RandomState(self.seed); k = self.k
        self.mu = r.mean(); y = r - self.mu
        P = rng.normal(0, .05, (nu, k)); Q = rng.normal(0, .05, (ni, k))
        bu = np.zeros(nu); bi = np.zeros(ni)
        for _ in range(self.iters):
            X = np.hstack([Q[i], np.ones((len(i), 1))])
            W = _batched_solve(u, X, y - bi[i], nu, np.r_[np.full(k, self.lam_p), self.lam_bu])
            P, bu = W[:, :k], W[:, k]
            X = np.hstack([P[u], np.ones((len(u), 1))])
            W = _batched_solve(i, X, y - bu[u], ni, np.r_[np.full(k, self.lam_q), self.lam_bi])
            Q, bi = W[:, :k], W[:, k]
        self.P, self.Q, self.bu, self.bi = P, Q, bu, bi
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + (self.P[u] * self.Q[i]).sum(1)


# ----------------------------------------------------------------------------- time-aware SVD++ (Adam)
class _Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1; b1, b2 = .9, .999
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for k, g in grads.items():
            m, v = self.m[k], self.v[k]
            m *= b1; m += (1 - b1) * g
            v *= b2; v += (1 - b2) * g * g
            self.p[k] -= self.lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)


def _dev(t, tmean):
    d = (t - tmean) / 86400.
    return np.sign(d) * np.abs(d) ** 0.4 / 10.


class TimeSVDpp:
    """r ~ mu + bu + a_u*dev_u(t) + b_{u,day} + bi + bi_bin(t) + (p_u + |N(u)|^-1/2 sum_{N(u)} y_j).q_i
    Full-batch Adam on squared error + absolute L2 penalties. N(u) = all known pairs of u."""
    def __init__(self, k=20, epochs=150, lr=0.01, lam_b=3., lam_bin=20., lam_ud=50., lam_a=10.,
                 lam_p=60., lam_q=60., lam_y=60., nbins=30, seed=0):
        self.k, self.epochs, self.lr, self.nbins, self.seed = k, epochs, lr, nbins, seed
        self.lam = dict(bu=lam_b, bi=lam_b, bbin=lam_bin, bud=lam_ud, au=lam_a, P=lam_p, Q=lam_q, Y=lam_y)

    def _bin(self, t):
        x = (np.clip(t, self.tmin, self.tmax) - self.tmin) / (self.tmax - self.tmin + 1)
        return np.minimum((x * self.nbins).astype(int), self.nbins - 1)

    def _daykey(self, u, t):
        return u.astype(np.int64) * 1_000_000 + np.clip((t // 86400).astype(np.int64) - self.day0, 0, 999_999)

    def fit(self, u, i, t, r, nu, ni, imp_u, imp_i):
        rng = np.random.RandomState(self.seed); k = self.k
        self.tmin, self.tmax = np.percentile(t, 0.5), np.percentile(t, 99.5)
        self.day0 = int(self.tmin // 86400) - 10
        cnt = np.bincount(u, minlength=nu).astype(float)
        self.tmean = np.bincount(u, t, minlength=nu) / np.maximum(cnt, 1)
        N = sp.csr_matrix((np.ones(len(imp_u)), (imp_u, imp_i)), shape=(nu, ni)); N.data[:] = 1
        N = (sp.diags(1 / np.sqrt(np.maximum(np.asarray(N.sum(1)).ravel(), 1))) @ N).tocsr()
        self.N = N; NT = N.T.tocsr()
        self.ukeys, inv = np.unique(self._daykey(u, t), return_inverse=True); nd = len(self.ukeys)
        self.mu = r.mean()
        P = {'bu': np.zeros(nu), 'bi': np.zeros(ni), 'bbin': np.zeros((ni, self.nbins)), 'bud': np.zeros(nd),
             'au': np.zeros(nu), 'P': rng.normal(0, .01, (nu, k)), 'Q': rng.normal(0, .01, (ni, k)),
             'Y': np.zeros((ni, k))}
        binx = self._bin(t); dv = _dev(t, self.tmean[u]); flat = i * self.nbins + binx
        E = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(nu, ni))
        order = np.lexsort((i, u)); pos = np.empty(len(u), np.int64); pos[order] = np.arange(len(u))
        opt = _Adam(P, self.lr); L = self.lam
        for _ in range(self.epochs):
            Z = P['P'] + N @ P['Y']
            e = (self.mu + P['bu'][u] + P['bi'][i] + P['bbin'][i, binx] + P['bud'][inv] + P['au'][u] * dv
                 + np.einsum('ij,ij->i', Z[u], P['Q'][i]) - r)
            E.data[pos] = e
            dZ = E @ P['Q']
            g = {'bu': np.bincount(u, e, minlength=nu), 'bi': np.bincount(i, e, minlength=ni),
                 'bbin': np.bincount(flat, e, minlength=ni * self.nbins).reshape(ni, self.nbins),
                 'bud': np.bincount(inv, e, minlength=nd), 'au': np.bincount(u, e * dv, minlength=nu),
                 'Q': E.T @ Z, 'P': dZ, 'Y': NT @ dZ}
            for kk in g: g[kk] = g[kk] + L[kk] * P[kk]
            opt.step(g)
        self.P = P
        return self

    def predict_parts(self, u, i, t):
        P = self.P
        Z = P['P'][u] + self.N[u] @ P['Y']
        key = self._daykey(u, t)
        j = np.minimum(np.searchsorted(self.ukeys, key), len(self.ukeys) - 1)
        hit = self.ukeys[j] == key
        bud = np.where(hit, P['bud'][j], 0.)
        base = self.mu + P['bu'][u] + P['bi'][i] + P['bbin'][i, self._bin(t)] + P['au'][u] * _dev(t, self.tmean[u])
        dot = np.einsum('ij,ij->i', Z, P['Q'][i])
        return base + bud + dot, base, bud, dot


# ----------------------------------------------------------------------------- neighbour features
RES_KS = (5, 20, 50, 200)                      # residual-cosine kNN
CO_KS = (5, 10, 20, 50, 100, 200, 100000)      # co-occurrence kNN (100000 = all)
SHARP = (2., 4., 8.)                           # co-occurrence weights s**alpha over all neighbours
TDEC = ((1., 7.), (1., 60.), (1., 365.), (4., 30.), (4., 365.))   # (alpha, half-life days): s**alpha * decay


def knn_names():
    n = []
    for K in RES_KS: n += [f'k{K}', f'k{K}_w']
    n += ['k_max']
    for K in CO_KS: n += [f'c{K}_res', f'c{K}_raw', f'c{K}_w']
    for a in SHARP: n += [f's{a:g}_res', f's{a:g}_raw']
    for a, h in TDEC: n += [f't{a:g}_{h:g}_res', f't{a:g}_{h:g}_raw', f't{a:g}_{h:g}_w']
    n += ['c_max', 'c_mean', 'j_max']
    return n


def knn_feats(U, I, T, res, raw, nu, ni, qu, qi, qt, side, Uall, Iall, shrink=10., chunk=256):
    """side='item': neighbours = items the user rated (sim to query item).
       side='user': neighbours = users who rated the item (sim to query user).
       Residual cosine uses rated rows only; co-occurrence cosine / Jaccard use all known pairs.
       Time-decayed variants weight each neighbour by 0.5**(|t_neighbour - t_query| / half-life)."""
    R = sp.csr_matrix((res, (U, I)), shape=(nu, ni))
    RAW = sp.csr_matrix((raw, (U, I)), shape=(nu, ni))
    TM = sp.csr_matrix((T + 1.0, (U, I)), shape=(nu, ni))       # +1 so no stored value is 0
    B = sp.csr_matrix((np.ones_like(res), (U, I)), shape=(nu, ni))
    A = sp.csr_matrix((np.ones(len(Uall)), (Uall, Iall)), shape=(nu, ni)); A.data[:] = 1
    if side == 'user':
        R, RAW, TM, B, A = R.T.tocsr(), RAW.T.tocsr(), TM.T.tocsr(), B.T.tocsr(), A.T.tocsr()
        qa, qb = qi, qu
    else:
        qa, qb = qu, qi
    for M in (R, RAW, TM): M.sort_indices()
    RT, BT, AT = R.T.tocsr(), B.T.tocsr(), A.T.tocsr()
    norm = np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel()) + 1e-9
    cnt = np.asarray(A.sum(0)).ravel()
    names = knn_names()
    out = np.zeros((len(qa), len(names)), np.float32)
    order = np.argsort(qb, kind='stable')
    ub, starts = np.unique(qb[order], return_index=True)
    starts = np.r_[starts, len(order)]
    hl = [h * 86400. for _, h in TDEC]
    for c in range(0, len(ub), chunk):
        ents = ub[c:c + chunk]
        D = (RT[ents] @ R).toarray()
        N = (BT[ents] @ B).toarray()
        S = D / (norm[ents][:, None] * norm[None, :]) * (N / (N + shrink))
        NA = (AT[ents] @ A).toarray()
        S2 = NA / np.sqrt(cnt[ents][:, None] * cnt[None, :] + 1e-9)
        J = NA / (cnt[ents][:, None] + cnt[None, :] - NA + 1e-9)
        for e in range(len(ents)):
            for q in order[starts[c + e]:starts[c + e + 1]]:
                a = qa[q]
                lo, hi = R.indptr[a], R.indptr[a + 1]
                cols = R.indices[lo:hi]
                keep = cols != qb[q]
                cols = cols[keep]
                if len(cols) == 0: continue
                v = R.data[lo:hi][keep]; rw = RAW.data[lo:hi][keep]
                dt = np.abs(TM.data[lo:hi][keep] - 1.0 - qt[q])
                o = out[q]; p = 0
                s = S[e, cols]; srt = np.argsort(-s); s1, v1 = s[srt], v[srt]
                for K in RES_KS:
                    ss = np.maximum(s1[:K], 0); den = ss.sum()
                    o[p] = (ss * v1[:K]).sum() / den if den > 0 else 0.; o[p + 1] = den; p += 2
                o[p] = s1[0]; p += 1
                s2a = S2[e, cols]; srt = np.argsort(-s2a); s2, v2, r2 = s2a[srt], v[srt], rw[srt]
                for K in CO_KS:
                    w = s2[:K]; den = w.sum()
                    if den > 0: o[p] = (w * v2[:K]).sum() / den; o[p + 1] = (w * r2[:K]).sum() / den
                    o[p + 2] = den; p += 3
                for al in SHARP:
                    w = s2 ** al; den = w.sum()
                    if den > 0: o[p] = (w * v2).sum() / den; o[p + 1] = (w * r2).sum() / den
                    p += 2
                for (al, _), h in zip(TDEC, hl):
                    w = s2a ** al * np.exp2(-dt / h); den = w.sum()
                    if den > 0: o[p] = (w * v).sum() / den; o[p + 1] = (w * rw).sum() / den
                    o[p + 2] = den; p += 3
                o[p] = s2[0]; o[p + 1] = s2.mean(); o[p + 2] = J[e, cols].max()
    return out, names


# ----------------------------------------------------------------------------- static feature table
def ent_stats(idx, r, n, prefix):
    cnt = np.bincount(idx, minlength=n).astype(float)
    s = np.bincount(idx, r, minlength=n); s2 = np.bincount(idx, r * r, minlength=n)
    mean = s / np.maximum(cnt, 1); std = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0))
    cols = {f'{prefix}_n': cnt, f'{prefix}_mean': mean, f'{prefix}_std': std}
    for v in range(1, 6):
        cols[f'{prefix}_f{v}'] = np.bincount(idx, (r == v).astype(float), minlength=n) / np.maximum(cnt, 1)
    return cols


def build_static(Ball, AP, qu, qi, qt, nu, ni):
    """Ball: history incl. zeros (u, i, timestamp, rating). AP: all known pairs (u, i, timestamp)."""
    t0 = time.time()
    B = Ball[Ball.rating > 0]
    U, I, R, T = B.u.values, B.i.values, B.rating.values.astype(float), B.timestamp.values.astype(float)
    F = {}
    bias = BiasMF(k=0, lam_bu=1, lam_bi=3).fit(U, I, R, nu, ni)
    F['bias'] = bias.predict(qu, qi); F['bu'] = bias.bu[qu]; F['bi'] = bias.bi[qi]
    mf = BiasMF(k=10, lam_p=30, lam_q=30, lam_bu=1, lam_bi=3).fit(U, I, R, nu, ni)
    F['mf'] = mf.predict(qu, qi); F['mf_dot'] = (mf.P[qu] * mf.Q[qi]).sum(1)
    log(f'  ALS MF done {time.time() - t0:.0f}s')
    ts = TimeSVDpp().fit(U, I, T, R, nu, ni, AP.u.values, AP.i.values)
    F['tsvd'], F['tsvd_base'], F['tsvd_bud'], F['tsvd_dot'] = ts.predict_parts(qu, qi, qt)
    log(f'  timeSVD++ done {time.time() - t0:.0f}s')
    res = R - bias.predict(U, I)
    for side, pre in (('item', 'i'), ('user', 'u')):
        arr, names = knn_feats(U, I, T, res, R, nu, ni, qu, qi, qt, side, AP.u.values, AP.i.values)
        for j, nm in enumerate(names): F[pre + nm] = arr[:, j]
        log(f'  {side} kNN done {time.time() - t0:.0f}s')
    for name, idx, q, n in [('u', U, qu, nu), ('i', I, qi, ni)]:
        for k, v in ent_stats(idx, R, n, name).items(): F[k] = v[q]
    Ua, Ia = Ball.u.values, Ball.i.values
    z = (Ball.rating.values == 0).astype(float)
    for name, idx, q, n in [('u', Ua, qu, nu), ('i', Ia, qi, ni)]:
        c = np.bincount(idx, minlength=n).astype(float)
        F[f'{name}_nall'] = c[q]
        F[f'{name}_zero'] = (np.bincount(idx, z, minlength=n) / np.maximum(c, 1))[q]
    ipop = np.bincount(Ia, minlength=ni).astype(float); uact = np.bincount(Ua, minlength=nu).astype(float)
    F['u_avgpop'] = (np.bincount(Ua, np.log1p(ipop[Ia]), minlength=nu) / np.maximum(uact, 1))[qu]
    F['i_avgact'] = (np.bincount(Ia, np.log1p(uact[Ua]), minlength=ni) / np.maximum(ipop, 1))[qi]
    F['u_avg_bi'] = (np.bincount(U, bias.bi[I], minlength=nu) / np.maximum(np.bincount(U, minlength=nu), 1))[qu]
    F['i_avg_bu'] = (np.bincount(I, bias.bu[U], minlength=ni) / np.maximum(np.bincount(I, minlength=ni), 1))[qi]
    return pd.DataFrame(F).astype(np.float32)


# ----------------------------------------------------------------------------- timeline features
def _positions(sid, st, qid, qt):
    lo = np.searchsorted(sid, qid, 'left'); hi = np.searchsorted(sid, qid, 'right')
    # position of qt inside each entity's sorted time slice, vectorised via a combined sort key
    pos = np.empty(len(qid), np.int64)
    for q in range(len(qid)):
        pos[q] = lo[q] + np.searchsorted(st[lo[q]:hi[q]], qt[q])
    return lo, hi, pos


def _nearest_feats(sid, st, sr, qid, qt, Ks=(1, 3, 10, 30), prefix='u'):
    """Stats of the ratings of the interactions closest in time (by rank), same-day stats and gaps."""
    F = {}
    key_lo, key_hi, pos = _positions(sid, st, qid, qt)
    maxK = max(Ks)
    offs = np.arange(-maxK, maxK)
    idx = pos[:, None] + offs[None, :]
    valid = (idx >= key_lo[:, None]) & (idx < key_hi[:, None])
    idxc = np.clip(idx, 0, len(st) - 1)
    T = np.where(valid, st[idxc], np.nan); R = np.where(valid, sr[idxc].astype(float), np.nan)
    dt = np.abs(T - qt[:, None])
    dt_sort = np.where(np.isnan(dt), np.inf, dt)
    order = np.argsort(dt_sort, 1)
    Rs = np.take_along_axis(R, order, 1); Ds = np.take_along_axis(dt_sort, order, 1)
    Rnz = np.where(Rs > 0, Rs, np.nan)
    with np.errstate(all='ignore'):
        for K in Ks:
            F[f'{prefix}_near{K}_mean'] = np.nanmean(Rnz[:, :K], 1)
            F[f'{prefix}_near{K}_zero'] = np.nanmean(np.where(np.isnan(Rs[:, :K]), np.nan, (Rs[:, :K] == 0)), 1)
        F[f'{prefix}_gap1'] = np.log1p(Ds[:, 0])
        F[f'{prefix}_gap3'] = np.log1p(Ds[:, min(2, Ds.shape[1] - 1)])
        same = Ds == 0
        F[f'{prefix}_same_n'] = same.sum(1)
        F[f'{prefix}_same_mean'] = np.nanmean(np.where(same, Rnz, np.nan), 1)
        within = Ds < 86400
        F[f'{prefix}_day_n'] = within.sum(1)
        F[f'{prefix}_day_mean'] = np.nanmean(np.where(within, Rnz, np.nan), 1)
        Rraw = np.where(R > 0, R, np.nan)   # unsorted window: directional
        F[f'{prefix}_prev5_mean'] = np.nanmean(Rraw[:, maxK - 5:maxK], 1)
        F[f'{prefix}_next5_mean'] = np.nanmean(Rraw[:, maxK:maxK + 5], 1)
        F[f'{prefix}_prev1'] = np.nanmean(Rraw[:, maxK - 1:maxK], 1)
        F[f'{prefix}_next1'] = np.nanmean(Rraw[:, maxK:maxK + 1], 1)
    cnt = key_hi - key_lo
    F[f'{prefix}_relpos'] = (pos - key_lo) / np.maximum(cnt, 1)
    return F


def time_feats(hist, qu, qi, qt):
    """hist: DataFrame u, i, timestamp, rating (zeros allowed)."""
    F = {}
    t = hist.timestamp.values.astype(float); r = hist.rating.values
    for name, ent, q in (('u', hist.u.values, qu), ('i', hist.i.values, qi)):
        o = np.lexsort((t, ent))
        F.update(_nearest_feats(ent[o], t[o], r[o], q, qt, prefix=name))
    df = pd.DataFrame({'u': hist.u.values, 'i': hist.i.values, 't': t})
    ifirst = df.groupby('i').t.quantile(0.02); ufirst = df.groupby('u').t.min(); ulast = df.groupby('u').t.max()
    F['item_age'] = np.log1p(np.maximum(qt - ifirst.reindex(qi).values, 0))
    F['item_age_neg'] = (qt < ifirst.reindex(qi).values).astype(float)
    F['user_tenure'] = np.log1p(np.maximum(qt - ufirst.reindex(qu).values, 0))
    F['user_left'] = np.log1p(np.maximum(ulast.reindex(qu).values - qt, 0))
    yr = pd.to_datetime(np.clip(qt, 0, 2e9), unit='s')
    F['year'] = yr.year.values + yr.dayofyear.values / 366
    return F


def resid_time_feats(hist, res_all, qu, qi, qt, Ks=(3, 10, 30), halflives=(7, 30, 180, 720), maxK=50):
    """res_all: bias residual per hist row (nan for unrated). Nearest-in-time and time-decayed means."""
    F = {}
    t = hist.timestamp.values.astype(float)
    for name, ent, q in (('u', hist.u.values, qu), ('i', hist.i.values, qi)):
        o = np.lexsort((t, ent)); sid, st, sv = ent[o], t[o], res_all[o]
        lo, hi, pos = _positions(sid, st, q, qt)
        offs = np.arange(-maxK, maxK)
        idx = pos[:, None] + offs[None, :]
        valid = (idx >= lo[:, None]) & (idx < hi[:, None]); idxc = np.clip(idx, 0, len(st) - 1)
        V = np.where(valid, sv[idxc], np.nan); D = np.where(valid, np.abs(st[idxc] - qt[:, None]), np.inf)
        D = np.where(np.isnan(V), np.inf, D)
        order = np.argsort(D, 1); Vs = np.take_along_axis(V, order, 1); Ds = np.take_along_axis(D, order, 1)
        with np.errstate(all='ignore'):
            for K in Ks:
                F[f'{name}_rnear{K}'] = np.nanmean(Vs[:, :K], 1)
            for h in halflives:
                w = np.where(np.isfinite(Ds), 0.5 ** (Ds / (h * 86400)), 0)
                F[f'{name}_rdecay{h}'] = np.nansum(w * np.nan_to_num(Vs), 1) / (w.sum(1) + 1)
                F[f'{name}_rdecay{h}_w'] = w.sum(1)
    return F


def activity_feats(AP, qu, qi, qt):
    """Counts of all known pairs (train + test, rated or not) near the query time."""
    F = {}
    t = AP.timestamp.values.astype(float)
    for name, ent, q in (('u', AP.u.values, qu), ('i', AP.i.values, qi)):
        o = np.lexsort((t, ent)); sid, st = ent[o], t[o]
        S = np.int64(200_000_000_000)                # > range of clipped timestamps
        key = sid.astype(np.int64) * S + np.clip(st, -9e10, 9e10).astype(np.int64)
        for d in (1, 7, 30):
            a = np.searchsorted(key, q.astype(np.int64) * S + np.clip(qt - d * 86400, -9e10, 9e10).astype(np.int64), 'left')
            b = np.searchsorted(key, q.astype(np.int64) * S + np.clip(qt + d * 86400, -9e10, 9e10).astype(np.int64), 'right')
            F[f'{name}_act{d}'] = (b - a).astype(float)
        F[f'{name}_act_all'] = np.bincount(ent, minlength=q.max() + 1)[q].astype(float)
    return F


def build_all(hist, AP, qu, qi, qt, nu, ni):
    qt = qt.astype(float)
    X = build_static(hist, AP, qu, qi, qt, nu, ni)
    T = pd.DataFrame(time_feats(hist, qu, qi, qt))
    nz = hist.rating.values > 0
    bias = BiasMF(k=0).fit(hist.u.values[nz], hist.i.values[nz], hist.rating.values[nz].astype(float), nu, ni)
    res = np.where(nz, hist.rating.values - bias.predict(hist.u.values, hist.i.values), np.nan)
    R = pd.DataFrame(resid_time_feats(hist, res, qu, qi, qt))
    A = pd.DataFrame(activity_feats(AP, qu, qi, qt))
    log('  time features done')
    return pd.concat([X, T.astype(np.float32), R.astype(np.float32), A.astype(np.float32)], axis=1)


# ----------------------------------------------------------------------------- pipeline
def load():
    tr = pd.read_csv(TRAIN); te = pd.read_csv(TEST)
    users = pd.Index(tr.user.unique()); items = pd.Index(tr.item.unique())
    tr['u'] = users.get_indexer(tr.user); tr['i'] = items.get_indexer(tr.item)
    te['u'] = users.get_indexer(te.user); te['i'] = items.get_indexer(te.item)
    return tr, te, len(users), len(items)


def cached(name, fn):
    if CACHE:
        p = Path(CACHE) / f'{name}.pkl'
        if p.exists(): return pd.read_pickle(p)
        X = fn(); p.parent.mkdir(parents=True, exist_ok=True); X.to_pickle(p); return X
    return fn()


GBM_PARAMS = dict(max_iter=900, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=200,
                  l2_regularization=1.0, early_stopping=False)


def fit_predict_gbm(XH, yH, XQ, seeds=(0, 1, 2)):
    ps = []
    for s in seeds:
        g = HistGradientBoostingRegressor(random_state=s, **GBM_PARAMS)
        g.fit(XH, yH); ps.append(g.predict(np.asarray(XQ, np.float64))); del g; log(f'  GBM seed {s} done')
    return np.clip(np.mean(ps, 0), 1, 5)


def main(validate):
    import warnings; warnings.filterwarnings('ignore')
    tr, te, nu, ni = load()
    AP = pd.concat([tr[['u', 'i', 'timestamp']], te[['u', 'i', 'timestamp']]], ignore_index=True)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(np.where(tr.rating.values > 0)[0])
    if validate:
        q_idx = perm[:N_VAL]; S_idx = perm[N_VAL:] if N_STACK is None else perm[N_VAL:N_VAL + N_STACK]
    else:
        q_idx = np.array([], int); S_idx = perm if N_STACK is None else perm[:N_STACK]
    keep = np.ones(len(tr), bool); keep[q_idx] = False
    folds = np.array_split(S_idx, K_FOLDS)
    XH, yH, cols = None, [], None
    row = 0
    for f, fidx in enumerate(folds):
        log(f'stacker fold {f + 1}/{K_FOLDS}: {len(fidx)} rows')
        m = keep.copy(); m[fidx] = False
        H = tr.iloc[fidx]
        tag = f'{"v" if validate else "t"}_N{len(S_idx)}_K{K_FOLDS}_f{f}'
        Xf = cached(tag, lambda: build_all(tr[m], AP, H.u.values, H.i.values, H.timestamp.values, nu, ni))
        if XH is None:   # one preallocated float64 matrix: sklearn would copy float32 to float64 anyway
            cols = list(Xf.columns); XH = np.empty((len(S_idx), len(cols)), np.float64)
        XH[row:row + len(Xf)] = Xf[cols].values; row += len(Xf); del Xf
        yH.append(H.rating.values)
    yH = np.concatenate(yH)
    if validate:
        Q = tr.iloc[q_idx]
        log('validation features (history = train minus holdout)')
        XQ = cached('val', lambda: build_all(tr[keep], AP, Q.u.values, Q.i.values, Q.timestamp.values, nu, ni))[cols]
        y = Q.rating.values
        log(f'bias MSE      {mse(np.clip(XQ.bias, 1, 5), y):.4f}')
        log(f'ALS MF MSE    {mse(np.clip(XQ.mf, 1, 5), y):.4f}')
        log(f'timeSVD++ MSE {mse(np.clip(XQ.tsvd, 1, 5), y):.4f}')
        pred = fit_predict_gbm(XH, yH, XQ)
        log(f'stack MSE     {mse(pred, y):.4f}  (honest: full-history features, non-zero ratings)')
        return
    log('test features (history = ALL training interactions)')
    XQ = cached('test', lambda: build_all(tr, AP, te.u.values, te.i.values, te.timestamp.values, nu, ni))[cols]
    pred = fit_predict_gbm(XH, yH, XQ)
    out = te[['user', 'item', 'timestamp']].copy(); out['rating'] = pred
    out.to_csv('task_3_predictions.csv', index=False)
    log('wrote task_3_predictions.csv')


if __name__ == '__main__':
    main('--validate' in sys.argv)
