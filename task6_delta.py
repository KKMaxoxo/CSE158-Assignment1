"""
Task 6 - predict the time until the user's next interaction (CSE 158 Assignment 1).

Test rows: one per user, (user, item of the NEXT interaction); that item is never in the user's
training history, so delta = t_next - t_last_train.

Training examples: hold out a user's k-th last interaction and predict the gap into it from everything
before. 'back' = how many steps before the user's TRUE final interaction that gap is. Task 6 train lacks
each user's final interaction (that is the test), so its examples have back = 1..5. The task 1-2 and 3-4
training files hold complete timelines of other users (different users, separately hashed items), so
they provide real FINAL gaps (back = 0), which is what the test asks for. Final gaps differ from earlier
ones (they cluster just before the crawl date), and adding them cut MAE on held-out real final gaps
by ~3% vs. task-6-only + rescaling. Targets are modelled as log1p(delta seconds) with
absolute loss, i.e. the conditional median -> optimal for absolute error in seconds or in log time.

Features
  user timeline : count, span, time since first, log gaps (last 1/2/3/5, mean, median, std, quantiles,
                  exp-weighted), activity in the last day/week/month, rating & zero of last items
  calendar      : last timestamp, log(time from last to end of data)
  next item     : popularity, first / 5% / median appearance time relative to the user's last time,
                  share of the item's interactions after the user's last time, item's typical
                  gap-before value from other users, co-occurrence of the next item with the user's
                  recent items (series / binge signal)
Model: HistGradientBoostingRegressor(loss='absolute_error') on log1p(delta), 3 seeds.

Usage:  python task6_delta.py      (prints local validation, then writes task_6_predictions.csv)
"""
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingRegressor

DATA = Path('/Users/kmaxx/Desktop/CSE 158/assignment1_2026')
TRAIN = DATA / 'task_6_train.csv'
EXTERNAL = [DATA / 'tasks_1_2_train.csv', DATA / 'tasks_3_4_train.csv']   # full timelines -> final gaps
TEST = DATA / 'task_6_test.csv'
OUT = 'task_6_predictions.csv'
SEED = 0
N_BACK = 5        # simulated examples per user: gaps into the last 5 training interactions
W_LAST = 2.0      # weight of the newest simulated position (closest to the test situation)



def log(*a): print(*a, flush=True)


def item_stats(hist, ni):
    """Per-item time statistics and typical log-gap-before from the users who read it."""
    t = hist.timestamp.values.astype(float); i = hist.i.values
    df = pd.DataFrame({'i': i, 't': t, 'g': hist.gap_before.values})
    q = df.groupby('i').t.quantile([0.0, 0.05, 0.5]).unstack().reindex(range(ni))
    o = np.lexsort((t, i))
    return dict(cnt=np.bincount(i, minlength=ni).astype(float), q=q.values,
                key=i[o].astype(np.float64) * 1e11 + t[o],
                gmed=df.groupby('i').g.median().reindex(range(ni)).values,
                gmean=df.groupby('i').g.mean().reindex(range(ni)).values)


def make_features(hist, qu, qi, t_last_q, IS, A, AT, t_end):
    """hist: interactions available as history (sorted by user, time), with columns u,i,timestamp,
    rating,gap_before. qu, qi: query user / next item. t_last_q: user's last history timestamp."""
    F = {}
    u = hist.u.values; t = hist.timestamp.values.astype(float); r = hist.rating.values
    starts = np.searchsorted(u, qu, 'left'); ends = np.searchsorted(u, qu, 'right')
    n = len(qu)
    cnt = ends - starts
    F['u_cnt'] = np.log1p(cnt)
    F['u_span'] = np.log1p(np.maximum(t[ends - 1] - t[starts], 0))
    F['t_last'] = t_last_q
    F['to_end'] = np.log1p(np.maximum(t_end - t_last_q, 0))
    lg = np.log1p(np.maximum(np.diff(t), 0)); lg = np.r_[lg, 0]      # lg[k] = gap from k to k+1
    W = 30
    idx = (ends - 1)[:, None] - np.arange(1, W + 1)[None, :]          # gaps ending at last item and before
    valid = idx >= starts[:, None]
    G = np.where(valid, lg[np.clip(idx, 0, len(lg) - 1)], np.nan)     # G[:,0] = gap into the last item
    with np.errstate(all='ignore'):
        for k in (1, 2, 3, 5):
            F[f'g_last{k}'] = np.nanmean(G[:, :k], 1)
        F['g_mean'] = np.nanmean(G, 1); F['g_med'] = np.nanmedian(G, 1); F['g_std'] = np.nanstd(G, 1)
        for qq in (10, 25, 75, 90): F[f'g_q{qq}'] = np.nanpercentile(G, qq, axis=1)
        w = 0.7 ** np.arange(W); F['g_ewm'] = np.nansum(G * w, 1) / np.nansum(np.where(np.isnan(G), 0, w), 1)
        F['g_min3'] = np.nanmin(G[:, :3], 1); F['g_max3'] = np.nanmax(G[:, :3], 1)
    # activity bursts before last
    for nm, dt in (('day', 86400), ('week', 7 * 86400), ('month', 30 * 86400), ('year', 365 * 86400)):
        a = np.searchsorted(hist.ukey.values, qu * 1e11 + t_last_q - dt, 'left')
        F[f'n_{nm}'] = np.log1p(np.maximum(ends - np.maximum(a, starts), 0))
    F['r_last'] = r[ends - 1]
    F['zero_last'] = (r[ends - 1] == 0).astype(float)
    with np.errstate(all='ignore'):
        R5 = np.where(valid[:, :5], r[np.clip((ends - 1)[:, None] - np.arange(5)[None, :], 0, None)], np.nan)
        F['zero_last5'] = np.nanmean(R5 == 0, 1); F['r_last5'] = np.nanmean(np.where(R5 > 0, R5, np.nan), 1)
    # next item
    F['i_pop'] = np.log1p(IS['cnt'][qi])
    for k, nm in enumerate(('first', 'q05', 'med')):
        d = IS['q'][qi, k] - t_last_q
        F[f'i_{nm}_minus_last'] = np.sign(d) * np.log1p(np.abs(np.nan_to_num(d)))
    F['i_first_after'] = (IS['q'][qi, 0] > t_last_q).astype(float)
    a = np.searchsorted(IS['key'], qi * 1e11 + t_last_q, 'right'); b = np.searchsorted(IS['key'], qi * 1e11 + 1e11 - 1, 'right')
    F['i_share_after'] = (b - a) / np.maximum(IS['cnt'][qi], 1)
    F['i_n_after'] = np.log1p(b - a)
    # when do OTHER readers pick up the next item after the user's last time?
    keyt = IS['key'] - np.floor(IS['key'] / 1e11) * 1e11          # times, aligned with key
    cntq = IS['cnt'][qi].astype(int)
    for nm, frac in (('after_first', 0.0), ('after_q25', 0.25), ('after_med', 0.5)):
        k = a + np.floor(frac * np.maximum(b - a - 1, 0)).astype(int)
        ok = (b - a) > 0
        F[f'i_{nm}'] = np.where(ok, np.log1p(np.maximum(keyt[np.clip(k, 0, len(keyt) - 1)] - t_last_q, 0)), np.nan)
    for nm, dt in (('30d', 30 * 86400), ('365d', 365 * 86400)):
        c = np.searchsorted(IS['key'], qi * 1e11 + t_last_q + dt, 'right')
        F[f'i_n_{nm}'] = np.log1p(c - a); F[f'i_sh_{nm}'] = (c - a) / np.maximum(b - a, 1)
    # and before: how recently others read it
    k = a - 1; okb = (k >= 0) & (np.floor(IS['key'][np.clip(k, 0, None)] / 1e11) == qi)
    F['i_before_last'] = np.where(okb, np.log1p(np.maximum(t_last_q - keyt[np.clip(k, 0, None)], 0)), np.nan)
    F['i_gap_med'] = IS['gmed'][qi]; F['i_gap_mean'] = IS['gmean'][qi]
    # co-occurrence of next item with the user's last items
    icnt = np.asarray(A.sum(0)).ravel()
    sims = np.zeros((n, 3))
    for k in range(3):
        j = hist.i.values[np.clip(ends - 1 - k, 0, None)]
        ok = (ends - 1 - k) >= starts
        co = np.asarray(AT[qi].multiply(AT[j]).sum(1)).ravel()
        sims[:, k] = np.where(ok, co / np.sqrt(np.maximum(icnt[qi] * icnt[j], 1)), np.nan)
    F['sim_last1'] = sims[:, 0]; F['sim_last3_max'] = np.nanmax(sims, 1)
    return pd.DataFrame(F)


def prep(df, users, items):
    df = df.copy()
    df['u'] = users.get_indexer(df.user); df['i'] = items.get_indexer(df.item)
    df = df.sort_values(['u', 'timestamp'], kind='stable').reset_index(drop=True)
    t = df.timestamp.values.astype(float)
    same = np.r_[False, df.u.values[1:] == df.u.values[:-1]]
    df['gap_before'] = np.where(same, np.log1p(np.maximum(np.r_[0, np.diff(t)], 0)), np.nan)
    df['ukey'] = df.u.values * 1e11 + t
    return df


def cooc_mats(hist, nu, ni):
    A = sp.csr_matrix((np.ones(len(hist)), (hist.u.values, hist.i.values)), shape=(nu, ni)); A.data[:] = 1
    return A, A.T.tocsr()


def simulated(tr, nu, ni, items_n, t_end, back):
    """Examples where the target is the gap into a user's (back+1)-th last interaction."""
    rank_from_end = tr.groupby('u').cumcount(ascending=False).values
    tgt = tr[rank_from_end == back]
    hist = tr[rank_from_end > back]
    hist = hist[hist.u.isin(tgt.u)]
    tgt = tgt[tgt.u.isin(hist.u)]
    hist = hist.reset_index(drop=True)
    # item stats from everything except each user's held-out tail
    IS = item_stats(tr[rank_from_end > back], items_n)
    A, AT = cooc_mats(tr[rank_from_end > back], nu, ni)
    qu = tgt.u.values; qi = tgt.i.values
    tl = hist.groupby('u').timestamp.max().reindex(qu).values.astype(float)
    X = make_features(hist, qu, qi, tl, IS, A, AT, t_end)
    X['back'] = back
    add_target_set_feats(X, qi, tgt.i.values, tr.i.values[rank_from_end > back], items_n)
    y = np.log1p(np.maximum(tgt.timestamp.values - tl, 0))
    return X, y, qu


def add_target_set_feats(X, qi, target_items, hist_items, ni):
    """How often the query item appears among ALL users' target interactions (the test file in the
    final run, the held-out rows of the same position in simulation): a book many users read last is
    a recent release."""
    tc = np.bincount(target_items, minlength=ni).astype(float)
    hc = np.bincount(hist_items, minlength=ni).astype(float)
    X['i_tgt_cnt'] = np.log1p(tc[qi]); X['i_tgt_share'] = tc[qi] / (tc[qi] + hc[qi])


def gbm(seed):
    return HistGradientBoostingRegressor(loss='absolute_error', max_iter=2500, learning_rate=0.02,
                                         max_leaf_nodes=127, min_samples_leaf=100, l2_regularization=1.0,
                                         random_state=seed)


def report(name, p, y):
    log(f'{name:28s} MAE(log)={np.mean(np.abs(p - y)):.4f}  MSE(log)={np.mean((p - y) ** 2):.4f}  '
        f'MAE(days)={np.mean(np.abs(np.expm1(p) - np.expm1(y))) / 86400:.1f}')


def dataset_examples(path, offset, backs, extra_items=None):
    """Gap examples from one training file; offset = interactions missing at the end of each timeline."""
    raw = pd.read_csv(path)
    users = pd.Index(raw.user.unique())
    it = raw.item if extra_items is None else pd.concat([raw.item, extra_items])
    items = pd.Index(pd.unique(it)); nu, ni = len(users), len(items)
    tr = prep(raw, users, items); t_end = np.percentile(tr.timestamp.values, 99.9)
    Xs, ys = [], []
    for b in range(backs):
        X, y, _ = simulated(tr, nu, ni, ni, t_end, b)
        X['back'] = b + offset; Xs.append(X); ys.append(y)
    return pd.concat(Xs, ignore_index=True), np.concatenate(ys), (tr, users, items, t_end)


def main():
    import warnings; warnings.filterwarnings('ignore')
    te = pd.read_csv(TEST)
    X6, y6, (tr, users, items, t_end) = dataset_examples(TRAIN, 1, N_BACK, extra_items=te.item)
    log(f'task 6 examples: {len(X6):,}')
    ext = [dataset_examples(p, 0, N_BACK) for p in EXTERNAL]
    for p, (X, _, _) in zip(EXTERNAL, ext): log(f'{p.name}: {len(X):,} examples')
    (Xa, ya, _), (Xb, yb, _) = ext

    # validation: real final gaps of the second external file's users (never trained on here)
    vb = Xb.back.values == 0
    XV = pd.concat([X6, Xa], ignore_index=True); yVt = np.r_[y6, ya]
    g = gbm(0).fit(XV, yVt, sample_weight=np.where(XV.back.values == 0, W_LAST, 1.))
    report('median baseline', np.full(vb.sum(), np.median(y6)), yb[vb])
    report('GBM (final gaps, held out)', g.predict(Xb[vb]), yb[vb])

    # final model: all three files
    Xs = pd.concat([X6, Xa, Xb], ignore_index=True); ys = np.r_[y6, ya, yb]
    wts = np.where(Xs.back.values == 0, W_LAST, 1.)
    ni = len(items)
    IS = item_stats(tr, ni); A, AT = cooc_mats(tr, len(users), ni)
    qu = users.get_indexer(te.user); qi = items.get_indexer(te.item)
    tl = tr.groupby('u').timestamp.max().reindex(qu).values.astype(float)
    Xt = make_features(tr, qu, qi, tl, IS, A, AT, t_end); Xt['back'] = 0
    add_target_set_feats(Xt, qi, qi, tr.i.values, ni)
    Xt = Xt[Xs.columns]
    p = np.mean([gbm(s).fit(Xs, ys, sample_weight=wts).predict(Xt) for s in range(3)], 0)
    out = te[['user', 'item']].copy()
    out['delta'] = np.round(np.expm1(np.maximum(p, 0))).astype(np.int64)
    out.to_csv(OUT, index=False)
    log(f'wrote {OUT}; median predicted delta {np.median(out.delta) / 86400:.1f} days')


if __name__ == '__main__':
    main()
