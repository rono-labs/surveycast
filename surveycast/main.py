"""surveycast: calibrated answer distributions for held-out survey respondents.

predict() fits everything on the visible rows of the frame it is given:

  1. backoff tables: answer shares within cells of the most informative GIVEN
     items, shrunk level by level toward coarser cells, shrinkage picked by
     leave-one-out likelihood;
  2. a multi-task softmax regression on one-hot GIVEN items;
  3. a multi-task MLP with a shared trunk and one softmax head per item.

The neural models are trained with K-fold cross-fitting, so every visible row
gets an out-of-fold prediction; per-item mixture weights over the components
are then fitted by EM on those predictions. Gates whose parent is a GIVEN item
are applied exactly when the visible rows never contradict them.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

FOLDS = 5
SEED = 0
EPS = 1e-9

# The run budget covers import and all three predict() calls. Phase 1 gives
# 900 s, phase 2 gives 3600 s; keep a margin for the harness itself.
BUDGET = {1: 900.0, 2: 3600.0}
MARGIN = 0.7
_STARTED = time.time()
_CALLS = [0]
_DEADLINE = [float("inf")]


class OutOfTime(Exception):
    pass


def _check_time():
    if time.time() > _DEADLINE[0]:
        raise OutOfTime

torch.set_num_threads(int(os.environ.get("SURVEYCAST_THREADS", min(8, os.cpu_count() or 1))))


def _device():
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda").sum().item()
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


DEVICE = _device()


# ------------------------------------------------------------------ encode --

def _codes(column, values):
    lookup = {v: i for i, v in enumerate(values)}
    out = np.full(len(column), -1, dtype=np.int64)
    for i, v in enumerate(column):
        if not pd.isna(v):
            out[i] = lookup.get(v, -1)
    return out


class Problem:
    def __init__(self, frame, schema):
        items = schema["items"]
        gated = schema["gated_value"]
        self.order = [k for k, r in items.items()
                      if r["class"] in ("GIVEN", "PREDICT") and k in frame.columns]
        self.given = [k for k in self.order if items[k]["class"] == "GIVEN"]
        self.targets = [k for k in self.order if items[k]["class"] == "PREDICT"]
        self.options = {}
        for k in self.order:
            r = items[k]
            self.options[k] = list(r["values"]) + ([gated] if r.get("gate") else [])
        self.gates = {k: items[k].get("gate") for k in self.targets}

        X = np.stack([_codes(frame[k].to_numpy(object), self.options[k])
                      for k in self.given], 1) if self.given else \
            np.zeros((len(frame), 0), np.int64)
        # Unseen or missing GIVEN values get their own level.
        self.cards = [len(self.options[k]) + 1 for k in self.given]
        for j, c in enumerate(self.cards):
            X[X[:, j] < 0, j] = c - 1
        self.X = X
        self.Y = np.stack([_codes(frame[k].to_numpy(object), self.options[k])
                           for k in self.targets], 1)
        self.K = np.array([len(self.options[k]) for k in self.targets])
        self.kmax = int(self.K.max())
        observed = self.Y >= 0
        self.train = np.where(observed.all(1))[0]
        self.test = np.where(~observed.all(1))[0]

    def onehot(self):
        offsets = np.concatenate([[0], np.cumsum(self.cards)[:-1]]).astype(np.int64)
        return self.X + offsets[None, :], int(sum(self.cards))


# --------------------------------------------------------------- backoff --

def _mutual_information(x, y, kx, ky):
    joint = np.zeros((kx, ky))
    np.add.at(joint, (x, y), 1.0)
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def _level(key_f, key_o, y, K, base_fit, base_out, strengths):
    """One backoff level: cell counts under key, shrunk toward base. Returns
    (loo ll, loo probs, out probs) at the best strength."""
    uniq, inv_f = np.unique(key_f, return_inverse=True)
    table = np.zeros((len(uniq), K))
    np.add.at(table, (inv_f, y), 1.0)
    rows = np.arange(len(y))
    n_f = table[inv_f]
    n_f[rows, y] -= 1.0
    tot_f = n_f.sum(1, keepdims=True)
    best, best_ll = None, -np.inf
    for a in strengths:
        p = (n_f[rows, y] + a * base_fit[rows, y]) / (tot_f[:, 0] + a)
        ll = np.log(p + EPS).mean()
        if ll > best_ll:
            best, best_ll = a, ll
    pos = np.clip(np.searchsorted(uniq, key_o), 0, len(uniq) - 1)
    hit = uniq[pos] == key_o
    n_o = np.where(hit[:, None], table[pos], 0.0)
    fit = (n_f + best * base_fit) / (tot_f + best)
    out = (n_o + best * base_out) / (n_o.sum(1, keepdims=True) + best)
    return best_ll, fit, out


def backoff_tables(P, rows_fit, rows_out, depth=4, greedy=0, first=0,
                   strengths=(0.5, 2.0, 8.0, 32.0, 128.0)):
    """Per item: hierarchical Dirichlet backoff, each level splitting the
    previous level's cells by one more GIVEN item, with shrinkage toward the
    coarser level picked by leave-one-out likelihood. The item order is by
    mutual information, or with greedy=1 by LOO gain at each level. Returns
    LOO predictions on rows_fit and predictions on rows_out."""
    X, Y = P.X, P.Y
    J = len(P.targets)
    loo = np.zeros((len(rows_fit), J, P.kmax))
    out = np.zeros((len(rows_out), J, P.kmax))
    Xf, Xo = X[rows_fit], X[rows_out]
    for j in range(J):
        K = P.K[j]
        y = Y[rows_fit, j]
        counts_y = np.bincount(y, minlength=K).astype(float)
        base_fit = (counts_y[None, :] - np.eye(K)[y] + 0.5) / (len(y) - 1 + 0.5 * K)
        base_out = np.tile((counts_y + 0.5) / (len(y) + 0.5 * K), (len(rows_out), 1))
        ll = np.log(base_fit[np.arange(len(y)), y] + EPS).mean()
        key_f = np.zeros(len(rows_fit), np.int64)
        key_o = np.zeros(len(rows_out), np.int64)
        if greedy:
            candidates = list(range(X.shape[1]))
        else:
            mi = [_mutual_information(Xf[:, g], y, P.cards[g], K)
                  for g in range(X.shape[1])]
            ranked = list(np.argsort(mi)[::-1])
            ranked.insert(0, ranked.pop(min(first, len(ranked) - 1)))
            candidates = ranked[:depth]
        for _ in range(depth):
            options = candidates if greedy else candidates[:1]
            best = None
            for g in options:
                kf = key_f * P.cards[g] + Xf[:, g]
                ko = key_o * P.cards[g] + Xo[:, g]
                res = _level(kf, ko, y, K, base_fit, base_out, strengths)
                if best is None or res[0] > best[0]:
                    best = (res[0], g, kf, ko, res[1], res[2])
            if best is None or best[0] <= ll:
                break
            ll, g, key_f, key_o, base_fit, base_out = best
            candidates.remove(g)
        loo[:, j, :K] = base_fit
        out[:, j, :K] = base_out
    return loo, out


# ------------------------------------------------------------ neural nets --

class MultiHead(torch.nn.Module):
    def __init__(self, n_in, J, kmax, hidden, dropout, layers=2):
        super().__init__()
        self.J, self.kmax = J, kmax
        if hidden:
            self.emb = torch.nn.EmbeddingBag(n_in, hidden, mode="sum")
            trunk = [torch.nn.GELU(), torch.nn.Dropout(dropout)]
            for _ in range(layers - 1):
                trunk += [torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                          torch.nn.Dropout(dropout)]
            self.trunk = torch.nn.Sequential(*trunk)
            self.head = torch.nn.Linear(hidden, J * kmax)
        else:
            self.emb = torch.nn.EmbeddingBag(n_in, J * kmax, mode="sum")
            self.trunk = None
            self.head = None
        self.bias = torch.nn.Parameter(torch.zeros(J * kmax))

    def forward(self, idx, mask):
        h = self.emb(idx)
        if self.trunk is not None:
            h = self.head(self.trunk(h))
        logits = (h + self.bias).view(-1, self.J, self.kmax)
        return torch.log_softmax(logits + mask, -1)


def train_net(P, rows_fit, rows_val, rows_out, hidden, dropout, wd,
              epochs=80, batch=256, lr=3e-3, patience=8, seed=0, layers=2):
    """Train on rows_fit; per-item early stopping on rows_val. Returns
    probabilities on rows_val and rows_out at each item's best epoch."""
    torch.manual_seed(seed)
    idx_all, n_in = P.onehot()
    J, kmax = len(P.targets), P.kmax
    mask = torch.full((J, kmax), -1e4)
    for j, k in enumerate(P.K):
        mask[j, :k] = 0.0
    mask = mask.to(DEVICE)
    net = MultiHead(n_in, J, kmax, hidden, dropout, layers).to(DEVICE)
    with torch.no_grad():
        Yf = P.Y[rows_fit]
        for j, k in enumerate(P.K):
            prior = np.log((np.bincount(Yf[:, j], minlength=kmax)[:kmax] + 0.5)
                           / (len(Yf) + 0.5 * k))
            net.bias[j * kmax:(j + 1) * kmax] = torch.tensor(prior, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)

    def tens(rows):
        return (torch.tensor(idx_all[rows], device=DEVICE),
                torch.tensor(np.maximum(P.Y[rows], 0), device=DEVICE))

    xf, yf = tens(rows_fit)
    xv, yv = tens(rows_val)
    xo = torch.tensor(idx_all[rows_out], device=DEVICE)
    best = torch.full((J,), np.inf, device=DEVICE)
    best_val = torch.zeros((len(rows_val), J, kmax), device=DEVICE)
    best_out = torch.zeros((len(rows_out), J, kmax), device=DEVICE)
    stale = 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(rows_fit), generator=g).to(DEVICE)
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            logp = net(xf[b], mask)
            loss = -logp.gather(2, yf[b].unsqueeze(2)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv = net(xv, mask)
            item_loss = -lv.gather(2, yv.unsqueeze(2)).squeeze(2).mean(0)
            better = item_loss < best - 1e-5
            if better.any():
                lo = net(xo, mask)
                best = torch.where(better, item_loss, best)
                best_val[:, better] = lv[:, better].exp()
                best_out[:, better] = lo[:, better].exp()
                stale = 0
            else:
                stale += 1
        if stale >= patience:
            break
        _check_time()
    return best_val.cpu().numpy(), best_out.cpu().numpy()


def crossfit(P, folds, rows_out, seeds=1, seed=0, nfolds=FOLDS, **kw):
    n = len(P.train)
    if seed or nfolds != FOLDS:
        folds = np.random.default_rng(SEED + seed).permutation(n) % nfolds
    oof = np.zeros((n, len(P.targets), P.kmax))
    out = np.zeros((len(rows_out), len(P.targets), P.kmax))
    for f in range(nfolds):
        val = folds == f
        for s in range(seeds):
            v, o = train_net(P, P.train[~val], P.train[val], rows_out,
                             seed=SEED + f + 100 * (s + seed), **kw)
            oof[val] += v / seeds
            out += o / (nfolds * seeds)
            _check_time()
    return oof, out


# ---------------------------------------------------------------- stacking --

def mixture_weights(preds, y, prior, iters=200):
    """EM for linear-pool weights. preds (M, n, kmax), y (n,)."""
    lik = preds[:, np.arange(len(y)), y] + EPS            # (M, n)
    w = np.full(len(preds), 1.0 / len(preds))
    for _ in range(iters):
        r = w[:, None] * lik
        r /= r.sum(0, keepdims=True)
        w = (r.sum(1) + prior) / (len(y) + prior.sum())
    return w


def stack(components_oof, components_out, P):
    M = len(components_oof)
    J = len(P.targets)
    y = P.Y[P.train]
    weights = np.zeros((J, M))
    glob = mixture_weights(
        np.stack([c.reshape(-1, c.shape[2]) for c in components_oof]),
        y.reshape(-1), np.zeros(M))
    for j in range(J):
        weights[j] = mixture_weights(
            np.stack([c[:, j] for c in components_oof]), y[:, j], 20.0 * glob)
    out = sum(weights[None, :, m, None] * components_out[m] for m in range(M))
    return out, weights


# ------------------------------------------------------------------ gates --

def apply_given_gates(P, probs, rows):
    for j, item in enumerate(P.targets):
        gate = P.gates[item]
        if not gate or gate["parent"] not in P.given:
            continue
        g = P.given.index(gate["parent"])
        opts = P.options[gate["parent"]]
        allowed = np.array([o in gate["observed_if"] for o in opts] + [False])
        sentinel = P.K[j] - 1
        asked_train = allowed[P.X[P.train, g]]
        ytr = P.Y[P.train, j]
        if not np.all((ytr == sentinel) == ~asked_train):
            continue
        asked = allowed[P.X[rows, g]]
        p = probs[:, j, :P.K[j]]
        p[~asked] = 0.0
        p[~asked, sentinel] = 1.0
        p[asked, sentinel] = 0.0
        p[asked] /= np.maximum(p[asked].sum(1, keepdims=True), EPS)
        probs[:, j, :P.K[j]] = p
    return probs


# -------------------------------------------------------------- components --

def comp_backoff(P, folds, rows_out, depth=4, greedy=0, chains=1):
    if chains == 1:
        return backoff_tables(P, P.train, rows_out, depth=depth, greedy=greedy)
    loo, out = 0.0, 0.0
    for c in range(chains):
        l, o = backoff_tables(P, P.train, rows_out, depth=depth, first=c)
        loo, out = loo + l / chains, out + o / chains
    return loo, out


def comp_marginal(P, folds, rows_out, **kw):
    n, J = len(P.train), len(P.targets)
    oof = np.zeros((n, J, P.kmax))
    out = np.zeros((len(rows_out), J, P.kmax))
    Y = P.Y[P.train]
    for f in range(FOLDS):
        val = folds == f
        for j, k in enumerate(P.K):
            c = np.bincount(Y[~val, j], minlength=k) + 0.5
            oof[val, j, :k] = c / c.sum()
    for j, k in enumerate(P.K):
        c = np.bincount(Y[:, j], minlength=k) + 0.5
        out[:, j, :k] = c / c.sum()
    return oof, out


def comp_linear(P, folds, rows_out, **kw):
    args = dict(hidden=0, dropout=0.0, wd=1e-4, lr=1e-2)
    args.update(kw)
    return crossfit(P, folds, rows_out, **args)


def comp_mlp(P, folds, rows_out, **kw):
    args = dict(hidden=256, dropout=0.3, wd=1e-3, batch=1024)
    args.update(kw)
    return crossfit(P, folds, rows_out, **args)


def _parse(name):
    """'mlp:hidden=512,dropout=0.5' -> ('mlp', {'hidden': 512, 'dropout': 0.5})"""
    base, _, rest = name.partition(":")
    kw = {}
    for part in filter(None, rest.split(",")):
        k, v = part.split("=")
        kw[k] = int(v) if v.lstrip("-").isdigit() else float(v)
    return base, kw


def comp_lgbm(P, folds, rows_out, **kw):
    import lightgbm as lgb
    n, J = len(P.train), len(P.targets)
    oof = np.zeros((n, J, P.kmax))
    out = np.zeros((len(rows_out), J, P.kmax))
    X = P.X.astype(np.float64)
    Xt, Xo = X[P.train], X[rows_out]
    params = dict(objective="multiclass", learning_rate=0.05, num_leaves=15,
                  min_data_in_leaf=40, lambda_l2=1.0, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, max_cat_to_onehot=8,
                  cat_smooth=20, verbose=-1, num_threads=torch.get_num_threads())
    params.update(kw)
    for j, k in enumerate(P.K):
        y = P.Y[P.train, j]
        if k < 2 or np.bincount(y).max() == len(y):
            oof[:, j, :k] = np.eye(k)[np.bincount(y, minlength=k).argmax()]
            out[:, j, :k] = oof[0, j, :k]
            continue
        for f in range(FOLDS):
            val = folds == f
            dtr = lgb.Dataset(Xt[~val], y[~val], categorical_feature="auto",
                              free_raw_data=False)
            dva = lgb.Dataset(Xt[val], y[val], reference=dtr)
            booster = lgb.train(dict(params, num_class=int(k)), dtr, 400,
                                valid_sets=[dva],
                                callbacks=[lgb.early_stopping(20, verbose=False)])
            oof[val, j, :k] = booster.predict(Xt[val])
            out[:, j, :k] += booster.predict(Xo) / FOLDS
        _check_time()
    return oof, out


COMPONENTS = {"lgbm": comp_lgbm, "backoff": comp_backoff, "linear": comp_linear, "mlp": comp_mlp,
              "marginal": comp_marginal}
# Most valuable first: when time runs short the tail is dropped.
DEFAULT = ("backoff", "mlp", "linear", "mlp:seed=1", "mlp:seed=2",
           "mlp:seed=3", "mlp:seed=4")


def apply_predicted_gates(P, probs, mix=0.5):
    """For a gate whose parent is a PREDICT item and which the visible rows
    never contradict, P(child never asked) = P(parent not in observed_if).
    Average the child's own estimate of that mass with the one implied by its
    parent's prediction. Items are visited in schema order, parents first."""
    col = {k: j for j, k in enumerate(P.targets)}
    Ytr = P.Y[P.train]
    for j, item in enumerate(P.targets):
        gate = P.gates[item]
        if not gate or gate["parent"] not in col:
            continue
        pj = col[gate["parent"]]
        allowed = np.array([o in gate["observed_if"] for o in P.options[gate["parent"]]])
        sentinel = P.K[j] - 1
        if not np.array_equal(Ytr[:, j] == sentinel, ~allowed[Ytr[:, pj]]):
            continue
        parent = probs[:, pj, :P.K[pj]]
        skip = parent[:, ~allowed].sum(1) / np.maximum(parent.sum(1), EPS)
        child = probs[:, j, :P.K[j]]
        child = child / np.maximum(child.sum(1, keepdims=True), EPS)
        target = mix * skip + (1 - mix) * child[:, sentinel]
        rest = child[:, :sentinel]
        rest = rest / np.maximum(rest.sum(1, keepdims=True), EPS)
        probs[:, j, :sentinel] = rest * (1 - target[:, None])
        probs[:, j, sentinel] = target
    return probs


# ----------------------------------------------------------------- predict --

def fit_components(P, names=DEFAULT):
    """Fit components cheapest first; one that runs past the deadline is
    dropped and the ones after it are skipped."""
    rng = np.random.default_rng(SEED)
    folds = rng.permutation(len(P.train)) % FOLDS
    preds = {}
    for name in ("marginal",) + tuple(n for n in names if n != "marginal"):
        base, kw = _parse(name)
        try:
            if name != "marginal":
                _check_time()
            preds[name] = COMPONENTS[base](P, folds, P.test, **kw)
        except OutOfTime:
            break
        except Exception:
            if name == "marginal":
                raise
            continue
    return preds


def _set_deadline(P, schema):
    split = schema.get("split") or {}
    final = len(P.train) > split.get("n_train", len(P.train))
    budget = BUDGET[2 if final else 1] * MARGIN
    spent = time.time() - _STARTED
    calls_left = max(1, 3 - _CALLS[0])
    _DEADLINE[0] = time.time() + max(0.0, budget - spent) / calls_left
    _CALLS[0] += 1


def predict(frame, schema):
    started = time.time()
    P = Problem(frame, schema)
    _set_deadline(P, schema)
    preds = fit_components(P)
    names = list(preds)
    probs, weights = stack([preds[n][0] for n in names],
                           [preds[n][1] for n in names], P)
    probs = apply_given_gates(P, probs, P.test)
    probs = apply_predicted_gates(P, probs)
    if os.environ.get("SURVEYCAST_VERBOSE"):
        print("weights", dict(zip(names, weights.mean(0).round(3))),
              "seconds %.1f" % (time.time() - started),
              "allowance %.0f" % (_DEADLINE[0] - started))
    return to_vectors(frame, P, probs)


def to_vectors(frame, P, probs):
    """Canonical order: rows top to bottom, PREDICT items in schema order,
    one vector per NaN cell."""
    vectors = []
    row_pos = {r: i for i, r in enumerate(P.test)}
    values = frame[P.targets].to_numpy(object)
    for r in range(len(frame)):
        for j, item in enumerate(P.targets):
            if pd.isna(values[r, j]):
                p = np.clip(probs[row_pos[r], j, :P.K[j]], 1e-6, None)
                vectors.append((p / p.sum()).tolist())
    return vectors
