"""Build realistic proxy datasets (official-kit format) from public survey microdata.

    python proxy/build_proxies.py [--only NAME ...]

Downloads into work/proxy_raw/ (re-uses files already there) and writes
work/proxy/<name>/{respondents.parquet,schema.json}. Downloaded files are only
ever read as data (parquet via pyarrow); nothing in them is executed.

Sources
  afro_*  : Afrobarometer rounds 6/8/9, HF repos electricsheepafrica/africa-afrobarometer-<country>
            (CC-BY-4.0 repackaging of public Afrobarometer microdata)
  eph_arg : INDEC Encuesta Permanente de Hogares (individual), HF Matiasdn/eph-microdata-parquet
"""
import argparse, glob, json, os, re, sys, urllib.error, urllib.request
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "work", "proxy_raw")
OUT = os.path.join(ROOT, "work", "proxy")
GATED = "NA_GATED"
MAX_LEVELS = 40
HF = "https://huggingface.co"


# ---------------------------------------------------------------- download --
def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as fh:
                fh.write(r.read())
            os.replace(dest + ".part", dest)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # missing round for a country is normal
                return False
        except Exception as exc:
            print("  retry", url.split("/")[-1], exc)
    print("  FAILED", url)
    return False


def afro_repos():
    cache = os.path.join(RAW, "afro", "repos.txt")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with urllib.request.urlopen(HF + "/api/datasets?search=africa-afrobarometer&limit=100") as r:
            names = [d["id"].split("/")[1] for d in json.load(r)]
        open(cache, "w").write("\n".join(names))
    return open(cache).read().split()


def afro_download(rd):
    for repo in afro_repos():
        fetch("%s/datasets/electricsheepafrica/%s/resolve/main/data/round_%d-00000-of-00001.parquet" % (HF, repo, rd),
              os.path.join(RAW, "afro", "%s_r%d.parquet" % (repo, rd)))


def afro_load(rd):
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, "afro", "*_r%d.parquet" % rd))):
        d = pd.read_parquet(f)
        d = d.loc[:, d.notna().any()]
        up = [c.upper() for c in d.columns]
        d.columns = up
        if len(set(up)) != len(up):  # same variable under several case variants
            cols = {}
            for c in dict.fromkeys(up):
                idx = [i for i, x in enumerate(up) if x == c]
                s = d.iloc[:, idx[0]]
                for i in idx[1:]:
                    s = s.where(s.notna(), d.iloc[:, i])
                cols[c] = s
            d = pd.DataFrame(cols)
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    # Same country can appear under two repo names (Cabo Verde / Cape Verde ...)
    d = d.drop_duplicates(subset=["RESPNO"]) if "RESPNO" in d else d
    return d


# ------------------------------------------------------------ preparation --
def band_numeric(s, max_bands=10):
    """Numeric series -> ordered string bands (NaN stays NaN)."""
    x = pd.to_numeric(s, errors="coerce")
    ok = x.dropna()
    if ok.empty:
        return pd.Series(np.nan, index=s.index, dtype=object)
    uniq = np.sort(ok.unique())
    if len(uniq) <= 12:
        lab = {u: (str(int(u)) if float(u).is_integer() else str(round(float(u), 4))) for u in uniq}
        return x.map(lab).astype(object)
    out = pd.Series(np.nan, index=s.index, dtype=object)
    zero = x == 0
    pos = x[(x != 0) & x.notna()]
    nb = max_bands - (1 if zero.any() and zero.mean() > 0.08 else 0)
    base = pos if zero.mean() > 0.08 else ok
    qs = np.unique(np.quantile(base, np.linspace(0, 1, nb + 1)[1:-1]))
    edges = [-np.inf] + list(qs) + [np.inf]
    idx = np.searchsorted(qs, x.to_numpy(float), side="right")
    names = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        names.append("%02d. %s" % (i + 1, ("<%g" % hi) if lo == -np.inf else (">=%g" % lo) if hi == np.inf else "%g-<%g" % (lo, hi)))
    lab = np.array(names, dtype=object)[idx]
    out[x.notna()] = lab[x.notna().to_numpy()]
    if zero.mean() > 0.08:
        out[zero] = "00. 0"
    return out


def age_bands(s):
    x = pd.to_numeric(s, errors="coerce")
    bins = [0, 24, 29, 34, 39, 44, 49, 54, 64, 200]
    labs = ["<=24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-64", "65+"]
    return pd.cut(x, bins, labels=labs).astype(object)


def natural_key(v, freq):
    m = re.match(r"^\s*(\d+)\b", v)
    return (0, int(m.group(1)), v) if m else (1, -freq.get(v, 0), v)


def find_gate(child_na, parents, codes, nlev):
    """Search a parent column whose value set deterministically opens the child."""
    n = len(child_na)
    best = None
    for p, (c, k) in codes.items():
        cnt = np.bincount(c, minlength=k)
        nas = np.bincount(c, weights=child_na.astype(float), minlength=k)
        rate = np.divide(nas, cnt, out=np.zeros(k), where=cnt > 0)
        closed = (rate >= 0.5)
        viol = np.where(closed, cnt - nas, nas).sum()  # rows that break the rule
        if closed.all() or (~closed).all() or viol > 0.01 * n:
            continue
        # tiny parent groups with an uncertain rate are not evidence; ignore them
        score = viol + 0.01 * k
        if best is None or score < best[0]:
            best = (score, p, closed, viol)
    return best


def finalize(raw, given, predict, questions, na_attempts, rng, min_gate_share=0.005):
    """raw: DataFrame of strings/NaN. Applies NA handling, level caps, gating."""
    items = list(dict.fromkeys(given + predict))
    work = {}
    # 1. NA-aware masks, tentative levels
    na1 = {}
    for c in items:
        s = raw[c].astype(object)
        s = s.where(~s.isin(["", "nan", "None"]), np.nan)
        work[c] = s
        na1[c] = na_attempts[0](s)
    # 2. parents: columns that are (almost) never NA
    codes, nlev = {}, {}
    for c in items:
        if na1[c].mean() < 0.002:
            f, u = pd.factorize(work[c].fillna("Missing"))
            if 2 <= len(u) <= 60:
                codes[c] = (f, len(u))
    gates = {}
    for c in items:
        if c in given:
            continue
        for att in na_attempts:
            m = att(work[c])
            if not (min_gate_share <= m.mean() <= 0.99):
                continue
            cand = {p: v for p, v in codes.items() if p != c}
            b = find_gate(m.to_numpy(), None, cand, None)
            if b is None:
                continue
            _, p, closed, _ = b
            vals = pd.factorize(work[p].fillna("Missing"))[1]
            gates[c] = (p, att, [str(vals[i]) for i in np.where(~closed)[0]])
            break
    # 3. build columns
    out, schema_items = {}, {}
    for c in items:
        s = work[c].copy()
        if c in gates:
            p, att, _ = gates[c]
            s = s.where(~att(s), np.nan)  # gated cells -> NaN, filled below
            gated_mask = att(work[c]) 
        s = s.fillna("Missing") if c not in gates else s
        out[c] = s
    for c in items:
        s = out[c]
        if c in gates:
            p, att, obs = gates[c]
            gm = ~out[p].isin(obs)  # rows whose gate is closed
            vis = s.where(~gm)
            vis = vis.fillna("Missing")  # open gate but blank -> ordinary non-response
            vc = vis[~gm].value_counts()
        else:
            vis, gm = s, pd.Series(False, index=s.index)
            vc = s.value_counts()
        if len(vc) > MAX_LEVELS:
            keep = set(vc.index[:MAX_LEVELS - 1])
            vis = vis.where(vis.isin(keep) | gm, "Other")
            vc = vis[~gm].value_counts()
        out[c] = vis.where(~gm, GATED)
        freq = vc.to_dict()
        vals = sorted(vc.index, key=lambda v: natural_key(v, freq))
        schema_items[c] = (vals, gates.get(c), gm)
    frame = pd.DataFrame(out)
    return frame, schema_items, gates


def build(name, raw, given, predict, questions, desc, na_attempts, seed=0, max_rows=None, min_gate_share=0.005):
    rng = np.random.default_rng([seed, 17])
    perm = rng.permutation(len(raw))
    if max_rows:
        perm = perm[:max_rows]
    raw = raw.iloc[perm].reset_index(drop=True)
    frame, sitems, gates = finalize(raw, given, predict, questions, na_attempts, rng, min_gate_share)
    # drop degenerate PREDICT items (single level / >99.5% one level among visible cells)
    keep_pred = []
    for c in predict:
        vals, g, gm = sitems[c]
        vis = frame[c][~gm] if g else frame[c]
        if len(vals) >= 2 and vis.value_counts(normalize=True).iloc[0] <= 0.995:
            keep_pred.append(c)
    # a gate parent must stay in the schema
    needed = {gates[c][0] for c in keep_pred if c in gates}
    for p in needed:
        if p not in keep_pred and p not in given:
            keep_pred.append(p)
    predict = [c for c in predict if c in keep_pred] + [p for p in keep_pred if p not in predict]
    n = len(frame)
    n_dev, n_test = int(round(0.06 * n)), int(round(0.14 * n))
    n_train = n - n_dev - n_test
    schema = {"dataset": {"n_rows": n, "version": "proxy-1.0", "description": desc},
              "split": {"n_train": n_train, "n_dev": n_dev, "n_test": n_test},
              "items": {}}
    for c in given + predict:
        vals, g, gm = sitems[c]
        rec = {"question": questions.get(c, c), "class": "GIVEN" if c in given else "PREDICT",
               "values": vals, "gate": None}
        if g and c in predict:
            rec["gate"] = {"parent": g[0].lower(), "observed_if": g[2]}
        schema["items"][c.lower()] = rec
    # drop gates whose parent vanished
    cols = {c.lower(): frame[c] for c in given + predict}
    fr = pd.DataFrame(cols)
    fr.insert(0, "respondent_id", ["R%06d" % i for i in range(1, n + 1)])
    fr["role"] = np.repeat(np.array(["TRAIN", "DEV", "TEST"], dtype=object), [n_train, n_dev, n_test])
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    fr.astype(object).to_parquet(os.path.join(d, "respondents.parquet"), index=False)
    with open(os.path.join(d, "schema.json"), "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    ng = sum(1 for r in schema["items"].values() if r["gate"])
    print("%s: n=%d GIVEN=%d PREDICT=%d gated=%d" % (name, n, len(given), len(predict), ng))


# --------------------------------------------------------------- datasets --
AFRO_NA1 = lambda s: s.isna() | s.astype(str).str.match(r"(?i)^not (asked|applicable)")
AFRO_NA2 = lambda s: s.astype(str).str.match(r"(?i)^not asked in")
AFRO_DESC = ("Proxy built from Afrobarometer round %d public-opinion survey microdata (%s). One row per "
             "respondent; GIVEN = country, locality type, gender, age band, education, work status, housing; "
             "PREDICT = attitude, trust, governance, living-condition and behaviour items. Item text is the "
             "variable name only (the per-round question wording is not reliably aligned in the source copy). "
             "'Not asked in this country' / 'Not applicable' cells are the gate sentinel, with the gate parent "
             "recovered from the data.")


def afro(name, rd, n_countries=None, max_rows=None, n_pred=120, given_cols=None, seed=0):
    afro_download(rd)
    d = afro_load(rd)
    rng = np.random.default_rng([seed, rd])
    if n_countries:
        cs = sorted(d["COUNTRY"].dropna().unique())
        pick = list(rng.choice(cs, n_countries, replace=False))
        d = d[d["COUNTRY"].isin(pick)]
    d = d.copy()
    d["AGE"] = age_bands(d["Q1"])
    gmap = {"COUNTRY": "Country", "URBRUR": "Urban or rural primary sampling unit",
            "THISINT": "Respondent gender", "AGE": "Respondent age band", "Q97": "Education level",
            "Q95A": "Employment status", "Q104": "Type of dwelling", "Q93A": "Main source of drinking water"}
    given = [c for c in (given_cols or gmap) if c in d.columns]
    bad = re.compile(r"^(Q1|Q2|Q9[7-9]|Q1[0-9][0-9]|Q9[1-6]A?|COUNTRY.*|NOCALL.*)$")
    cand = [c for c in d.columns if re.match(r"^(Q\d+[A-Z]?(_\d+)?|COV\d+)$", c) and c not in given
            and not bad.match(c) and c not in ("Q95A", "Q93A", "Q104")]
    cand += [c for c in ("Q91A", "Q92A", "Q92B", "Q92C", "Q96A", "Q98B", "Q94A", "Q94B", "Q95B", "Q93B", "Q106", "Q108")
             if c in d.columns and c not in given]
    cand = [c for c in dict.fromkeys(cand) if d[c].nunique() <= 60 and d[c].notna().mean() > 0.9]
    cand = [c for c in cand if d[c].nunique() >= 2]
    if len(cand) > n_pred:
        cand = sorted(rng.choice(cand, n_pred, replace=False), key=cand.index)
    raw = d[given + cand].copy()
    for c in given + cand:
        raw[c] = raw[c].astype(object).where(raw[c].notna(), np.nan)
    qs = {c: gmap.get(c, "Afrobarometer round %d item %s" % (rd, c)) for c in given + cand}
    yrs = {6: "2014-15", 7: "2016-18", 8: "2019-21", 9: "2021-23"}[rd]
    build(name, raw, given, cand, qs, AFRO_DESC % (rd, yrs), [AFRO_NA1, AFRO_NA2], seed, max_rows)


def eph_hogar(name="eph_arg_hh", quarters=(("2023", "t4"), ("2023", "t3"), ("2022", "t4")), max_rows=25000, seed=3):
    parts = []
    for y, t in quarters:
        dest = os.path.join(RAW, "eph", "hogar_%s_%s.parquet" % (y, t))
        fetch("%s/datasets/Matiasdn/eph-microdata-parquet/resolve/main/data/hogar_%s_%s.parquet" % (HF, y, t), dest)
        parts.append(pd.read_parquet(dest))
    d = pd.concat(parts, ignore_index=True)
    d["QUARTER"] = d["ANO4"].astype(str) + d["TRIMESTRE"].astype(str)
    given = ["REGION", "AGLOMERADO", "MAS_500", "IX_TOT", "IX_MEN10", "QUARTER"]
    skip = re.compile(r"^(CODUSU|ANO4|TRIMESTRE|NRO_HOGAR|REALIZADA|PONDERA|PONDIH|.*_ESP|[IRGPA]DEC.*)$")
    cand = [c for c in d.columns if c not in given and not skip.match(c) and d[c].nunique() > 1]
    raw = pd.DataFrame({c: (band_numeric(d[c]) if pd.api.types.is_numeric_dtype(d[c]) else d[c].astype(object))
                        for c in given + cand})
    raw["QUARTER"] = d["QUARTER"].astype(object)
    cand = [c for c in cand if raw[c].nunique() <= 60][:150]
    zero_or_na = lambda s: s.isna() | s.isin(["0", "00. 0"])
    desc = ("Proxy built from INDEC Argentina Encuesta Permanente de Hogares, household file, 2022 Q4 + 2023 Q3/Q4 "
            "(public microdata). One row per household; GIVEN = region, agglomeration, size-of-city flag, household "
            "size, number of children; PREDICT = dwelling quality, tenure, sanitation, household income-source "
            "indicators, total / per-capita income bands and deciles. Numeric variables banded (<=10 bands). "
            "Blank / zero 'not applicable' cells are the gate sentinel where the routing parent is recoverable.")
    qs = {c: "EPH household variable %s" % c for c in given + cand}
    build(name, raw, given, cand, qs, desc, [zero_or_na, lambda s: s.isna()], seed, max_rows)


def eph(name="eph_arg", quarters=(("2023", "t4"), ("2023", "t3")), max_rows=30000, seed=0):
    parts = []
    for y, t in quarters:
        dest = os.path.join(RAW, "eph", "individual_%s_%s.parquet" % (y, t))
        fetch("%s/datasets/Matiasdn/eph-microdata-parquet/resolve/main/data/individual_%s_%s.parquet" % (HF, y, t), dest)
        parts.append(pd.read_parquet(dest))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["CH06"] >= 0].copy()
    d["AGE"] = age_bands(d["CH06"])
    d["QUARTER"] = d["TRIMESTRE"].astype(str)
    given = ["REGION", "AGLOMERADO", "MAS_500", "CH04", "AGE", "CH03", "CH07", "NIVEL_ED", "QUARTER"]
    skip = re.compile(r"^(CODUSU|ANO4|TRIMESTRE|NRO_HOGAR|COMPONENTE|PONDERA|PONDII|PONDIH|PONDIIO|CH05|.*_COD|.*_ESP|"
                      r"PP04B3_.*|PP05B2_.*|PP11B2_.*|PP11G_.*|PP04D_COD|IMPUTA|H15|P_.*|ID?EC.*|[IRGPA]DEC.*)$")
    cand = [c for c in d.columns if c not in given and c != "AGE" and not skip.match(c)
            and d[c].nunique() > 1]
    raw = pd.DataFrame(index=d.index)
    for c in given + cand:
        if c in ("AGE", "QUARTER"):
            raw[c] = d[c].astype(object)
        elif pd.api.types.is_numeric_dtype(d[c]):
            raw[c] = band_numeric(d[c])
        else:
            raw[c] = d[c].astype(object)
    # drop high-cardinality leftovers
    cand = [c for c in cand if raw[c].nunique() <= 60]
    cand = cand[:150]
    zero_or_na = lambda s: s.isna() | s.isin(["0", "00. 0"])
    na_only = lambda s: s.isna()
    desc = ("Proxy built from INDEC Argentina Encuesta Permanente de Hogares, individual file, 2023 Q3+Q4 "
            "(public microdata). One row per person; GIVEN = region, agglomeration, sex, age band, relation to head, "
            "marital status, education level; PREDICT = labour-force status, occupation module (asked only of the "
            "employed / unemployed / inactive), income bands and household-income deciles. Numeric variables are "
            "banded (<=10 bands). Blank / zero 'not applicable' cells are the gate sentinel where the routing "
            "parent can be recovered from the data.")
    qs = {c: "EPH variable %s" % c for c in given + cand}
    build(name, raw, given, cand, qs, desc, [zero_or_na, na_only], seed, max_rows)


DATASETS = {
    "afro_r8_multi": lambda: afro("afro_r8_multi", 8, max_rows=30000, n_pred=120),
    "afro_r6_multi": lambda: afro("afro_r6_multi", 6, max_rows=12000, n_pred=60,
                                  given_cols=["COUNTRY", "URBRUR", "THISINT", "AGE"], seed=1),
    "afro_r9_small": lambda: afro("afro_r9_small", 9, n_countries=6, max_rows=8000, n_pred=40,
                                  given_cols=["COUNTRY", "URBRUR", "THISINT", "AGE", "Q97", "Q95A", "Q104", "Q93A"], seed=2),
    "eph_arg": lambda: eph(),
    "eph_arg_hh": lambda: eph_hogar(),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for k, f in DATASETS.items():
        if not a.only or k in a.only:
            f()
