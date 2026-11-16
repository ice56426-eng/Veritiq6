"""VERITIQ: spending-anomaly detector for small businesses.
Prototype, not financial advice. Run: python veritiq.py   (needs pandas numpy scikit-learn matplotlib)"""
import difflib, html, json, re
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, average_precision_score

START = pd.Timestamp(2025, 1, 1)
SUBS = {"Slack": 480, "AWS": 3200, "Zoom": 150, "Adobe": 360, "Notion": 200,
        "HubSpot": 1100, "Google Workspace": 540, "Datadog": 900}
ONEOFF = ["Office Depot", "Delta Air", "Upwork Contractor", "Staples", "Uber", "Catering Co"]
FEATS = ["prior_n", "ratio_median", "rel_diff_prev", "days_since_prev", "is_sub", "amount"]

# ---------- 1. Synthetic data (labels injected by us) ----------
def make_data(seed=42, months=24):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        for v, amt in SUBS.items():
            noise = 0.12 if v == "AWS" else 0.0          # AWS is usage-based, so it varies
            rows.append((START + pd.DateOffset(months=m, days=int(rng.integers(0, 3))),
                         v, round(amt * (1 + rng.normal(0, noise)), 2), "subscription", 0, "normal"))
        for _ in range(int(rng.integers(35, 50))):
            rows.append((START + pd.DateOffset(months=m, days=int(rng.integers(0, 28))),
                         rng.choice(ONEOFF), round(float(rng.lognormal(5.3, 0.6)), 2), "one-off", 0, "normal"))
    df = pd.DataFrame(rows, columns=["date", "vendor", "amount", "kind", "is_anomaly", "type"])
    extra = []
    pool = df.index[df.date >= START + pd.DateOffset(months=2)]   # keep early history clean
    for i in rng.choice(pool, size=int(0.12 * len(pool)), replace=False):
        r = df.loc[i]
        t = rng.choice(["duplicate", "price_hike", "spike"])
        if t == "price_hike" and r.kind != "subscription": t = "duplicate"
        if t == "spike" and r.kind != "one-off": t = "duplicate"
        if t == "duplicate":                               # 60% exact, 40% near-exact (fees/tax)
            a = r.amount if rng.random() < 0.6 else round(r.amount * rng.uniform(1.005, 1.03), 2)
            extra.append((r.date + pd.Timedelta(days=int(rng.integers(0, 3))), r.vendor, a, r.kind, 1, "duplicate"))
        else:
            f = rng.uniform(1.25, 1.7) if t == "price_hike" else rng.uniform(4, 8)
            df.loc[i, ["amount", "is_anomaly", "type"]] = [round(r.amount * f, 2), 1, t]
    df = pd.concat([df, pd.DataFrame(extra, columns=df.columns)], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]); df["amount"] = df["amount"].astype(float); df["is_anomaly"] = df["is_anomaly"].astype(int)
    df = df.sort_values(["date", "vendor", "amount"], kind="stable").reset_index(drop=True)
    # Identical pairs cannot be told apart, so the label always goes on the LATER row (fixes swapped-pair errors)
    for _, idx in df[df.duplicated(["date", "vendor", "amount"], keep=False)].groupby(["date", "vendor", "amount"]).groups.items():
        idx = list(idx)
        if (df.loc[idx, "type"] == "duplicate").any():
            df.loc[idx, "is_anomaly"] = 0; df.loc[idx, "type"] = "normal"
            df.loc[idx[-1], "is_anomaly"] = 1; df.loc[idx[-1], "type"] = "duplicate"
    return df

# ---------- 2. Features: each charge only sees the past ----------
def add_features(df):
    df = df.copy(); g = df.groupby("vendor")
    df["prior_n"] = g.cumcount()
    df["prior_median"] = g["amount"].transform(lambda s: s.shift().expanding().median())
    df["prev_amount"] = g["amount"].shift()
    df["days_since_prev"] = g["date"].diff().dt.days.fillna(99)
    df["ratio_median"] = (df.amount / df.prior_median).fillna(1.0)
    df["prior_median"] = df.prior_median.fillna(df.amount)
    df["rel_diff_prev"] = ((df.amount - df.prev_amount).abs() / df.prev_amount).fillna(9.0)
    df["is_sub"] = (df.kind == "subscription").astype(int)
    return df

# ---------- 3. Baselines ----------
def baseline_big(df): return (df.amount > 2000).astype(int)
def baseline_dup(df): return ((df.rel_diff_prev == 0) & (df.days_since_prev <= 3)).astype(int)

# ---------- 4. Messy real-world input: cleaning + vendor-name matching ----------
STOP = {"inc", "llc", "ltd", "co", "corp", "payment", "pos", "purchase", "debit", "card", "us", "usa", "com", "www", "the"}
def _norm(s):
    t = [w for w in re.sub(r"[^a-z ]", " ", str(s).lower()).split() if w not in STOP and len(w) > 1]
    return " ".join(t) or str(s).lower().strip()

def normalize_vendors(names):
    """Merge name variants ('SLACK*1234', 'Slack Technologies Inc') onto one vendor. Limitation: can over-merge (e.g. 'Uber' / 'Uber Eats')."""
    uniq = sorted(set(names)); keys = {n: _norm(n) for n in uniq}; canon, mp = [], {}
    for n in sorted(uniq, key=lambda x: (len(keys[x]), x)):
        k = keys[n]
        hit = next((c for c in canon if k == c or k.startswith(c + " ") or difflib.SequenceMatcher(None, k, c).ratio() >= 0.88), None)
        if hit is None: canon.append(k); hit = k
        mp[n] = hit.title()
    return mp

def clean_transactions(raw):
    """Validate a user CSV. Returns (clean_df, warnings). Raises ValueError with a readable message."""
    df = raw.copy(); df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"date", "vendor", "amount"} - set(df.columns)
    if missing: raise ValueError(f"CSV is missing column(s): {', '.join(sorted(missing))}. Needed: date, vendor, amount.")
    n0 = len(df); w = []
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(r"[$,\s]", "", regex=True), errors="coerce")
    df["vendor"] = df["vendor"].fillna("").map(str).str.strip()
    bad = df.date.isna() | df.amount.isna() | df.vendor.isin(["", "nan"])
    if bad.sum(): w.append(f"Skipped {int(bad.sum())} row(s) with a missing/unreadable date, vendor or amount.")
    df = df[~bad]
    neg = df.amount <= 0
    if neg.sum(): w.append(f"Ignored {int(neg.sum())} refund/zero row(s) (amount <= 0).")
    df = df[~neg]
    if len(df) < 10: raise ValueError(f"Only {len(df)} usable charges out of {n0}. Need at least 10 for the checks to mean anything.")
    mp = normalize_vendors(df.vendor); df["vendor"] = df.vendor.map(mp)
    if len(set(mp.values())) < len(mp): w.append("Some vendor names were treated as the same vendor (e.g. variants of one name).")
    return df[["date", "vendor", "amount"]].reset_index(drop=True), w

# ---------- 5. Explanations, recommendations, and the plain-English version ----------
def classify(r):
    if r.rel_diff_prev < 0.04 and r.days_since_prev <= 3: return "duplicate"
    if r.ratio_median > 1.2 and r.is_sub: return "price_hike"
    if r.ratio_median > 2.5: return "spike"
    return "other"

def explain(r):
    t = classify(r)
    if t == "duplicate": return f"Possible duplicate: same vendor, ~same amount as {int(r.days_since_prev)} day(s) ago"
    if t == "price_hike": return f"Subscription price is {r.ratio_median:.2f}x its usual amount"
    if t == "spike": return f"Charge is {r.ratio_median:.1f}x this vendor's usual amount"
    return "Pattern looks unusual compared with earlier charges"

def est_impact(r):
    """Rough, NOT guaranteed: duplicate = amount recoverable once; price hike = extra cost x 12; spike = excess over usual."""
    t = classify(r)
    if t == "duplicate": return round(float(r.amount), 2)
    if t == "price_hike": return round(float(r.amount - r.prior_median) * 12, 2)
    if t == "spike": return round(float(r.amount - r.prior_median), 2)
    return 0.0

def _t(r): return r.get("type") or classify(r)

def _m(x): return f"${x:,.2f}"

def recommendation(r):
    t, v = _t(r), r.vendor
    return {"duplicate": f"Check whether {v} billed you twice and request a refund of the extra charge.",
            "price_hike": f"Review your {v} plan and ask about the higher price or a cheaper plan.",
            "spike": f"Confirm this {v} purchase was expected and find the receipt."}.get(t, f"Take a quick look at this {v} charge.")

def plain_explanation(r):
    """Small-business-owner wording: what happened, what to do, how sure we are. No jargon, no guarantees."""
    t, v = _t(r), r.vendor
    if t == "duplicate":
        what = f"It looks like {v} may have charged you twice: {_m(r.amount)}, almost the same as a charge {int(r.days_since_prev)} day(s) earlier."
        steps = ["Find both charges on your statement.", f"If you only bought once, contact {v} and ask for the second charge back.", "Keep the receipt or email in case you need to show it."]
        money = f"If it is a mistake, you could get about {_m(r.amount)} back."
    elif t == "price_hike":
        diff = r.amount - r.prior_median
        what = f"Your {v} bill is {_m(r.amount)}, but it is usually about {_m(r.prior_median)}. That is {_m(diff)} more this time."
        steps = ["Look in your email for a price-change notice.", "Check whether you added users or upgraded your plan.", f"If not, contact {v} and ask for the old price, or switch to a cheaper plan."]
        money = f"If the higher price stays, that is roughly {_m(diff * 12)} more per year."
    elif t == "spike":
        what = f"This {v} charge ({_m(r.amount)}) is about {r.ratio_median:.1f} times bigger than your usual {v} charges."
        steps = ["Check whether it was a planned purchase.", "Ask the person who made it for the receipt.", "If nobody recognizes it, call your card company and dispute it."]
        money = f"About {_m(max(r.amount - r.prior_median, 0))} above your normal spending with {v}."
    else:
        what = f"This {v} charge ({_m(r.amount)}) does not match your usual pattern."
        steps = ["Look at the charge on your statement.", "Check it against your receipts."]
        money = "We can't estimate a dollar amount for this one."
    conf = "fairly sure" if r.score >= 0.8 else "worth checking, but not certain"
    return {"what": what, "steps": steps, "money": money,
            "caveat": f"We are {conf}. This is only a tip to check, not proof of a mistake. It is not financial advice."}

def cards_html(flags, limit=15):
    """One card per flagged charge with an 'Explain this simply' button (HTML <details>, no server round trip). All user text is escaped."""
    e = lambda x: html.escape(str(x))
    css = ("<style>.vq{border:1px solid rgba(139,92,246,.45);border-radius:12px;padding:12px 14px;margin:10px 0}"
           ".vq summary{cursor:pointer;display:inline-block;background:#7c3aed;color:#fff;padding:6px 14px;border-radius:8px;font-weight:600;list-style:none}"
           ".vq summary::-webkit-details-marker{display:none}.vq .s{margin-top:10px;padding:10px 12px;border-left:3px solid #a78bfa;background:rgba(139,92,246,.10)}"
           ".vq .m{opacity:.75;font-size:.9em}</style>")
    if len(flags) == 0: return css + "<p>No suspicious charges found.</p>"
    out = [css]
    for _, r in flags.head(limit).iterrows():
        p = plain_explanation(r)
        out.append(f'<div class="vq"><b>{e(r.vendor)}</b> · {_m(r.amount)} · {e(str(r.date)[:10])}'
                   f'<p><b>Recommendation:</b> {e(recommendation(r))}</p>'
                   f'<p class="m">Estimated impact (not guaranteed): {_m(r.est_impact)} · Match score {r.score:.2f}</p>'
                   f'<details><summary>Explain this simply</summary><div class="s"><p>{e(p["what"])}</p><p><b>What to do:</b></p><ol>'
                   + "".join(f"<li>{e(s)}</li>" for s in p["steps"]) + f'</ol><p>{e(p["money"])}</p><p class="m">{e(p["caveat"])}</p></div></details></div>')
    if len(flags) > limit: out.append(f"<p class='m'>Showing the top {limit} of {len(flags)} flags (see the table for all).</p>")
    return "".join(out)

# ---------- 6. Scoring ----------
def train_final_model(seed=42):
    d = add_features(make_data(seed))
    return GradientBoostingClassifier(random_state=0).fit(d[FEATS], d.is_anomaly)

def flag_transactions(raw, model, threshold=0.2):
    """raw: DataFrame(date, vendor, amount). Model is trained on SYNTHETIC data: results on real ledgers are unvalidated."""
    df, warnings = clean_transactions(raw)
    st = df.groupby("vendor")["amount"].agg(["count", "mean", "std", "median"]).fillna(0)
    mad = df.groupby("vendor")["amount"].apply(lambda a: (a - a.median()).abs().median())
    # subscription = steady amounts overall OR steady around the median (so one price hike doesn't hide a subscription)
    subs = st[(st["count"] >= 3) & ((st["std"] / st["mean"] < 0.15) | (1.4826 * mad.reindex(st.index) / st["median"] < 0.1))].index
    df["kind"] = np.where(df.vendor.isin(subs), "subscription", "one-off")
    f = add_features(df.sort_values(["date", "vendor", "amount"], kind="stable").reset_index(drop=True))
    f["score"] = model.predict_proba(f[FEATS])[:, 1]
    out = f[f.score >= threshold].copy()
    out["reason"] = out.apply(explain, axis=1); out["type"] = out.apply(classify, axis=1); out["est_impact"] = out.apply(est_impact, axis=1)
    out = out.sort_values("score", ascending=False)[["date", "vendor", "amount", "score", "type", "reason", "est_impact", "ratio_median", "prior_median", "days_since_prev"]]
    out.attrs["warnings"] = warnings
    return out

# ---------- 7. Evaluation ----------
def _metrics(y, p, s=None):
    d = {"precision": float(precision_score(y, p, zero_division=0)), "recall": float(recall_score(y, p)), "flagged": int(p.sum())}
    if s is not None: d["pr_auc"] = float(average_precision_score(y, s))
    return d

def evaluate(seed=42, months=24, miss_cost=5):
    """Time split: train first half, pick threshold on next quarter, retrain on both, test on the last quarter."""
    df = add_features(make_data(seed, months))
    c1, c2 = START + pd.DateOffset(months=months // 2), START + pd.DateOffset(months=months * 3 // 4)
    tr, va, te = df[df.date < c1], df[(df.date >= c1) & (df.date < c2)], df[df.date >= c2].copy()
    mk = lambda: GradientBoostingClassifier(random_state=0)
    m = mk().fit(tr[FEATS], tr.is_anomaly); pv = m.predict_proba(va[FEATS])[:, 1]; yv = va.is_anomaly.values
    grid = np.linspace(0.05, 0.9, 35)
    cost = [miss_cost * ((yv == 1) & (pv < t)).sum() + ((yv == 0) & (pv >= t)).sum() for t in grid]
    thr = float(grid[int(np.argmin(cost))])
    full = pd.concat([tr, va]); m = mk().fit(full[FEATS], full.is_anomaly)
    te["score"] = m.predict_proba(te[FEATS])[:, 1]; te["pred"] = (te.score >= thr).astype(int)
    y = te.is_anomaly
    res = {"seed": seed, "threshold": thr, "n_test": len(te), "n_anom": int(y.sum()),
           "model": _metrics(y, te.pred, te.score), "base_big": _metrics(y, baseline_big(te)), "base_dup": _metrics(y, baseline_dup(te))}
    return res, te, m

def multi_seed(n=20):
    runs = [evaluate(s)[0] for s in range(n)]; out = {}
    for k in ("model", "base_big", "base_dup"):
        out[k] = {q: [float(np.mean([r[k][q] for r in runs])), float(np.std([r[k][q] for r in runs]))] for q in ("precision", "recall")}
    return out, n

def make_charts(te, model, multi):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ths = np.linspace(0.05, 0.95, 50); y = te.is_anomaly.values
    pr = [precision_score(y, te.score >= t, zero_division=0) for t in ths]; rc = [recall_score(y, te.score >= t) for t in ths]
    plt.figure(figsize=(6, 3.6)); plt.plot(ths, pr, label="Precision", color="#7c3aed"); plt.plot(ths, rc, label="Recall", color="#f59e0b")
    plt.xlabel("Flagging threshold"); plt.ylabel("Score"); plt.title("Precision / recall vs threshold (test set)"); plt.legend(); plt.tight_layout(); plt.savefig("assets/threshold_curve.png", dpi=150); plt.close()
    names = {"base_big": "Charge > $2,000", "base_dup": "Exact duplicates", "model": "VERITIQ model"}; x = np.arange(3)
    plt.figure(figsize=(6, 3.6))
    for j, (q, c) in enumerate([("precision", "#7c3aed"), ("recall", "#f59e0b")]):
        plt.bar(x + j * 0.38, [multi[k][q][0] for k in names], 0.38, yerr=[multi[k][q][1] for k in names], label=q.capitalize(), color=c, capsize=3)
    plt.xticks(x + 0.19, names.values()); plt.ylim(0, 1.05); plt.title("Mean over 20 random datasets (error bars = std)"); plt.legend(); plt.tight_layout(); plt.savefig("assets/model_vs_baselines.png", dpi=150); plt.close()
    imp = sorted(zip(FEATS, model.feature_importances_), key=lambda a: a[1])
    plt.figure(figsize=(6, 3.4)); plt.barh([a[0] for a in imp], [a[1] for a in imp], color="#7c3aed"); plt.title("What the model relies on"); plt.tight_layout(); plt.savefig("assets/feature_importance.png", dpi=150); plt.close()

if __name__ == "__main__":
    res, te, m = evaluate(42)
    print(f"Seed 42 held-out test: {res['n_test']} charges, {res['n_anom']} anomalies, threshold={res['threshold']:.2f}")
    for k, n in (("base_big", "Baseline A: charge > $2,000"), ("base_dup", "Baseline B: exact duplicates"), ("model", "VERITIQ model")):
        d = res[k]; print(f"{n:<30} precision {d['precision']:.2f} recall {d['recall']:.2f} flagged {d['flagged']}")
    bt = {t: {"n": int(len(g)), "model": float(g.pred.mean()), "dup_rule": float(baseline_dup(g).mean())} for t, g in te[te.is_anomaly == 1].groupby("type")}
    for t, d in bt.items(): print(f"  {t:<11} n={d['n']:<3} model {d['model']:.0%}  dup-rule {d['dup_rule']:.0%}")
    multi, n = multi_seed(20)
    print(f"\nMean ± std over {n} random datasets:")
    for k in multi: print(f"  {k:<9} precision {multi[k]['precision'][0]:.2f}±{multi[k]['precision'][1]:.2f}  recall {multi[k]['recall'][0]:.2f}±{multi[k]['recall'][1]:.2f}")
    print("\nMISSED:"); [print(f"  {r.date:%b %d} {r.vendor:<17} ${r.amount:>8,.2f} [{r.type}] ratio={r.ratio_median:.2f}") for _, r in te[(te.is_anomaly == 1) & (te.pred == 0)].head(5).iterrows()]
    print("FALSE ALARMS:"); [print(f"  {r.date:%b %d} {r.vendor:<17} ${r.amount:>8,.2f} ratio={r.ratio_median:.2f}") for _, r in te[(te.is_anomaly == 0) & (te.pred == 1)].head(5).iterrows()]
    make_charts(te, m, multi)
    json.dump({"single": res, "by_type": bt, "multi": multi, "n_seeds": n}, open("results.json", "w"), indent=1)
