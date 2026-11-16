"""VERITIQ insights: turns flagged charges + the cleaned ledger into a plain-English summary, a prioritised fix list,
spend breakdowns and rule-based answers. No AI model needed: every number comes from code, so it always works offline.
(The optional LLM in pro.py can reword or answer extra questions, but never produces the numbers.)"""
import html, math, re
import numpy as np, pandas as pd
from veritiq import recommendation

TYPE_LABEL = {"duplicate": "Duplicate charges", "price_hike": "Price hikes", "spike": "Unusual spikes",
              "other": "Other unusual", "new_vendor": "New vendor, large"}
PRI_RANK = {"High": 0, "Medium": 1, "Low": 2}
NOISY_RATE = 0.25          # if more than 25% of charges are flagged, the flags are not useful

def _m(x): return f"${x:,.2f}"
def _m0(x): return f"${x:,.0f}"
def _e(x): return html.escape(str(x))

# ---------------------------------------------------------------- analysis
def _priority(impact, amount, t):
    if t == "new_vendor": return "Medium" if amount >= 500 else "Low"
    return "High" if impact >= 500 else "Medium" if impact >= 100 else "Low"

def _suggest_threshold(scores, n_charges, target_rate=0.12):
    """Threshold that would flag roughly `target_rate` of the charges (never below the current one)."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    k = min(len(s) - 1, max(0, math.ceil(target_rate * n_charges) - 1))
    return float(min(0.9, round(s[k] + 0.005, 2)))

def spend_stats(clean):
    total = float(clean.amount.sum())
    g = clean.groupby("vendor").amount.agg(["sum", "count", "mean", "std", "median"]).fillna(0).sort_values("sum", ascending=False)
    months = max(1, int(clean.date.dt.to_period("M").nunique()))
    top = [(v, float(r["sum"]), float(r["sum"]) / total) for v, r in g.head(6).iterrows()]
    steady = g[(g["count"] >= 3) & (g["std"] / g["mean"].replace(0, np.nan) < 0.15)]
    rec = [(v, float(r["sum"]) / months) for v, r in steady.sort_values("sum", ascending=False).iterrows()]
    mt = clean.groupby(clean.date.dt.to_period("M")).amount.sum()
    trend = None
    if len(mt) >= 6:
        recent, earlier = float(mt.iloc[-3:].mean()), float(mt.iloc[:-3].mean())
        trend = (recent / earlier - 1) if earlier else None
    return {"total": total, "months": months, "monthly_avg": total / months, "top": top, "n_vendors": int(len(g)),
            "top3_share": float(sum(t[2] for t in top[:3])), "recurring": rec,
            "recurring_monthly": float(sum(r[1] for r in rec)), "trend": trend}

def _new_vendor_watch(clean, flagged_keys):
    """A vendor's very first charge has no history, so the model cannot judge it. Surface big ones for a manual look."""
    cnt = clean.groupby("vendor").amount.transform("count"); med = float(clean.amount.median())
    c = clean[(cnt == 1) & (clean.amount >= max(3 * med, 500))]
    return [r for r in c.itertuples() if (r.vendor, round(float(r.amount), 2)) not in flagged_keys]

def analyze(flags, clean, threshold=0.2, src=None):
    n = len(clean); nf = len(flags)
    a = {"n_charges": n, "n_flagged": nf, "flag_rate": nf / n if n else 0.0, "threshold": float(threshold), "src": src,
         "warnings": list(flags.attrs.get("warnings", []))}
    a["spend"] = spend_stats(clean)
    by = {}
    for t, g in flags.groupby("type"): by[t] = {"n": int(len(g)), "impact": float(g.est_impact.sum())}
    a["by_type"] = by
    a["total_at_risk"] = float(flags.est_impact.sum()) if nf else 0.0
    fixes = []
    for _, r in flags.iterrows():
        imp = float(r.est_impact); amt = float(r.amount)
        fixes.append({"vendor": r.vendor, "date": str(r.date)[:10], "amount": amt, "type": r.type, "impact": imp,
                      "score": float(r.score), "priority": _priority(imp, amt, r.type), "action": recommendation(r), "reason": r.reason})
    keys = {(f["vendor"], round(f["amount"], 2)) for f in fixes}
    nv = _new_vendor_watch(clean, keys); a["new_vendor_watch"] = len(nv)
    for r in nv:
        amt = float(r.amount)
        fixes.append({"vendor": r.vendor, "date": str(r.date)[:10], "amount": amt, "type": "new_vendor", "impact": 0.0, "score": 0.0,
                      "priority": _priority(0, amt, "new_vendor"),
                      "action": f"First charge from {r.vendor} and it is large. Confirm you ordered it and keep the invoice.",
                      "reason": "New vendor with no history to compare against (not scored by the model)"})
    fixes.sort(key=lambda f: (PRI_RANK[f["priority"]], -f["impact"], -f["amount"]))
    a["fixes"] = fixes
    vend = {}
    for f in fixes: vend[f["vendor"]] = vend.get(f["vendor"], 0.0) + f["impact"]
    a["top_vendors"] = sorted(vend.items(), key=lambda kv: -kv[1])[:5]
    a["suggested_threshold"] = None
    if a["flag_rate"] > NOISY_RATE:
        a["status"] = "noisy"
        a["suggested_threshold"] = _suggest_threshold(flags.score, n)
        if a["suggested_threshold"] <= a["threshold"]: a["suggested_threshold"] = min(0.9, round(a["threshold"] + 0.15, 2))
    elif nf == 0 and not nv: a["status"] = "clean"
    else: a["status"] = "attention"
    return a

# ---------------------------------------------------------------- text
def headline(a):
    if a["status"] == "noisy":
        return (f"Too many flags to trust: {a['n_flagged']} of {a['n_charges']} charges ({a['flag_rate']:.0%}) were flagged. "
                f"Raise the sensitivity threshold to about {a['suggested_threshold']:.2f} and re-check.")
    if a["status"] == "clean": return "No suspicious charges found in this statement."
    return (f"{a['n_flagged']} charge(s) need a look, with about {_m0(a['total_at_risk'])} at stake (rough estimate)."
            + (f" Plus {a['new_vendor_watch']} large first-time charge(s) to check by hand." if a["new_vendor_watch"] else ""))

def problems(a):
    out, bt = [], a["by_type"]
    if "duplicate" in bt: out.append(f"{bt['duplicate']['n']} possible duplicate charge(s), about {_m(bt['duplicate']['impact'])} you may be able to get back.")
    if "price_hike" in bt: out.append(f"{bt['price_hike']['n']} subscription price increase(s): roughly {_m(bt['price_hike']['impact'])} more per year if the new price stays.")
    if "spike" in bt: out.append(f"{bt['spike']['n']} unusually large charge(s), about {_m(bt['spike']['impact'])} above normal spending.")
    if "other" in bt: out.append(f"{bt['other']['n']} charge(s) look odd but cannot be given a dollar estimate.")
    if a["new_vendor_watch"]: out.append(f"{a['new_vendor_watch']} large charge(s) from a vendor seen only once. The model cannot judge these, so check them by hand.")
    if a["status"] == "noisy": out.append(f"Flag rate is {a['flag_rate']:.0%}, which is far too high. Most flags are probably false alarms.")
    return out

def observations(a):
    s, out = a["spend"], []
    top = s["top"][0]
    out.append(f"Your biggest cost is {top[0]}: {_m0(top[1])} ({top[2]:.0%} of all spending). The top 3 vendors make up {s['top3_share']:.0%}.")
    if s["recurring"]:
        out.append(f"{len(s['recurring'])} steady recurring charge(s) total about {_m0(s['recurring_monthly'])}/month. Worth a yearly look for unused or cheaper plans.")
    if s["trend"] is not None and abs(s["trend"]) >= 0.1:
        out.append(f"Spending over the last 3 months is {abs(s['trend']):.0%} {'higher' if s['trend'] > 0 else 'lower'} than earlier months.")
    return out

def report_md(a):
    L = ["### Summary", headline(a), ""]
    p = problems(a)
    if p: L += ["### Problems found"] + [f"- {x}" for x in p] + [""]
    if a["fixes"]:
        L += ["### What to fix first (biggest losses at the top)"]
        for i, f in enumerate(a["fixes"][:6], 1):
            money = f"est. {_m(f['impact'])}" if f["impact"] else f"{_m(f['amount'])} charge"
            L.append(f"{i}. **{f['vendor']}** ({f['date']}, {_m(f['amount'])}) [{f['priority']}]: {f['action']} _{money}_")
        L.append("")
    L += ["### Where your money goes"] + [f"- {x}" for x in observations(a)] + [""]
    if a["warnings"]: L += ["### Data notes"] + [f"- {w}" for w in a["warnings"]] + [""]
    L.append("_Prototype, not financial advice. Estimates are rough and not guaranteed. Flags are leads to review, not verdicts._")
    return "\n".join(L)

def llm_context(a):
    L = [f"- {f['date']} {f['vendor']} ${f['amount']:,.2f} ({f['type']}, priority {f['priority']}): {f['reason']}" for f in a["fixes"][:15]]
    s = a["spend"]
    L.append(f"Spend facts: {a['n_charges']} charges, total ${s['total']:,.2f} over {s['months']} month(s). Top vendors: "
             + "; ".join(f"{v} ${x:,.0f}" for v, x, _ in s["top"][:5]) + f". Recurring about ${s['recurring_monthly']:,.0f}/month.")
    return "\n".join(L)

def lacking_md(a):
    """Answer to 'where are we lacking / where should we put resources': problems, then where attention or money should go."""
    L = ["### Where you are losing money or attention"] + [f"- {x}" for x in (problems(a) or ["No problems found in the flagged charges."])] + ["", "### Where to put resources"]
    d = a["by_type"]
    if "duplicate" in d: L.append(f"- **Billing checks:** recover about {_m(d['duplicate']['impact'])} in possible duplicates, then have someone review each statement for double billing.")
    if "price_hike" in d: L.append(f"- **Vendor negotiation:** re-check the plans behind {d['price_hike']['n']} price increase(s) (about {_m(d['price_hike']['impact'])}/year).")
    if "spike" in d or a["new_vendor_watch"]: L.append("- **Purchase approval:** large one-off and first-time charges are where unplanned spending enters. Get a receipt for each.")
    s = a["spend"]
    if s["recurring"]: L.append(f"- **Subscription audit:** {len(s['recurring'])} recurring charges cost about {_m0(s['recurring_monthly'])}/month. Cancel or downgrade any that are unused.")
    L += [f"- {x}" for x in observations(a)]
    L += ["", "_These are leads based only on your charges. VERITIQ cannot see revenue, headcount or goals, so treat this as a starting point, not advice._"]
    return "\n".join(L)

def answer(q, a):
    """Rule-based answers to the common questions. Returns markdown."""
    t = (q or "").lower()
    if a is None: return "Check a statement first: attach a CSV or type **demo**."
    has = lambda *w: any(re.search(r"\b" + re.escape(x), t) for x in w)
    fx = a["fixes"]
    if has("summary", "overview", "how is", "how are", "everything"): return report_md(a)
    if has("lacking", "allocat", "analy", "resource", "better"): return lacking_md(a)
    if has("spend", "vendor", "allocat", "where", "cost", "resource", "lacking"):
        return "\n".join(f"- {x}" for x in observations(a))
    if has("first", "matter", "priorit", "important", "biggest", "worst"):
        if not fx: return "Nothing is flagged, so there is nothing urgent."
        f = fx[0]; return f"Start with **{f['vendor']}** ({f['date']}, {_m(f['amount'])}): {f['action']} Estimated impact {_m(f['impact'])}."
    if has("next", "do", "fix", "step", "action") and not has("does"):
        if not fx: return "Nothing to fix right now."
        return "Do these in order:\n" + "\n".join(f"{i}. {f['vendor']}: {f['action']}" for i, f in enumerate(fx[:4], 1))
    if has("total", "how much", "lose", "loss", "leak", "save", "money at"):
        parts = ", ".join(f"{TYPE_LABEL.get(k, k)}: {_m(v['impact'])}" for k, v in a["by_type"].items() if v["impact"])
        return f"About **{_m(a['total_at_risk'])}** is at stake (rough estimate, not guaranteed). {parts}"
    if has("duplicate", "twice", "double"):
        d = [f for f in fx if f["type"] == "duplicate"]
        return ("Possible duplicates:\n" + "\n".join(f"- {f['vendor']} {f['date']} {_m(f['amount'])}" for f in d)) if d else "No duplicates flagged."
    if has("price", "hike", "subscription", "increase"):
        d = [f for f in fx if f["type"] == "price_hike"]
        return ("Price increases:\n" + "\n".join(f"- {f['vendor']} {f['date']}: {_m(f['amount'])}, about {_m(f['impact'])}/year extra" for f in d)) if d else "No price hikes flagged."
    if has("spike", "large", "unusual", "big"):
        d = [f for f in fx if f["type"] in ("spike", "new_vendor")]
        return ("Large or unusual charges:\n" + "\n".join(f"- {f['vendor']} {f['date']} {_m(f['amount'])}" for f in d)) if d else "No unusual spikes flagged."
    if has("sensitiv", "false", "noisy", "too many", "threshold"):
        if a["status"] == "noisy": return f"Yes, {a['flag_rate']:.0%} flagged is too many. Use **Apply suggested sensitivity** to re-check at {a['suggested_threshold']:.2f}."
        return f"{a['flag_rate']:.0%} of charges are flagged at threshold {a['threshold']:.2f}, which is a reasonable level. Raise it to see fewer, stronger flags."
    return ("I can answer from your flagged charges: try **summary**, **which flag matters most**, **what should I do next**, "
            "**how much am I losing**, **duplicates**, **price hikes**, or **where does my money go**. "
            "Turn on the AI engine in Setup for open-ended questions.")

# ---------------------------------------------------------------- HTML panels (styled by app.py)
PILL = {"High": "hi", "Medium": "md", "Low": "lo"}

def kpi_html(a=None):
    if a is None: v = ["-", "-", "-", "-"]
    else:
        v = [f"{a['n_charges']:,}", f"{a['n_flagged']:,}", _m0(a["total_at_risk"]), _m0(a["spend"]["recurring_monthly"])]
    lab = ["Charges checked", "Flagged for review", "Estimated at stake", "Recurring / month"]
    sub = ["after cleaning", "" if a is None else f"{a['flag_rate']:.0%} of charges", "rough estimate", "steady subscriptions"]
    return '<div class="kpis">' + "".join(f'<div class="kpi"><div class="kl">{l}</div><div class="kv">{x}</div><div class="ks">{s}</div></div>'
                                          for l, x, s in zip(lab, v, sub)) + "</div>"

def quick_summary_html(a=None):
    if a is None:
        return ('<div class="qs"><div class="qh"><span class="pill lo">Waiting</span><b>Quick summary</b></div>'
                '<p class="qm">Attach a CSV or type <b>demo</b>. Your summary, problems and what to fix first will show up here.</p></div>')
    cls = {"noisy": "md", "clean": "ok", "attention": "hi"}[a["status"]]
    lab = {"noisy": "Check sensitivity", "clean": "Looks clean", "attention": "Needs attention"}[a["status"]]
    p = "".join(f"<li>{_e(x)}</li>" for x in problems(a)[:4]) or "<li>No problems found.</li>"
    first = ""
    if a["fixes"]:
        f = a["fixes"][0]
        first = f'<div class="qf"><span>Do this first</span><b>{_e(f["vendor"])}</b> {_m(f["amount"])}: {_e(f["action"])}</div>'
    return (f'<div class="qs"><div class="qh"><span class="pill {cls}">{lab}</span><b>Quick summary</b></div>'
            f'<p class="qm">{_e(headline(a))}</p><ul>{p}</ul>{first}'
            '<p class="tiny">Open the AI Assistant (top right) for the full report and to ask questions.</p></div>')

def fixes_html(a=None, limit=8):
    if a is None or not a["fixes"]:
        msg = "Your prioritised fix list will appear here after a check." if a is None else "Nothing to fix right now."
        return f'<div class="vpanel"><div class="ph">What to fix</div><p class="tiny">{msg}</p></div>'
    rows = []
    for f in a["fixes"][:limit]:
        money = f"est. {_m(f['impact'])}" if f["impact"] else "amount to verify"
        rows.append(f'<div class="fx"><span class="pill {PILL[f["priority"]]}">{f["priority"]}</span>'
                    f'<div class="fb"><b>{_e(f["vendor"])}</b> <span class="m">{_e(f["date"])} · {_m(f["amount"])} · {TYPE_LABEL.get(f["type"], f["type"])}</span>'
                    f'<div>{_e(f["action"])}</div></div><div class="fm">{money}</div></div>')
    more = f'<p class="tiny">Showing {limit} of {len(a["fixes"])}. See the full table below.</p>' if len(a["fixes"]) > limit else ""
    return f'<div class="vpanel"><div class="ph">What to fix, biggest losses first</div>{"".join(rows)}{more}</div>'

def _bars(items, title, color):
    if not items: return ""
    mx = max(v for _, v, _ in items) or 1
    rows = "".join(f'<div class="br"><div class="bl">{_e(n)}</div><div class="bt"><div class="bf" style="width:{max(2, v / mx * 100):.0f}%;background:{color}"></div></div>'
                   f'<div class="bv">{lbl}</div></div>' for n, v, lbl in items)
    return f'<div class="ph">{title}</div>{rows}'

def charts_html(a=None):
    if a is None:
        return '<div class="vpanel"><div class="ph">Charts</div><p class="tiny">Money at stake by type and spending by vendor will appear here.</p></div>'
    t = [(TYPE_LABEL.get(k, k), v["impact"], _m0(v["impact"])) for k, v in sorted(a["by_type"].items(), key=lambda kv: -kv[1]["impact"]) if v["impact"] > 0]
    s = [(n, x, f"{_m0(x)} · {sh:.0%}") for n, x, sh in a["spend"]["top"]]
    return '<div class="vpanel">' + (_bars(t, "Estimated money at stake, by problem type", "#f59e0b") if t else "") + _bars(s, "Where your money goes (top vendors)", "#8b5cf6") + "</div>"
