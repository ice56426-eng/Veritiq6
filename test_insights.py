import numpy as np, pandas as pd, veritiq as v, insights as ins

MODEL = v.train_final_model()

def _ledger():
    rows = [(f"2025-{m:02d}-05", "Slack", 480) for m in range(1, 13)] + [("2025-12-06", "Slack", 480), ("2025-09-17", "Hobart Service", 2640)]
    rows += [(f"2025-{m:02d}-{d:02d}", "Flour", 300 + 7 * m + d) for m in range(1, 13) for d in (4, 18)]
    return pd.DataFrame(rows, columns=["date", "vendor", "amount"])

def _a(thr=0.2, raw=None):
    raw = _ledger() if raw is None else raw
    return ins.analyze(v.flag_transactions(raw, MODEL, thr), v.clean_transactions(raw)[0], thr)

def test_summary_and_fix_order():
    a = _a(); assert a["status"] == "attention" and a["fixes"]
    ranks = [ins.PRI_RANK[f["priority"]] for f in a["fixes"]]; assert ranks == sorted(ranks)
    assert "Summary" in ins.report_md(a) and "not financial advice" in ins.report_md(a)

def test_new_vendor_watch_catches_big_first_charge():
    a = _a(); assert any(f["type"] == "new_vendor" and f["vendor"].startswith("Hobart") for f in a["fixes"])

def test_noisy_flags_get_a_higher_suggested_threshold():
    a = _a(0.02, pd.DataFrame({"date": (pd.Timestamp("2025-01-01") + pd.to_timedelta(np.random.default_rng(1).integers(0, 300, 500), "D")).strftime("%Y-%m-%d"),
                               "vendor": [f"Vendor {i % 120}" for i in range(500)], "amount": np.random.default_rng(2).lognormal(5, 1, 500).round(2)}))
    assert a["status"] == "noisy" and a["suggested_threshold"] > a["threshold"]

def test_html_panels_escape_vendor_names():
    a = _a(); a["fixes"][0]["vendor"] = "<script>x</script>"
    assert "<script>" not in ins.fixes_html(a) and "<script>" not in ins.quick_summary_html(a)

def test_answers_and_empty_state():
    a = _a(); assert "Start with" in ins.answer("which flag matters most?", a)
    assert "biggest cost" in ins.answer("where does my money go?", a)
    assert "Check a statement" in ins.answer("hi", None)
    for fn in (ins.kpi_html, ins.quick_summary_html, ins.fixes_html, ins.charts_html): assert fn(None)

def test_lacking_prompt_gets_resource_advice():
    a = _a(); r = ins.answer("Analyze my company's dataset and tell me where we are lacking as a company and what we can better allocate its resources to.", a)
    assert "Where to put resources" in r and "not advice" in r
