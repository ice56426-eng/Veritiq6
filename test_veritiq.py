import pandas as pd, veritiq as v

def _raw(rows): return pd.DataFrame(rows, columns=["date", "vendor", "amount"])

def test_vendor_merge():
    mp = v.normalize_vendors(["SLACK*1234", "Slack Technologies Inc", "Slack", "Delta Air", "Delta Dental"])
    assert mp["SLACK*1234"] == mp["Slack Technologies Inc"] == mp["Slack"]
    assert mp["Delta Air"] != mp["Delta Dental"]

def test_clean_handles_bad_rows():
    rows = [(f"2025-{m:02d}-05", "Slack", "$480.00") for m in range(1, 13)] + [("bad", "Slack", "1"), ("2025-09-05", "Zoom", "-50"), ("2025-09-06", "", "10"), ("2025-09-07", "Zoom", "abc")]
    df, w = v.clean_transactions(_raw(rows))
    assert len(df) == 12 and len(w) >= 2

def test_clean_errors_are_readable():
    for bad, msg in ((pd.DataFrame({"a": [1]}), "missing column"), (_raw([("2025-01-01", "X", 5)]), "usable")):
        try: v.clean_transactions(bad); assert False
        except ValueError as e: assert msg in str(e)

def test_duplicate_label_goes_on_later_row():
    df = v.make_data(1, 24)
    g = df.groupby(["date", "vendor", "amount"]).is_anomaly.agg(["size", "sum"])
    assert (g[g["size"] > 1]["sum"] <= 1).all()

def _flags():
    rows = [(f"2025-{m:02d}-05", "Slack", 480) for m in range(1, 9)] + [("2025-08-06", "Slack", 480), ("2025-09-05", "Slack", 700)]
    return v.flag_transactions(_raw(rows), v.train_final_model(), threshold=0.2)

def test_cards_and_simple_button():
    f = _flags(); assert len(f) >= 1
    h = v.cards_html(f)
    assert "Explain this simply" in h and "Recommendation:" in h and "not financial advice" in h
    assert "guarantee" not in h.lower().replace("not guaranteed", "")

def test_cards_escape_user_text():
    f = _flags().copy(); f["vendor"] = "<script>alert(1)</script>"
    h = v.cards_html(f); assert "<script>" not in h and "&lt;script&gt;" in h

def test_plain_text_per_type():
    base = dict(vendor="Zoom", amount=300.0, prior_median=150.0, ratio_median=2.0, days_since_prev=1.0, score=0.9, is_sub=1, rel_diff_prev=0.0)
    for t, kw in (("duplicate", {}), ("price_hike", dict(rel_diff_prev=1.0, days_since_prev=30.0)), ("spike", dict(rel_diff_prev=1.0, days_since_prev=30.0, is_sub=0, ratio_median=5.0))):
        r = pd.Series({**base, **kw}); assert v.classify(r) == t
        p = v.plain_explanation(r); assert p["what"] and len(p["steps"]) >= 2 and "$" in p["money"]

def test_blank_vendor_row_does_not_crash():
    rows = [(f"2025-{m:02d}-05", "Slack", 480) for m in range(1, 13)] + [("2025-05-02", None, 40)]
    df, w = v.clean_transactions(_raw(rows)); assert len(df) == 12 and any("Skipped" in x for x in w)

def test_price_hike_does_not_hide_subscription():
    rows = [(f"2025-{m:02d}-05", "QuickBooks", 30) for m in range(1, 10)] + [(f"2025-{m:02d}-05", "QuickBooks", 55) for m in range(10, 13)]
    f = v.flag_transactions(_raw(rows), v.train_final_model(), threshold=0.2)
    assert (f.type == "price_hike").any()
