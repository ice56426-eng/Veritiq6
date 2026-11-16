"""Hand-written 'real-style' test for VERITIQ (NOT synthetic-generator data): a fictional bakery's 2025 card statement
with planted problems and an answer key. Run from the repo folder:  python test_data/real_style_test.py
Writes bakery_ledger.csv, answer_key.csv, messy_statement.csv, then scores the model."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import veritiq as v
D = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(7)
rows, key = [], []
def add(d, ven, amt, issue=None):
    rows.append((d, ven, round(amt, 2)))
    if issue: key.append((d, ven, round(amt, 2), issue))
for m in range(1, 13):
    d = lambda day: f"2025-{m:02d}-{day:02d}"
    add(d(1), "Main Street Landlord", 2200.00)
    add(d(3), "Square Fees", rng.uniform(58, 92))
    add(d(5), "QuickBooks Online", 55.00 if m >= 10 else 30.00, "price_hike" if m == 10 else None)
    add(d(6), "Mailchimp", 45.00); add(d(8), "Google Workspace", 36.00); add(d(12), "Verizon Wireless", 118.50)
    add(d(20), "City Electric", rng.uniform(180, 260))
    for day in (4, 18): add(d(day), "Valley Mills Flour", rng.uniform(300, 420))
    add(d(10), "PackRight Supplies", rng.uniform(80, 200))
    add(d(15), "Market Stall Fee", 85.00)
    add(d(22), "Costco Business", rng.uniform(150, 260))
add("2025-06-06", "Mailchimp", 45.00, "duplicate")
add("2025-03-13", "Verizon Wireless", 118.50, "duplicate")
add("2025-08-20", "Valley Mills Flour", 389.40); add("2025-08-21", "Valley Mills Flour", 389.40, "duplicate")
add("2025-09-17", "Hobart Service Co", 2640.00, "spike_new_vendor")
add("2025-11-14", "PackRight Supplies", 1150.00, "spike")
df = pd.DataFrame(rows, columns=["date", "vendor", "amount"]).sort_values(["date", "vendor"]).reset_index(drop=True)
df.to_csv(f"{D}/bakery_ledger.csv", index=False)
pd.DataFrame(key, columns=["date", "vendor", "amount", "issue"]).to_csv(f"{D}/answer_key.csv", index=False)
# messy bank-style export: variant names, $ and commas, a refund, blank/bad rows
msy = [(f"2025-{m:02d}-0{3+m%3}", np.random.default_rng(m).choice(["SLACK*T0{}".format(m), "Slack Technologies Inc", "SLACK"]), f"${480:,.2f}") for m in range(1, 13)]
msy += [("2025-12-04", "SLACK*T012", "$480.00"), ("2025-07-09", "Refund - Zoom", "-150.00"), ("not a date", "Zoom", "150"), ("2025-05-02", "", "40"), ("2025-10-11", "Delta Air Lines", "1,420.55")]
pd.DataFrame(msy, columns=["Date", "Description", "Amount"]).rename(columns={"Description": "Vendor"}).to_csv(f"{D}/messy_statement.csv", index=False)

if __name__ == "__main__":
    model = v.train_final_model()
    for thr in (0.55, 0.2):
        fl = v.flag_transactions(df, model, threshold=thr); fl["k"] = list(zip(fl.date.dt.strftime("%Y-%m-%d"), fl.vendor.str.lower(), fl.amount.round(2)))
        pk = {(a, b.title().lower(), c): i for a, b, c, i in key}; got = set(fl.k)
        caught = [i for k, i in pk.items() if k in got]; missed = [i for k, i in pk.items() if k not in got]
        print(f"\nthreshold {thr}: {len(df)} charges, {len(key)} planted issues -> flagged {len(fl)}, caught {len(caught)}, missed {missed}, false alarms {len(got - set(pk))}")
        for _, r in fl[~fl.k.isin(pk)].head(6).iterrows(): print(f"   false alarm: {r.date:%Y-%m-%d} {r.vendor} ${r.amount:,.2f} ({r.type})")
    out = v.flag_transactions(pd.read_csv(f"{D}/messy_statement.csv"), model, threshold=0.2)
    print("\nmessy_statement.csv warnings:", out.attrs["warnings"]); print(out[["date", "vendor", "amount", "type"]].to_string())
