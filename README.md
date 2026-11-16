<p align="center"><img src="assets/logo.png" width="260" alt="VERITIQ logo"></p>

# VERITIQ: Analyze · Optimize · Evolve

**Spending-anomaly detector for small businesses.** Flags duplicate charges, subscription price hikes and one-off spikes. Each recommendation has an **"Explain this simply"** button that rewrites it in plain English for a business owner.

**Track:** AI + Finance (own idea) · **Team:** _add names_ · Veritas AI International High School Hackathon 2026

> This project is a prototype and is not financial advice.

## The problem
Small-business owners and bookkeepers handle hundreds of charges a month and rarely catch double billing or silent price increases. Common tools only flag exact duplicates or large amounts.

## What it does
Input: a CSV of charges (`date, vendor, amount`). A sensitivity slider (default 0.2, matching our evaluation) trades false alarms against misses. Output: flagged charges, each with a recommendation, an estimated impact (not guaranteed), and a button that explains it simply: *what happened, what to do (2-3 steps), roughly how much money is involved, and how sure we are*.

## Quick start
```bash
pip install -r requirements.txt
python veritiq.py      # full experiment: results, charts in assets/, results.json
python app.py          # Gradio demo app
python -m pytest       # tests
```
Colab: upload `VERITIQ_Colab.ipynb` to Colab, upload `veritiq_project.zip` when asked, run the cells in order, and open the `gradio.live` link it prints.

## How it works
- **Data:** synthetic 24-month ledger (8 subscriptions, 6 one-off vendors, ~1,300 charges, ~10% injected anomalies): duplicates (60% exact, 40% near-exact), price hikes, spikes. No real financial data. Identical duplicate pairs are labelled on the later charge so the model is never scored on an impossible coin-flip.
- **Features (past only):** ratio to the vendor's prior median, difference from the previous charge, days since previous charge, subscription flag, amount.
- **Model:** scikit-learn gradient boosting. Threshold chosen on validation data assuming a missed anomaly costs 5x a false alarm.
- **Time split:** train first 12 months, tune threshold on months 13-18, retrain on 1-18, test on months 19-24 (never seen). Repeated on **20 random datasets**.
- **Real-world input handling:** CSV validation with readable errors, skips bad rows, ignores refunds, merges vendor-name variants (`SLACK*1234` = `Slack Technologies Inc`).
- **Explanations and recommendations:** rule-based templates from the same features. No AI language model is needed, and all numbers come from code.

## Results (held-out test, 20 random datasets)
| Method | Precision (mean ± std) | Recall (mean ± std) |
|---|---|---|
| Baseline A: charge > $2,000 | 0.30 ± 0.09 | 0.07 ± 0.02 |
| Baseline B: exact duplicates only | 1.00 ± 0.00 | 0.33 ± 0.07 |
| **VERITIQ model** | 0.73 ± 0.14 | 0.77 ± 0.09 |

One example dataset (seed 42, 320 test charges, 42 anomalies): model precision 0.77, recall 0.81 (PR-AUC 0.84); duplicate rule precision 1.00, recall 0.26.
Recall by type on that example: duplicate 92% (rule 42%, n=26); price hike 75% (rule 0%, n=4); spike 58% (rule 0%, n=12).

![Model vs baselines](assets/model_vs_baselines.png)
![Threshold curve](assets/threshold_curve.png)
![Feature importance](assets/feature_importance.png)

## Where it fails
- **Baseline B's perfect precision is partly an artifact:** our synthetic exact duplicates are always errors. In real ledgers, two identical charges can be legitimate, so its real precision would be lower.
- **Precision varies a lot between datasets** (0.73 ± 0.14); expect false alarms, e.g. a legitimate large flight looks like a spike.
- **Usage-based bills (like AWS)** vary naturally, so price hikes on them are often missed.
- **Near-exact duplicates** (fees added) are the hardest; some are missed.
- **Synthetic data is cleaner than real ledgers.** The app applies a synthetic-trained model to uploaded CSVs, so results on real data are **unvalidated**.
- **Vendor matching can over-merge** similar names (e.g. "Uber" / "Uber Eats").
- The "explain simply" text and savings estimates are templates and rough estimates, not advice or guarantees.

## Real-style test (hand-written, not from our generator)
A fictional bakery's 2025 card statement (150 charges) with 6 planted problems and an answer key, in `test_data/`. Run `python test_data/real_style_test.py`.
- **Result at the default threshold (0.2):** caught 4 of 6 (3 duplicates and the QuickBooks price hike), plus 2 false alarms among the 144 normal charges. At a stricter 0.55: still 4 of 6 with 0 false alarms.
- **Missed:** a $2,640 charge from a vendor seen for the first time (nothing to compare it with), and a $1,150 PackRight spike (about 8x usual). The model likely missed the second because, in our synthetic training data, one-off vendors are charged several times a month while PackRight is monthly: a distribution-shift problem.
- **Bugs this test found and we fixed:** a blank vendor name crashed the app, and a price hike made a subscription look like a one-off.
- Only 6 planted issues, so treat it as a sanity check, not proof. Also in `test_data/`: `messy_statement.csv` (name variants, `$` and commas, a refund, bad rows) to try the cleaning.

## Advanced (Pro) features (Advanced tab, off by default)
Opt-in persistent memory/history (local SQLite), recommendation tracker (New → Implemented), estimated vs reported savings (accepting a recommendation is never counted as saved; "actual" is user-entered and unverified), optional AI-reworded explanations via any OpenAI-compatible endpoint (key kept in memory only), CSV export, clear workspace. The backend refuses to store anything until Pro is enabled **and** consent is given. Not implemented: Google Drive/Sheets/Gmail, scheduled reports, alerts.

## Testing
18 tests (`test_veritiq.py`, `test_pro.py`, `test_insights.py`) cover vendor matching, CSV cleaning, label fix, HTML escaping, the simple-explanation text for each anomaly type, and Pro access/consent/tracking/impact math. The Gradio screen and the optional AI connection have not been run in the build environment; check them once on your machine.

## Next steps
Test on real, permissioned data; test with a real bookkeeper (does the simple explanation help them decide?); add subscription-overlap detection.

## Sources
scikit-learn, pandas, NumPy, matplotlib, Gradio. Built with help from ChatGPT and Claude (AI tools); the team reviewed and can explain every part.

## AI Assistant (new)
Top-right and sidebar **AI Assistant** button: types out a summary, the problems found, a ranked fix list and where the money goes (`insights.py`, rule-based, no API key needed). Ask questions in the chat; "Apply suggested sensitivity" re-checks when too many charges are flagged. Optional AI engine (Setup) adds an AI take and open-ended answers. Also flags large first-time vendor charges for a manual look.
