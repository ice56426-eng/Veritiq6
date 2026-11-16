<p align="center"><img src="assets/logo.png" width="260" alt="VERITIQ logo"></p>

# VERITIQ: Analyze · Optimize · Evolve

**Spending-anomaly detector for small businesses.** Flags duplicate charges, subscription price hikes and one-off spikes, then an **AI Assistant** summarizes the problems, ranks what to fix first, and answers questions in plain English.

**Track:** AI + Finance (own idea) · **Team:** Ian Ensley & Arya Patel · Veritas AI International High School Hackathon 2026
**Demo video:** [Watch on Google Drive](https://drive.google.com/file/d/1mp2FQSodtVqcYolgGqRDgWsHAZv3ZolB/view)

> This project is a prototype and is not financial advice.

## The problem
Small-business owners and bookkeepers handle hundreds of charges a month and rarely catch double billing or silent price increases. Common tools only flag exact duplicates or large amounts.

## What it does
Input: a CSV of charges (`date, vendor, amount`). A sensitivity slider (default 0.2, matching our evaluation) trades false alarms against misses. Output: flagged charges, each with a recommendation, an estimated impact (not guaranteed), and an **"Explain this simply"** button that says *what happened, what to do (2-3 steps), roughly how much money is involved, and how sure we are*.

### AI Assistant
Open it from the top-right button, the sidebar, or the Quick summary panel.
- **Types out a full report:** summary, problems found, a fix list ranked by money at stake, and where the money goes.
- **Chat:** ask "Which flag matters most?", "What should I do next?", "How much am I losing?", or "Where are we lacking?".
- **Corrects its own noise:** if too many charges are flagged (over 25%), it says so and **Apply suggested sensitivity** re-checks at a better threshold. In our test, 55% flagged dropped to 12%.
- **Works with no API key.** All numbers come from code (`insights.py`). An optional AI engine (Setup tab) adds a short written take and open-ended answers.
- **Checks big first-time vendors by hand,** because the model has no history to judge them.

### Dashboard
Stat tiles (charges checked, flagged, estimated money at stake, recurring cost per month), a prioritized fix list, and charts of money at stake by problem type and spending by vendor.

## Quick start
```bash
pip install -r requirements.txt
python veritiq.py      # full experiment: results, charts in assets/, results.json
python app.py          # Gradio demo app
python -m pytest       # tests
```
**Google Colab:** run these in separate cells.
```python
!git clone https://github.com/ice56426-eng/Veritiq6.git
%cd /content/Veritiq6
!pip install -q -r requirements.txt
```
```python
import app
app.demo.launch(share=True)
```
Click the `gradio.live` link it prints.

## How it works
- **Data:** synthetic 24-month ledger (8 subscriptions, 6 one-off vendors, ~1,300 charges, ~10% injected anomalies): duplicates (60% exact, 40% near-exact), price hikes, spikes. No real financial data. Identical duplicate pairs are labelled on the later charge so the model is never scored on an impossible coin-flip.
- **Features (past only):** ratio to the vendor's prior median, difference from the previous charge, days since previous charge, subscription flag, amount.
- **Model:** scikit-learn gradient boosting. Threshold chosen on validation data assuming a missed anomaly costs 5x a false alarm.
- **Time split:** train first 12 months, tune threshold on months 13-18, retrain on 1-18, test on months 19-24 (never seen). Repeated on **20 random datasets**.
- **Real-world input handling:** CSV validation with readable errors, skips bad rows, ignores refunds, merges vendor-name variants (`SLACK*1234` = `Slack Technologies Inc`).
- **Explanations, summaries and recommendations:** rule-based templates from the same features. No AI language model is needed, and all numbers come from code.

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
- **Too many flags on some real files:** one 500-charge company file had 274 flagged (55%) at a low threshold. The AI Assistant now detects this and suggests a higher threshold, but the underlying model is still tuned for synthetic data.
- The "explain simply" text, savings estimates and AI Assistant advice are templates and rough estimates, not advice or guarantees. The assistant sees only charges, not revenue, headcount or goals.

## Real-style test (hand-written, not from our generator)
A fictional bakery's 2025 card statement (150 charges) with 6 planted problems and an answer key, in `test_data/`. Run `python test_data/real_style_test.py`.
- **Model at the default threshold (0.2):** caught 4 of 6 (3 duplicates and the QuickBooks price hike), plus 2 false alarms among the 144 normal charges.
- **Missed by the model:** a $2,640 charge from a vendor seen for the first time, and a $1,150 PackRight spike (about 8x usual). The second was likely missed because in our synthetic training data one-off vendors are charged several times a month while PackRight is monthly: a distribution-shift problem.
- **First-time-vendor check:** a simple rule (not the model) now surfaces the $2,640 charge for a manual look, so 5 of 6 planted problems are surfaced. The PackRight spike is still missed.
- **Bugs this test found and we fixed:** a blank vendor name crashed the app, and a price hike made a subscription look like a one-off.
- Only 6 planted issues, so treat it as a sanity check, not proof. Also in `test_data/`: `messy_statement.csv` (name variants, `$` and commas, a refund, bad rows) to try the cleaning.

## Advanced (Pro) features (Advanced tab, off by default)
Opt-in persistent memory/history (local SQLite), recommendation tracker (New → Implemented), estimated vs reported savings (accepting a recommendation is never counted as saved; "actual" is user-entered and unverified), optional AI-reworded explanations and chat via any supported provider (key kept in memory only), CSV export, clear workspace. The backend refuses to store anything until Pro is enabled **and** consent is given. When the AI engine is on, flagged charges and a short spend summary by vendor are sent to the provider you chose. Not implemented: Google Drive/Sheets/Gmail, scheduled reports, alerts.

## Testing
19 tests (`test_veritiq.py`, `test_pro.py`, `test_insights.py`) cover vendor matching, CSV cleaning, label fix, HTML escaping, the simple-explanation text for each anomaly type, Pro access/consent/tracking/impact math, and the AI Assistant's summary, fix ordering, first-time-vendor check and sensitivity suggestion. The Gradio screen and the optional AI connection should be checked once on your own machine.

## Next steps
Test on real, permissioned data; test with a real bookkeeper (does the simple explanation help them decide?); add subscription-overlap detection.

## Files
`app.py` (Gradio app) · `insights.py` (AI Assistant summaries) · `veritiq.py` (model, cleaning, evaluation) · `pro.py` (Pro features) · `test_*.py` (tests) · `assets/` (logo, charts) · `test_data/` (bakery test, messy statement) · `results.json`

## Sources and AI credit
scikit-learn, pandas, NumPy, matplotlib, Gradio.
**AI tools used:** the VERITIQ logo was generated with ChatGPT. Claude (Anthropic) helped write and improve the code, including the AI Assistant, the dashboard layout and the tests. The team designed the project, reviewed the code, and can explain every part.
