"""VERITIQ demo app (Gradio). Run: python app.py
Standard use stores nothing. The Setup panel turns on opt-in Pro features (see pro.py)."""
import base64, io, os, tempfile, time, gradio as gr, pandas as pd
import pro, insights as ins
from veritiq import make_data, train_final_model, flag_transactions, cards_html, clean_transactions

MODEL = train_final_model()
DISCLAIMER = "Prototype, not financial advice. Model trained on synthetic data; flags are leads to review, not verdicts."
COLS = ["id", "tx_date", "vendor", "amount", "type", "est_impact", "status", "actual_savings", "reason"]
SHOW = ["date", "vendor", "amount", "score", "type", "est_impact", "reason"]

# ---------- logo (found in assets/ or the repo root, embedded so the layout needs no file paths) ----------
def _logo_src():
    for p in (os.path.join("assets", "logo.png"), "logo.png"):
        if os.path.exists(p):
            try:
                from PIL import Image
                im = Image.open(p).convert("RGBA"); im.thumbnail((360, 360))
                buf = io.BytesIO(); im.save(buf, "PNG"); data = buf.getvalue()
            except Exception:
                data = open(p, "rb").read()
            return "data:image/png;base64," + base64.b64encode(data).decode()
    return ""
LOGO_SRC = _logo_src()

# ---------- logic ----------
N_OUT = 9   # table, cards, status, kpis, quick summary, fixes, charts, ai fixes, state

def _path(f):
    if f is None: return None
    if isinstance(f, dict): return f.get("path") or f.get("name")
    return getattr(f, "name", f)

def _keep(): return [gr.update() for _ in range(N_OUT)]

def _say(text):
    k = _keep(); k[2] = text; return tuple(k)

def run(path, use_demo, sens):
    demo_mode = bool(use_demo or path is None)
    try:
        raw = make_data()[["date", "vendor", "amount"]] if demo_mode else pd.read_csv(path)
        flags = flag_transactions(raw, MODEL, threshold=float(sens))
        clean, _ = clean_transactions(raw)
    except Exception as e:   # readable message instead of a crash
        return (pd.DataFrame(), "", f"Could not check this file: {e}", *[gr.update()] * 6)
    a = ins.analyze(flags, clean, float(sens), src=(path, demo_mode))
    if pro.get("llm_enabled") == "1":   # optional: reword top 5 only; sends vendor, amount, reason to the provider you chose
        for i in flags.index[:5]:
            flags.loc[i, "reason"] = pro.llm_reword(flags.loc[i, "vendor"], flags.loc[i, "amount"], flags.loc[i, "reason"])
    note = ""
    if pro.is_enabled(): pro.save_run(flags, len(raw)); note = " Saved to your Pro workspace."
    warn = " ".join(flags.attrs.get("warnings", []))
    out = flags[SHOW].copy(); out["score"] = out.score.round(2); out["date"] = out.date.astype(str).str[:10]
    status = f"**{len(raw)} charges checked, {len(flags)} flagged.**{note} {warn}\n\n_{DISCLAIMER}_"
    if a["status"] == "noisy":
        status = f"**Heads up: {a['flag_rate']:.0%} of charges were flagged, which is too many to be useful.** Open the AI Assistant and press *Apply suggested sensitivity*.\n\n" + status
    fx = ins.fixes_html(a)
    return out, cards_html(flags), status, ins.kpi_html(a), ins.quick_summary_html(a), fx, ins.charts_html(a), fx, a

def smart_answer(q, a):
    """LLM if the user turned it on (flagged list + spend summary only), otherwise the built-in rule-based answers."""
    if a is None: return "Check a statement first: attach a CSV or type **demo**."
    if pro.get("llm_enabled") == "1":
        try: return pro.llm_ask(q, ins.llm_context(a)) + "\n\n_AI-written from the flagged list and spend summary only. Check it against the table._"
        except Exception as e: return f"_The AI could not answer ({e}). Built-in answer instead:_\n\n" + ins.answer(q, a)
    return ins.answer(q, a) + "\n\n_Built-in answer. Turn on the AI engine in Setup for open-ended questions._"

def chat(msg, sens, a):
    msg = msg or {}
    text = (msg.get("text") or "").strip()
    files = [p for p in (_path(f) for f in (msg.get("files") or [])) if p]
    csvs = [p for p in files if p.lower().endswith(".csv")]
    others = [p for p in files if not p.lower().endswith(".csv")]
    if csvs:
        res = list(run(csvs[0], False, sens))
        if others: res[2] = f"Read the CSV and ignored {len(others)} other file(s): VERITIQ only reads CSV statements.\n\n" + str(res[2])
        return tuple(res)
    if others:
        return _say("I can only read CSV statements with the columns `date, vendor, amount`. Images and PDFs aren't supported yet, so export your statement as a CSV.")
    if text.lower() in ("demo", "demo data", "try demo"):
        return run(None, True, sens)
    if text:
        if a is None: return _say("Check a statement first: attach a CSV or type **demo**. Then you can ask questions about the flags.")
        return _say(smart_answer(text, a))
    return _say("Attach a CSV with `date, vendor, amount`, type **demo**, or ask a question about your flags.")

# ---------- AI assistant ----------
def stream_report(a):
    """Types the report out word by word. Numbers come from code (insights.py); the optional AI take is added at the end."""
    if a is None:
        yield "### No data yet\nGo to **Check charges** and attach a CSV or type **demo**, or press **Analyze demo data** below."
        return
    text = ins.report_md(a); words = text.split(" ")
    for i in range(0, len(words), 3):
        yield " ".join(words[:i + 3]) + " ▌"; time.sleep(0.03)
    final = text
    if pro.get("llm_enabled") == "1":
        yield final + "\n\n_Asking the AI engine for a plain-English take..._"
        try:
            take = pro.llm_ask("In 3 plain sentences: what is going wrong and what should the owner do first?", ins.llm_context(a))
            final += f"\n\n### AI take\n{take}\n\n_AI-written from the data above only. Check it against the report._"
        except Exception as e:
            final += f"\n\n_AI take unavailable: {e}_"
    yield final

def apply_fix(a, sens):
    """'Correct' the main problem we can fix automatically: a sensitivity that flags too much."""
    if a is None or not a.get("src"):
        return (*_say("Nothing to correct yet. Check a statement first."), gr.update())
    thr = a.get("suggested_threshold")
    if not thr:
        return (*_say(f"Sensitivity {a['threshold']:.2f} already looks reasonable ({a['flag_rate']:.0%} flagged). Nothing to change."), gr.update())
    path, demo_mode = a["src"]
    return (*run(path, demo_mode, thr), gr.update(value=thr))

def render_convo(hist):
    return "\n\n---\n\n".join(f"**You:** {q}\n\n**VERITIQ AI:** {r}" for q, r in (hist or [])[-8:])

def ask(q, hist, a):
    q = (q or "").strip()
    if not q: return hist, render_convo(hist), ""
    hist = (hist or []) + [(q, smart_answer(q, a))]
    return hist, render_convo(hist), ""

def tracker(status):
    try:
        recs = pro.list_recs(status); s = pro.impact_summary()
    except pro.ProDisabled as e:
        return pd.DataFrame(columns=COLS), str(e)
    df = pd.DataFrame(recs, columns=COLS) if recs else pd.DataFrame(columns=COLS)
    md = (f"**Identified (estimated):** ${s['identified_total']:,.2f} across {s['n_recs']} flags  \n"
          f"**Accepted/planned/in progress/implemented (estimated, NOT realized):** ${s['accepted_est']:,.2f}  \n"
          f"**Implemented:** {s['n_implemented']}  ·  **Pending estimate:** ${s['pending_est']:,.2f}  \n"
          f"**Projected vs reported actual** (for {s['n_with_outcome']} items with an outcome): "
          f"${s['projected_for_reported']:,.2f} vs ${s['reported_actual']:,.2f} (difference ${s['difference']:,.2f}).  \n"
          "_Estimates are not guaranteed. 'Reported actual' is what you typed in; VERITIQ does not verify it._")
    return df, md

def update(rec_id, status, actual, notes):
    try:
        pro.update_rec(int(rec_id), status, None if actual in (None, "") else float(actual), notes or "")
        return "Updated."
    except Exception as e:
        return f"Could not update: {e}"

def history():
    try: return pd.DataFrame(pro.history())
    except pro.ProDisabled as e: return pd.DataFrame({"message": [str(e)]})

PROV0 = pro.get("llm_provider") if pro.get("llm_provider") in pro.PROVIDERS else "gemini"

def prov_note(pid):
    p = pro.PROVIDERS[pid]
    link = f' <a href="{p["key"]}" target="_blank" rel="noopener">Get an API key</a>' if p["key"] else ""
    return f'<p class="tiny">{p["note"]}{link}</p>'

def prov_change(pid):
    m = pro.PROVIDERS[pid]["model"]
    return prov_note(pid), gr.update(choices=[m], value=m)

def save_ai(on, pid, endpoint, model, key):
    pro.set_("llm_enabled", "1" if on else "0"); pro.set_("llm_provider", pid)
    pro.set_("llm_endpoint", endpoint or ""); pro.set_("llm_model", model or ""); pro.set_api_key(key, pid)
    n = pro.PROVIDERS[pid]["n"]
    if on and pid != "custom" and pid not in pro._KEYS: return f"Saved, but {n} needs an API key before AI will work."
    return f"Saved. AI is {'ON' if on else 'OFF'} ({n}). Press Test to check the connection."

def list_ui(pid, endpoint, key):
    try: ids = pro.list_models(pid, endpoint, key)
    except Exception as e: return gr.update(), f"Could not list models: {e}"
    return gr.update(choices=ids), f"{len(ids)} models found. Pick one in the Model box."

def save_pro(enabled, consent):
    pro.set_("pro_enabled", "1" if enabled else "0"); pro.set_("consent", "1" if consent else "0")
    if enabled and not consent: return "Saved, but persistent storage stays OFF until you tick the consent box."
    return "Saved. Pro workspace is " + ("ON." if pro.is_enabled() else "OFF (nothing is stored).")

def export():
    try:
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False); f.write(pro.export_csv()); f.close(); return f.name
    except pro.ProDisabled: return None

def clear(confirm):
    if not confirm: return "Tick the confirmation box first."
    try: pro.clear_workspace(); return "Workspace cleared (settings kept)."
    except pro.ProDisabled as e: return str(e)

# ---------- look ----------
STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;1,400&family=Inter:wght@400;500;600&display=swap');
.gradio-container{--body-background-fill:#0a0714;--body-text-color:#f3f0ff;--body-text-color-subdued:#a39bc9;
--background-fill-primary:#120f22;--background-fill-secondary:#171329;--block-background-fill:#120f22;
--panel-background-fill:#120f22;--block-border-color:#2b2547;--border-color-primary:#2b2547;--border-color-accent:#8b5cf6;
--input-background-fill:#171329;--input-border-color:#2b2547;--block-label-background-fill:#171329;--block-label-text-color:#a39bc9;
--block-title-text-color:#d8d0ff;--table-even-background-fill:#120f22;--table-odd-background-fill:#171329;--table-row-focus:#241d44;
--button-secondary-background-fill:#171329;--button-secondary-background-fill-hover:#241d44;--button-secondary-text-color:#f3f0ff;
--button-secondary-border-color:#2b2547;--button-primary-background-fill:#7c3aed;--button-primary-background-fill-hover:#8b5cf6;
--button-primary-text-color:#fff;--checkbox-label-background-fill:#171329;--checkbox-background-color:#171329;
background:#0a0714!important;font-family:'Inter',system-ui,sans-serif}
#shell{gap:0!important;align-items:stretch}
#side{background:#0d0a1b;border-right:1px solid #2b2547;padding:22px 14px!important;min-height:94vh;gap:4px!important}
.brand img{width:148px;display:block}
.slogan{color:#a39bc9;font-size:.82rem;line-height:1.45;margin:12px 0 22px}
button.nav{justify-content:flex-start;text-align:left;border:none!important;background:transparent!important;color:#cfc7f5!important;box-shadow:none!important;border-radius:10px!important;font-weight:500}
button.nav:hover{background:rgba(139,92,246,.12)!important;color:#fff!important}
button.nav.primary{background:rgba(139,92,246,.20)!important;color:#fff!important;box-shadow:inset 3px 0 0 #a78bfa!important}
#spacer{flex:1 1 auto}
#setup-btn{margin-top:auto;border-top:1px solid #2b2547!important;border-radius:0!important;padding-top:14px}
#main{padding:16px 2.5vw 30px!important}
.hero{margin:0 0 14px;padding:28px 30px;border-radius:22px;border:1px solid #2b2547;
background:radial-gradient(70% 140% at 20% 0%,rgba(124,58,237,.30),rgba(124,58,237,0) 70%),#0f0c1e}
.hero h1{font-family:'Fraunces',Georgia,serif;font-weight:400;font-size:clamp(2.2rem,5vw,3.4rem);line-height:1.06;margin:0 0 14px;color:#fff}
.hero h1 em{color:#c4b5fd}
.hero p{color:#bdb5e0;margin:0;max-width:52ch;line-height:1.55}
#chat,#sens,.bubble,.cards,.chips{max-width:none}
#chat{border:1px solid #3b2f6b;border-radius:18px;background:#120f22;box-shadow:0 14px 44px rgba(124,58,237,.14)}
#chat:focus-within{border-color:#8b5cf6;box-shadow:0 0 0 3px rgba(139,92,246,.22),0 14px 44px rgba(124,58,237,.18)}
#chat textarea{background:transparent!important;color:#fff!important}
.chips{gap:8px!important;flex-wrap:wrap}
button.chip{flex:0 0 auto!important;width:auto!important;min-width:0!important;border-radius:999px!important;background:#171329!important;border:1px solid #2b2547!important;color:#d8d0ff!important;font-size:.84rem!important;padding:6px 14px!important}
button.chip:hover{border-color:#8b5cf6!important;color:#fff!important}
.bubble{border-left:3px solid #8b5cf6;padding:4px 0 4px 14px;color:#f3f0ff}
.cards .vq{background:#120f22;border-color:#3b2f6b}
.card{background:#120f22;border:1px solid #2b2547!important;border-radius:16px!important;padding:18px 20px!important;margin-bottom:14px}
.card h3{margin-top:0;color:#fff}
.pills .wrap{gap:8px!important}
.pills label{border-radius:999px!important;background:#171329!important;border:1px solid #2b2547!important;padding:7px 16px!important;color:#d8d0ff!important}
.pills label.selected{background:#7c3aed!important;border-color:#a78bfa!important;color:#fff!important}
.tiny{color:#a39bc9;font-size:.85rem;margin:2px 0 8px}.tiny a{color:#c4b5fd}
.page-title{font-family:'Fraunces',Georgia,serif;font-weight:400;font-size:2rem;margin:0 0 12px;color:#fff}
.fine{color:#8f87b5;font-size:.8rem;text-align:center;margin-top:28px}
.gradio-container{max-width:100%!important}
#topbar{align-items:center;gap:12px!important;margin-bottom:6px}
.crumb{color:#a39bc9;font-size:.9rem}.crumb b{color:#fff}
#ai-top{flex:0 0 auto!important;min-width:0!important;width:auto!important;border-radius:999px!important;background:linear-gradient(90deg,#7c3aed,#c026d3)!important;border:none!important;color:#fff!important;font-weight:600!important;box-shadow:0 6px 24px rgba(168,85,247,.35)!important;padding:8px 20px!important}
#ai-open{border-radius:12px!important;background:linear-gradient(90deg,#7c3aed,#c026d3)!important;border:none!important;color:#fff!important;font-weight:600!important}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
.kpi{background:#120f22;border:1px solid #2b2547;border-radius:16px;padding:16px 18px}
.kl{color:#a39bc9;font-size:.75rem;text-transform:uppercase;letter-spacing:.06em}
.kv{font-family:'Fraunces',Georgia,serif;font-size:2.2rem;color:#fff;line-height:1.2}
.ks{color:#8f87b5;font-size:.78rem;min-height:1em}
.vpanel,.qs{background:#120f22;border:1px solid #2b2547;border-radius:16px;padding:16px 18px;color:#e7e2ff}
.ph{color:#d8d0ff;font-weight:600;margin:4px 0 10px}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;font-size:.74rem;font-weight:600;margin-right:8px;white-space:nowrap}
.pill.hi{background:rgba(239,68,68,.18);color:#fca5a5}.pill.md{background:rgba(245,158,11,.18);color:#fcd34d}
.pill.lo{background:rgba(139,92,246,.22);color:#c4b5fd}.pill.ok{background:rgba(34,197,94,.18);color:#86efac}
.fx{display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-top:1px solid #231d3d}.ph + .fx{border-top:none}
.fb{flex:1;color:#e7e2ff;font-size:.9rem}.fb .m{color:#8f87b5;font-size:.78rem}.fm{color:#fcd34d;font-size:.85rem;white-space:nowrap}
.br{display:grid;grid-template-columns:150px 1fr 130px;gap:10px;align-items:center;margin:7px 0;font-size:.85rem;color:#e7e2ff}
.br + .ph{margin-top:20px}
.bl{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bt{background:#1d1736;border-radius:6px;height:12px;overflow:hidden}.bf{height:100%;border-radius:6px}
.bv{text-align:right;color:#a39bc9}
.qs ul{margin:8px 0 8px 18px;color:#cfc7f5;font-size:.9rem}.qm{color:#fff;margin:6px 0}.qh{margin-bottom:6px;color:#fff}
.qf{background:rgba(139,92,246,.12);border-left:3px solid #a78bfa;padding:8px 10px;border-radius:8px;margin:8px 0;color:#e7e2ff;font-size:.88rem}
.qf span{display:block;color:#a39bc9;font-size:.7rem;text-transform:uppercase;letter-spacing:.06em}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:4px}
.how div{background:#120f22;border:1px solid #2b2547;border-radius:16px;padding:16px 18px;color:#bdb5e0;font-size:.9rem;line-height:1.5}
.how b{display:block;color:#fff;font-size:1rem;margin-bottom:4px}.how i{font-style:normal;color:#a78bfa;font-weight:600}
@media(max-width:900px){.kpis{grid-template-columns:repeat(2,1fr)}.how{grid-template-columns:1fr}.br{grid-template-columns:100px 1fr 90px}}
</style>"""

HERO = ('<div class="hero"><h1>Your statement.<br><em>Every charge checked.</em></h1>'
        '<p>Drop a CSV with date, vendor and amount, or type <b>demo</b>. In standard mode nothing is saved.</p></div>')
BRAND = (f'<div class="brand"><img src="{LOGO_SRC}" alt="VERITIQ"><p class="slogan">Analyze. Optimize. Evolve. Today, with VERITIQ.</p></div>')
HOW = ('<h3 class="ph" style="margin-top:18px">How VERITIQ works</h3><div class="how">'
       '<div><i>1</i><b>Drop in your statement</b>Any CSV with date, vendor and amount. Messy names, $ signs and refunds are cleaned for you.</div>'
       '<div><i>2</i><b>Every charge is checked</b>Duplicate charges, subscription price hikes and one-off spikes are flagged, plus big first-time vendors to check by hand.</div>'
       '<div><i>3</i><b>Fix the biggest first</b>The AI Assistant summarises the problems, ranks what to fix by money at stake, and answers your questions.</div></div>')
VIEWS = ["check", "tracker", "history", "ai", "setup"]
LACKING = "Analyze my company's dataset and tell me where we are lacking as a company and what we can better allocate its resources to."
SUGGEST = [("Where are we lacking?", LACKING), ("Which flag matters most?", "Which flagged charge should I look at first, and why?"),
           ("What should I do next?", "What should I do next about these flags?"), ("How much am I losing?", "How much am I losing?"),
           ("Where does my money go?", "Where does my money go?")]

def go(name):
    return ([gr.update(visible=(v == name)) for v in VIEWS] +
            [gr.update(variant="primary" if v == name else "secondary") for v in VIEWS])

def _nav_fn(name):
    def f(): return go(name)
    return f

with gr.Blocks(title="VERITIQ", theme=gr.themes.Base(primary_hue="purple")) as demo:
    gr.HTML(STYLE)
    st = gr.State(None)      # latest analysis (summary, fixes, spend) shared by every screen
    hist = gr.State([])      # AI chat history
    with gr.Row(equal_height=False, elem_id="shell"):
        # ----- sidebar -----
        with gr.Column(scale=0, min_width=250, elem_id="side"):
            gr.HTML(BRAND)
            n_check = gr.Button("Check charges", variant="primary", elem_classes="nav")
            n_track = gr.Button("Tracker (Pro)", variant="secondary", elem_classes="nav")
            n_hist = gr.Button("History (Pro)", variant="secondary", elem_classes="nav")
            n_ai = gr.Button("\u2726 AI Assistant", variant="secondary", elem_classes="nav")
            gr.HTML("", elem_id="spacer")
            n_setup = gr.Button("Setup", variant="secondary", elem_classes="nav", elem_id="setup-btn")
        # ----- main area -----
        with gr.Column(scale=1, elem_id="main"):
            with gr.Row(elem_id="topbar"):
                gr.HTML('<div class="crumb"><b>Spending checkup</b> \u00b7 prototype, not financial advice</div>')
                top_ai = gr.Button("\u2726 AI Assistant", variant="primary", elem_id="ai-top", size="sm")
            # ===== Check charges =====
            with gr.Column(visible=True) as v_check:
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5):
                        gr.HTML(HERO)
                        mm = gr.MultimodalTextbox(placeholder="Attach a statement CSV, type demo, or ask about your flags", show_label=False,
                                                  file_count="multiple", elem_id="chat")
                        with gr.Row(elem_classes="chips"):
                            demo_btn = gr.Button("Try the demo data", size="sm", elem_classes="chip")
                            q1 = gr.Button("Which flag matters most?", size="sm", elem_classes="chip")
                            q2 = gr.Button("What should I do next?", size="sm", elem_classes="chip")
                            q3 = gr.Button("Where are we lacking?", size="sm", elem_classes="chip")
                        with gr.Accordion("Sensitivity (lower catches more, with more false alarms)", open=False, elem_id="sens"):
                            sens = gr.Slider(0.05, 0.9, value=0.2, step=0.05, label="Threshold (0.2 matches our test)")
                        status = gr.Markdown(elem_classes="bubble")
                    with gr.Column(scale=3, min_width=320):
                        qsum = gr.HTML(ins.quick_summary_html(None))
                        ai_open = gr.Button("\u2726 Open full AI summary", elem_id="ai-open")
                kpis = gr.HTML(ins.kpi_html(None))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5): fixes = gr.HTML(ins.fixes_html(None))
                    with gr.Column(scale=4): charts = gr.HTML(ins.charts_html(None))
                gr.HTML(HOW)
                cards = gr.HTML(elem_classes="cards")
                with gr.Accordion("All flagged charges (table)", open=False):
                    table = gr.Dataframe()
            # ===== Tracker =====
            with gr.Column(visible=False) as v_tracker:
                gr.HTML('<h2 class="page-title">Tracker</h2>')
                flt = gr.Dropdown(["All"] + pro.STATUSES, value="All", label="Filter by status"); rb = gr.Button("Refresh")
                tt = gr.Dataframe(label="Recommendations"); sm = gr.Markdown(); rb.click(tracker, flt, [tt, sm])
                gr.Markdown("### Update a recommendation")
                rid = gr.Number(label="ID", precision=0); stt = gr.Dropdown(pro.STATUSES, value="Accepted", label="Status")
                act = gr.Number(label="Actual savings you measured (optional, set after implementing)"); nt = gr.Textbox(label="Notes")
                ub = gr.Button("Update"); um = gr.Markdown(); ub.click(update, [rid, stt, act, nt], um)
            # ===== History =====
            with gr.Column(visible=False) as v_history:
                gr.HTML('<h2 class="page-title">History</h2>')
                hb = gr.Button("Load past runs"); ht = gr.Dataframe(); hb.click(history, None, ht)
            # ===== AI Assistant =====
            with gr.Column(visible=False) as v_ai:
                gr.HTML('<h2 class="page-title">\u2726 AI Assistant</h2><p class="tiny">Summary, problems and what to fix, written from your checked statement. '
                        'The numbers always come from code, so this works without an API key.</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        with gr.Column(elem_classes="card"):
                            report = gr.Markdown("### No data yet\nCheck a statement first, or press **Analyze demo data**.")
                            with gr.Row():
                                b_re = gr.Button("Re-run summary", size="sm")
                                b_demo = gr.Button("Analyze demo data", size="sm")
                                b_fix = gr.Button("Apply suggested sensitivity", variant="primary", size="sm")
                        with gr.Column(elem_classes="card"):
                            gr.Markdown("### Ask a question")
                            convo = gr.Markdown()
                            with gr.Row():
                                qbox = gr.Textbox(placeholder="e.g. Which flag matters most? Where am I overspending?", show_label=False, scale=5)
                                qsend = gr.Button("Ask", variant="primary", scale=1)
                            with gr.Row(elem_classes="chips"):
                                sug = [gr.Button(lab, size="sm", elem_classes="chip") for lab, _ in SUGGEST]
                    with gr.Column(scale=2, min_width=320):
                        ai_fixes = gr.HTML(ins.fixes_html(None))
            # ===== Setup =====
            with gr.Column(visible=False) as v_setup:
                gr.HTML('<h2 class="page-title">Setup</h2>')
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### AI engine\nOptional. Pick a provider, paste your own API key, and VERITIQ can reword explanations and answer open-ended questions about your flags. "
                                "The flagging and the summary numbers never use AI. Free: Gemini, Groq, OpenRouter (:free models). ChatGPT and Claude APIs are paid.")
                    prov = gr.Radio([(v["n"], k) for k, v in pro.PROVIDERS.items()], value=PROV0, label="Provider preset", elem_classes="pills")
                    pnote = gr.HTML(prov_note(PROV0))
                    ky = gr.Textbox(label="API key", type="password", placeholder="Paste key. Kept in memory only; clears when the app stops.")
                    m0 = pro.get("llm_model") or pro.PROVIDERS[PROV0]["model"]
                    mdl = gr.Dropdown(choices=[m0], value=m0, allow_custom_value=True, label="Model")
                    ep = gr.Textbox(label="Endpoint (only used for Custom / local)", value=pro.get("llm_endpoint"))
                    lo = gr.Checkbox(label="Turn AI on (sends vendor, amount and reason of flagged charges, plus a short spend summary by vendor, to the provider you chose)", value=pro.get("llm_enabled") == "1")
                    with gr.Row():
                        sv = gr.Button("Save", variant="primary"); tc = gr.Button("Test"); lm = gr.Button("List models")
                    aim = gr.Markdown()
                    prov.change(prov_change, prov, [pnote, mdl])
                    sv.click(save_ai, [lo, prov, ep, mdl, ky], aim); tc.click(lambda: pro.llm_test(), None, aim)
                    lm.click(list_ui, [prov, ep, ky], [mdl, aim])
                    gr.HTML('<p class="tiny">Anyone who opens a shared link to this running app can use the key you enter, so only add a key on a session you control.</p>')
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### Pro workspace\nOff by default. Turning it on stores your results in a local file (`veritiq_pro.db`) on this computer.")
                    e = gr.Checkbox(label="Enable Pro features (persistent memory, history, tracker)", value=pro.get("pro_enabled") == "1")
                    c = gr.Checkbox(label="I consent to saving flagged charges and my decisions locally", value=pro.get("consent") == "1")
                    sv2 = gr.Button("Save Pro settings", variant="primary"); sm2 = gr.Markdown(); sv2.click(save_pro, [e, c], sm2)
                    eb = gr.Button("Export history (CSV)"); ef = gr.File(); eb.click(export, None, ef)
                    cf = gr.Checkbox(label="Yes, permanently delete all saved runs and recommendations"); cb = gr.Button("Clear workspace"); cm = gr.Markdown(); cb.click(clear, cf, cm)
                with gr.Column(elem_classes="card"):
                    gr.Markdown("### Not implemented\nGoogle Drive / Sheets / Gmail, scheduled reports and alerts. These are roadmap items, not working features.")
            gr.HTML('<p class="fine">This project is a prototype and is not financial advice.</p>')

    views, btns = [v_check, v_tracker, v_history, v_ai, v_setup], [n_check, n_track, n_hist, n_ai, n_setup]
    outs = [table, cards, status, kpis, qsum, fixes, charts, ai_fixes, st]

    # ----- Check charges wiring -----
    mm.submit(chat, [mm, sens, st], outs)
    demo_btn.click(lambda s: run(None, True, s), sens, outs)
    q1.click(lambda s, a: chat({"text": "Which flagged charge should I look at first, and why?"}, s, a), [sens, st], outs)
    q2.click(lambda s, a: chat({"text": "What should I do next about these flags?"}, s, a), [sens, st], outs)
    q3.click(lambda s, a: chat({"text": LACKING}, s, a), [sens, st], outs)

    # ----- navigation (opening the AI view always types out a fresh summary) -----
    for b, name in zip(btns, VIEWS):
        if name != "ai": b.click(_nav_fn(name), None, views + btns)
    for b in (n_ai, top_ai, ai_open):
        b.click(_nav_fn("ai"), None, views + btns).then(stream_report, st, report)

    # ----- AI Assistant wiring -----
    b_re.click(stream_report, st, report)
    b_demo.click(lambda s: run(None, True, s), sens, outs).then(stream_report, st, report)
    b_fix.click(apply_fix, [st, sens], outs + [sens]).then(stream_report, st, report)
    qbox.submit(ask, [qbox, hist, st], [hist, convo, qbox]); qsend.click(ask, [qbox, hist, st], [hist, convo, qbox])
    for btn, (_, t) in zip(sug, SUGGEST):
        btn.click(lambda h, a, t=t: ask(t, h, a), [hist, st], [hist, convo, qbox])

if __name__ == "__main__":
    demo.launch(share=True)
