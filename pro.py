"""VERITIQ Advanced (Pro) features: opt-in persistent memory, recommendation/impact tracking,
optional LLM-reworded explanations, export and delete. Standard use never touches the database.
Stdlib only. Backend-enforced: every persistent call checks the Advanced settings."""
import sqlite3, csv, io, json, os, urllib.request, urllib.parse, urllib.error
from datetime import datetime

DB = os.environ.get("VERITIQ_DB", "veritiq_pro.db")
STATUSES = ["New", "Under Review", "Accepted", "Planned", "In Progress", "Implemented", "Rejected", "Deferred"]
DEFAULTS = {"pro_enabled": "0", "consent": "0", "impact_tracker": "1", "llm_enabled": "0",
            "llm_provider": "gemini", "llm_endpoint": "http://localhost:11434/v1", "llm_model": ""}
_KEYS = {}   # API keys live in memory only (one per provider), never in the database or on disk

class ProDisabled(Exception): pass

def _c():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row; return con

def init_db():
    with _c() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS settings(k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, ts TEXT, n_charges INT, n_flagged INT, est_total REAL);
        CREATE TABLE IF NOT EXISTS recs(id INTEGER PRIMARY KEY, run_id INT, identified TEXT, tx_date TEXT, vendor TEXT,
          amount REAL, type TEXT, reason TEXT, score REAL, est_impact REAL, status TEXT DEFAULT 'New',
          accepted_on TEXT, implemented_on TEXT, actual_savings REAL, notes TEXT DEFAULT '');
        CREATE INDEX IF NOT EXISTS ix_recs_status ON recs(status);""")

def get(k):
    init_db()
    with _c() as c:
        r = c.execute("SELECT v FROM settings WHERE k=?", (k,)).fetchone()
    return r["v"] if r else DEFAULTS.get(k, "")

def set_(k, v):
    init_db()
    with _c() as c: c.execute("INSERT OR REPLACE INTO settings VALUES(?,?)", (k, str(v)))

def is_enabled(): return get("pro_enabled") == "1" and get("consent") == "1"

def require_pro():
    if not is_enabled():
        raise ProDisabled("Advanced features are off, or storage consent has not been given (Advanced tab).")

def set_api_key(k, provider=None):
    if k: _KEYS[provider or get("llm_provider")] = k.strip()

def save_run(flags, n_charges):
    require_pro(); now = datetime.now().isoformat(timespec="seconds")
    with _c() as c:
        cur = c.execute("INSERT INTO runs(ts,n_charges,n_flagged,est_total) VALUES(?,?,?,?)",
                        (now, n_charges, len(flags), float(flags.est_impact.sum()) if len(flags) else 0.0))
        rid = cur.lastrowid
        for _, r in flags.iterrows():
            c.execute("INSERT INTO recs(run_id,identified,tx_date,vendor,amount,type,reason,score,est_impact) VALUES(?,?,?,?,?,?,?,?,?)",
                      (rid, now, str(r.date)[:10], r.vendor, float(r.amount), r.type, r.reason, float(r.score), float(r.est_impact)))
    return rid

def list_recs(status=None):
    require_pro()
    q, a = "SELECT * FROM recs", ()
    if status and status != "All": q, a = q + " WHERE status=?", (status,)
    with _c() as c: return [dict(r) for r in c.execute(q + " ORDER BY id DESC", a)]

def update_rec(rec_id, status, actual=None, notes=""):
    require_pro()
    if status not in STATUSES: raise ValueError(f"Unknown status: {status}")
    today = datetime.now().date().isoformat()
    with _c() as c:
        r = c.execute("SELECT * FROM recs WHERE id=?", (rec_id,)).fetchone()
        if not r: raise ValueError(f"No recommendation with id {rec_id}")
        acc = r["accepted_on"] or (today if status in ("Accepted", "Planned", "In Progress", "Implemented") else None)
        imp = r["implemented_on"] or (today if status == "Implemented" else None)
        c.execute("UPDATE recs SET status=?, accepted_on=?, implemented_on=?, actual_savings=COALESCE(?,actual_savings), notes=? WHERE id=?",
                  (status, acc, imp, actual, notes or r["notes"], rec_id))

def impact_summary():
    """Accepting a recommendation is NOT counted as savings. 'Reported actual' is what the user entered after
    implementation; VERITIQ does not verify it against ledger data."""
    require_pro()
    with _c() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM recs")]
    acc = [r for r in rows if r["status"] in ("Accepted", "Planned", "In Progress", "Implemented")]
    imp = [r for r in rows if r["status"] == "Implemented"]
    rep = [r for r in imp if r["actual_savings"] is not None]
    proj_rep = sum(r["est_impact"] for r in rep); act = sum(r["actual_savings"] for r in rep)
    return {"identified_total": round(sum(r["est_impact"] for r in rows), 2), "n_recs": len(rows),
            "accepted_est": round(sum(r["est_impact"] for r in acc), 2), "n_implemented": len(imp),
            "pending_est": round(sum(r["est_impact"] for r in acc if r["status"] != "Implemented"), 2),
            "projected_for_reported": round(proj_rep, 2), "reported_actual": round(act, 2),
            "difference": round(act - proj_rep, 2), "n_with_outcome": len(rep)}

def history():
    require_pro()
    with _c() as c: return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC")]

def export_csv():
    require_pro(); recs = list_recs()
    buf = io.StringIO()
    if recs:
        w = csv.DictWriter(buf, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    return buf.getvalue()

def clear_workspace():
    require_pro()
    with _c() as c: c.execute("DELETE FROM recs"); c.execute("DELETE FROM runs")

# ---- Optional AI layer: provider presets (same set as MoneyMaker). Keys stay in memory only. ----
PROVIDERS = {
    "openrouter": {"n": "OpenRouter", "kind": "openai", "model": "meta-llama/llama-3.3-70b-instruct:free", "base": "https://openrouter.ai/api/v1",
                   "note": "Free models end in :free (small daily cap). Use List models.", "key": "https://openrouter.ai/keys"},
    "groq": {"n": "Groq", "kind": "openai", "model": "llama-3.3-70b-versatile", "base": "https://api.groq.com/openai/v1",
             "note": "Free tier, no card (per-model daily limits).", "key": "https://console.groq.com/keys"},
    "openai": {"n": "ChatGPT / OpenAI", "kind": "openai", "model": "gpt-4.1-mini", "base": "https://api.openai.com/v1",
               "note": "Paid API. A ChatGPT subscription does not include API credit.", "key": "https://platform.openai.com/api-keys"},
    "anthropic": {"n": "Claude", "kind": "anthropic", "model": "claude-haiku-5-5", "base": "https://api.anthropic.com/v1",
                  "note": "Paid API. A Claude.ai plan does not include API credit.", "key": "https://console.anthropic.com/settings/keys"},
    "gemini": {"n": "Gemini", "kind": "gemini", "model": "gemini-2.5-flash", "base": "https://generativelanguage.googleapis.com/v1beta",
               "note": "Free tier, no card (Flash models; daily limits change often).", "key": "https://aistudio.google.com/apikey"},
    "custom": {"n": "Custom / local (Ollama)", "kind": "openai", "model": "llama3.1", "base": "http://localhost:11434/v1",
               "note": "Any OpenAI-compatible endpoint. Key is optional for local Ollama.", "key": ""},
}
_UA = "VERITIQ/1.0"

def _cfg():
    pid = get("llm_provider") if get("llm_provider") in PROVIDERS else "gemini"
    p = PROVIDERS[pid]
    base = get("llm_endpoint").rstrip("/") if pid == "custom" else p["base"]
    return pid, p, base, (get("llm_model") or p["model"])

def _http(url, headers, body=None, timeout=25):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": _UA, **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        try: msg = json.loads(e.read().decode()).get("error", {}); msg = msg.get("message", msg) if isinstance(msg, dict) else msg
        except Exception: msg = ""
        raise RuntimeError(f"HTTP {e.code}: {msg or e.reason}")

def _chat(messages, max_tokens=120, timeout=25):
    if get("llm_enabled") != "1": raise RuntimeError("AI is turned off (Setup).")
    pid, p, base, model = _cfg(); key = _KEYS.get(pid, "")
    if not key and pid != "custom": raise RuntimeError(f"{p['n']}: add an API key in Setup.")
    sysmsg = "\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    if p["kind"] == "gemini":
        d = _http(f"{base}/models/{urllib.parse.quote(model)}:generateContent", {"x-goog-api-key": key},
                  {"systemInstruction": {"parts": [{"text": sysmsg}]},
                   "contents": [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in turns],
                   "generationConfig": {"maxOutputTokens": max(max_tokens * 4, 1024)}}, timeout)
        t = "".join(x.get("text", "") for x in (d.get("candidates") or [{}])[0].get("content", {}).get("parts", []))
    elif p["kind"] == "anthropic":
        d = _http(f"{base}/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"},
                  {"model": model, "max_tokens": max_tokens, "system": sysmsg, "messages": turns}, timeout)
        t = "".join(x.get("text", "") for x in d.get("content", []))
    else:
        d = _http(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"} if key else {},
                  {"model": model, "messages": messages, "max_tokens": max_tokens}, timeout)
        t = d["choices"][0]["message"]["content"]
    if not t.strip(): raise RuntimeError(f"{p['n']} returned no text.")
    return t.strip()

def list_models(pid, endpoint="", key=""):
    p = PROVIDERS[pid]; key = key or _KEYS.get(pid, ""); base = (endpoint or p["base"]).rstrip("/") if pid == "custom" else p["base"]
    if not key and pid not in ("openrouter", "custom"): raise RuntimeError("Enter the API key first.")
    if p["kind"] == "gemini":
        d = _http(f"{base}/models?pageSize=100", {"x-goog-api-key": key})
        ids = [m["name"].replace("models/", "") for m in d.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])]
    elif p["kind"] == "anthropic":
        ids = [m["id"] for m in _http(f"{base}/models", {"x-api-key": key, "anthropic-version": "2023-06-01"}).get("data", [])]
    else:
        ids = [m["id"] for m in _http(f"{base}/models", {"Authorization": f"Bearer {key}"} if key else {}).get("data", [])]
        if pid == "openrouter": ids = [i for i in ids if i.endswith(":free")]
    return ids

def llm_test():
    try: _chat([{"role": "user", "content": "Reply with the word OK."}]); return "Connected."
    except Exception as e: return f"Connection failed: {e}"

def llm_reword(vendor, amount, reason):
    """Sends ONLY vendor, amount and the rule-based reason to the configured provider. Falls back to the original reason on any error."""
    try:
        return _chat([{"role": "system", "content": "Rewrite in one plain sentence for a small-business owner. Use only the facts given. Do not add numbers."},
                      {"role": "user", "content": f"{vendor}, ${amount:,.2f}: {reason}"}])
    except Exception:
        return reason

def llm_ask(question, flags_text):
    """Answers a question using ONLY the flagged-charge lines passed in. Raises on error (caller shows it)."""
    return _chat([{"role": "system", "content": "You help a small-business owner review flagged charges. Use ONLY the flagged charges given. "
                   "Never invent vendors, dates or amounts. Be brief and plain. This is not financial advice."},
                  {"role": "user", "content": f"Flagged charges:\n{flags_text}\n\nQuestion: {question}"}], max_tokens=400, timeout=40)
