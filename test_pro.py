import os, tempfile, pandas as pd, pytest
os.environ["VERITIQ_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import pro, veritiq

def flags():
    d = [f"2025-{m:02d}-05" for m in range(1, 13)] + ["2025-12-06"]
    raw = pd.DataFrame({"date": d, "vendor": ["Slack"]*13, "amount": [480]*13})
    return veritiq.flag_transactions(raw, veritiq.train_final_model(), threshold=0.2)

def test_backend_blocks_when_off():
    with pytest.raises(pro.ProDisabled): pro.list_recs()
    with pytest.raises(pro.ProDisabled): pro.save_run(flags(), 13)

def test_consent_required():
    pro.set_("pro_enabled", "1")
    with pytest.raises(pro.ProDisabled): pro.history()
    pro.set_("consent", "1"); assert pro.is_enabled()

def test_flow_and_impact():
    pro.set_("pro_enabled", "1"); pro.set_("consent", "1")
    f = flags(); assert len(f) >= 1
    pro.save_run(f, 13); recs = pro.list_recs(); assert recs and recs[0]["status"] == "New"
    rid = recs[0]["id"]; est = recs[0]["est_impact"]
    pro.update_rec(rid, "Accepted")
    s = pro.impact_summary(); assert s["accepted_est"] == est and s["reported_actual"] == 0   # accepted != realized
    pro.update_rec(rid, "Implemented", actual=est * 0.8)
    s = pro.impact_summary(); assert s["n_implemented"] == 1 and s["difference"] == pytest.approx(-est * 0.2, abs=0.02)
    with pytest.raises(ValueError): pro.update_rec(rid, "Bogus")
    assert "vendor" in pro.export_csv()
    pro.clear_workspace(); assert pro.list_recs() == [] and pro.get("pro_enabled") == "1"

def test_llm_off_falls_back():
    assert pro.llm_reword("Slack", 480, "orig") == "orig"
