import re

from scholar.config import Config
from scholar.research import Researcher, grade, quote_found

from .conftest import PAGES, RoutingLLM


def _extract(prompt):
    """Pretend to read: quote the real page text, except invent a quote for the blog."""
    sid = int(re.search(r"\[S(\d+)\]", prompt).group(1))
    if "blog.example.com" in prompt:
        return {"relevant": True, "facts": [{"fact": "prices rose", "quote": "prices tripled overnight everywhere"}]}
    for url, text in PAGES.items():
        if url.split("/")[2].removeprefix("www.") in prompt:
            return {"relevant": True, "facts": [{"fact": f"fact from S{sid}", "quote": text[:70]}]}
    return {"relevant": False}


def _ids(prompt):
    return {d: int(i) for i, d in re.findall(r"\[S(\d+)\] tier \d (\S+):", prompt)}


def _synth(prompt):
    ids = _ids(prompt)
    return {
        "answer": f"Prices rose about 12% [S{ids['bls.gov']}].",
        "key_findings": [
            {"claim": "Resale prices rose 12% in 2025", "source_ids": [ids["bls.gov"]], "confidence": 0.99},
            {"claim": "Market is ~$10B", "source_ids": [f"S{ids['reuters.com']}"], "confidence": 0.9},
            {"claim": "Everyone gets rich", "source_ids": [], "confidence": 0.9},
            {"claim": "Made up source", "source_ids": [999], "confidence": 0.9},
        ],
        "risks": ["Hype cycles"], "assumptions": ["US market only"],
        "startup_costs": [{"item": "Inventory", "cost": 500, "source_ids": []}, {"item": "Bad", "cost": "lots"}],
        "scenarios": [{"name": "expected", "units_per_month": 10, "price": 200, "unit_cost": 150,
                       "fees_pct": 10, "fixed_costs_per_month": 50}],
        "verdict": "maybe — margins are thin",
    }


def _plan(prompt):
    return {"restated": "How fast are sneaker resale prices rising?", "sub_questions": ["price trend"],
            "queries": [{"q": "sneaker resale prices", "where": "web"}, {"q": "resale study", "where": "papers"},
                        {"q": "sneakers", "where": "wiki"}]}


def test_quote_found():
    text = "The Bureau reports that resale sneaker prices rose 12 percent in 2025."
    assert quote_found("resale sneaker prices rose 12 percent", text)
    assert quote_found("Resale  sneaker prices, rose 12 percent!", text)
    assert not quote_found("prices tripled overnight everywhere", text)
    assert not quote_found("12", text)


def test_grade_rules():
    srcs = {1: {"domain": "bls.gov", "tier": 1}, 2: {"domain": "reuters.com", "tier": 2},
            3: {"domain": "blog.com", "tier": 3}, 4: {"domain": "reddit.com", "tier": 4}}
    assert grade([], srcs, set()) == "assumption"
    assert grade([99], srcs, set()) == "assumption"
    assert grade([1], srcs, {1}) == "verified"           # primary with a checked quote
    assert grade([1], srcs, set()) == "reported"         # primary but quote unchecked
    assert grade([2, 3], srcs, set()) == "verified"      # two independent domains
    assert grade([3, 4], srcs, set()) == "reported"      # tier 4 doesn't count toward corroboration
    assert grade([2, 3], srcs, set(), conflicting=True) == "disputed"


def test_investigate_end_to_end(tb, tmp_path):
    llm = RoutingLLM(plan=_plan, extract=_extract, synth=_synth)
    logs = []
    cfg = Config(home=tmp_path)
    r = Researcher(tb, llm, cfg, log=logs.append).investigate("Are sneaker resale prices rising?", "quick", "plan")
    d = r["data"]

    by_claim = {f["claim"]: f for f in d["findings"]}
    assert by_claim["Resale prices rose 12% in 2025"]["status"] == "verified"
    assert by_claim["Resale prices rose 12% in 2025"]["confidence"] == 0.95  # capped by code
    assert by_claim["Market is ~$10B"]["status"] == "reported"
    assert by_claim["Everyone gets rich"]["status"] == "assumption"
    assert by_claim["Everyone gets rich"]["confidence"] == 0.4
    assert by_claim["Made up source"]["status"] == "assumption" and by_claim["Made up source"]["source_ids"] == []
    # the invented blog quote is caught
    assert d["quotes_checked"] == "2/3"  # bls + reuters real, blog invented (doi page unreadable)
    # Wikipedia failing (rate limit) doesn't sink the research
    assert any("skipped" in m for m in logs)
    # plan numbers are computed by code, junk costs dropped
    assert d["startup_total"] == 500 and [c["item"] for c in d["startup_costs"]] == ["Inventory"]
    s = d["scenarios"][0]
    assert s["revenue_per_month"] == 2000 and s["profit_per_month"] == 250 and s["payback_months"] == 2
    # findings are in the library, report saved in DB and on disk
    assert len(tb.db.findings(limit=50)) == 4
    assert tb.db.get_report(r["report_id"])["mode"] == "plan"
    assert r["path"] and open(r["path"]).read().startswith("# Business plan:")
    md = r["markdown"]
    assert "✅ verified" in md and "## Revenue scenarios (computed)" in md and "$500.00" in md
    assert "## Sources" in md and "bls.gov" in md
    # web text reaches the model fenced as untrusted data
    extract_prompts = [p for p in llm.prompts if "Extract only facts" in p]
    assert all("<<<SOURCE" in p and "untrusted" in p for p in extract_prompts)


def test_investigate_survives_empty_model_output(tb, tmp_path):
    r = Researcher(tb, RoutingLLM(), Config(home=tmp_path)).investigate("anything?", "quick")
    assert r["data"]["findings"] == [] and "not enough to answer" in r["markdown"]


def test_choose_prefers_credible_and_diverse():
    hits = [{"url": f"u{i}", "domain": d, "tier": t, "rank": i}
            for i, (d, t) in enumerate([("a.com", 3), ("a.com", 3), ("a.com", 3), ("gov.gov", 1), ("b.org", 2)])]
    picked = Researcher.choose(hits, 4)
    assert [h["domain"] for h in picked] == ["gov.gov", "b.org", "a.com", "a.com"]
