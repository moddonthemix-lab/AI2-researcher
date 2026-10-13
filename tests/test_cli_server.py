import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from scholar.cli import main
from scholar.config import Config
from scholar.db import Store
from scholar.research import Researcher
from scholar.server import make_handler
from scholar.tools import Toolbox

from .conftest import FakeWeb, RoutingLLM


def test_cli_library_roundtrip(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SCHOLAR_HOME", str(tmp_path))
    assert main(["remember", "Boss", "likes", "sneakers", "--topic", "interests"]) == 0
    assert main(["finding", "add", "Resale", "margins", "are", "thin", "--status", "reported", "--source", "3"]) == 0
    assert main(["finding", "update", "1", "--status", "disputed", "--reason", "new survey disagrees"]) == 0
    assert main(["finding", "show", "1"]) == 0
    assert main(["calc", "(320-210)/320*100"]) == 0
    assert main(["economics", "--price", "45", "--cost", "12", "--fees-pct", "13"]) == 0
    assert main(["card"]) == 0
    assert main(["calc", "open('x')"]) == 1
    out = capsys.readouterr().out
    assert "F1 is now disputed" in out and "new survey disagrees" in out and "34.375" in out
    assert '"profit_per_unit": 27.15' in out and '"family_role": "Researcher"' in out
    assert (tmp_path / "scholar.db").exists()


def _serve(store, researcher=None, token=""):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, None, researcher, token))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _req(url, method="GET", body=None, token=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_api_card_findings_and_research(tmp_path):
    store = Store()
    fid = store.add_finding("Sneaker margins are thin", "reported", 0.6, source_ids=[])
    researcher = Researcher(Toolbox(store, FakeWeb()), RoutingLLM(), Config(home=tmp_path))
    httpd, base = _serve(store, researcher)
    try:
        assert _req(base + "/api/health") == (200, {"ok": True, "agent": "scholar"})
        card = _req(base + "/api/agent")[1]
        assert card["id"] == "scholar" and card["library"]["findings"] == 1
        assert _req(base + "/api/findings?q=margins")[1][0]["id"] == fid
        f = _req(f"{base}/api/findings/{fid}")[1]
        assert f["history"][0]["status"] == "reported"
        status, body = _req(base + "/api/research", "POST", {"question": "q?", "depth": "quick", "from": "amo"})
        assert status == 200 and body["from"] == "scholar" and body["report_id"]
        assert _req(f"{base}/api/reports/{body['report_id']}")[0] == 200
        assert _req(base + "/api/research", "POST", {"question": "q?", "depth": "forever"})[0] == 400
        assert _req(base + "/api/ask", "POST", {"text": "hi"})[0] == 503
        assert _req(base + "/api/nope")[0] == 404
    finally:
        httpd.shutdown()


def test_api_token():
    httpd, base = _serve(Store(), token="s3cret")
    try:
        assert _req(base + "/api/agent")[0] == 401
        assert _req(base + "/api/agent", token="s3cret")[0] == 200
    finally:
        httpd.shutdown()
