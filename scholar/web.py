"""Reading the outside world (standard library only).

Everything here is read-only: Scholar can search and fetch pages, never
post, log in or download programs. Fetched text is untrusted data — the
assistant prompt says so, and pages on your own network (localhost, LAN
addresses) are refused so a malicious page can't point him at your devices.

Sources:
- web search: your own SearXNG if ``SCHOLAR_SEARXNG_URL`` is set, else DuckDuckGo
- Wikipedia (good map of a topic; follow its citations)
- papers: arXiv + Crossref (journal articles with DOIs)
- any URL: HTML pages, plain text/JSON, PDFs (with the optional ``pypdf`` extra)
- .onion sites, only when a Tor SOCKS proxy is configured (``SCHOLAR_TOR_PROXY``)
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import socket
import ssl
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser

from . import credibility

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) Scholar/0.1 (personal research assistant)"
# Public APIs (Wikipedia, arXiv, Crossref) ask bots to identify themselves honestly.
API_AGENT = "Scholar/0.1 (personal research assistant; https://github.com/moddonthemix-lab/AI2-researcher)"
MAX_BYTES = 4_000_000
MAX_TEXT = 60_000  # characters of readable text kept per source


class WebError(RuntimeError):
    pass


# ---------------------------------------------------------------- HTML -> text
class _Readable(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside",
            "iframe", "button", "select", "template"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
             "table", "section", "article", "blockquote", "pre", "dd", "dt", "figcaption"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.in_title = False
        self.title = ""
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key and a.get("content"):
                self.meta[key] = a["content"]
        if tag in self.BLOCK:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3") and not self.skip:
            self.parts.append("## ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag == "title":
            self.in_title = False
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.parts.append(data)


def html_to_text(html: str) -> dict:
    p = _Readable()
    p.feed(html)
    text = "".join(p.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop menu crumbs: very short lines that aren't headings.
    text = "\n".join(ln for ln in lines if len(ln) > 30 or ln.startswith("##") or not ln).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    m = p.meta
    return {
        "title": (m.get("og:title") or p.title).strip(),
        "published": (m.get("article:published_time") or m.get("date") or m.get("dc.date")
                      or m.get("citation_publication_date") or "")[:10],
        "description": m.get("description") or m.get("og:description") or "",
        "text": text,
    }


def _strip_tags(s: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


# ------------------------------------------------------------------ Tor / SOCKS
def _socks5_connect(proxy: str, host: str, port: int, timeout: float) -> socket.socket:
    """Open a TCP connection to host:port through a SOCKS5 proxy (remote DNS, so .onion works)."""
    phost, _, pport = proxy.rpartition(":")
    sock = socket.create_connection((phost or "127.0.0.1", int(pport or 9050)), timeout=timeout)
    try:
        sock.sendall(b"\x05\x01\x00")
        if sock.recv(2) != b"\x05\x00":
            raise WebError("Tor proxy refused the handshake")
        h = host.encode("idna")
        sock.sendall(b"\x05\x01\x00\x03" + bytes([len(h)]) + h + struct.pack(">H", port))
        reply = sock.recv(4)
        if len(reply) < 4 or reply[1] != 0:
            raise WebError(f"Tor could not reach {host} (SOCKS error {reply[1] if len(reply) > 1 else '?'})")
        skip = {1: 4, 4: 16}.get(reply[3])
        skip = skip if skip is not None else sock.recv(1)[0]
        sock.recv(skip + 2)
        return sock
    except Exception:
        sock.close()
        raise


# ------------------------------------------------------------------------ Web
class Web:
    def __init__(self, searxng_url: str = "", tor_proxy: str = "", timeout: float = 25,
                 allow_private: bool = False):
        self.searxng_url = searxng_url.rstrip("/")
        self.tor_proxy = tor_proxy
        self.timeout = timeout
        self.allow_private = allow_private
        web = self

        class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
            # A public page must not be able to bounce Scholar onto your own network.
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                web._check_target(newurl)
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        self._opener = urllib.request.build_opener(_CheckedRedirects)

    # ---- low-level GET
    def _check_target(self, url: str) -> urllib.parse.ParseResult:
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise WebError(f"only http(s) URLs can be read, not {url!r}")
        if u.hostname.endswith(".onion"):
            if not self.tor_proxy:
                raise WebError(".onion sites need Tor: run Tor and set SCHOLAR_TOR_PROXY=127.0.0.1:9050")
            return u
        if not self.allow_private:
            try:
                infos = socket.getaddrinfo(u.hostname, None)
            except OSError:
                infos = []  # can't resolve locally (e.g. behind a proxy) — the proxy decides
            for info in infos:
                ip = ipaddress.ip_address(info[4][0].split("%")[0])
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    raise WebError(f"refusing to read {u.hostname}: it is on your private network")
        return u

    def get(self, url: str, accept: str = "*/*", agent: str = USER_AGENT) -> tuple[str, str, bytes]:
        """Return (final_url, content_type, body)."""
        u = self._check_target(url)
        if u.hostname.endswith(".onion"):
            return self._get_via_tor(url, accept)
        req = urllib.request.Request(url, headers={"User-Agent": agent, "Accept": accept,
                                                   "Accept-Language": "en-US,en;q=0.8"})
        for attempt in (1, 2):
            try:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    return resp.geturl(), resp.headers.get("Content-Type", ""), resp.read(MAX_BYTES)
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt == 1:  # rate limited: wait once, politely
                    retry = e.headers.get("Retry-After", "")
                    time.sleep(min(float(retry) if retry.isdigit() else 2.0, 8.0))
                    continue
                raise WebError(f"{url} returned HTTP {e.code}"
                               + (" (rate limited — try again in a minute)" if e.code == 429 else "")) from e
            except (urllib.error.URLError, OSError) as e:
                raise WebError(f"could not reach {url}: {getattr(e, 'reason', e)}") from e
        raise WebError(f"could not read {url}")

    def _get_via_tor(self, url: str, accept: str, redirects: int = 3) -> tuple[str, str, bytes]:
        u = urllib.parse.urlparse(url)
        port = u.port or (443 if u.scheme == "https" else 80)
        try:
            sock = _socks5_connect(self.tor_proxy, u.hostname, port, self.timeout)
            if u.scheme == "https":
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=u.hostname)
            conn = http.client.HTTPConnection(u.hostname, port, timeout=self.timeout)
            conn.sock = sock
            path = (u.path or "/") + (f"?{u.query}" if u.query else "")
            conn.request("GET", path, headers={"User-Agent": USER_AGENT, "Accept": accept,
                                               "Host": u.netloc})
            resp = conn.getresponse()
            if resp.status in (301, 302, 303, 307, 308) and redirects and resp.getheader("Location"):
                nxt = urllib.parse.urljoin(url, resp.getheader("Location"))
                conn.close()
                self._check_target(nxt)
                if urllib.parse.urlparse(nxt).hostname.endswith(".onion"):
                    return self._get_via_tor(nxt, accept, redirects - 1)
                return self.get(nxt, accept)
            if resp.status >= 400:
                raise WebError(f"{url} returned HTTP {resp.status}")
            body = resp.read(MAX_BYTES)
            ctype = resp.getheader("Content-Type", "")
            conn.close()
            return url, ctype, body
        except (OSError, http.client.HTTPException) as e:
            raise WebError(f"could not reach {url} over Tor: {e}") from e

    def get_json(self, url: str) -> dict:
        _, _, body = self.get(url, "application/json", API_AGENT)
        try:
            return json.loads(body)
        except ValueError as e:
            raise WebError(f"{url} did not return JSON") from e

    # ---- search
    def search(self, query: str, limit: int = 8) -> list[dict]:
        """General web search. Each result: title, url, snippet, domain, tier."""
        results = self._searxng(query, limit) if self.searxng_url else self._duckduckgo(query, limit)
        return [_with_rating(r) for r in results[:limit]]

    def _searxng(self, query: str, limit: int) -> list[dict]:
        data = self.get_json(f"{self.searxng_url}/search?" + urllib.parse.urlencode(
            {"q": query, "format": "json"}))
        return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")}
                for r in data.get("results", [])[:limit] if r.get("url")]

    def _duckduckgo(self, query: str, limit: int) -> list[dict]:
        _, _, body = self.get("https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query}),
                              "text/html")
        results = parse_duckduckgo(body.decode("utf-8", "replace"))
        if not results and "result__a" not in body.decode("utf-8", "replace"):
            raise WebError("DuckDuckGo sent no results page (it rate-limits automated searches). "
                           "Try again shortly, or run your own SearXNG and set SCHOLAR_SEARXNG_URL.")
        return results[:limit]

    def wikipedia(self, query: str, limit: int = 3) -> list[dict]:
        data = self.get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1, "exlimit": "max",
            "inprop": "url", "format": "json"}))
        pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        return [_with_rating({"title": p.get("title", ""), "url": p.get("fullurl", ""),
                              "snippet": (p.get("extract") or "")[:1500]}) for p in pages]

    def papers(self, query: str, limit: int = 6) -> list[dict]:
        """Academic papers from arXiv and Crossref, interleaved."""
        out: list[dict] = []
        errors = []
        for fn in (self._arxiv, self._crossref):
            try:
                out.append(fn(query, limit))
            except WebError as e:
                errors.append(str(e))
                out.append([])
        merged = [r for pair in zip(*out) for r in pair] + out[0][len(out[1]):] + out[1][len(out[0]):]
        if not merged and errors:
            raise WebError("; ".join(errors))
        return merged[:limit]

    def _arxiv(self, query: str, limit: int) -> list[dict]:
        _, _, body = self.get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": " AND ".join(f"all:{w}" for w in re.findall(r"\w+", query)[:8]),
             "max_results": limit}), "application/atom+xml", API_AGENT)
        return parse_arxiv(body)

    def _crossref(self, query: str, limit: int) -> list[dict]:
        data = self.get_json("https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query": query, "rows": limit,
             "select": "DOI,title,abstract,published,container-title,author,URL"}))
        return parse_crossref(data)

    # ---- read one page
    def read(self, url: str) -> dict:
        """Fetch a URL and return readable text plus credibility info."""
        final, ctype, body = self.get(url)
        ctype = ctype.lower()
        if "pdf" in ctype or body[:5] == b"%PDF-":
            doc = {"title": "", "published": "", "text": _pdf_text(body)}
        elif "html" in ctype or body.lstrip()[:1] == b"<":
            doc = html_to_text(body.decode(_charset(ctype), "replace"))
        else:
            doc = {"title": "", "published": "", "text": body.decode(_charset(ctype), "replace")}
        if _is_bot_wall(doc):
            raise WebError(f"{final} showed a bot-check page instead of the content")
        rating = credibility.rate(final)
        kind = "onion" if rating["domain"].endswith(".onion") else (
            "paper" if rating["domain"] in ("arxiv.org", "doi.org") or "pdf" in ctype else
            "reference" if rating["domain"].endswith("wikipedia.org") else "web")
        return {"url": final, "title": doc["title"] or final, "published": doc.get("published", ""),
                "text": doc["text"][:MAX_TEXT], "kind": kind, **rating}


# ------------------------------------------------------------------- parsers
def parse_duckduckgo(html: str) -> list[dict]:
    out = []
    blocks = re.split(r'<div[^>]+class="[^"]*result results_links', html)[1:]
    for b in blocks:
        a = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
        if not a:
            continue
        href = unescape(a.group(1))
        q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        url = q.get("uddg", [href])[0]
        if url.startswith("//"):
            url = "https:" + url
        if "duckduckgo.com/y.js" in url:  # ads
            continue
        snip = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', b, re.S)
        out.append({"title": _strip_tags(a.group(2)), "url": url,
                    "snippet": _strip_tags(snip.group(1)) if snip else ""})
    return out


_ATOM = "{http://www.w3.org/2005/Atom}"


def parse_arxiv(xml: bytes) -> list[dict]:
    root = ET.fromstring(xml)
    out = []
    for e in root.iter(f"{_ATOM}entry"):
        get = lambda tag: re.sub(r"\s+", " ", (e.findtext(f"{_ATOM}{tag}") or "")).strip()  # noqa: E731
        authors = [a.findtext(f"{_ATOM}name") or "" for a in e.findall(f"{_ATOM}author")]
        out.append(_with_rating({
            "title": get("title"), "url": get("id"), "snippet": get("summary")[:1500],
            "published": get("published")[:10], "authors": ", ".join(authors[:4]),
            "venue": "arXiv preprint (not peer-reviewed)"}))
    return out


def parse_crossref(data: dict) -> list[dict]:
    out = []
    for it in data.get("message", {}).get("items", []):
        parts = (it.get("published") or {}).get("date-parts") or [[]]
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])]
        url = f"https://doi.org/{it['DOI']}" if it.get("DOI") else it.get("URL", "")
        if not url:
            continue
        out.append({
            "title": " ".join(it.get("title") or []), "url": url,
            "snippet": _strip_tags(it.get("abstract", ""))[:1500],
            "published": "-".join(str(p) for p in parts[0]), "authors": ", ".join(authors[:4]),
            "venue": " ".join(it.get("container-title") or []),
            "domain": "doi.org", "tier": 1, "tier_label": credibility.TIER_LABELS[1]})
    return out


def _with_rating(r: dict) -> dict:
    return {**r, **credibility.rate(r.get("url", ""))}


_BOT_WALLS = ("client challenge", "just a moment", "attention required", "verify you are human",
              "are you a robot", "access denied", "enable javascript and cookies")


def _is_bot_wall(doc: dict) -> bool:
    title = doc.get("title", "").lower()
    return any(w in title for w in _BOT_WALLS) or (
        len(doc.get("text", "")) < 600 and any(w in doc.get("text", "").lower() for w in _BOT_WALLS))


def _charset(ctype: str) -> str:
    m = re.search(r"charset=([\w-]+)", ctype)
    return m.group(1) if m else "utf-8"


def _pdf_text(body: bytes) -> str:
    try:
        from io import BytesIO

        from pypdf import PdfReader
    except ImportError as e:
        raise WebError("reading PDFs needs the pdf extra: pip install -e '.[pdf]'") from e
    reader = PdfReader(BytesIO(body))
    return "\n\n".join((p.extract_text() or "") for p in reader.pages[:60])
