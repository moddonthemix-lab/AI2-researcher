# Scholar — the Researcher of the AI family

> *"A scholar and highly intelligent member — his knowledge goes beyond books and the internet.
> He does deep research, digs into the fine details, thinks before he speaks, sticks to the facts,
> sounds nerdy and professional at the same time, and uses credible sources whenever he can."*

Scholar is a private research analyst and business strategist that runs on your own machine
through [Ollama](https://ollama.com). No AI subscription, and your questions never leave the
MacBook — only the searches and page reads he does go out to the web.

What a conversation looks like (illustrative — the figures are made up for the example):

```
you> is reselling refurbished iPhones still worth it in 2026?
scholar> Short answer: it can be, but the margin is narrower than the hype suggests.
         Refurbished-phone volumes grew 7% last year [S14] — verified (IDC + Counterpoint agree).
         Typical wholesale-to-retail spread on grade-A iPhone 13s is about 18% before fees [S9] —
         reported, single source, so treat it as a lead. After eBay's 13.25% fee and shipping,
         unit_economics puts you at $21 profit per phone, 6.9% margin. Thin. The real risk is
         grading disputes and returns [S11]. Shall I run a full `plan` on it?
```

## Where he fits in the family

| Member | Role | Repo |
|---|---|---|
| **Amo** | main everyday companion | [AMOAI](https://github.com/moddonthemix-lab/AMOAI) |
| **Scholar** | researcher — deep research, business & income analysis | **this repo** |
| Builder | coder & hacker — build and test | coming |
| Baby | the curious one — explores and learns (Scholar's study partner) | coming |
| Butler | oversees the family, suggests improvements | coming |

Scholar is forked from Amo's skeleton (same stdlib-only Python, Ollama tool loop, SQLite
memory, CLI and JSON API) with the studio/trading/reselling trackers swapped for a research
library, web research tools and a deep-research pipeline. The rest of the family talks to him
through his JSON API today, and through the Family Hub forum later.

## What makes him a researcher and not just a prompt

- **He reads, not just searches.** Web (your own SearXNG, or DuckDuckGo), academic papers
  (arXiv + Crossref), Wikipedia, any URL or PDF — and `.onion` sites when you turn Tor on.
- **Every source gets a credibility tier** — 1 primary/peer-reviewed/official, 2 reputable
  secondary, 3 general web, 4 unvetted (social media, forums, the dark web).
- **Facts vs. assumptions is enforced by code.** A finding is *verified* only with two
  independent domains, or one tier-1 source whose quote was found on the actual page.
  One source = *reported*. No source = *assumption*. The model can't override this.
- **Quotes are checked.** If the model "quotes" something that isn't on the page, it's flagged.
- **Numbers come from code.** Margins, break-even, revenue scenarios and payback are calculated,
  never guessed.
- **A research library that remembers and updates.** Every source he has read and every finding
  he has recorded is searchable. When new evidence comes in, findings are revised and the history
  keeps *why* he changed his mind.
- **He's careful.** Web content is treated as untrusted data (prompt-injection on a page doesn't
  steer him); he can't reach your own network through a link; he has no shell and only writes
  to `~/.scholar`.

## Setup (MacBook)

1. Install Ollama and pull a model that handles tools and JSON well:
   ```bash
   ollama pull qwen2.5:7b         # 16GB Mac. With 32GB+, qwen2.5:14b reasons noticeably better.
   ```
2. Install Scholar (Python 3.10+):
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .              # core: standard library only
   pip install -e '.[pdf]'       # optional: read PDFs
   ```
3. Check and start:
   ```bash
   scholar doctor
   scholar chat
   ```

Settings are environment variables; see [`.env.example`](.env.example). Everything lives in
`~/.scholar/` — `scholar.db` (the library) and `reports/` (markdown reports).

### Better search: your own SearXNG (recommended)

DuckDuckGo works out of the box but rate-limits automated searches. A private
[SearXNG](https://docs.searxng.org) instance queries many engines at once and is yours:

```bash
docker run -d --name searxng -p 8888:8080 searxng/searxng
# enable the JSON format: add "- json" under search.formats in the container's settings.yml, restart
export SCHOLAR_SEARXNG_URL=http://localhost:8888
```

### Dark web sources (optional, read-only)

```bash
brew install tor && brew services start tor
export SCHOLAR_TOR_PROXY=127.0.0.1:9050
scholar read http://<something>.onion/
```

Only `.onion` addresses go through Tor. Everything from there is tier 4 and has to be
corroborated before it can count toward a *verified* finding. Scholar reads and reports; he
doesn't log in, buy, download or post anywhere. Keep what you ask him to look at legal where you live.

## Using him

| Command | What it does |
|---|---|
| `scholar chat` | talk to him; he searches, reads, calculates and records findings as needed |
| `scholar research "question" [--depth quick\|standard\|deep]` | full investigation → cited report |
| `scholar plan "business idea"` | research + startup costs, steps, milestones, computed revenue scenarios, verdict |
| `scholar search "query" [--in web\|papers\|wiki]` | search without the model |
| `scholar read URL` | read a page/PDF into the library |
| `scholar finding list [query] [--status verified]` · `finding show 3` | browse conclusions with sources and history |
| `scholar finding update 3 --status disputed --reason "new survey"` | revise a conclusion yourself |
| `scholar sources [query]` · `scholar reports` · `scholar report 2` | the library |
| `scholar calc "(320-210-30)/320*100"` · `scholar economics --price 45 --cost 12 --fees-pct 13` | exact math |
| `scholar remember ...` · `recall` · `forget` | notes about you and your interests |
| `scholar card` | his identity card + library stats |

### How `research` works

1. **plan** — breaks the question into sub-questions and queries, including one that looks for
   *counter-evidence* (he challenges the obvious answer)
2. **gather** — web, papers and Wikipedia; every hit is saved as a source
3. **read** — the most credible pages, at most two per domain so one site can't dominate
4. **extract** — facts plus exact quotes from each page
5. **check** — quotes are matched against the page text
6. **synthesize** — answer, key findings, disagreements, assumptions, risks, open questions
7. **grade** — the code assigns verified / reported / assumption / disputed from the evidence
8. **save** — findings go into the library; the report goes to the database and `~/.scholar/reports/`

`quick` reads ~4 sources, `standard` ~8, `deep` ~14.

## JSON API (for the rest of the family)

```bash
SCHOLAR_API_HOST=0.0.0.0 SCHOLAR_API_TOKEN=change-me scholar serve     # port 8766
```

| Method | Path | |
|---|---|---|
| GET | `/api/health` | liveness |
| GET | `/api/agent` | identity card: role, specialties, permissions, library stats |
| GET | `/api/findings?q=&status=` · `/api/findings/<id>` | conclusions (with sources + history) |
| GET | `/api/sources?q=` | what he's read |
| GET | `/api/reports` · `/api/reports/<id>` | reports (markdown + structured data) |
| POST | `/api/ask` `{"text": "...", "from": "amo"}` | ask him something |
| POST | `/api/research` `{"question": "...", "depth": "standard", "from": "baby"}` | full investigation |
| POST | `/api/plan` `{"idea": "...", "from": "amo"}` | researched business plan |

`from` tells him who's asking — he's warmer and more explanatory with Baby, for example.
Send `Authorization: Bearer <SCHOLAR_API_TOKEN>` when a token is set. Always set one before
binding to `0.0.0.0`.

## Development

```bash
pip install -e '.[dev]'
pytest
```

Tests use a fake model and a fake web, so they need neither Ollama nor the internet.

Layout: `scholar/research.py` (deep-research pipeline) · `scholar/web.py` (search, reading, Tor) ·
`scholar/credibility.py` (source tiers) · `scholar/db.py` (library) · `scholar/tools.py` (what the
model can call) · `scholar/assistant.py` (personality + chat loop) · `scholar/calc.py` (business math) ·
`scholar/agent.py` (identity card) · `scholar/server.py` · `scholar/cli.py`.
