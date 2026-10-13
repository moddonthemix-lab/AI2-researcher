# Scholar — the Researcher of the AI family

> *"A scholar and highly intelligent member — his knowledge goes beyond books and the internet.
> He does deep research, digs into the fine details, thinks before he speaks, sticks to the facts,
> sounds nerdy and professional at the same time, and uses credible sources whenever he can."*

Scholar is a private research analyst and business strategist that runs on your own machine
through [Ollama](https://ollama.com). No AI subscription, and your questions never leave the
MacBook. Only the searches and page reads he does go out to the web.

The goal isn't to retrieve *more* information. It's to find what other systems miss: he breaks
a question into dozens of smaller ones, follows connections from source to source, reads whole
documents (tables and appendices included), tries to disprove his own conclusions, checks the
numbers himself, and remembers what he learned for next time.

What a conversation looks like (illustrative — the figures are made up for the example):

```
you> is reselling refurbished iPhones still worth it in 2026?
scholar> Short answer: it can be, but the margin is narrower than the hype suggests.
         Refurbished-phone volumes grew 7% last year [S14] — verified (two independent sources).
         Typical wholesale-to-retail spread on grade-A iPhone 13s is about 18% before fees [S9] —
         reported, single source, so treat it as a lead. After eBay's 13.25% fee and shipping,
         unit_economics puts you at $21 profit per phone, 6.9% margin. Thin. My stress test gives
         a 31% chance of losing money in a given month. Shall I run a full `plan` on it?
```

## Where he fits in the family

| Member | Role | Repo |
|---|---|---|
| **Amo** | main everyday companion | [AMOAI](https://github.com/moddonthemix-lab/AMOAI) |
| **Scholar** | researcher — deep research, business & income analysis | **this repo** |
| Builder | coder & hacker — build and test | coming |
| Baby | the curious one — explores and learns (Scholar's study partner) | coming |
| Butler | oversees the family, suggests improvements | coming |

Scholar is forked from Amo's skeleton (stdlib Python, Ollama tool loop, SQLite, CLI, JSON API).
The rest of the family talks to him through his JSON API today, and through the Family Hub forum later.

## His knowledge, in four layers

Nothing here retrains the model. **Retrieval lets him consult knowledge; it doesn't change the
model's weights.** He "learns" by remembering and looking things up, which is also why you can
inspect and correct what he knows. Fine-tuning can come later, once there's a quality dataset
of his work and your corrections.

| Layer | What | Where |
|---|---|---|
| **1 · Live knowledge** | web search, SEC company filings (EDGAR), scholarly databases (OpenAlex 250M+ works, arXiv, Crossref), Wikipedia, any document or page, `.onion` sites over Tor (opt-in) | `web.py` |
| **2 · Personal library** | your PDFs, manuals, reports, books you have rights to, notes, plus every document he has read, chunked and searchable by meaning | `scholar ingest`, `memory.py` |
| **3 · Structured knowledge** | findings + their evidence (exact quotes, for *and* against), entities + relationships (the research graph), question trees, extracted tables, glossary, all with sources and dates. SQLite now; a graph DB only if research ever needs it | `db.py` |
| **4 · Experience** | lessons from **your** feedback (preferences, corrections, strategies), searches that produced evidence, hypotheses that failed | `scholar feedback`, `memory.py` |

Before every investigation he **recalls only what's relevant**: similar past investigations,
related findings, hypotheses that already failed (so he won't repeat them), searches that worked
on similar topics, glossary terms, your lessons, and matching passages from your library. Semantic
search runs on [Qdrant](https://qdrant.tech) if you run it, or inside `scholar.db` if you don't.

Lessons come only from you (CLI, or the API without a `from` agent). Never from a web page,
and never from another agent.

## How `research` finds what others miss

```
recall ─► plan question tree ─► round 1 ─► gaps + multi-hop ─► round 2 … ─► synthesize ─► skeptic ─► numbers ─► remember
```

**A. Recursive research planning.** The question becomes a tree of sub-questions (6 / 12 / 30 for
quick / standard / deep), each with its own searches. Business questions *always* cover demand,
competitors, customer complaints, costs, margins, regulations, distribution and evidence of actual
willingness to pay. After each round he reviews the tree, notes what's still unanswered and what
the sources raised, and adds follow-up questions.

**B. Multi-hop investigation.** Every source he reads yields entities and relationships (company
*subsidiary_of*, *supplies*, *customer_of*, *competes_with*, *cites*…), saved with source and date
in the research graph. Newly discovered companies, products and studies become new questions:
companies get SEC filing searches, studies get paper searches. That's how a filing leads to a
subsidiary, the subsidiary to a supplier, and the supplier to the opportunity.

**C. Full-document analysis.** PDFs, Word, PowerPoint, Excel and SEC filings are parsed in full
([Docling](https://github.com/docling-project/docling) recovers real tables and layout). Tables are saved as CSV,
appendices and methodology sections are searched, and the reference list is followed back to the
original studies through OpenAlex (most-cited first). Whole documents are downloaded only where the
site's robots.txt permits.

**D. Adversarial fact-checking.** After the preliminary conclusion, a separate skeptic pass tries to
*disprove* each key finding: it searches for contradicting evidence, newer data and methodology
criticism, reads it, and checks every quote. On top of that, rule checks flag single-source claims,
outdated numbers (more than 3 years old), self-reported claims (a company's own site praising
the company), retracted papers, and quotes that don't appear on the page. Findings get downgraded
to *disputed* or *refuted*, and the reason is kept.

**E. Persistent memory.** See the four layers above.

**F. Quantitative analysis.** A controlled Python sandbox (no network, no other programs, writes only
in its own folder, time and memory limits) for statistics, financial models, simulations, data
cleaning and charts on the tables he extracted. Business plans get computed scenarios and a
**Monte Carlo stress test** over ranges, so you see the chance of losing money instead of one
optimistic number.

### What the code enforces (so a small model can't fake it)

- A finding is **verified** only with two independent domains, or one tier-1 source whose quote was
  found in the page. One source = **reported**. No valid source = **assumption**.
- Invented quotes are caught; numbers in the answer that appear in no source are flagged.
- With no evidence, he writes no answer, rather than answering from the model's imagination.
- Every source gets a credibility tier: 1 primary/peer-reviewed/official · 2 reputable secondary ·
  3 general web · 4 unvetted (social media, forums, the dark web, retracted papers).
- Text from pages and documents is fenced off as data. Instructions hidden in a page don't steer him.

## Setup (MacBook)

1. **Ollama + models**
   ```bash
   ollama pull qwen2.5:7b          # 16GB Mac. With 32GB+, qwen2.5:14b reasons noticeably better.
   ollama pull nomic-embed-text    # semantic memory
   ```
   Small models (1–3B) are not enough: they skip tools and get the math wrong.
2. **Scholar** (Python 3.10+)
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e '.[all]'         # Docling, pypdf, pandas/numpy/matplotlib for the sandbox
   # or just: pip install -e .     # core runs on the standard library alone
   ```
3. **Keys and contact** (both free)
   ```bash
   export SCHOLAR_OPENALEX_KEY=...           # https://openalex.org/settings/api
   export SCHOLAR_CONTACT=you@example.com    # SEC EDGAR requires contact info to search filings
   ```
4. **Qdrant** (optional; without it vectors live in `scholar.db`, which is fine for a personal library)
   ```bash
   docker run -d --name qdrant -p 6333:6333 -v ~/.scholar/qdrant:/qdrant/storage qdrant/qdrant
   export SCHOLAR_QDRANT_URL=http://localhost:6333
   scholar reindex                 # if you add Qdrant after he already has memories
   ```
5. **Check and start**
   ```bash
   scholar doctor
   scholar ingest ~/Documents/research     # optional: your own library
   scholar chat
   ```

All settings: [`.env.example`](.env.example). Everything lives in `~/.scholar/`: `scholar.db`,
`reports/`, `documents/` (downloaded originals + extracted tables as CSV), `analysis/` (sandbox runs).

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

Only `.onion` addresses go through Tor. Everything from there is tier 4 and has to be corroborated
before it can count toward a *verified* finding. Scholar reads and reports; he doesn't log in, buy,
download or post anywhere. Keep what you ask him to look at legal where you live.

## Using him

| Command | What it does |
|---|---|
| `scholar chat` | talk to him; he searches, reads, follows citations and filings, runs analyses, records findings |
| `scholar research "question" [--depth quick\|standard\|deep]` | full investigation → cited report |
| `scholar plan "business idea"` | research + startup costs, steps, milestones, scenarios, Monte Carlo, verdict |
| `scholar search "query" --in web\|papers\|wiki\|filings` | search without the model |
| `scholar read URL` | read a page/PDF/filing in full: tables, sections, references |
| `scholar citations DOI [--cited-by]` | what a paper cites (most-cited first), or who cites it since |
| `scholar ingest PATH [--tier 1-4]` | add your own documents to his library |
| `scholar investigations` · `scholar questions [ID]` | past investigations and their question trees (and what's still unanswered) |
| `scholar graph [entity]` | the research graph: top entities, or one entity's connections |
| `scholar finding list\|show\|add\|update` | conclusions with evidence for/against, skeptic flags and history |
| `scholar tables [--source ID]` · `scholar table ID` | tables extracted from documents |
| `scholar analyze script.py --table 3 --table 7` | run your (or his) analysis in the sandbox on those tables |
| `scholar stress --price 180 220 260 --cost 150 --units 5 10 25 --fees-pct 9 13` | Monte Carlo stress test |
| `scholar calc` · `scholar economics` | exact arithmetic, unit economics |
| `scholar feedback "..." --kind preference\|correction\|strategy\|source_note` · `scholar lessons` | teach him; retire a lesson with `lessons --retire ID` |
| `scholar glossary [term]` · `scholar sources` · `scholar reports` · `scholar report ID` | the library |
| `scholar reindex` · `scholar card` · `scholar doctor` | maintenance |

How long it takes depends on your Mac and model. Each source he reads is one model call, plus a few
for planning, gaps, synthesis and the skeptic. Rough guide with `qwen2.5:7b` on Apple Silicon:
**quick** ≈ 15 calls (a few minutes), **standard** ≈ 35 (10–20 min), **deep** ≈ 90 (30–60+ min).
Docling's first run downloads its layout models.

## JSON API (for the rest of the family)

```bash
SCHOLAR_API_HOST=0.0.0.0 SCHOLAR_API_TOKEN=change-me scholar serve     # port 8766 (Amo uses 8765)
```

| Method | Path | |
|---|---|---|
| GET | `/api/health` · `/api/agent` | liveness · identity card (role, skills, permissions, library stats) |
| GET | `/api/findings?q=&status=` · `/api/findings/<id>` | conclusions, with evidence for/against + history |
| GET | `/api/investigations` · `/api/investigations/<id>` | investigations with question tree + graph |
| GET | `/api/questions?status=unanswered` | open questions across investigations |
| GET | `/api/graph?entity=` · `/api/tables?source=` · `/api/glossary?q=` · `/api/lessons` | structured knowledge |
| GET | `/api/sources?q=` · `/api/reports` · `/api/reports/<id>` | sources and reports |
| POST | `/api/ask` `{"text": "...", "from": "amo"}` | ask him something |
| POST | `/api/research` `{"question": "...", "depth": "standard", "from": "baby"}` | full investigation |
| POST | `/api/plan` `{"idea": "...", "from": "amo"}` | researched business plan |
| POST | `/api/feedback` `{"text": "...", "kind": "preference"}` | a lesson. **Boss only:** refused when `from` is another agent |

Send `Authorization: Bearer <SCHOLAR_API_TOKEN>` when a token is set. Always set one before
binding to `0.0.0.0`.

## Development

```bash
pip install -e '.[dev]'
pytest
```

Tests use a fake model, fake web and a fake embedder, so they need neither Ollama nor the internet.

Layout: `research.py` (the investigation method) · `memory.py` (four-layer memory, recall, ingest) ·
`vectors.py` (embeddings; Qdrant or built-in store) · `documents.py` (Docling + fallbacks: tables,
sections, references) · `web.py` (search, OpenAlex, EDGAR, robots.txt, Tor) · `sandbox.py` (controlled
Python) · `calc.py` (business math, Monte Carlo) · `credibility.py` · `db.py` · `tools.py` (what the
model can call in chat) · `assistant.py` (personality) · `agent.py` (identity card) · `server.py` · `cli.py`.
