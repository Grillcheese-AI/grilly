# elephant-coder 0.2.1 Plan D: Research & Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RSS news reader, research notes MCP tools, external model validation via OpenRouter, and news briefing generation.

**Architecture:** `news_reader.py` handles RSS fetching, XML parsing, HTML-to-text extraction, and deduplication. `research_engine.py` handles OpenRouter API calls for external validation/auditing. Both store results in the global knowledge store. MCP tools expose everything to Claude.

**Tech Stack:** Python 3.10+, httpx (async HTTP), xml.etree (RSS parsing), re (HTML stripping)

**Plugin source:** `C:\Users\grill\grilly-plugins\elephant-coder\`

**Depends on:** Plans A + B + C (complete)

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `news_reader.py` | RSS fetching, XML parsing, HTML text extraction, deduplication, briefing generation |
| `research_engine.py` | OpenRouter API calls, external validation, auditing |
| `tests/test_news_reader.py` | News reader tests |
| `tests/test_research_engine.py` | Research engine tests |

### Modified Files

| File | Changes |
|------|---------|
| `server.py` | Add `get_news_briefing()`, `take_note()`, `recall_notes()`, `get_external_review()`, `request_audit()` MCP tools |
| `hooks/hooks.json` | Add news fetching to SessionStart |
| `pyproject.toml` | Add httpx dependency, register new modules |
| `settings.py` | Add RSS feed defaults to DEFAULT_SETTINGS |

---

## Task 1: News Reader Module

**Files:**
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\news_reader.py`
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\tests\test_news_reader.py`

Tests must cover:
- `parse_rss_xml` — parse a sample RSS XML string into article dicts
- `strip_html` — convert HTML to plain text
- `truncate_text` — truncate to max chars at word boundary
- `deduplicate_articles` — filter out already-seen URLs

Implementation:
- `fetch_feeds(urls, max_per_feed=5)` — fetch all feeds concurrently with httpx
- `parse_rss_xml(xml_text)` — extract title, link, summary, published from RSS/Atom XML
- `strip_html(html)` — regex-based HTML to text (strip script/style/nav, extract text)
- `fetch_full_article(url)` — GET the URL, extract article text from HTML
- `deduplicate_articles(articles, existing_urls)` — filter already-stored
- `generate_briefing(articles, max_articles=20)` — format as readable briefing text

Keep it under 200 lines. Use xml.etree.ElementTree for parsing (stdlib, no deps). Use httpx for HTTP.

Commit: "feat: add news reader with RSS parsing and full article extraction"

---

## Task 2: Research Engine (OpenRouter)

**Files:**
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\research_engine.py`
- Create: `C:\Users\grill\grilly-plugins\elephant-coder\tests\test_research_engine.py`

Tests must cover:
- `build_review_prompt` — constructs the adversarial review prompt correctly
- `build_audit_prompt` — constructs the audit prompt correctly
- `parse_review_response` — extracts issues from model response text

Implementation:
- `get_external_review(plan, objectives, evidence, api_key, model)` — call OpenRouter
- `request_audit(task_desc, files_changed, test_results, api_key, model)` — call OpenRouter for audit
- `build_review_prompt(plan, objectives, evidence)` — construct the prompt
- `build_audit_prompt(task_desc, files_changed, test_results)` — construct audit prompt
- `parse_review_response(text)` — extract structured issues

Default model: `google/gemini-3.1-flash-lite-preview`
Keep it under 150 lines.

Commit: "feat: add research engine with OpenRouter external validation"

---

## Task 3: MCP Tools for Research + News

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\server.py`

Add these MCP tools:

```python
@mcp.tool()
def get_news_briefing(topics: str = "") -> str:
    """Fetch and summarize today's news from configured RSS feeds.

    Reads RSS feeds, follows links to full articles when needed,
    stores new articles as research notes, returns a briefing.

    Args:
        topics: Optional comma-separated topic filter
    """

@mcp.tool()
def take_note(topic: str, summary: str, source: str = "", tags: str = "") -> str:
    """Save a research note for future reference.

    Use this when you find something interesting during work —
    papers, techniques, ideas, patterns. Notes persist across sessions
    and are cross-referenced for creative sparks.

    Args:
        topic: Brief topic name
        summary: What you learned
        source: Where you found it (URL, paper ID, etc.)
        tags: Comma-separated tags for categorization
    """

@mcp.tool()
def recall_notes(query: str, limit: int = 10) -> str:
    """Search your research notes.

    Args:
        query: Search keywords
        limit: Max results
    """

@mcp.tool()
def get_external_review(plan: str, context: str = "") -> str:
    """Get an adversarial review of a plan from an external model.

    Sends the plan to Gemini 3.1 Flash Lite via OpenRouter for
    an independent review. The reviewer looks for flaws, missed
    edge cases, and incorrect assumptions.

    Requires external_validation.enabled=true and an OpenRouter API key.

    Args:
        plan: The plan text to review
        context: Additional context (objectives, constraints)
    """

@mcp.tool()
def request_audit(task_id: str, files_changed: str = "", test_results: str = "") -> str:
    """Request an independent audit of completed work.

    Sends task description + changes to Gemini 3.1 Flash Lite
    to verify the implementation matches the spec.

    Args:
        task_id: Task ID that was completed
        files_changed: Summary of files changed
        test_results: Test output
    """
```

Commit: "feat: add news briefing, research notes, and external review MCP tools"

---

## Task 4: Update Settings Defaults + Hooks + pyproject

**Files:**
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\settings.py` — add RSS defaults
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\hooks\hooks.json` — add news to SessionStart
- Modify: `C:\Users\grill\grilly-plugins\elephant-coder\pyproject.toml` — add httpx dep + modules

Settings defaults to add:
```python
"rss_feeds": [
    "https://hackernoon.com/feed",
    "https://globalnews.ca/feed/",
    "https://feedx.net/rss/ap.xml",
    "https://www.theverge.com/rss/index.xml",
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://techcrunch.com/feed/",
    "https://blog.bytebytego.com/feed",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://www.wired.com/feed/category/ideas/latest/rss",
    "https://rss.arxiv.org/rss/math.QA",
    "https://rss.arxiv.org/rss/cs.ai",
    "https://www.reddit.com/r/news/.rss",
    "https://www.reddit.com/r/LocalLLaMA/.rss",
    "https://www.reddit.com/r/singularity/.rss",
],
"rss_max_articles_per_feed": 5,
"rss_fetch_full_articles": True,
```

SessionStart hook: add step 6: "Run get_news_briefing() to see today's relevant news."

pyproject: add httpx to main deps, add news_reader and research_engine to py-modules.

Commit: "feat: configure RSS feeds, register Plan D modules, enhance SessionStart"

---

## Plan D Complete — Summary

| Feature | Description |
|---------|-------------|
| RSS reader | Fetch 14 feeds, parse XML, follow links to full articles |
| Article storage | Dedup by URL, store in global knowledge as research notes |
| News briefing | Auto-generated summary injected at session start |
| Research notes | take_note() / recall_notes() for persistent findings |
| External review | OpenRouter → Gemini 3.1 Flash Lite adversarial plan review |
| Independent audit | Same model audits completed work against spec |
