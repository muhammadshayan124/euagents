"""MCP server: read-only search over the six EU digital-law acts.

Run over stdio by the agent (`python -m euagents.servers.lawsearch`). Three tools, all read-only, so
nothing here needs human approval.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from euagents import law

mcp = FastMCP("lawsearch")
_index: law.Index | None = None


def index() -> law.Index:
    global _index
    if _index is None:
        _index = law.Index(law.load(Path(os.environ.get("EUAGENTS_DATA", law.DATA_DIR))))
    return _index


@mcp.tool()
def list_acts() -> list[dict]:
    """List the acts that can be searched, with their codes and article counts."""
    return [{"code": c, "name": n, "articles": law.EXPECTED_ARTICLES[c]} for c, (_, n) in law.ACTS.items()]


@mcp.tool()
def search_articles(query: str, act: str | None = None, k: int = 5) -> list[dict]:
    """Keyword (BM25) search over articles. `act` optionally restricts to one act code (AI, GDPR, DSA, DATA, NIS2, CRA).
    Returns the best-matching articles with a short snippet. Use specific legal terms, not whole sentences."""
    if act and act.upper() not in law.ACTS:
        raise ValueError(f"unknown act {act!r}; valid codes: {', '.join(law.ACTS)}")
    k = max(1, min(int(k), 10))
    hits = index().search(query, act.upper() if act else None, k)
    return [{"key": a.key, "heading": a.heading, "score": round(s, 3), "snippet": a.text[:220]} for a, s in hits]


@mcp.tool()
def get_article(act: str, number: str) -> dict:
    """Return the full text of one article, e.g. act='GDPR', number='17'. Errors if the article does not exist."""
    art = index().get(act, number)
    if art is None:
        raise ValueError(f"no article {number!r} in act {act!r}")
    return {"key": art.key, "heading": art.heading, "text": art.text[:6000], "truncated": len(art.text) > 6000}


if __name__ == "__main__":
    mcp.run(transport="stdio")
