"""MCP server: the one tool with a side effect, writing a research memo to disk.

The agent graph never calls this without a human approval step in front of it; the server also
refuses path tricks, so even a mistaken call cannot write outside the memo directory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("memo")


def memo_dir() -> Path:
    d = Path(os.environ.get("EUAGENTS_MEMOS", "memos")).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]
    if not s:
        raise ValueError("title must contain letters or digits")
    return s


@mcp.tool()
def save_memo(title: str, body: str) -> dict:
    """Save a memo as <slug>.md in the memo directory. Refuses to overwrite an existing memo."""
    if len(body) > 20_000:
        raise ValueError("memo too long")
    path = memo_dir() / f"{slug(title)}.md"
    if path.exists():
        raise FileExistsError(f"memo {path.name} already exists")
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return {"saved": path.name, "bytes": path.stat().st_size}


@mcp.tool()
def list_memos() -> list[str]:
    """List saved memo file names."""
    return sorted(p.name for p in memo_dir().glob("*.md"))


if __name__ == "__main__":
    mcp.run(transport="stdio")
