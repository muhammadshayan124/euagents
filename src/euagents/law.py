"""The corpus the tools search: six EU digital-law acts from the Publications Office (CELLAR), parsed to articles.

Vendored in compact form from the companion `eurag` project so this repo stands alone. The text is
European Union legislation, reusable with attribution (Commission Decision 2011/833/EU); it is fetched
with `euagents fetch`, not committed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lxml import etree

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CELLAR = "http://publications.europa.eu/resource/celex/{celex}"
ACTS = {
    "AI": ("32024R1689", "AI Act"),
    "GDPR": ("32016R0679", "General Data Protection Regulation"),
    "DSA": ("32022R2065", "Digital Services Act"),
    "DATA": ("32023R2854", "Data Act"),
    "NIS2": ("32022L2555", "NIS2 Directive"),
    "CRA": ("32024R2847", "Cyber Resilience Act"),
}
EXPECTED_ARTICLES = {"AI": 113, "GDPR": 99, "DSA": 93, "DATA": 50, "NIS2": 46, "CRA": 71}
STOP = frozenset(
    "a an and are as at be been being by can could do does for from has have how if in into is it its may must of on or shall should "
    "than that the their them then there these they this those to under was were what when where which who whom why will with would".split()
)


def fetch(directory: Path = DATA_DIR) -> list[Path]:
    import httpx

    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for key, (celex, _) in ACTS.items():
        r = httpx.get(CELLAR.format(celex=celex), headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"}, timeout=120, follow_redirects=True)
        r.raise_for_status()
        path = directory / f"{key}.xhtml"
        path.write_bytes(r.content)
        out.append(path)
    return out


@dataclass(frozen=True)
class Article:
    act: str
    number: str
    title: str
    paragraphs: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.act}:{self.number}"

    @property
    def heading(self) -> str:
        return f"{ACTS[self.act][1]}, Article {self.number}: {self.title}"

    @property
    def text(self) -> str:
        return "\n".join(self.paragraphs)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def parse_act(act: str, raw: bytes) -> list[Article]:
    root = etree.fromstring(raw, etree.HTMLParser())
    out = []
    for div in root.xpath('//div[@class="eli-subdivision" and starts-with(@id,"art_")]'):
        label = _clean("".join(div.xpath('.//p[@class="oj-ti-art"]//text()')))
        m = re.search(r"(\d+[a-z]?)\s*$", label)
        if not m:
            continue
        title = _clean("".join(div.xpath('.//p[@class="oj-sti-art"]//text()')))
        paras = [_clean("".join(p.itertext())) for p in div.xpath('./div[re:test(@id, "^\\d{3}\\.\\d{3}$")]', namespaces={"re": "http://exslt.org/regular-expressions"})]
        if not paras:
            paras = [_clean("".join(p.itertext())) for p in div.xpath('.//p[@class="oj-normal"]')]
        out.append(Article(act, m.group(1), title, tuple(p for p in paras if p)))
    return out


def load(directory: Path = DATA_DIR) -> list[Article]:
    arts: list[Article] = []
    for key in ACTS:
        parsed = parse_act(key, (directory / f"{key}.xhtml").read_bytes())
        if len(parsed) != EXPECTED_ARTICLES[key]:
            raise ValueError(f"{key}: parsed {len(parsed)} articles, expected {EXPECTED_ARTICLES[key]}")
        arts += parsed
    return arts


def stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            if suf in ("ies", "ied"):
                return base + "y"
            if suf == "es" and not base.endswith(("s", "x", "z", "ch", "sh")):
                base = w[:-1]
            elif suf == "s" and base.endswith("s"):
                return w
            return base[:-1] if base.endswith("e") and len(base) > 3 else base
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class Index:
    """BM25 over whole articles. Deliberately lexical: it is the tool the agent has to learn to query well."""

    def __init__(self, articles: list[Article], k1: float = 1.2, b: float = 0.75):
        self.articles = articles
        self.by_key = {a.key: a for a in articles}
        self.k1, self.b = k1, b
        docs = [tokenize(a.heading + " " + a.text) for a in articles]
        self.lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.lengths.mean())
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings.setdefault(term, []).append((i, tf))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, act: str | None = None, k: int = 5) -> list[tuple[Article, float]]:
        s = np.zeros(len(self.articles), dtype=np.float32)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                s[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
        if act:
            mask = np.array([a.act == act for a in self.articles])
            s = np.where(mask, s, 0.0)
        top = [i for i in np.argsort(-s, kind="stable")[:k] if s[i] > 0]
        return [(self.articles[i], float(s[i])) for i in top]

    def get(self, act: str, number: str) -> Article | None:
        return self.by_key.get(f"{act.upper()}:{str(number).strip().lstrip('0')}")
