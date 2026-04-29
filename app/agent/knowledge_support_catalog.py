"""知识库支持类问题意图目录。

该模块把 ETC 等具体业务对象从 planner 编排规则中移出。
planner 只判断“这个问题是否应走知识库”，具体业务对象、处置意图和检索关键词维护在 JSON 目录中。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_KNOWLEDGE_SUPPORT_CATALOG_PATH = (
    Path(__file__).with_name("data") / "knowledge_support_catalog.json"
)


@dataclass(frozen=True, slots=True)
class KnowledgeSupportMatch:
    """命中的知识库支持类目录项。"""

    name: str
    query_type: str
    focus: str
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KnowledgeSupportEntry:
    """可配置的业务支持类匹配项。"""

    name: str
    subjects: tuple[str, ...]
    intents: tuple[str, ...]
    query_type: str
    focus: str
    keywords: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "KnowledgeSupportEntry":
        return cls(
            name=str(payload.get("name") or "").strip(),
            subjects=_normalize_terms(payload.get("subjects")),
            intents=_normalize_terms(payload.get("intents")),
            query_type=str(payload.get("query_type") or "knowledge_query").strip()
            or "knowledge_query",
            focus=str(payload.get("focus") or "业务办理与处置流程").strip()
            or "业务办理与处置流程",
            keywords=_normalize_terms(payload.get("keywords")),
        )

    def match(self, message: str) -> KnowledgeSupportMatch | None:
        """同时命中业务对象和处置意图时，才认为应走知识库。"""

        normalized_message = message.strip()
        lowered_message = normalized_message.lower()
        if not self.subjects or not self.intents:
            return None
        if not _contains_any(normalized_message, lowered_message, self.subjects):
            return None
        if not _contains_any(normalized_message, lowered_message, self.intents):
            return None
        return KnowledgeSupportMatch(
            name=self.name,
            query_type=self.query_type,
            focus=self.focus,
            keywords=self.keywords,
        )


@dataclass(frozen=True, slots=True)
class KnowledgeSupportCatalog:
    """用于把业务支持类问题路由到 RAG 的目录。"""

    entries: tuple[KnowledgeSupportEntry, ...]

    @classmethod
    def empty(cls) -> "KnowledgeSupportCatalog":
        return cls(entries=())

    @classmethod
    def from_json_payload(cls, payload: dict[str, object]) -> "KnowledgeSupportCatalog":
        """从 JSON 配置构建目录，并跳过无效条目。"""

        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            return cls.empty()
        entries = []
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            entry = KnowledgeSupportEntry.from_payload(raw_entry)
            if entry.name and entry.subjects and entry.intents:
                entries.append(entry)
        return cls(entries=tuple(entries))

    @classmethod
    def load_default(cls) -> "KnowledgeSupportCatalog":
        """加载默认目录；文件缺失时返回空目录，避免影响主链路。"""

        if not DEFAULT_KNOWLEDGE_SUPPORT_CATALOG_PATH.exists():
            return cls.empty()
        payload = json.loads(
            DEFAULT_KNOWLEDGE_SUPPORT_CATALOG_PATH.read_text(encoding="utf-8")
        )
        return cls.from_json_payload(payload if isinstance(payload, dict) else {})

    def match(self, message: str) -> KnowledgeSupportMatch | None:
        """按目录顺序返回第一个匹配项。"""

        for entry in self.entries:
            matched = entry.match(message)
            if matched is not None:
                return matched
        return None


def _normalize_terms(value: object) -> tuple[str, ...]:
    """规整 JSON 里的词表，去空值并按大小写去重。"""

    if not isinstance(value, list):
        return ()
    terms: list[str] = []
    seen: set[str] = set()
    for item in value:
        term = str(item).strip()
        if not term:
            continue
        dedupe_key = term.lower()
        if dedupe_key in seen:
            continue
        terms.append(term)
        seen.add(dedupe_key)
    return tuple(terms)


def _contains_any(message: str, lowered_message: str, terms: tuple[str, ...]) -> bool:
    """同时支持中文原文匹配和英文大小写不敏感匹配。"""

    for term in terms:
        if term in message or term.lower() in lowered_message:
            return True
    return False


@lru_cache(maxsize=1)
def load_default_knowledge_support_catalog() -> KnowledgeSupportCatalog:
    """按进程缓存默认目录，避免每轮 planner 都读取 JSON。"""

    return KnowledgeSupportCatalog.load_default()
