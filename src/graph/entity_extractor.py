"""
Domain-agnostic entity extraction for HippoRAG knowledge graphs.

Supports three strategies (configured via rag_config.json → graph.entity_extraction):
  - gliner : Zero-shot NER via GLiNER — fast, CPU-friendly, no hand-written rules
  - rule   : Regex patterns loaded from config/ontology.json
  - llm    : LLM-based extraction via the project's call_llm() convention

The extractor reads entity type definitions from config/ontology.json so that
switching research domains only requires editing that file.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.log import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ONTOLOGY_PATH = _PROJECT_ROOT / "config" / "ontology.json"


# ── dataclasses ──────────────────────────────────────────────

@dataclass
class EntityType:
    name: str
    label: str
    description: str
    patterns: List[str] = field(default_factory=list)


@dataclass
class Ontology:
    entity_types: Dict[str, EntityType] = field(default_factory=dict)
    min_entity_length: int = 2

    @classmethod
    def from_json(cls, path: Path) -> "Ontology":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        types: Dict[str, EntityType] = {}
        for name, cfg in raw.get("entity_types", {}).items():
            types[name] = EntityType(
                name=name,
                label=cfg.get("label", name.lower()),
                description=cfg.get("description", ""),
                patterns=cfg.get("patterns", []),
            )
        return cls(
            entity_types=types,
            min_entity_length=int(raw.get("min_entity_length", 2)),
        )

    @property
    def type_names(self) -> List[str]:
        return list(self.entity_types.keys())

    @property
    def gliner_labels(self) -> List[str]:
        """Labels suitable for GLiNER predict_entities()."""
        return [et.label for et in self.entity_types.values()]

    def label_to_type_name(self, label: str) -> str:
        """Map a GLiNER label back to an ontology type name."""
        for et in self.entity_types.values():
            if et.label == label:
                return et.name
        return "OTHER"

    def describe_for_prompt(self) -> str:
        """Render entity types as a bullet list for LLM prompts."""
        lines = []
        for et in self.entity_types.values():
            lines.append(f"- {et.name}: {et.description}")
        return "\n".join(lines)


# ── GLiNER singleton loader ──────────────────────────────────

_gliner_models: Dict[Tuple[str, str], object] = {}
_gliner_lock = threading.Lock()


def _load_gliner(model_name: str, device: str = "cpu"):
    key = (model_name, device or "cpu")
    if key in _gliner_models:
        return _gliner_models[key]
    with _gliner_lock:
        if key in _gliner_models:
            return _gliner_models[key]
        try:
            from gliner import GLiNER  # type: ignore[import-untyped]
            logger.info("Loading GLiNER model: %s (device=%s) …", model_name, device)
            model = GLiNER.from_pretrained(model_name)
            if device and device != "cpu":
                model = model.to(device)
            logger.info("GLiNER model loaded successfully")
            _gliner_models[key] = model
        except Exception as exc:
            logger.warning("Failed to load GLiNER model (%s): %s", model_name, exc)
            return None
    return _gliner_models.get(key)


# ── Entity result (shared with hippo_rag) ────────────────────

@dataclass
class ExtractedEntity:
    name: str
    type: str
    mentions: List[str] = field(default_factory=list)


# ── Main extractor class ─────────────────────────────────────

@dataclass
class ExtractorConfig:
    strategy: str = "gliner"
    fallback: str = "rule"
    ontology_path: str = str(_DEFAULT_ONTOLOGY_PATH)
    gliner_model: str = "urchade/gliner_base"
    gliner_threshold: float = 0.4
    gliner_device: str = "cpu"
    llm_provider: str = "deepseek"
    llm_max_tokens: int = 1000


class EntityExtractor:
    """Unified entity extractor with pluggable backends."""

    def __init__(self, config: Optional[ExtractorConfig] = None):
        self.config = config or ExtractorConfig()
        ontology_path = Path(self.config.ontology_path)
        if not ontology_path.is_absolute():
            ontology_path = _PROJECT_ROOT / ontology_path
        if ontology_path.exists():
            self.ontology = Ontology.from_json(ontology_path)
            logger.info(
                "Ontology loaded: %d entity types from %s",
                len(self.ontology.entity_types), ontology_path,
            )
        else:
            logger.warning("Ontology file not found: %s — using empty ontology", ontology_path)
            self.ontology = Ontology()

        self._compiled_patterns: Optional[Dict[str, List[re.Pattern]]] = None

    # ── public API ────────────────────────────────────────────

    def extract(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        strategy = self.config.strategy
        try:
            return self._dispatch(strategy, text, chunk_id)
        except Exception as exc:
            if strategy != self.config.fallback:
                logger.warning(
                    "Entity extraction failed with strategy '%s': %s; falling back to '%s'",
                    strategy, exc, self.config.fallback,
                )
                return self._dispatch(self.config.fallback, text, chunk_id)
            logger.error("Entity extraction failed with fallback strategy '%s': %s", strategy, exc)
            return []

    # ── dispatch ──────────────────────────────────────────────

    def _dispatch(self, strategy: str, text: str, chunk_id: str) -> List[ExtractedEntity]:
        if strategy == "gliner":
            return self._extract_gliner(text, chunk_id)
        if strategy == "rule":
            return self._extract_rule(text, chunk_id)
        if strategy == "llm":
            return self._extract_llm(text, chunk_id)
        raise ValueError(f"Unknown extraction strategy: {strategy}")

    # ── GLiNER backend ────────────────────────────────────────

    def _extract_gliner(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        model = _load_gliner(self.config.gliner_model, self.config.gliner_device)
        if model is None:
            raise RuntimeError("GLiNER model is unavailable")

        labels = self.ontology.gliner_labels
        if not labels:
            return []

        try:
            raw_entities = model.predict_entities(
                text[:4096],
                labels,
                threshold=self.config.gliner_threshold,
            )
        except Exception as exc:
            raise RuntimeError(f"GLiNER prediction failed: {exc}") from exc

        seen: Dict[str, ExtractedEntity] = {}
        for ent in raw_entities:
            name = ent["text"].strip().lower()
            if len(name) < self.ontology.min_entity_length:
                continue
            type_name = self.ontology.label_to_type_name(ent["label"])
            if name not in seen:
                seen[name] = ExtractedEntity(name=name, type=type_name, mentions=[chunk_id])
            else:
                if chunk_id not in seen[name].mentions:
                    seen[name].mentions.append(chunk_id)
        return list(seen.values())

    # ── Rule-based backend ────────────────────────────────────

    def _ensure_compiled_patterns(self):
        if self._compiled_patterns is not None:
            return
        self._compiled_patterns = {}
        for name, et in self.ontology.entity_types.items():
            compiled: List[re.Pattern] = []
            for p in et.patterns:
                try:
                    compiled.append(re.compile(p))
                except re.error as exc:
                    logger.warning("Bad regex in ontology [%s]: %s — %s", name, p, exc)
            self._compiled_patterns[name] = compiled

    def _extract_rule(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        self._ensure_compiled_patterns()
        assert self._compiled_patterns is not None

        seen: Dict[str, ExtractedEntity] = {}
        min_len = self.ontology.min_entity_length
        for type_name, patterns in self._compiled_patterns.items():
            for pat in patterns:
                for match in pat.finditer(text):
                    name = match.group(0).strip().lower()
                    if len(name) < min_len:
                        continue
                    if name not in seen:
                        seen[name] = ExtractedEntity(name=name, type=type_name, mentions=[chunk_id])
                    else:
                        if chunk_id not in seen[name].mentions:
                            seen[name].mentions.append(chunk_id)
        return list(seen.values())

    # ── LLM backend ───────────────────────────────────────────

    def _extract_llm(self, text: str, chunk_id: str) -> List[ExtractedEntity]:
        try:
            from src.generation.llm_client import call_llm
        except ImportError:
            logger.error("call_llm unavailable; cannot use llm strategy")
            return []

        from src.utils.prompt_manager import PromptManager
        pm = PromptManager()

        entity_type_desc = self.ontology.describe_for_prompt()
        prompt = pm.render(
            "hippo_entity_extract.txt",
            text=text[:3000],
            entity_types=entity_type_desc,
        )

        result_text = call_llm(
            provider=self.config.llm_provider,
            system="You are a named-entity extraction assistant. Return ONLY valid JSON.",
            user_prompt=prompt,
            max_tokens=self.config.llm_max_tokens,
        )

        if result_text.startswith("[ERROR]"):
            logger.error("LLM entity extraction failed: %s", result_text)
            return []

        try:
            parsed = json.loads(result_text)
            if not isinstance(parsed, list):
                parsed = parsed.get("entities", [])
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON for entity extraction")
            return []

        entities: List[ExtractedEntity] = []
        valid_types = set(self.ontology.type_names)
        for item in parsed:
            name = str(item.get("name", "")).strip().lower()
            etype = str(item.get("type", "OTHER")).upper()
            if etype not in valid_types:
                etype = "OTHER"
            if len(name) >= self.ontology.min_entity_length:
                entities.append(ExtractedEntity(name=name, type=etype, mentions=[chunk_id]))
        return entities
