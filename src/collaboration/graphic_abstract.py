from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.llm.llm_manager import get_manager
from src.services.media_store import detect_image_media_details, get_media_store

GRAPHIC_ABSTRACT_DEFAULT_MODEL = "nanobanana 2"
GRAPHIC_ABSTRACT_FAILURE_MD = "\n> *Graphic abstract generation failed.*"
GRAPHIC_ABSTRACT_MODEL_MAP = {
    "nanobanana 2": ("gemini", "gemini-2.5-flash-image"),
    "nanobanana pro": ("gemini", "gemini-3-pro-image-preview"),
    "gpt-image-1.5": ("openai", "gpt-image-1.5"),
    "qwen-image-2.0": ("qwen", "qwen-image-2.0"),
    # Backward compatibility for previously shipped Gemini image ids.
    "gemini-3.1-flash-image-preview": ("gemini", "gemini-2.5-flash-image"),
}
GRAPHIC_ABSTRACT_PROVIDER_DEFAULTS = {
    "gemini": GRAPHIC_ABSTRACT_MODEL_MAP[GRAPHIC_ABSTRACT_DEFAULT_MODEL],
    "openai": GRAPHIC_ABSTRACT_MODEL_MAP["gpt-image-1.5"],
    "qwen": GRAPHIC_ABSTRACT_MODEL_MAP["qwen-image-2.0"],
}

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "rag_config.json"
_PROMPT_TRUNCATION = {
    "chat": 2000,
    "deep_research": 12000,
}


def detect_image_extension(image_bytes: bytes) -> str:
    return detect_image_media_details(image_bytes).file_ext


@dataclass(frozen=True)
class GraphicAbstractResult:
    provider: str
    model: str
    image_url: str
    markdown: str
    session_id: Optional[str] = None
    turn_id: Optional[str] = None
    asset_key: Optional[str] = None
    content_type: Optional[str] = None
    storage_backend: Optional[str] = None


def resolve_graphic_abstract_model(model_raw: Optional[str]) -> tuple[str, str]:
    normalized = (model_raw or GRAPHIC_ABSTRACT_DEFAULT_MODEL).strip() or GRAPHIC_ABSTRACT_DEFAULT_MODEL
    lowered = normalized.lower()

    mapped = GRAPHIC_ABSTRACT_MODEL_MAP.get(normalized) or GRAPHIC_ABSTRACT_MODEL_MAP.get(lowered)
    if mapped is not None:
        return mapped

    provider_default = GRAPHIC_ABSTRACT_PROVIDER_DEFAULTS.get(lowered)
    if provider_default is not None:
        return provider_default

    if lowered == "kimi" or lowered.startswith("kimi-"):
        return GRAPHIC_ABSTRACT_PROVIDER_DEFAULTS["gemini"]

    if lowered.startswith("gpt"):
        return "openai", normalized
    if lowered.startswith("qwen"):
        return "qwen", normalized
    if lowered.startswith("gemini"):
        return "gemini", normalized
    return "gemini", normalized


def _generate_graphic_abstract_image(provider: str, model: str, prompt: str) -> bytes:
    manager = get_manager(str(_CONFIG_PATH))
    return manager.generate_image(provider=provider, model=model, prompt=prompt)


def build_graphic_abstract_prompt(source_text: str, *, content_kind: str) -> str:
    max_chars = _PROMPT_TRUNCATION.get(content_kind, _PROMPT_TRUNCATION["chat"])
    trimmed = (source_text or "").strip()[:max_chars]
    source_label = "Research report" if content_kind == "deep_research" else "Research summary"
    focus = (
        "Summarize the full report into one integrated poster with the main problem, approach, evidence, "
        "and conclusions."
        if content_kind == "deep_research"
        else "Summarize the answer into one integrated poster with the main findings, relationships, and takeaways."
    )
    return (
        "Create a clean, professional scientific graphic abstract as a single infographic poster. "
        "Use a white or light background, academic colors, clear section hierarchy, and concise labels. "
        "Show the core concepts, evidence flow, and conclusions with icons, callouts, and arrows. "
        "Avoid decorative borders, screenshots, and dense paragraphs. "
        f"{focus}\n\n"
        f"{source_label}:\n{trimmed}"
    )


def render_graphic_abstract_markdown(
    source_text: str,
    *,
    model_raw: Optional[str],
    content_kind: str,
    heading: str,
    session_id: Optional[str] = None,
    turn_id: Optional[str] = None,
) -> GraphicAbstractResult:
    provider, model = resolve_graphic_abstract_model(model_raw)
    prompt = build_graphic_abstract_prompt(source_text, content_kind=content_kind)
    image_bytes = _generate_graphic_abstract_image(provider, model, prompt)
    image_details = detect_image_media_details(image_bytes)

    if session_id and turn_id:
        logical_name = f"turn_{turn_id}"
    else:
        logical_name = uuid.uuid4().hex

    asset = get_media_store().store_bytes(
        category="graphic-abstract",
        content=image_bytes,
        content_type=image_details.content_type,
        file_ext=image_details.file_ext,
        session_id=session_id,
        logical_name=logical_name,
    )
    image_url = asset.url

    markdown = f"\n\n{heading}\n\n![Graphic Abstract]({image_url})\n"

    return GraphicAbstractResult(
        provider=provider,
        model=model,
        image_url=image_url,
        markdown=markdown,
        session_id=session_id,
        turn_id=str(turn_id) if turn_id is not None else None,
        asset_key=asset.key,
        content_type=asset.content_type,
        storage_backend=asset.backend,
    )
