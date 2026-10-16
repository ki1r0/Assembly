from __future__ import annotations

import base64
import json
import urllib.request
from io import BytesIO
from typing import Any, Dict, List, Sequence

from .config import (
    COSMOS_API_KEY,
    COSMOS_MODEL,
    IMAGE_FORMAT,
    IMAGE_MIME,
    IMAGE_QUALITY,
    RESOLUTION,
    TIMEOUT_SEC,
    cosmos_chat_completions_url,
    cosmos_is_configured,
)


def cosmos_endpoint() -> str:
    return cosmos_chat_completions_url()


def _frame_to_b64(frame: Any) -> str:
    if isinstance(frame, str):
        with open(frame, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    from PIL import Image

    img = Image.fromarray(frame)
    try:
        max_w = int(RESOLUTION[0]) if isinstance(RESOLUTION, (tuple, list)) else 640
        max_h = int(RESOLUTION[1]) if isinstance(RESOLUTION, (tuple, list)) else 480
    except Exception:
        max_w, max_h = 640, 480
    max_w = max(192, min(max_w, 448))
    max_h = max(192, min(max_h, 448))
    if img.width > max_w or img.height > max_h:
        img = img.resize((max_w, max_h))

    buf = BytesIO()
    img.save(buf, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_message_parts(frames: Sequence[Any], prompt: str) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    total = len(frames)
    for i, frame in enumerate(frames):
        parts.append({"type": "text", "text": f"Frame {i + 1}/{total} ordered oldest->newest"})
        parts.append({"type": "image_url", "image_url": {"url": f"data:{IMAGE_MIME};base64,{_frame_to_b64(frame)}"}})
    return parts


def _trim_words(text: str, max_words: int = 50) -> str:
    words = str(text or "").strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def observe_physics(
    *,
    frames: Sequence[Any],
    object_ids: Sequence[str],
    trigger: Dict[str, Any] | None = None,
    user_text: str = "",
    observer_instruction: str = "",
) -> str:
    """Cosmos observer: returns concise physical delta text only."""
    not_initiated = "Cosmos is not initiated."

    frames_list = list(frames or [])
    names = [str(n) for n in (object_ids or []) if str(n or "").strip()]
    trigger_type = str((trigger or {}).get("type") or "unknown")

    if not cosmos_is_configured():
        return not_initiated
    if not frames_list:
        return "No visual frames provided."

    system_prompt = (
        "You are a physical observation engine. Look at these 5 frames. "
        "Describe ONLY the physical movements, occlusions, or interactions that occurred. "
        "Keep it under 50 words. Do NOT infer human intentions, beliefs, or plans. "
        "Mention occlusion only if clearly seen. Use canonical object IDs exactly as provided. "
        "Do not format as JSON."
    )

    user_prompt = (
        f"Trigger: {trigger_type}. "
        f"Canonical object IDs: {', '.join(names) if names else '(none)'}\n"
        f"Optional human context: {str(user_text or '').strip()[:200]}\n"
        f"Observer instruction: {str(observer_instruction or '').strip()[:220]}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_message_parts(frames_list, user_prompt)},
    ]

    payload = {
        "model": COSMOS_MODEL,
        "messages": messages,
        "max_tokens": 96,
        "temperature": 0.1,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if COSMOS_API_KEY:
        headers["Authorization"] = f"Bearer {COSMOS_API_KEY}"

    endpoint = cosmos_endpoint()
    if not endpoint:
        return not_initiated
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
        content = str((json.loads(raw).get("choices") or [{}])[0].get("message", {}).get("content", "")).strip()
        if not content:
            return "No clear physical delta observed."
        return _trim_words(content, max_words=50)
    except Exception as exc:
        return f"Physics observer error: {str(exc)[:120]}"
