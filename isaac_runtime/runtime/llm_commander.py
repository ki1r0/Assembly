from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .config import (
    COMMANDER_API_KEY,
    COMMANDER_BASE_URL,
    COMMANDER_GEMINI_BASE_URL,
    COMMANDER_LLM_MODEL,
    COMMANDER_TIMEOUT_SEC,
    IMAGE_FORMAT,
    IMAGE_MIME,
    IMAGE_QUALITY,
    RESOLUTION,
    cosmos_is_configured,
)


class CommanderActionType(str, Enum):
    combine = "combine"
    separate = "separate"
    noop = "noop"


class CommanderAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: CommanderActionType
    args: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_args(self) -> "CommanderAction":
        args = self.args if isinstance(self.args, dict) else {}
        if self.type == CommanderActionType.combine:
            required = ("partA", "partB", "plug", "socket")
            missing = [k for k in required if not str(args.get(k) or "").strip()]
            if missing:
                raise ValueError(f"combine requires args {required}; missing={missing}")
        elif self.type == CommanderActionType.separate:
            if not str(args.get("part") or "").strip():
                raise ValueError("separate requires args.part")
        return self


class CommanderOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    belief_state_update: Dict[str, Any] = Field(default_factory=dict)
    action: Optional[CommanderAction] = None
    reply_to_human: str


class CommanderRouting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_cosmos: bool = False
    observer_instruction: str = ""
    quick_observation: str = ""


class CommanderHypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_inferred: str = ""
    proposed_action: str = "noop"
    target_object: str = ""
    destination: str = ""
    action_args: Dict[str, Any] = Field(default_factory=dict)


class CommanderPhysicsValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_physically_valid: bool = True
    reason: str = ""


_WARNED_AUTH = False
_WARNED_NO_KEY = False
_GEMINI_MODELS_CACHE_TS = 0.0
_GEMINI_MODELS_CACHE: List[str] = []
_GEMINI_MODELS_CACHE_TTL_SEC = 300.0
_GEMINI_BAD_MODELS: set[str] = set()
_GEMINI_WORKING_MODEL = ""


def _provider_for_model(model: str) -> str:
    name = str(model or "").strip().lower()
    if name.startswith("gemini"):
        return "gemini"
    return "openai"


def _system_prompt() -> str:
    return (
        "You are the Cognitive Commander for a robotics scene.\n"
        "Conflict Resolution Policy (absolute):\n"
        "1) Priority Rule: Physics Observation > System State > Human Chat.\n"
        "2) Contradictions: if human says X but observation implies Y, update belief to Y and gently correct the human.\n"
        "3) No Hallucinations: update only entities explicitly present in observation, chat, or current state.\n"
        "Action contract:\n"
        "- action.type must be one of: combine | separate | noop.\n"
        "- combine args must include: partA, partB, plug, socket.\n"
        "- separate args must include: part.\n"
        "- If action is not fully grounded, return noop.\n"
        "Return JSON matching schema exactly."
    )


def _routing_prompt() -> str:
    return (
        "You are routing between two brains: Commander and Cosmos physics observer.\n"
        "Decide if Cosmos should be called on these frames.\n"
        "Set use_cosmos=true for uncertain temporal motion/occlusion/interaction cases.\n"
        "Set use_cosmos=false only if confident from current evidence.\n"
        "Return strict JSON only."
    )


def _hypothesis_prompt() -> str:
    return (
        "You are Node 1 (Static Analysis & Action Proposal) in a dual-brain A2A system.\n"
        "Infer human intent from chat + state and propose one assistive action candidate.\n"
        "For assembly tasks prefer canonical actions: combine / separate / noop.\n"
        "Return strict JSON only matching schema."
    )


def _physics_gate_prompt() -> str:
    return (
        "You are Node 4 (Physics Validation Router) in a dual-brain A2A system.\n"
        "Given a proposed action and Cosmos physics observation, decide if action is physically valid.\n"
        "Return strict JSON only matching schema."
    )


def _extract_json_text(raw: str) -> str:
    txt = str(raw or "").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?", "", txt, flags=re.IGNORECASE).strip()
        txt = re.sub(r"```$", "", txt).strip()
    start = txt.find("{")
    end = txt.rfind("}")
    if start >= 0 and end > start:
        return txt[start : end + 1]
    return txt


def _normalize_action_type(value: Any) -> str:
    if isinstance(value, Enum):
        return str(value.value).strip().lower()
    text = str(value or "").strip()
    if text.lower().startswith("commanderactiontype."):
        text = text.split(".", 1)[1]
    return text.lower()


def _coerce_hypothesis_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Hypothesis payload must be JSON object, got {type(payload)}")

    intent = str(
        payload.get("intent_inferred")
        or payload.get("intent")
        or payload.get("goal")
        or ""
    ).strip()
    proposed_action = payload.get("proposed_action")
    target_object = str(payload.get("target_object") or payload.get("object") or "").strip()
    destination = str(payload.get("destination") or payload.get("target") or "").strip()
    action_args = payload.get("action_args")
    if not isinstance(action_args, dict):
        action_args = {}

    if isinstance(proposed_action, dict):
        pa_type = _normalize_action_type(
            proposed_action.get("type")
            or proposed_action.get("action_type")
            or proposed_action.get("action")
            or ""
        )
        if not target_object:
            target_object = str(
                proposed_action.get("target_object")
                or proposed_action.get("object")
                or proposed_action.get("partA")
                or ""
            ).strip()
        if not destination:
            destination = str(
                proposed_action.get("destination")
                or proposed_action.get("target")
                or proposed_action.get("partB")
                or ""
            ).strip()
        args = proposed_action.get("args")
        if isinstance(args, dict):
            action_args = args
        else:
            passthrough = {
                k: v
                for k, v in proposed_action.items()
                if k not in {"type", "action", "action_type", "target", "target_object", "object", "destination", "args"}
            }
            if isinstance(passthrough, dict) and passthrough:
                action_args = passthrough
        proposed_action = pa_type or "noop"
    else:
        proposed_action = _normalize_action_type(proposed_action or payload.get("action_type") or "noop")

    if not proposed_action:
        proposed_action = "noop"

    return {
        "intent_inferred": intent,
        "proposed_action": proposed_action,
        "target_object": target_object,
        "destination": destination,
        "action_args": action_args,
    }


def _coerce_commander_payload(payload: Any) -> Dict[str, Any]:
    """Normalize provider JSON into CommanderOutput shape before strict validation."""
    if not isinstance(payload, dict):
        raise ValueError(f"Commander payload must be a JSON object, got {type(payload)}")

    belief_update = payload.get("belief_state_update")
    if not isinstance(belief_update, dict):
        alt = payload.get("belief_update")
        belief_update = alt if isinstance(alt, dict) else {}

    reply = str(payload.get("reply_to_human") or "").strip()
    if not reply:
        reply = str(payload.get("reply") or "").strip()
    if not reply:
        reply = str(payload.get("response") or "").strip()
    if not reply:
        reply = str(payload.get("stm_observation") or "").strip()
    if not reply:
        reply = str(payload.get("utterance") or "").strip()
    if not reply:
        reply = str(payload.get("reply_to_user") or "").strip()
    if not reply:
        reply = str(payload.get("assistant_reply") or "").strip()
    if not reply:
        reply = str(payload.get("final_reply") or "").strip()
    if not reply:
        reply = str(payload.get("explanation") or "").strip()
    if not reply:
        reply = str(payload.get("summary") or "").strip()

    action_raw = payload.get("action")
    action_out: Any = None
    if isinstance(action_raw, Enum):
        at = _normalize_action_type(action_raw)
        if at in {"combine", "separate", "noop"}:
            action_out = {"type": at, "args": {}}
        else:
            action_out = {"type": "noop", "args": {}}
    elif isinstance(action_raw, str):
        at = action_raw.strip().lower()
        if at in {"combine", "separate", "noop"}:
            action_out = {"type": at, "args": {}}
        elif at in {"none", "null", ""}:
            action_out = None
        else:
            action_out = {"type": "noop", "args": {}}
    elif isinstance(action_raw, dict):
        at = _normalize_action_type(action_raw.get("type") or action_raw.get("action_type") or "")
        if at in {"respond", "reply", "say", "assistant"}:
            action_out = None
            at = ""
        if at and at not in {"combine", "separate", "noop"}:
            at = "noop"
        if at:
            args = action_raw.get("args")
            if not isinstance(args, dict):
                args = {
                    k: v
                    for k, v in action_raw.items()
                    if k
                    not in {
                        "type",
                        "action_type",
                        "reply_to_human",
                        "utterance",
                        "message",
                        "response",
                    }
                }
            if not isinstance(args, dict):
                args = {}
            action_out = {"type": at, "args": args}
        if not reply:
            reply = str(
                action_raw.get("utterance")
                or action_raw.get("reply_to_human")
                or action_raw.get("message")
                or action_raw.get("response")
                or ""
            ).strip()
    elif action_raw in (None, False):
        action_out = None
    else:
        action_out = {"type": "noop", "args": {}}

    if not reply:
        # Last-resort deterministic synthesis from returned machine state.
        objects = belief_update.get("objects")
        if isinstance(objects, dict) and objects:
            names = [str(k) for k in objects.keys() if str(k or "").strip()]
            if names:
                shown = ", ".join(names[:4])
                if len(names) > 4:
                    shown += ", ..."
                reply = f"Belief updated for: {shown}."
        if not reply and isinstance(action_out, dict):
            a_t = str(action_out.get("type") or "noop").strip().lower()
            if a_t and a_t != "noop":
                reply = f"Proposed action: {a_t}."
        if not reply:
            raise ValueError("Commander payload missing non-empty reply_to_human")

    return {
        "belief_state_update": belief_update,
        "action": action_out,
        "reply_to_human": reply,
    }


def _coerce_physics_validation_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Physics validation payload must be JSON object, got {type(payload)}")

    valid_raw = None
    for key in ("is_physically_valid", "is_valid", "valid", "feasible", "is_feasible"):
        if key in payload:
            valid_raw = payload.get(key)
            break
    if isinstance(valid_raw, str):
        v = valid_raw.strip().lower()
        is_valid = v in {"true", "1", "yes", "y", "valid", "ok"}
    else:
        is_valid = bool(valid_raw)

    reason = ""
    for key in ("reason", "message", "details", "explanation", "why"):
        txt = str(payload.get(key) or "").strip()
        if txt:
            reason = txt
            break

    return {"is_physically_valid": is_valid, "reason": reason}


def _brief(text: Any, limit: int = 180) -> str:
    s = str(text or "").strip().replace("\n", " ")
    return s if len(s) <= limit else (s[: limit - 3] + "...")


def _is_grounding_trigger(trigger: Dict[str, Any]) -> bool:
    trig_type = str((trigger or {}).get("type") or "").strip().lower()
    mode = str((trigger or {}).get("mode") or "").strip().lower()
    return trig_type == "grounding" and mode == "initialization"


def _collect_grounding_required_ids(trigger: Dict[str, Any]) -> List[str]:
    required: List[str] = []
    seen = set()

    def _add(name: Any) -> None:
        n = str(name or "").strip()
        if n and n not in seen:
            seen.add(n)
            required.append(n)

    raw_ids = (trigger or {}).get("init_interactables")
    if isinstance(raw_ids, list):
        for item in raw_ids:
            _add(item)

    if not required:
        form = (trigger or {}).get("init_belief_form")
        if isinstance(form, dict):
            objects = ((form.get("belief_state_update") or {}).get("objects") or {})
            if isinstance(objects, dict):
                for key in objects.keys():
                    _add(key)
    return required


def _validate_grounding_output(output: CommanderOutput, required_ids: Sequence[str]) -> None:
    required = [str(x).strip() for x in required_ids if str(x).strip()]
    if not required:
        return
    update = output.belief_state_update if isinstance(output.belief_state_update, dict) else {}
    objects = update.get("objects")
    if not isinstance(objects, dict):
        raise ValueError("Grounding output requires belief_state_update.objects as an object")

    missing = [name for name in required if name not in objects]
    if missing:
        raise ValueError(
            f"Grounding output missing required interactables: {missing[:12]}"
            + ("..." if len(missing) > 12 else "")
        )

    bad_shape = [name for name in required if not isinstance(objects.get(name), dict)]
    if bad_shape:
        raise ValueError(
            f"Grounding output has non-object entries for interactables: {bad_shape[:12]}"
            + ("..." if len(bad_shape) > 12 else "")
        )

def _make_grounding_reply_json(objects: Dict[str, Any], required_ids: Sequence[str]) -> str:
    required = [str(x).strip() for x in required_ids if str(x).strip()]
    states: Dict[str, Any] = {}
    for name in required:
        value = objects.get(name)
        states[name] = value if isinstance(value, dict) else {}
    return json.dumps({"interactable_states": states}, ensure_ascii=True)


def _normalize_grounding_reply_json(output: CommanderOutput, required_ids: Sequence[str]) -> CommanderOutput:
    update = output.belief_state_update if isinstance(output.belief_state_update, dict) else {}
    objects = update.get("objects") if isinstance(update.get("objects"), dict) else {}
    fallback_reply = _make_grounding_reply_json(objects, required_ids)
    reply_txt = str(output.reply_to_human or "").strip()
    if not reply_txt:
        return output.model_copy(update={"reply_to_human": fallback_reply})
    try:
        payload = json.loads(_extract_json_text(reply_txt))
    except Exception:
        return output.model_copy(update={"reply_to_human": fallback_reply})
    if not isinstance(payload, dict):
        return output.model_copy(update={"reply_to_human": fallback_reply})
    states = payload.get("interactable_states")
    if not isinstance(states, dict):
        return output.model_copy(update={"reply_to_human": fallback_reply})
    missing = [name for name in required_ids if str(name or "").strip() and str(name) not in states]
    if missing:
        merged = dict(states)
        for name in missing:
            key = str(name)
            merged[key] = objects.get(key) if isinstance(objects.get(key), dict) else {}
        return output.model_copy(update={"reply_to_human": json.dumps({"interactable_states": merged}, ensure_ascii=True)})
    return output


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
    max_w = max(192, min(max_w, 384))
    max_h = max(192, min(max_h, 384))
    if img.width > max_w or img.height > max_h:
        img = img.resize((max_w, max_h))

    buf = BytesIO()
    img.save(buf, format=IMAGE_FORMAT, quality=IMAGE_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_multimodal_user_content(text: str, frames: Sequence[Any]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    total = len(frames)
    for i, frame in enumerate(frames):
        content.append({"type": "text", "text": f"Frame {i + 1}/{total} (oldest->newest)"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{IMAGE_MIME};base64,{_frame_to_b64(frame)}"},
            }
        )
    return content


def _http_post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout_sec: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.time()
    print(
        f"[COMMANDER] POST {url} | payload={len(data)/1024:.1f}KB | timeout={timeout_sec:.1f}s",
        flush=True,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(exc)
        print(
            f"[COMMANDER] HTTP ERROR {exc.code} in {time.time()-t0:.2f}s | body={_brief(body, 500)!r}",
            flush=True,
        )
        raise
    print(f"[COMMANDER] HTTP OK in {time.time()-t0:.2f}s | bytes={len(raw)}", flush=True)
    return json.loads(raw)


def _http_get_json(url: str, timeout_sec: float) -> Dict[str, Any]:
    t0 = time.time()
    print(f"[COMMANDER] GET {url} | timeout={timeout_sec:.1f}s", flush=True)
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    print(f"[COMMANDER] HTTP OK in {time.time()-t0:.2f}s | bytes={len(raw)}", flush=True)
    return json.loads(raw)


def _normalize_gemini_model_name(name: str) -> str:
    m = str(name or "").strip()
    if m.startswith("models/"):
        m = m[len("models/") :]
    return m


def _list_gemini_generate_models(api_key: str, timeout_sec: float) -> List[str]:
    global _GEMINI_MODELS_CACHE_TS, _GEMINI_MODELS_CACHE
    now = time.time()
    if _GEMINI_MODELS_CACHE and (now - _GEMINI_MODELS_CACHE_TS) < _GEMINI_MODELS_CACHE_TTL_SEC:
        return list(_GEMINI_MODELS_CACHE)

    endpoint = f"{COMMANDER_GEMINI_BASE_URL}/models?key={api_key}"
    try:
        payload = _http_get_json(endpoint, timeout_sec)
    except Exception as exc:
        print(f"[COMMANDER] listModels failed: {_brief(exc, 160)}", flush=True)
        return list(_GEMINI_MODELS_CACHE)

    out: List[str] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            continue
        name = _normalize_gemini_model_name(str(item.get("name") or ""))
        if name:
            out.append(name)
    _GEMINI_MODELS_CACHE = out
    _GEMINI_MODELS_CACHE_TS = time.time()
    return list(out)


def _gemini_model_candidates(requested: str, available: List[str]) -> List[str]:
    global _GEMINI_WORKING_MODEL, _GEMINI_BAD_MODELS
    req = _normalize_gemini_model_name(requested)
    candidates: List[str] = []

    def _add(name: str) -> None:
        n = _normalize_gemini_model_name(name)
        if n and n not in candidates:
            candidates.append(n)

    if _GEMINI_WORKING_MODEL:
        _add(_GEMINI_WORKING_MODEL)
    _add(req)
    if req.endswith("-pro"):
        _add(req + "-latest")
    if req.endswith("-flash"):
        _add(req + "-latest")
    if req == "gemini-1.5-pro":
        _add("gemini-1.5-pro-latest")
        _add("gemini-1.5-flash-latest")
    if req == "gemini-1.5-flash":
        _add("gemini-1.5-flash-latest")

    # Add available models with preference to requested family first.
    for name in available:
        if req and req in name:
            _add(name)
    for name in available:
        if name.startswith("gemini-"):
            _add(name)

    return [name for name in candidates if name not in _GEMINI_BAD_MODELS]


def _call_openai_structured(
    *,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    schema_name: str,
    schema: Dict[str, Any],
    timeout_sec: float,
) -> str:
    endpoint = COMMANDER_BASE_URL or "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        },
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    res = _http_post_json(endpoint, payload, headers, timeout_sec)
    return str(res["choices"][0]["message"]["content"])


def _gemini_parts_from_content(content: Any) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if isinstance(content, str):
        parts.append({"text": content})
        return parts
    if not isinstance(content, list):
        parts.append({"text": str(content)})
        return parts

    for item in content:
        if not isinstance(item, dict):
            continue
        typ = item.get("type")
        if typ == "text":
            parts.append({"text": str(item.get("text") or "")})
        elif typ == "image_url":
            url = str((item.get("image_url") or {}).get("url") or "")
            if url.startswith("data:") and ";base64," in url:
                prefix, b64 = url.split(",", 1)
                mime = "image/jpeg"
                m = re.match(r"data:([^;]+);base64", prefix)
                if m:
                    mime = m.group(1)
                parts.append({"inline_data": {"mime_type": mime, "data": b64}})
    return parts or [{"text": ""}]


def _call_gemini_json_mode(
    *,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    schema: Dict[str, Any],
    timeout_sec: float,
) -> str:
    global _GEMINI_WORKING_MODEL, _GEMINI_BAD_MODELS
    available_models = _list_gemini_generate_models(api_key, timeout_sec)
    candidates = _gemini_model_candidates(model, available_models)
    print(
        f"[COMMANDER] gemini model candidates: {candidates[:6]}{'...' if len(candidates) > 6 else ''}",
        flush=True,
    )

    system_text = ""
    contents: List[Dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user").lower()
        content = msg.get("content")
        if role == "system":
            system_text = f"{system_text}\n{str(content or '')}".strip()
        else:
            contents.append({"role": "user", "parts": _gemini_parts_from_content(content)})

    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents": contents or [{"role": "user", "parts": [{"text": "{}"}]}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json"}

    last_exc: Exception | None = None
    for candidate in candidates:
        endpoint = f"{COMMANDER_GEMINI_BASE_URL}/models/{candidate}:generateContent?key={api_key}"
        print(f"[COMMANDER] trying gemini model: {candidate}", flush=True)
        try:
            res = _http_post_json(endpoint, payload, headers, timeout_sec)
            gem_candidates = res.get("candidates") or []
            if not gem_candidates:
                raise RuntimeError(f"Gemini returned no candidates: {res}")
            parts = (((gem_candidates[0] or {}).get("content") or {}).get("parts") or [])
            if not parts:
                raise RuntimeError(f"Gemini returned no content parts: {res}")
            _GEMINI_WORKING_MODEL = candidate
            return str(parts[0].get("text") or "")
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 404:
                _GEMINI_BAD_MODELS.add(candidate)
                continue
            if exc.code in (429, 500, 503):
                # Transient overload/rate-limit: try next available model candidate.
                print(
                    f"[COMMANDER] gemini model {candidate} unavailable ({exc.code}) -> trying next candidate",
                    flush=True,
                )
                continue
            raise
        except Exception as exc:
            last_exc = exc
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No usable Gemini model candidates for generateContent.")


def _call_provider(
    *,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    schema_name: str,
    schema: Dict[str, Any],
    timeout_sec: float,
) -> str:
    provider = _provider_for_model(model)
    if provider == "gemini":
        return _call_gemini_json_mode(
            model=model,
            api_key=api_key,
            messages=messages,
            schema=schema,
            timeout_sec=timeout_sec,
        )
    return _call_openai_structured(
        model=model,
        api_key=api_key,
        messages=messages,
        schema_name=schema_name,
        schema=schema,
        timeout_sec=timeout_sec,
    )


def _is_auth_error(exc: Exception) -> bool:
    txt = str(exc).lower()
    return (
        "401" in txt
        or "403" in txt
        or "unauthorized" in txt
        or "invalid api key" in txt
        or "permission denied" in txt
        or ("api key" in txt and "missing" in txt)
    )


def _route_cosmos_decision(
    *,
    model: str,
    api_key: str,
    timeout_sec: float,
    belief_state: Dict[str, Any],
    user_chat: str,
    latest_observation: str,
    long_memory_snippets: List[str],
    frames: Sequence[Any],
    object_ids: Sequence[str],
    trigger: Dict[str, Any],
) -> CommanderRouting:
    frame_list = list(frames or [])
    route_payload = {
        "user_chat": str(user_chat or "").strip(),
        "latest_observation": str(latest_observation or "").strip(),
        "belief_state": belief_state if isinstance(belief_state, dict) else {},
        "long_memory_snippets": [str(s) for s in (long_memory_snippets or [])[:3]],
        "trigger": trigger if isinstance(trigger, dict) else {},
        "object_ids": [str(n) for n in (object_ids or [])[:40]],
        "frame_count": len(frame_list),
    }
    user_text = (
        "Decide whether to call Cosmos physics observer now.\n"
        "Return use_cosmos=true when temporal interpretation is uncertain.\n"
        f"Context:\n{json.dumps(route_payload, ensure_ascii=True)}"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": _routing_prompt()},
        {
            "role": "user",
            "content": _build_multimodal_user_content(user_text, frame_list[:5]),
        },
    ]

    raw = _call_provider(
        model=model,
        api_key=api_key,
        messages=messages,
        schema_name="CommanderRouting",
        schema=CommanderRouting.model_json_schema(),
        timeout_sec=timeout_sec,
    )
    parsed = json.loads(_extract_json_text(raw))
    return CommanderRouting.model_validate(parsed)


def _propose_hypothesis(
    *,
    model: str,
    api_key: str,
    timeout_sec: float,
    belief_state: Dict[str, Any],
    user_chat: str,
    latest_observation: str,
    long_memory_snippets: List[str],
    trigger: Dict[str, Any],
    object_ids: Sequence[str],
    feedback: str = "",
) -> CommanderHypothesis:
    payload = {
        "user_chat": str(user_chat or "").strip(),
        "latest_observation": str(latest_observation or "").strip(),
        "belief_state": belief_state if isinstance(belief_state, dict) else {},
        "long_memory_snippets": [str(s) for s in (long_memory_snippets or [])[:5]],
        "trigger": trigger if isinstance(trigger, dict) else {},
        "object_ids": [str(n) for n in (object_ids or [])[:64]],
        "feedback_from_physics": str(feedback or "").strip(),
        "task": "Infer intent and propose one assistive action candidate.",
    }
    messages = [
        {"role": "system", "content": _hypothesis_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    raw = _call_provider(
        model=model,
        api_key=api_key,
        messages=messages,
        schema_name="CommanderHypothesis",
        schema=CommanderHypothesis.model_json_schema(),
        timeout_sec=timeout_sec,
    )
    parsed = _coerce_hypothesis_payload(json.loads(_extract_json_text(raw)))
    return CommanderHypothesis.model_validate(parsed)


def _gate_physics_validation(
    *,
    model: str,
    api_key: str,
    timeout_sec: float,
    hypothesis: CommanderHypothesis,
    cosmos_observation: str,
    trigger: Dict[str, Any],
) -> CommanderPhysicsValidation:
    payload = {
        "hypothesis": hypothesis.model_dump(),
        "cosmos_observation": str(cosmos_observation or "").strip(),
        "trigger": trigger if isinstance(trigger, dict) else {},
    }
    messages = [
        {"role": "system", "content": _physics_gate_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
    ]
    raw = _call_provider(
        model=model,
        api_key=api_key,
        messages=messages,
        schema_name="CommanderPhysicsValidation",
        schema=CommanderPhysicsValidation.model_json_schema(),
        timeout_sec=timeout_sec,
    )
    parsed = _coerce_physics_validation_payload(json.loads(_extract_json_text(raw)))
    return CommanderPhysicsValidation.model_validate(parsed)


def _build_cosmos_validation_instruction(hypothesis: CommanderHypothesis) -> str:
    action = str(hypothesis.proposed_action or "noop").strip()
    target = str(hypothesis.target_object or "").strip()
    dest = str(hypothesis.destination or "").strip()
    args = hypothesis.action_args if isinstance(hypothesis.action_args, dict) else {}
    return (
        "A2A Physics Validation Request. "
        f"Proposed action={action}; target_object={target or '(none)'}; destination={dest or '(none)'}; "
        f"action_args={json.dumps(args, ensure_ascii=True)}. "
        "Assess physical feasibility, collisions/occlusions, and stability in current scene."
    )


def _build_reason2_delegate_text(
    *,
    hypothesis: CommanderHypothesis,
    user_chat: str,
    feedback: str = "",
) -> str:
    """Build explicit Commander->Cosmos task text for structured physics reasoning."""
    action = str(hypothesis.proposed_action or "noop").strip()
    target = str(hypothesis.target_object or "").strip()
    dest = str(hypothesis.destination or "").strip()
    args = hypothesis.action_args if isinstance(hypothesis.action_args, dict) else {}
    lines = [
        "A2A COMMAND FROM COMMANDER TO COSMOS_REASON2.",
        "Role: physics+spatial validator with structured JSON output.",
        f"Commander proposed action={action}",
        f"target_object={target or '(none)'}",
        f"destination={dest or '(none)'}",
        f"action_args={json.dumps(args, ensure_ascii=True)}",
        (
            "If proposal is physically valid in the provided frames, return action combine/separate/noop with full args. "
            "If invalid, set action.type=noop and explain constraints in stm_observation/reply."
        ),
    ]
    chat = str(user_chat or "").strip()
    if chat:
        lines.append(f"Human chat: {chat}")
    fb = str(feedback or "").strip()
    if fb:
        lines.append(f"Replan feedback: {fb}")
    return "\n".join(lines)


def _sanitize_action_from_cosmos(action_payload: Any) -> Dict[str, Any]:
    if not isinstance(action_payload, dict):
        return {"type": "noop", "args": {}}
    candidate = {
        "type": str(action_payload.get("type") or "").strip().lower() or "noop",
        "args": action_payload.get("args") if isinstance(action_payload.get("args"), dict) else {},
    }
    try:
        validated = CommanderAction.model_validate(candidate)
        return validated.model_dump(mode="json")
    except Exception:
        return {"type": "noop", "args": {}}


def _merge_belief_updates(commander_update: Dict[str, Any], cosmos_update: Dict[str, Any]) -> Dict[str, Any]:
    base = dict(commander_update) if isinstance(commander_update, dict) else {}
    incoming = dict(cosmos_update) if isinstance(cosmos_update, dict) else {}
    if not incoming:
        return base

    base_objects = base.get("objects")
    if not isinstance(base_objects, dict):
        base_objects = {}
    incoming_objects = incoming.get("objects")
    if isinstance(incoming_objects, dict):
        for name, obj in incoming_objects.items():
            if isinstance(obj, dict):
                base_objects[str(name)] = obj
    if base_objects:
        base["objects"] = base_objects

    base_static = base.get("static_context")
    if not isinstance(base_static, dict):
        base_static = {}
    incoming_static = incoming.get("static_context")
    if isinstance(incoming_static, dict):
        for name, obj in incoming_static.items():
            if isinstance(obj, dict):
                base_static[str(name)] = obj
    if base_static:
        base["static_context"] = base_static

    return base


def _run_a2a_hypothesize_verify(
    *,
    model: str,
    api_key: str,
    timeout_sec: float,
    belief_state: Dict[str, Any],
    user_chat: str,
    latest_observation: str,
    long_memory_snippets: List[str],
    frames: Sequence[Any],
    object_ids: Sequence[str],
    trigger: Dict[str, Any],
) -> Dict[str, Any]:
    from agent.reason2 import reason2_decide

    feedback = ""
    trace: List[Dict[str, Any]] = []
    hypothesis: CommanderHypothesis | None = None
    gate: CommanderPhysicsValidation | None = None
    cosmos_text = str(latest_observation or "").strip()
    cosmos_action: Dict[str, Any] = {"type": "noop", "args": {}}
    cosmos_belief_update: Dict[str, Any] = {}

    # one replan loop (initial + 1 retry)
    for attempt in range(2):
        t1 = time.time()
        hypothesis = _propose_hypothesis(
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            belief_state=belief_state,
            user_chat=user_chat,
            latest_observation=cosmos_text,
            long_memory_snippets=long_memory_snippets,
            trigger=trigger,
            object_ids=object_ids,
            feedback=feedback,
        )
        print(
            f"[COMMANDER][A2A] node1_hypothesis attempt={attempt+1} in {time.time()-t1:.2f}s "
            f"| intent={_brief(hypothesis.intent_inferred, 100)!r} "
            f"| action={_brief(hypothesis.proposed_action, 60)!r}",
            flush=True,
        )

        instruction = _build_cosmos_validation_instruction(hypothesis)
        delegate_text = _build_reason2_delegate_text(
            hypothesis=hypothesis,
            user_chat=user_chat,
            feedback=feedback,
        )
        print(
            f"[COMMANDER][A2A] node2_bridge attempt={attempt+1} | instruction={_brief(instruction, 180)!r}",
            flush=True,
        )

        t3 = time.time()
        cosmos_out = reason2_decide(
            frames=list(frames or []),
            user_text=delegate_text,
            trigger=trigger if isinstance(trigger, dict) else {},
            short_memory_summary=str(latest_observation or ""),
            long_memory_snippets=list(long_memory_snippets or []),
            prev_belief_json=belief_state if isinstance(belief_state, dict) else {},
        )
        if not isinstance(cosmos_out, dict):
            cosmos_out = {}
        cosmos_text = str(cosmos_out.get("stm_observation") or cosmos_out.get("reply") or "").strip()
        cosmos_action = _sanitize_action_from_cosmos(cosmos_out.get("action"))
        belief_candidate = cosmos_out.get("belief_update")
        cosmos_belief_update = belief_candidate if isinstance(belief_candidate, dict) else {}
        print(
            f"[COMMANDER][A2A] node3_cosmos_validation attempt={attempt+1} in {time.time()-t3:.2f}s "
            f"| observation={_brief(cosmos_text, 180)!r} | action={cosmos_action.get('type')}",
            flush=True,
        )

        t4 = time.time()
        gate = _gate_physics_validation(
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            hypothesis=hypothesis,
            cosmos_observation=cosmos_text,
            trigger=trigger,
        )
        print(
            f"[COMMANDER][A2A] node4_gate attempt={attempt+1} in {time.time()-t4:.2f}s "
            f"| valid={gate.is_physically_valid} | reason={_brief(gate.reason, 140)!r}",
            flush=True,
        )

        trace.append(
            {
                "attempt": attempt + 1,
                "hypothesis": hypothesis.model_dump(),
                "cosmos_observation": cosmos_text,
                "cosmos_action": cosmos_action,
                "validation": gate.model_dump(),
            }
        )

        if gate.is_physically_valid:
            break
        feedback = (
            f"Physics validation failed on attempt {attempt+1}: {gate.reason}. "
            "Replan with a physically feasible alternative."
        )

    return {
        "hypothesis": hypothesis.model_dump() if hypothesis else {},
        "validation": gate.model_dump() if gate else {},
        "observer_text": cosmos_text,
        "cosmos_action": cosmos_action,
        "cosmos_belief_update": cosmos_belief_update,
        "trace": trace,
        "used_cosmos": True,
    }


def commander_reason(
    *,
    belief_state: Dict[str, Any],
    user_chat: str,
    latest_observation: str,
    long_memory_snippets: Optional[List[str]] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    frames: Optional[Sequence[Any]] = None,
    object_ids: Optional[Sequence[str]] = None,
    trigger: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Commander LLM entrypoint with A2A routing to Cosmos observer."""

    global _WARNED_AUTH, _WARNED_NO_KEY

    use_model = str(model or COMMANDER_LLM_MODEL or "").strip() or "gpt-4o"
    use_api_key = str(api_key if api_key is not None else COMMANDER_API_KEY or "").strip()
    use_timeout = float(timeout_sec if timeout_sec is not None else COMMANDER_TIMEOUT_SEC)

    snippets = [str(s) for s in (long_memory_snippets or [])]
    frame_list = list(frames or [])
    trig = trigger if isinstance(trigger, dict) else {}
    trig_type = str(trig.get("type") or "unknown").strip().lower()
    is_grounding = _is_grounding_trigger(trig)
    grounding_required_ids = _collect_grounding_required_ids(trig) if is_grounding else []
    cosmos_available = bool(cosmos_is_configured())

    names = [str(n) for n in (object_ids or []) if str(n or "").strip()]
    if not names and isinstance(belief_state, dict):
        objs = belief_state.get("objects")
        if isinstance(objs, dict):
            names = [str(k) for k in objs.keys()]

    if not use_api_key:
        if not _WARNED_NO_KEY:
            print(
                "[COMMANDER][WARN] COMMANDER_API_KEY is not set; live commander calls are disabled.",
                flush=True,
            )
            _WARNED_NO_KEY = True
        raise RuntimeError(
            "COMMANDER_API_KEY is not set. Configure GEMINI_API_KEY/COMMANDER_API_KEY before running."
        )

    print(
        f"[COMMANDER] start | model={use_model} | trigger={trig_type} | frames={len(frame_list)} | "
        f"objects={len(names)} | chat={_brief(user_chat, 80)!r}",
        flush=True,
    )
    if is_grounding:
        print(
            f"[COMMANDER] grounding contract | required_ids={len(grounding_required_ids)} "
            f"| sample={grounding_required_ids[:8]}",
            flush=True,
        )

    routing: CommanderRouting | None = None
    observer_text = str(latest_observation or "").strip()
    used_cosmos = False
    hypothesis_payload: Dict[str, Any] = {}
    validation_payload: Dict[str, Any] = {}
    cosmos_action_payload: Dict[str, Any] = {"type": "noop", "args": {}}
    cosmos_belief_payload: Dict[str, Any] = {}
    a2a_trace: List[Dict[str, Any]] = []

    if frame_list and not is_grounding and cosmos_available:
        try:
            a2a = _run_a2a_hypothesize_verify(
                model=use_model,
                api_key=use_api_key,
                timeout_sec=use_timeout,
                belief_state=belief_state if isinstance(belief_state, dict) else {},
                user_chat=user_chat,
                latest_observation=observer_text,
                long_memory_snippets=snippets,
                frames=frame_list,
                object_ids=names,
                trigger=trig,
            )
            hypothesis_payload = a2a.get("hypothesis") if isinstance(a2a.get("hypothesis"), dict) else {}
            validation_payload = a2a.get("validation") if isinstance(a2a.get("validation"), dict) else {}
            cosmos_action_payload = _sanitize_action_from_cosmos(a2a.get("cosmos_action"))
            cosmos_belief_payload = (
                a2a.get("cosmos_belief_update")
                if isinstance(a2a.get("cosmos_belief_update"), dict)
                else {}
            )
            a2a_trace = a2a.get("trace") if isinstance(a2a.get("trace"), list) else []
            observer_text = str(a2a.get("observer_text") or observer_text).strip()
            used_cosmos = bool(a2a.get("used_cosmos"))
        except Exception as exc:
            print(
                f"[COMMANDER][A2A] hypothesize-verify failed -> fallback routing | err={_brief(exc, 180)}",
                flush=True,
            )
            # Fallback to legacy route path.
            routing = CommanderRouting(use_cosmos=True, observer_instruction="", quick_observation="")
    elif frame_list and not is_grounding and not cosmos_available:
        print(
            "[COMMANDER] Cosmos is not configured; bypassing optional observer and reasoning from commander-only context.",
            flush=True,
        )
        routing = CommanderRouting(use_cosmos=False, observer_instruction="", quick_observation="")
    elif frame_list:
        t_route = time.time()
        try:
            routing = _route_cosmos_decision(
                model=use_model,
                api_key=use_api_key,
                timeout_sec=use_timeout,
                belief_state=belief_state if isinstance(belief_state, dict) else {},
                user_chat=user_chat,
                latest_observation=observer_text,
                long_memory_snippets=snippets,
                frames=frame_list,
                object_ids=names,
                trigger=trig,
            )
            print(
                f"[COMMANDER] routing decided in {time.time()-t_route:.2f}s | "
                f"use_cosmos={routing.use_cosmos} | hint={_brief(routing.observer_instruction, 90)!r}",
                flush=True,
            )
        except Exception as exc:
            # Safe default in uncertainty: use Cosmos.
            routing = CommanderRouting(
                use_cosmos=cosmos_available,
                observer_instruction="",
                quick_observation="",
            )
            print(
                f"[COMMANDER] routing failed after {time.time()-t_route:.2f}s -> "
                f"fallback use_cosmos={cosmos_available} | err={_brief(exc, 140)}",
                flush=True,
            )

    if frame_list and routing is not None:
        if routing.use_cosmos and cosmos_available:
            from .vlm_cosmos import observe_physics

            t_obs = time.time()
            observer_text = observe_physics(
                frames=frame_list,
                object_ids=names,
                trigger=trig,
                user_text=user_chat,
                observer_instruction=routing.observer_instruction,
            )
            used_cosmos = True
            print(
                f"[COMMANDER] cosmos observation in {time.time()-t_obs:.2f}s | text={_brief(observer_text)!r}",
                flush=True,
            )
        elif routing.quick_observation:
            observer_text = str(routing.quick_observation).strip()
            print(
                f"[COMMANDER] using commander quick_observation | text={_brief(observer_text)!r}",
                flush=True,
            )
        elif routing.use_cosmos and not cosmos_available:
            print(
                "[COMMANDER] Cosmos routing requested but Cosmos is not configured; continuing without observer input.",
                flush=True,
            )

    payload = {
        "latest_observation": observer_text,
        "user_chat": str(user_chat or "").strip(),
        "belief_state": belief_state if isinstance(belief_state, dict) else {},
        "long_memory_snippets": [str(s) for s in snippets[:5]],
        "trigger": trig,
        "object_ids": names[:64],
        "task": (
            "Update belief_state_update from observation first, reconcile with user_chat, "
            "and return action only when clearly required."
        ),
    }
    if hypothesis_payload:
        payload["hypothesis"] = hypothesis_payload
    if validation_payload:
        payload["physics_validation"] = validation_payload
    if cosmos_action_payload and cosmos_action_payload.get("type") not in ("", "noop"):
        payload["cosmos_action_proposal"] = cosmos_action_payload
    if cosmos_belief_payload:
        payload["cosmos_belief_update"] = cosmos_belief_payload
    if a2a_trace:
        payload["a2a_trace"] = a2a_trace[-2:]
    if is_grounding:
        payload["task"] = (
            "INITIALIZATION/GROUNDING. Perform static visual analysis of this setup before PLAY. "
            "Fill belief_state_update.objects for every required object id. "
            "Do not rename, remove, or skip required ids. "
            "Use the frame as source of truth; set temporal_change='initial_state' for static scene grounding. "
            "Keep action null/noop unless an explicit assembly command is requested. "
            "Set reply_to_human to JSON only (no markdown, no prose) with this exact top-level key: "
            "{\"interactable_states\": {\"<object_id>\": { ...state fields... }}}."
        )
        payload["required_object_ids"] = grounding_required_ids
        init_form = trig.get("init_belief_form")
        if isinstance(init_form, dict):
            payload["init_belief_form"] = init_form
        payload["reply_schema_example"] = {
            "interactable_states": {
                name: {
                    "belief_status": "present|occluded|contained|moving|static|unknown",
                    "visible": True,
                    "confidence": 0.0,
                    "inferred_container": "",
                    "temporal_change": "initial_state",
                }
                for name in grounding_required_ids[:24]
            }
        }

    system_prompt = _system_prompt()
    if is_grounding:
        system_prompt = (
            system_prompt
            + "\nGrounding Mode (strict):\n"
            + "1) Treat this as static setup analysis before simulation run.\n"
            + "2) belief_state_update.objects MUST include every required_object_id key.\n"
            + "3) reply_to_human MUST be valid JSON only with top-level key 'interactable_states'.\n"
            + "4) interactable_states MUST include every required_object_id key.\n"
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": _build_multimodal_user_content(
                json.dumps(payload, ensure_ascii=True), frame_list[:5]
            ),
        },
    ]

    raw_content = ""
    try:
        t_final = time.time()
        raw_content = _call_provider(
            model=use_model,
            api_key=use_api_key,
            messages=messages,
            schema_name="CommanderOutput",
            schema=CommanderOutput.model_json_schema(),
            timeout_sec=use_timeout,
        )
        print(
            f"[COMMANDER] final reasoning done in {time.time()-t_final:.2f}s",
            flush=True,
        )
        raw_payload = json.loads(_extract_json_text(raw_content))
        validated = CommanderOutput.model_validate(_coerce_commander_payload(raw_payload))
        if is_grounding:
            _validate_grounding_output(validated, grounding_required_ids)
            validated = _normalize_grounding_reply_json(validated, grounding_required_ids)
        out = validated.model_dump(mode="json")
        if cosmos_belief_payload:
            out["belief_state_update"] = _merge_belief_updates(
                out.get("belief_state_update") if isinstance(out.get("belief_state_update"), dict) else {},
                cosmos_belief_payload,
            )
        if cosmos_action_payload.get("type") not in ("", "noop"):
            # Physics executor (Cosmos Reason2) owns function-call actions.
            out["action"] = cosmos_action_payload
        if validation_payload and not bool(validation_payload.get("is_physically_valid", True)):
            out["action"] = {"type": "noop", "args": {}}
        out["_observer_text"] = observer_text
        out["_used_cosmos"] = used_cosmos
        out["_routing"] = routing.model_dump() if routing else {}
        out["_a2a"] = {
            "hypothesis": hypothesis_payload,
            "physics_validation": validation_payload,
            "cosmos_action": cosmos_action_payload,
            "attempts": len(a2a_trace),
        }
        return out
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        print(f"[COMMANDER] schema parse failed -> retry once | err={_brief(exc, 180)}", flush=True)
        correction = (
            "Your previous output failed strict schema validation. "
            "Return JSON only that matches CommanderOutput exactly. "
            f"Error: {exc}"
        )
        retry_messages = list(messages)
        if raw_content:
            retry_messages.append({"role": "assistant", "content": raw_content})
        retry_messages.append({"role": "user", "content": correction})
        retry_raw = ""
        try:
            t_retry = time.time()
            retry_raw = _call_provider(
                model=use_model,
                api_key=use_api_key,
                messages=retry_messages,
                schema_name="CommanderOutput",
                schema=CommanderOutput.model_json_schema(),
                timeout_sec=use_timeout,
            )
            print(
                f"[COMMANDER] retry reasoning done in {time.time()-t_retry:.2f}s",
                flush=True,
            )
            retry_payload = json.loads(_extract_json_text(retry_raw))
            retry_validated = CommanderOutput.model_validate(_coerce_commander_payload(retry_payload))
            if is_grounding:
                _validate_grounding_output(retry_validated, grounding_required_ids)
                retry_validated = _normalize_grounding_reply_json(retry_validated, grounding_required_ids)
            out = retry_validated.model_dump(mode="json")
            if cosmos_belief_payload:
                out["belief_state_update"] = _merge_belief_updates(
                    out.get("belief_state_update") if isinstance(out.get("belief_state_update"), dict) else {},
                    cosmos_belief_payload,
                )
            if cosmos_action_payload.get("type") not in ("", "noop"):
                out["action"] = cosmos_action_payload
            if validation_payload and not bool(validation_payload.get("is_physically_valid", True)):
                out["action"] = {"type": "noop", "args": {}}
            out["_observer_text"] = observer_text
            out["_used_cosmos"] = used_cosmos
            out["_routing"] = routing.model_dump() if routing else {}
            out["_a2a"] = {
                "hypothesis": hypothesis_payload,
                "physics_validation": validation_payload,
                "cosmos_action": cosmos_action_payload,
                "attempts": len(a2a_trace),
            }
            return out
        except Exception as retry_exc:
            raise RuntimeError(
                "Commander output schema validation failed after retry. "
                f"first_error={_brief(exc, 200)} retry_error={_brief(retry_exc, 200)} "
                f"first_raw={_brief(raw_content, 260)!r} retry_raw={_brief(retry_raw, 260)!r}"
            ) from retry_exc
    except Exception as exc:
        print(f"[COMMANDER] exception: {_brief(exc, 220)}", flush=True)
        if _is_auth_error(exc):
            if not _WARNED_AUTH:
                print(
                    "[COMMANDER][WARN] Commander authentication failed. "
                    "Set COMMANDER_API_KEY to enable live commander calls.",
                    flush=True,
                )
                _WARNED_AUTH = True
        raise


def validate_commander_output(payload: Dict[str, Any]) -> CommanderOutput:
    return CommanderOutput.model_validate(_coerce_commander_payload(payload))
