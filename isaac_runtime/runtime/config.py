import os
from urllib.parse import urlparse

from .paths import PROJECT_ROOT, TMP_DIR, load_runtime_env_defaults


load_runtime_env_defaults()


# -----------------------------
# SETTINGS
# -----------------------------
MEM0_API_KEY = (os.getenv("MEM0_API_KEY") or "").strip()

CAMERA_PRIM_PATH = "/Franka/head_camera"  # Copy Prim Path
# Scene default prim is /Root (not /World). The workbench is the `table_low`
# object; hover()/surface detection descends to the tabletop mesh under it.
TABLE_PRIM_PATH = os.getenv("TABLE_PRIM_PATH", "/Root/table_low_327")
# Fallback surface heights (world metres) used when the table prim can't be
# found by detection. The real tabletop mesh top measures ≈ -0.091; -0.1 is the
# canonical value shared with tools/scatter_parts.py (TABLE_SURFACE_Z).
TABLE_SURFACE_Z = float(os.getenv("TABLE_SURFACE_Z", "-0.1"))
FLOOR_SURFACE_Z = float(os.getenv("FLOOR_SURFACE_Z", "0.02"))
ROBOT_PRIM_PATH = os.getenv("ROBOT_PRIM_PATH", "").strip()  # optional: existing Franka prim path in stage
ROBOT_PRIM_EXPR = os.getenv("ROBOT_PRIM_EXPR", "").strip()  # optional regex expression to find robot prim(s)


# Franka control
# Enabled by default. Auto-discovers Franka/Panda robot in the scene.
# Disable by setting ENABLE_FRANKA_CONTROL=0 if your scene has non-Franka articulations.
ENABLE_FRANKA_CONTROL = os.getenv("ENABLE_FRANKA_CONTROL", "1").strip() not in ("0", "false", "False")
FRANKA_HOME_JOINT_POS = {
    # Default Panda home pose (matches many examples). Override in code if your robot differs.
    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
}

RESOLUTION = (640, 480)
IMAGE_FORMAT = "JPEG"
IMAGE_MIME = "image/jpeg"
IMAGE_QUALITY = 90
SAVE_EACH_SENT_FRAME = True

# Saved frames
_PROJECT_DIR = str(PROJECT_ROOT)
LAST_FRAME_PATH = str(TMP_DIR / "isaac_cam_latest.jpg")
SENT_FRAMES_ROOT = str(TMP_DIR / "sent_frames")
MEMORY_ROOT = str(TMP_DIR / "memory")
LTM_SQLITE_PATH = str(TMP_DIR / "memory" / "ltm.sqlite3")


def _int_env(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        return float(raw)
    except Exception:
        return float(default)


COSMOS_BASE_URL = (os.getenv("COSMOS_BASE_URL") or "").strip()
COSMOS_HOST = (os.getenv("COSMOS_HOST") or os.getenv("COSMOS_HOST_IP") or "").strip()
COSMOS_PORT = _int_env("COSMOS_PORT", 8000)
COSMOS_MODEL = (os.getenv("COSMOS_MODEL") or "/models/Cosmos-Reason2-8B").strip()
COSMOS_API_KEY = (os.getenv("COSMOS_API_KEY") or "").strip()
TIMEOUT_SEC = _float_env("COSMOS_TIMEOUT", 60.0)
FORCE_JSON_RESPONSE = os.getenv("FORCE_JSON_RESPONSE", "1").strip() not in ("0", "false", "False")
COSMOS_MAX_MODEL_LEN = _int_env("COSMOS_MAX_MODEL_LEN", 2048)


def _normalize_chat_completions_url(raw: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return ""
    parsed = urlparse(value)
    if not parsed.scheme:
        value = f"http://{value.lstrip('/')}"
        parsed = urlparse(value)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/chat/completions"):
        return value.rstrip("/")
    if path.endswith("/v1"):
        path = f"{path}/chat/completions"
    else:
        path = f"{path}/v1/chat/completions" if path else "/v1/chat/completions"
    return parsed._replace(path=path).geturl().rstrip("/")


def cosmos_chat_completions_url() -> str:
    explicit = (
        os.getenv("COSMOS_CHAT_COMPLETIONS_URL")
        or os.getenv("COSMOS_URL")
        or ""
    ).strip()
    if explicit:
        return _normalize_chat_completions_url(explicit)
    if COSMOS_BASE_URL:
        return _normalize_chat_completions_url(COSMOS_BASE_URL)
    if COSMOS_HOST:
        host = COSMOS_HOST
        if "://" in host:
            base = host
        else:
            base = f"http://{host}"
        parsed = urlparse(base)
        netloc = parsed.netloc or parsed.path
        path = parsed.path if parsed.netloc else ""
        if ":" not in netloc:
            netloc = f"{netloc}:{COSMOS_PORT}"
        return _normalize_chat_completions_url(parsed._replace(netloc=netloc, path=path).geturl())
    return ""


def cosmos_is_configured() -> bool:
    return bool(cosmos_chat_completions_url() and str(COSMOS_MODEL or "").strip())


COSMOS_URL = cosmos_chat_completions_url()
COSMOS_CHAT_COMPLETIONS_URL = COSMOS_URL
COSMOS_ENABLED = cosmos_is_configured()

# Commander LLM (cognitive layer)
COMMANDER_LLM_MODEL = os.getenv("COMMANDER_LLM_MODEL", "gpt-4o").strip()
COMMANDER_API_KEY = (
    os.getenv("COMMANDER_API_KEY")
    or os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
).strip()
COMMANDER_TIMEOUT_SEC = _float_env("COMMANDER_TIMEOUT_SEC", 45.0)
COMMANDER_BASE_URL = os.getenv("COMMANDER_BASE_URL", "").strip()
COMMANDER_GEMINI_BASE_URL = os.getenv("COMMANDER_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").strip()

MAX_TURNS_TO_KEEP = 12
MAX_LOG_LINES = 400

SYSTEM_MSG = None
STREAM_DEFAULT_PROMPT = "Describe what you see."

# Capture/inquiry defaults
CAPTURE_FPS = 5.0
INQUIRY_INTERVAL_SEC = 1.0
INQUIRY_FRAME_COUNT = 3
FRAME_BUFFER_SECONDS = 2.0

# Agent scheduling
MIN_INFER_INTERVAL_SEC = _float_env("MIN_INFER_INTERVAL_SEC", 0.8)

# ---------------------------------------------------------------------------
# Hybrid Theory-of-Mind: GT State Monitor
# ---------------------------------------------------------------------------
GT_POSITION_THRESHOLD = _float_env("GT_POSITION_THRESHOLD", 0.02)
GT_ORIENTATION_THRESHOLD = _float_env("GT_ORIENTATION_THRESHOLD", 0.05)
GT_COOLDOWN_SEC = _float_env("GT_COOLDOWN_SEC", 2.0)
RING_BUFFER_CAPACITY = _int_env("RING_BUFFER_CAPACITY", 5)
GT_TRACKED_PRIMS = [p.strip() for p in os.getenv("GT_TRACKED_PRIMS", "").split(",") if p.strip()]

# ---------------------------------------------------------------------------
# Robot Control Tuning (AI-CPS industrial-grade values)
# ---------------------------------------------------------------------------
ROBOT_SHOULDER_STIFFNESS = _float_env("ROBOT_SHOULDER_STIFFNESS", 400.0)
ROBOT_SHOULDER_DAMPING = _float_env("ROBOT_SHOULDER_DAMPING", 80.0)
ROBOT_FOREARM_STIFFNESS = _float_env("ROBOT_FOREARM_STIFFNESS", 400.0)
ROBOT_FOREARM_DAMPING = _float_env("ROBOT_FOREARM_DAMPING", 80.0)
ROBOT_HAND_STIFFNESS = _float_env("ROBOT_HAND_STIFFNESS", 2000.0)
ROBOT_HAND_DAMPING = _float_env("ROBOT_HAND_DAMPING", 100.0)
SOLVER_POSITION_ITERATIONS = _int_env("SOLVER_POSITION_ITERATIONS", 12)
SOLVER_VELOCITY_ITERATIONS = _int_env("SOLVER_VELOCITY_ITERATIONS", 1)
PREFER_DIFFIK = os.getenv("PREFER_DIFFIK", "0").strip() not in ("0", "false", "False")
