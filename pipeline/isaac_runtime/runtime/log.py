import time

try:
    import carb  # type: ignore
except ImportError:  # pragma: no cover - Isaac Sim runtime optional for pure-Python checks
    carb = None

from .state import STATE


def render_log():
    if STATE.log_label is None:
        return
    STATE.log_label.text = "\n".join(STATE.log_lines)

    # Try to auto-scroll to bottom (best-effort; depends on Kit build)
    try:
        if STATE.log_frame is not None:
            STATE.log_frame.scroll_y = 1e9
    except Exception:
        pass


def log_line(prefix, msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {prefix} {msg}"
    STATE.log_lines.append(line)
    render_log()
    if STATE.log_label is None:
        print(line, flush=True)


def log_info(msg):
    log_line("INFO ", str(msg))
    if carb is not None:
        carb.log_info(str(msg))
    else:
        print(f"[INFO] {msg}", flush=True)


def log_warn(msg):
    log_line("WARN ", str(msg))
    if carb is not None:
        carb.log_warn(str(msg))
    else:
        print(f"[WARN] {msg}", flush=True)


def log_error(msg):
    log_line("ERROR", str(msg))
    if carb is not None:
        carb.log_error(str(msg))
    else:
        print(f"[ERROR] {msg}", flush=True)
