from collections import deque

from .config import (
    MAX_LOG_LINES,
    MAX_TURNS_TO_KEEP,
    CAMERA_PRIM_PATH,
    LAST_FRAME_PATH,
    SENT_FRAMES_ROOT,
    CAPTURE_FPS,
    FRAME_BUFFER_SECONDS,
)


class State:
    def __init__(self):
        # UI models / widgets
        self.window = None
        self.log_label = None
        self.log_frame = None
        self.input_model = None
        self.attach_latest_model = None
        self.action_label = None
        self.belief_label = None
        self.ltm_label = None

        # Logs / chat
        self.log_lines = deque(maxlen=MAX_LOG_LINES)
        self.chat_history = deque(maxlen=2 * MAX_TURNS_TO_KEEP + 2)

        # Replicator / camera
        self.rp = None
        self.rgb_annot = None
        self.camera_ready = False
        self.active_camera_prim_path = CAMERA_PRIM_PATH
        self.last_rgb = None
        self.last_frame_path = LAST_FRAME_PATH
        self.sent_frames_root = SENT_FRAMES_ROOT
        self.sent_frames_dir = SENT_FRAMES_ROOT
        # Pre-play frame directory prepared by "Initiate Cosmos" so grounding and
        # the subsequent run can share the same sent_frames folder.
        self.pending_run_frames_dir = ""

        self.playing = False
        self.run_active = False
        self.run_start_time = 0.0
        self.run_id = 0

        # RAM frame buffer for high-rate capture
        self.frame_buffer = deque(maxlen=max(1, int(CAPTURE_FPS * FRAME_BUFFER_SECONDS)))
        self.capture_task = None
        self.inquiry_task = None
        self.run_bootstrap_task = None

        # Cosmos connection state
        self.cosmos_state = None  # None = unknown, True = connected, False = failed
        self.cosmos_last_detail = ""

        # VLM request guard
        self.vlm_busy = False
        self.vlm_busy_since = 0.0
        self.vlm_busy_trigger = ""    # "auto"|"user"|"gt_change"|"grounding" — what triggered current busy
        self.pause_inquiry_until = 0.0
        self.manual_pending = False   # True while a manual (user) request is in-flight
        self.last_vlm_response = ""
        self.last_infer_time = 0.0
        self.last_auto_infer_ts = 0.0

        # VLM status indicator (UI label set in build_ui)
        self.vlm_status_label = None

        # Theory-of-Mind / cognition modules (optional)
        self.belief_manager = None
        self.ghost_visualizer = None
        self.worker_in_q = None
        self.worker_out_q = None
        self.worker_thread = None
        self.worker_poll_task = None

        # Agent memory
        self.short_memory = None
        self.long_memory = None

        # Robot control (optional)
        self.robot_controller = None
        self.robot_update_task = None

        # Magic (kinematic) assembly manager
        self.magic_assembly = None
        # Startup scene snapshot for explicit /reset commands.
        self.initial_scene_snapshot = {}
        self.initial_scene_stage_id = ""

        # Sequence runner (loaded from assembly/gearbox_sequence.yaml)
        self.sequence_runner = None
        # Slash command to re-dispatch after the operator answers a paused
        # check_pose prompt (see _ui_pose_resolver / on_send in ui.py). None
        # when no pose query is awaiting an answer.
        self.pose_resume_cmd = None

        # Hybrid ToM: GT state monitor + ring buffer
        self.state_monitor = None
        self.ring_buffer = None
        self.last_trigger_event = None
        self.pending_user_text = ""  # User input waiting to be processed

        # Keep Kit event subscriptions alive (avoid GC).
        self._kit_subs = []

        # Startup guard: prevent any unintended auto-play until stage is ready.
        self.startup_stage_opened = False
        self.startup_autostop_done = False

        # Bootstrap recovery guard: when True, timeline STOP events are ignored
        # so that the bootstrap stop→play recovery cycle is not interrupted.
        self._bootstrap_recovery = False

        # One-shot static initialization (Initiate Cosmos grounding).
        # grounding_in_progress: True while the first VLM grounding call is in-flight.
        # grounding_complete: True once the grounding response has been received and
        #                     ShortTermMemory has been populated.
        self.grounding_in_progress = False
        self.grounding_complete = False

        # UI label that shows "Initializing..." / "Ready" / "Ready (no init)" near
        # the Initiate Cosmos button.  Set by build_ui(); updated by on_initiate_cosmos()
        # and _worker_poll_loop().
        self.init_status_label = None
        # Reference to the PLAY button so it can be disabled during grounding.
        self.play_button = None


STATE = State()
