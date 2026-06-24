from __future__ import annotations

import json as _json
import logging
import os
import sys
import threading
import time
import uuid
import contextlib
import re
import traceback
import platform
import subprocess
import runpy
import signal
from contextvars import ContextVar
from collections import defaultdict, deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Dict, Any, Deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_STUDIO_ROOT = APP_ROOT.parent
RUNTIME_HOME = Path(
    os.environ.get("QUANTUMSTUDIO_RUNTIME_HOME", str(DEFAULT_STUDIO_ROOT)),
)
STUDIO_ROOT = RUNTIME_HOME
MLX_ROOT = Path(
    os.environ.get("QUANTUMSTUDIO_MLX_ROOT", str(DEFAULT_STUDIO_ROOT.parent)),
)
BENCH_DIR = Path(
    os.environ.get("QUANTUMSTUDIO_BENCH_DIR", str(MLX_ROOT / "bench")),
)
RUNS_DIR = Path(
    os.environ.get("QUANTUMSTUDIO_RUNS_DIR", str(RUNTIME_HOME / "runs")),
)
SETTINGS_FILE = Path(
    os.environ.get(
        "QUANTUMSTUDIO_SETTINGS_FILE",
        str(RUNTIME_HOME / "settings.json"),
    ),
)
MLX_PYTHON = Path(
    os.environ.get("QUANTUMSTUDIO_MLX_PYTHON", str(MLX_ROOT / "src")),
)
LOG_DIR = Path(
    os.environ.get("QUANTUMSTUDIO_LOG_DIR", str(RUNTIME_HOME / "logs")),
)
REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")
SERVER_STARTED_AT = time.time()
MAX_LOG_READ_BYTES = 2 * 1024 * 1024
MAX_BENCHMARKS_PER_RUN = 32
MAX_ENV_OVERRIDES = 32
MAX_QASM_CONTENT_BYTES = 1_000_000
DEFAULT_RATE_LIMIT_WINDOW_SEC = 60
DEFAULT_RATE_LIMIT_MAX_REQUESTS = 240
HEAVY_RATE_LIMIT_WINDOW_SEC = 60
HEAVY_RATE_LIMIT_MAX_REQUESTS = 30
RUN_CREATE_RATE_LIMIT_WINDOW_SEC = 60
RUN_CREATE_RATE_LIMIT_MAX_REQUESTS = 12
API_AUTH_TOKEN = os.environ.get("QUANTUMSTUDIO_API_TOKEN", "").strip()

for directory in [RUNTIME_HOME, RUNS_DIR, BENCH_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

_log_handler = RotatingFileHandler(
    LOG_DIR / "backend.log",
    maxBytes=5 * 1024 * 1024,  # 5MB per file
    backupCount=3,
    encoding="utf-8",
)
_log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logger = logging.getLogger("quantumstudio")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(_log_handler)

    # Also add console handler for development
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(_console_handler)

if str(MLX_PYTHON) not in sys.path:
    sys.path.insert(0, str(MLX_PYTHON))


# ---------------------------------------------------------------------------
# Settings Management
# ---------------------------------------------------------------------------

def _get_default_max_concurrent() -> int:
    """Calculate default max concurrent jobs based on CPU cores."""
    try:
        cores = os.cpu_count() or 4
        return max(1, cores // 4)
    except Exception:
        return 2


def _load_settings() -> Dict[str, Any]:
    """Load settings from disk or return defaults."""
    defaults = {
        "max_concurrent_jobs": _get_default_max_concurrent(),
    }
    if SETTINGS_FILE.exists():
        try:
            data = _json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return {**defaults, **data}
        except Exception:
            logger.warning("Failed to load settings file %s", SETTINGS_FILE, exc_info=True)
    return defaults


def _save_settings(settings: Dict[str, Any]) -> None:
    """Persist settings to disk."""
    try:
        SETTINGS_FILE.write_text(_json.dumps(settings, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("Failed to save settings file %s", SETTINGS_FILE, exc_info=True)


SETTINGS = _load_settings()


# ---------------------------------------------------------------------------
# Error and Safety Helpers
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%s; using default=%d", name, raw, default)
        return default
    if value < min_value or value > max_value:
        logger.warning(
            "Out-of-range value for %s=%d (allowed %d..%d); using default=%d",
            name,
            value,
            min_value,
            max_value,
            default,
        )
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean for %s=%s; using default=%s", name, raw, default)
    return default


RATE_LIMIT_ENABLED = _env_bool("QUANTUMSTUDIO_RATE_LIMIT_ENABLED", True)
DEFAULT_RATE_LIMIT_WINDOW_SEC = _env_int(
    "QUANTUMSTUDIO_RATE_LIMIT_WINDOW_SEC",
    DEFAULT_RATE_LIMIT_WINDOW_SEC,
    min_value=1,
    max_value=3600,
)
DEFAULT_RATE_LIMIT_MAX_REQUESTS = _env_int(
    "QUANTUMSTUDIO_RATE_LIMIT_MAX_REQUESTS",
    DEFAULT_RATE_LIMIT_MAX_REQUESTS,
    min_value=10,
    max_value=10000,
)
HEAVY_RATE_LIMIT_WINDOW_SEC = _env_int(
    "QUANTUMSTUDIO_HEAVY_RATE_LIMIT_WINDOW_SEC",
    HEAVY_RATE_LIMIT_WINDOW_SEC,
    min_value=1,
    max_value=3600,
)
HEAVY_RATE_LIMIT_MAX_REQUESTS = _env_int(
    "QUANTUMSTUDIO_HEAVY_RATE_LIMIT_MAX_REQUESTS",
    HEAVY_RATE_LIMIT_MAX_REQUESTS,
    min_value=1,
    max_value=1000,
)
RUN_CREATE_RATE_LIMIT_WINDOW_SEC = _env_int(
    "QUANTUMSTUDIO_RUN_CREATE_RATE_LIMIT_WINDOW_SEC",
    RUN_CREATE_RATE_LIMIT_WINDOW_SEC,
    min_value=1,
    max_value=3600,
)
RUN_CREATE_RATE_LIMIT_MAX_REQUESTS = _env_int(
    "QUANTUMSTUDIO_RUN_CREATE_RATE_LIMIT_MAX_REQUESTS",
    RUN_CREATE_RATE_LIMIT_MAX_REQUESTS,
    min_value=1,
    max_value=200,
)

AUTH_EXEMPT_PATHS = {"/api/health"}
HEAVY_RATE_LIMIT_PATHS = {
    "/api/runs",
    "/api/qasm/visualize/ascii",
    "/api/qasm/visualize/image",
}

METRICS_LOCK = threading.Lock()
REQUEST_METRICS: Dict[str, Any] = {
    "total_requests": 0,
    "status_2xx": 0,
    "status_4xx": 0,
    "status_5xx": 0,
    "auth_rejected": 0,
    "rate_limited": 0,
}


class SlidingWindowRateLimiter:
    """Simple in-memory sliding window limiter keyed by endpoint+client."""

    def __init__(self):
        self._lock = threading.Lock()
        self._events: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._events[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= max_requests:
                retry_after = max(1, int(bucket[0] + window_seconds - now))
                return False, retry_after

            bucket.append(now)
            return True, 0

    def clear(self) -> None:
        with self._lock:
            self._events.clear()


RATE_LIMITER = SlidingWindowRateLimiter()

def _current_request_id() -> str:
    return REQUEST_ID_CTX.get()


def _truncate_for_log(value: str, max_chars: int = 2048) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}... [truncated]"


def _safe_remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning(
            "Failed to remove file %s (request_id=%s)",
            path,
            _current_request_id(),
            exc_info=True,
        )


def _read_log_text(path: Path, max_bytes: int = MAX_LOG_READ_BYTES) -> str:
    try:
        size = path.stat().st_size
        if size <= max_bytes:
            return path.read_text(encoding="utf-8", errors="replace")

        with path.open("rb") as fh:
            fh.seek(-max_bytes, os.SEEK_END)
            tail = fh.read()

        return (
            "[log truncated to last "
            f"{max_bytes // (1024 * 1024)}MB]\n"
            + tail.decode("utf-8", errors="replace")
        )
    except Exception as exc:
        logger.exception(
            "Failed to read run log %s (request_id=%s)",
            path,
            _current_request_id(),
        )
        raise HTTPException(status_code=500, detail=f"Failed to read run log: {exc}") from exc


def _http_detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    try:
        return _truncate_for_log(_json.dumps(detail))
    except Exception:
        return _truncate_for_log(str(detail))


def _increment_metric(name: str, amount: int = 1) -> None:
    with METRICS_LOCK:
        REQUEST_METRICS[name] = int(REQUEST_METRICS.get(name, 0)) + amount


def _observe_status(status_code: int) -> None:
    _increment_metric("total_requests")
    if 200 <= status_code < 300:
        _increment_metric("status_2xx")
    elif 400 <= status_code < 500:
        _increment_metric("status_4xx")
    elif status_code >= 500:
        _increment_metric("status_5xx")


def _client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _extract_auth_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return request.headers.get("x-api-token", "").strip()


def _is_auth_required(request: Request) -> bool:
    if not API_AUTH_TOKEN:
        return False
    if request.method.upper() == "OPTIONS":
        return False
    if request.url.path in AUTH_EXEMPT_PATHS:
        return False
    return request.url.path.startswith("/api/")


def _is_static_auth_required(request: Request) -> bool:
    if not API_AUTH_TOKEN:
        return False
    if request.method.upper() == "OPTIONS":
        return False
    return request.url.path.startswith("/bench/")


def _rate_limit_policy_for(request: Request) -> tuple[int, int]:
    if request.url.path == "/api/runs" and request.method.upper() == "POST":
        return RUN_CREATE_RATE_LIMIT_MAX_REQUESTS, RUN_CREATE_RATE_LIMIT_WINDOW_SEC
    if request.url.path in HEAVY_RATE_LIMIT_PATHS:
        return HEAVY_RATE_LIMIT_MAX_REQUESTS, HEAVY_RATE_LIMIT_WINDOW_SEC
    return DEFAULT_RATE_LIMIT_MAX_REQUESTS, DEFAULT_RATE_LIMIT_WINDOW_SEC


def _metrics_snapshot() -> Dict[str, Any]:
    uptime = max(0.0, time.time() - SERVER_STARTED_AT)
    with METRICS_LOCK:
        base = dict(REQUEST_METRICS)
    base["uptime_seconds"] = round(uptime, 3)
    base["rate_limit_enabled"] = RATE_LIMIT_ENABLED
    base["auth_enabled"] = bool(API_AUTH_TOKEN)
    return base


def _apply_api_response_headers(response: JSONResponse | PlainTextResponse) -> None:
    response.headers["cache-control"] = "no-store"
    response.headers["x-content-type-options"] = "nosniff"
    response.headers["x-frame-options"] = "DENY"
    response.headers["referrer-policy"] = "no-referrer"


def _set_thread_exception_hook() -> None:
    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        logger.error(
            "Unhandled thread exception in %s: %s",
            args.thread.name if args.thread else "unknown-thread",
            args.exc_value,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = _thread_excepthook


_set_thread_exception_hook()


# ---------------------------------------------------------------------------
# Job Queue Manager
# ---------------------------------------------------------------------------

class JobQueueManager:
    """Manages concurrent job execution with configurable limits."""

    def __init__(self):
        self._lock = threading.Lock()
        self._queue: Deque[str] = deque()  # run_ids waiting to execute
        self._running: Dict[str, subprocess.Popen] = {}  # run_id -> process
        self._worker_threads: Dict[str, threading.Thread] = {}

    @property
    def max_concurrent(self) -> int:
        return SETTINGS.get("max_concurrent_jobs", 2)

    def submit(self, run_id: str, payload: "RunRequest") -> None:
        """Submit a job to the queue."""
        with self._lock:
            self._queue.append(run_id)
            self._update_queue_positions()
        logger.info("Job submitted: %s (queue position: %d)", run_id, len(self._queue))
        self._try_start_next()

    def _update_queue_positions(self) -> None:
        """Update queue_position for all queued jobs."""
        with RUNS_LOCK:
            for idx, rid in enumerate(self._queue):
                if rid in RUNS:
                    RUNS[rid]["queue_position"] = idx + 1

    def _try_start_next(self) -> None:
        """Start next job if capacity allows."""
        with self._lock:
            while len(self._running) < self.max_concurrent and self._queue:
                run_id = self._queue.popleft()
                self._start_job(run_id)
            self._update_queue_positions()

    def _start_job(self, run_id: str) -> None:
        """Start a job in a subprocess."""
        with RUNS_LOCK:
            run_data = RUNS.get(run_id)
            if not run_data or run_data.get("status") != "queued":
                return
            payload_json = _json.dumps(run_data.get("_payload", {}))

        # Mark as running immediately to prevent over-scheduling
        # Use None as placeholder until actual process is created
        self._running[run_id] = None

        # Create worker thread that manages the subprocess
        thread = threading.Thread(
            target=self._run_job_subprocess,
            args=(run_id, payload_json),
            daemon=True
        )
        self._worker_threads[run_id] = thread
        thread.start()

    def _run_job_subprocess(self, run_id: str, payload_json: str) -> None:
        """Run job in subprocess and handle completion."""
        log_path = RUNS_DIR / f"{run_id}.log"
        started_at = _now_iso()
        should_run = True

        with RUNS_LOCK:
            run_data = RUNS.get(run_id)
            if not run_data:
                should_run = False
            elif run_data.get("status") != "queued":
                # Job may have been stopped/cancelled between queue pop and thread start.
                run_data["queue_position"] = None
                if run_data.get("ended_at") is None and run_data.get("status") in {"stopped", "cancelled"}:
                    run_data["ended_at"] = started_at
                should_run = False
            else:
                run_data["status"] = "running"
                run_data["started_at"] = started_at
                run_data["queue_position"] = None
        _persist_run(run_id)

        if not should_run:
            with self._lock:
                self._running.pop(run_id, None)
                self._worker_threads.pop(run_id, None)
            self._try_start_next()
            return

        # Run the job worker script as subprocess
        worker_script = APP_ROOT / "job_worker.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(MLX_PYTHON) + ":" + env.get("PYTHONPATH", "")

        log_file = None
        try:
            log_file = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                [sys.executable, str(worker_script), run_id, payload_json],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(MLX_ROOT),
                env=env,
            )

            with self._lock:
                self._running[run_id] = proc

            # Wait for completion
            exit_code = proc.wait()

        except Exception as e:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\nFailed to start job: {e}\n")
                f.write(traceback.format_exc())
            exit_code = 1
        finally:
            if log_file is not None:
                log_file.close()
            with self._lock:
                self._running.pop(run_id, None)
                self._worker_threads.pop(run_id, None)

        # Determine final status
        with RUNS_LOCK:
            current_status = RUNS.get(run_id, {}).get("status")
            # Only update if not already stopped
            if current_status == "running":
                RUNS[run_id]["status"] = "completed" if exit_code == 0 else "failed"
            RUNS[run_id]["ended_at"] = _now_iso()
            RUNS[run_id]["exit_code"] = exit_code

            # Collect outputs
            run_data = RUNS[run_id]
            existing = run_data.get("_existing_files", set())
            start_ts = run_data.get("_start_ts", 0)
            bench_names = run_data.get("benchmarks", [])
            outputs = _collect_outputs(existing, start_ts, bench_names)
            RUNS[run_id]["outputs"] = outputs
            final_status = RUNS[run_id]["status"]

        _persist_run(run_id)
        logger.info("Job finished: %s status=%s exit_code=%d", run_id, final_status, exit_code)

        # Try to start next job
        self._try_start_next()

    def stop_job(self, run_id: str) -> bool:
        """Stop a running job."""
        with self._lock:
            if run_id not in self._running:
                return False

            proc = self._running.get(run_id)
            if proc is None:
                # Job was just started, process not yet created
                # Wait a bit and retry
                pass

            try:
                if proc is not None:
                    # Send SIGTERM first, then SIGKILL if needed
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()

                with RUNS_LOCK:
                    RUNS[run_id]["status"] = "stopped"
                    RUNS[run_id]["ended_at"] = _now_iso()
                    RUNS[run_id]["queue_position"] = None
                _persist_run(run_id)
                return True
            except Exception:
                return False

    def cancel_job(self, run_id: str) -> bool:
        """Cancel a queued job."""
        with self._lock:
            if run_id in self._queue:
                self._queue.remove(run_id)
                self._update_queue_positions()
                with RUNS_LOCK:
                    RUNS[run_id]["status"] = "cancelled"
                    RUNS[run_id]["ended_at"] = _now_iso()
                    RUNS[run_id]["queue_position"] = None
                _persist_run(run_id)
                return True
            return False

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status."""
        with self._lock:
            return {
                "running_count": len(self._running),
                "queued_count": len(self._queue),
                "max_concurrent": self.max_concurrent,
                "running_ids": list(self._running.keys()),
                "queued_ids": list(self._queue),
            }

    def stop_all(self) -> int:
        """Stop all running jobs and clear queue."""
        stopped = 0
        with self._lock:
            # Cancel all queued
            while self._queue:
                run_id = self._queue.popleft()
                with RUNS_LOCK:
                    RUNS[run_id]["status"] = "cancelled"
                    RUNS[run_id]["ended_at"] = _now_iso()
                    RUNS[run_id]["queue_position"] = None
                _persist_run(run_id)
                stopped += 1

            # Stop all running - send SIGTERM without waiting
            for run_id, proc in list(self._running.items()):
                try:
                    if proc is not None:
                        proc.terminate()
                    with RUNS_LOCK:
                        RUNS[run_id]["status"] = "stopped"
                        RUNS[run_id]["ended_at"] = _now_iso()
                        RUNS[run_id]["queue_position"] = None
                    _persist_run(run_id)
                    stopped += 1
                except Exception:
                    pass
        return stopped


JOB_QUEUE = JobQueueManager()

BENCHMARKS = [
    {"name": "hamiltonian_simulation", "label": "Hamiltonian Simulation", "max_qubits": 30},
    {"name": "time_evolution", "label": "Time Evolution", "max_qubits": 30},
    {"name": "trotter", "label": "Trotter", "max_qubits": 30},
    {"name": "heisenberg", "label": "Heisenberg", "max_qubits": 30},
    {"name": "heisenberg_xxz", "label": "Heisenberg XXZ", "max_qubits": 25},
    {"name": "heisenberg_random_field", "label": "Heisenberg Random Field", "max_qubits": 25},
    {"name": "tfim", "label": "TFIM", "max_qubits": 25},
    {"name": "tfim_trotter2", "label": "TFIM Trotter2", "max_qubits": 25},
    {"name": "tfim_random_field", "label": "TFIM Random Field", "max_qubits": 25},
    {"name": "long_range_ising", "label": "Long-Range Ising", "max_qubits": 25},
    {"name": "ladder_heisenberg", "label": "Ladder Heisenberg", "max_qubits": 25},
    {"name": "steady_state", "label": "Steady State", "max_qubits": 12},
    {"name": "random_circuit", "label": "Random Circuit", "max_qubits": 25},
    {"name": "qcbm", "label": "QCBM", "max_qubits": 25},
    {"name": "phase_estimation", "label": "Phase Estimation", "max_qubits": 12},
    {"name": "qft", "label": "QFT", "max_qubits": 12},
    {"name": "qaoa", "label": "QAOA", "max_qubits": 25},
    {"name": "vqe", "label": "VQE", "max_qubits": 15},
    {"name": "variational_circuit", "label": "Variational Circuit", "max_qubits": 25},
    {"name": "grover", "label": "Grover", "max_qubits": 25},
    {"name": "ghz", "label": "GHZ", "max_qubits": 25},
    {"name": "qasm", "label": "QASM", "max_qubits": 18},
]


class BenchmarkConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    qubits_spec: str = Field("default", min_length=1, max_length=128)
    backend: str = Field("sv", min_length=1, max_length=32)
    simulate_cap: Optional[int] = Field(None, ge=1, le=256)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[a-z0-9_]+", cleaned):
            raise ValueError("Benchmark name must use lowercase letters, numbers, or underscores")
        return cleaned

    @field_validator("qubits_spec")
    @classmethod
    def _validate_qubits_spec(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Qubit spec cannot be empty")
        return cleaned

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not re.fullmatch(r"[a-z0-9_-]+", cleaned):
            raise ValueError("Backend must be alphanumeric")
        return cleaned


class RunRequest(BaseModel):
    benchmark_configs: List[BenchmarkConfig] = Field(
        ...,
        min_length=1,
        max_length=MAX_BENCHMARKS_PER_RUN,
    )
    save_plots: bool = True
    max_qubits: Optional[int] = Field(None, ge=1, le=128)
    qasm_max_qubits: Optional[int] = Field(None, ge=1, le=64)
    qasm_timeout_ms: Optional[int] = Field(None, ge=100, le=600_000)
    qasm_max_mem_mb: Optional[int] = Field(None, ge=16, le=1_048_576)
    qasm_include_large: bool = False
    qasm_simulate_limit: Optional[int] = Field(None, ge=1, le=10_000_000)
    benchpress: bool = False
    env_overrides: Dict[str, str] = Field(default_factory=dict)

    @field_validator("env_overrides")
    @classmethod
    def _validate_env_overrides(cls, value: Dict[str, str]) -> Dict[str, str]:
        if len(value) > MAX_ENV_OVERRIDES:
            raise ValueError(f"Too many env overrides. Max allowed is {MAX_ENV_OVERRIDES}")

        cleaned: Dict[str, str] = {}
        for key, raw_val in value.items():
            env_key = key.strip()
            if not env_key:
                raise ValueError("Environment variable names cannot be empty")
            if not re.fullmatch(r"[A-Z0-9_]{1,64}", env_key):
                raise ValueError(f"Invalid env variable name: {key}")

            env_val = str(raw_val)
            if len(env_val) > 2048:
                raise ValueError(f"Value too long for env variable: {env_key}")
            cleaned[env_key] = env_val
        return cleaned


class SettingsUpdate(BaseModel):
    max_concurrent_jobs: Optional[int] = Field(None, ge=1, le=16)


RUNS: Dict[str, Dict[str, Any]] = {}
RUNS_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _persist_run(run_id: str) -> None:
    """Save run metadata as JSON next to the log file."""
    with RUNS_LOCK:
        data = RUNS.get(run_id)
    if not data:
        return
    meta_path = RUNS_DIR / f"{run_id}.json"

    def _json_default(value: Any):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    try:
        meta_path.write_text(
            _json.dumps(data, indent=2, default=_json_default),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Failed to persist run metadata for %s", run_id, exc_info=True)


def _load_persisted_runs() -> None:
    """Restore run history from on-disk JSON files."""
    for meta_path in sorted(RUNS_DIR.glob("run_*.json")):
        try:
            data = _json.loads(meta_path.read_text(encoding="utf-8"))
            run_id = data.get("id")
            if run_id and run_id not in RUNS:
                # Ensure terminal state for old runs
                if data.get("status") in ("queued", "running"):
                    data["status"] = "failed"
                    data["ended_at"] = data.get("ended_at") or data.get("started_at")
                    data["exit_code"] = data.get("exit_code") or 1
                RUNS[run_id] = data
        except Exception:
            logger.warning("Failed to load persisted run file %s", meta_path, exc_info=True)


def _ensure_paths() -> None:
    if not MLX_ROOT.exists():
        raise RuntimeError(f"Missing MLX root at {MLX_ROOT}")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Paths verified: MLX_ROOT=%s, BENCH_DIR=%s", MLX_ROOT, BENCH_DIR)


def _parse_qubits_spec(spec: str) -> List[int]:
    cleaned = spec.replace(" ", "")
    if not cleaned:
        raise ValueError("empty qubit spec")
    parts = cleaned.split(",")
    qubits: List[int] = []
    for part in parts:
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0:
                raise ValueError("qubits must be positive integers")
            if start > end:
                raise ValueError("range start must be <= end")
            qubits.extend(range(start, end + 1))
        else:
            value = int(part)
            if value <= 0:
                raise ValueError("qubits must be positive integers")
            qubits.append(value)
    seen = set()
    ordered: List[int] = []
    for q in qubits:
        if q not in seen:
            seen.add(q)
            ordered.append(q)
    return ordered


def _default_qubits_for(name: str, max_qubits: int) -> List[int]:
    env_key = f"MLXQ_LIST_{name.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return [q for q in _parse_qubits_spec(env_val) if q <= max_qubits]
    list25 = [1, 2, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    list30 = list25 + [26, 27, 28]
    vqe_list = [1, 2, 5, 7, 10, 11, 12, 13, 14, 15]
    steady_list = [1, 2, 5, 7, 10, 11, 12]
    list30_keys = {
        "hamiltonian_simulation",
        "time_evolution",
        "trotter",
        "heisenberg",
        "heisenberg_xxz",
        "heisenberg_random_field",
        "tfim",
        "tfim_trotter2",
        "tfim_random_field",
        "long_range_ising",
        "ladder_heisenberg",
    }
    if name == "vqe":
        base = vqe_list
    elif name == "steady_state":
        base = steady_list
    elif name in list30_keys:
        base = list30
    else:
        base = list25
    return [q for q in base if q <= max_qubits]


@contextlib.contextmanager
def _temp_environ(overrides: Dict[str, Optional[str]]):
    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


_JOB_LOCK = threading.Lock()


@contextlib.contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_script(path: Path, log_file, argv: Optional[List[str]] = None) -> None:
    if not path.exists():
        log_file.write(f"Script not found: {path}\n")
        return
    original_argv = sys.argv[:]
    if argv is not None:
        sys.argv = argv
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit:
        pass
    except Exception:
        log_file.write(f"Script failed: {path}\n")
        log_file.write(traceback.format_exc())
    finally:
        sys.argv = original_argv


def _detect_hw_label() -> tuple[str, str]:
    gen = "Unknown"
    var = "Base"
    label = "Unknown Base"
    if platform.system() == "Darwin":
        brand = ""
        try:
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            brand = out.decode("utf-8", "ignore").strip()
        except Exception:
            brand = ""
        if "M1" in brand:
            gen = "M1"
        elif "M2" in brand:
            gen = "M2"
        elif "M3" in brand:
            gen = "M3"
        elif "M4" in brand:
            gen = "M4"
        if "Max" in brand:
            var = "Max"
        elif "Pro" in brand:
            var = "Pro"
        elif "Ultra" in brand:
            var = "Ultra"
        label = f"{gen} {var}"
    return f"{gen}_{var}", label


def _generate_ghz_distributions(log_file) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from mlxq.mlxQdevice import Device
    except Exception:
        log_file.write("GHZ plot generation unavailable (missing deps).\n")
        log_file.write(traceback.format_exc())
        return

    def ghz_ops(n: int):
        ops = [{"name": "H", "wires": [0]}]
        for i in range(1, n):
            ops.append({"name": "CNOT", "wires": [i - 1, i]})
        return ops

    outdir = BENCH_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    hw_prefix, hw_label = _detect_hw_label()
    for n in (4, 5, 6):
        dev = Device(n, shots=10000)
        dev.execute(ghz_ops(n))
        counts = dev.counts(shots=10000)
        total = max(1, sum(counts.values()))
        csv_path = outdir / f"ghz{n}_distribution_{hw_prefix}.csv"
        with csv_path.open("w") as f:
            f.write("bitstring,count,probability\n")
            for k in sorted(counts.keys()):
                v = counts[k]
                f.write(f"{k},{v},{v/total:.6f}\n")
        keys = sorted(counts.keys())
        probs = [counts[k] / total for k in keys]
        plt.figure(figsize=(12, 6))
        plt.bar(keys, probs, color="#00AA00")
        plt.axhline(0.5, color="red", linewidth=2, linestyle="--")
        plt.ylim(0, 0.6)
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        plt.title(f"GHZ-{n} Measurement Distribution (10000 shots on {hw_label})")
        plt.xlabel("Measurement Outcome")
        plt.ylabel("Probability")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        png_path = outdir / f"ghz{n}_distribution_{hw_prefix}.png"
        plt.savefig(png_path, dpi=100)
        plt.close()
    log_file.write("GHZ distributions generated.\n")


def _collect_outputs(
    existing: set[str],
    started_at: float,
    benchmark_names: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, str]]]:
    images: List[Dict[str, str]] = []
    data: List[Dict[str, str]] = []
    scan_roots: List[Path] = []
    for root in [BENCH_DIR, MLX_ROOT / "bench"]:
        if root.exists() and root not in scan_roots:
            scan_roots.append(root)

    seen_urls: set[str] = set()
    for root in scan_roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(root).as_posix()
            # Avoid emitting unstable top-level aliases when the canonical
            # artifact exists in runs/... (prevents 404 image links in UI).
            if "/" not in rel_path:
                runs_dir = root / "runs"
                if runs_dir.exists():
                    try:
                        if any(True for _ in runs_dir.glob(f"**/{path.name}")):
                            continue
                    except Exception:
                        pass
            if rel_path in existing and path.stat().st_mtime < started_at:
                continue
            # Filter to files matching the benchmarks that were actually run
            if benchmark_names:
                stem = path.stem.lower()
                if not any(bname in stem for bname in benchmark_names):
                    continue
            url = f"/bench/{rel_path}"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                images.append({"name": path.name, "url": url})
            elif path.suffix.lower() in {".csv", ".json", ".md", ".txt"}:
                data.append({"name": path.name, "url": url})
    return {"images": images, "data": data}


def _run_job(run_id: str, payload: RunRequest) -> None:
    log_path = RUNS_DIR / f"{run_id}.log"
    existing = {p.relative_to(BENCH_DIR).as_posix() for p in BENCH_DIR.rglob("*") if p.is_file()}
    start_ts = time.time()
    started_at = _now_iso()
    with RUNS_LOCK:
        RUNS[run_id]["status"] = "running"
        RUNS[run_id]["started_at"] = started_at

    base_env = os.environ.copy()
    base_env["MLXQ_SAVE_PLOTS"] = "1" if payload.save_plots else "0"
    default_max_qubits = payload.max_qubits or 25
    run_env_overrides = {k: v for k, v in payload.env_overrides.items() if str(v).strip()}

    exit_code = 0
    os.environ.setdefault("MPLBACKEND", "Agg")
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            from mlxq import bench as mlxq_bench
        except Exception:
            log_file.write("Failed to import mlxq benchmarks.\n")
            log_file.write(traceback.format_exc())
            exit_code = 1
        else:
            for cfg in payload.benchmark_configs:
                try:
                    if cfg.qubits_spec.lower() in {"default", "auto"}:
                        qubits = _default_qubits_for(cfg.name, default_max_qubits)
                    else:
                        qubits = _parse_qubits_spec(cfg.qubits_spec)
                except Exception as exc:
                    log_file.write(f"Invalid qubit spec '{cfg.qubits_spec}': {exc}\n")
                    exit_code = 1
                    break

                env_overrides: Dict[str, Optional[str]] = {
                    **run_env_overrides,
                    "MLXQ_BACKEND": cfg.backend or None,
                    "MLXQ_SAVE_PLOTS": base_env.get("MLXQ_SAVE_PLOTS"),
                }

                log_file.write(f"\n=== Running {cfg.name} (qubits={cfg.qubits_spec}, backend={cfg.backend}) ===\n")
                log_file.flush()

                try:
                    with _temp_environ(env_overrides), contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                        if cfg.name == "qasm":
                            qasm_max = payload.qasm_max_qubits or cfg.simulate_cap or (max(qubits) if qubits else None) or 18
                            qasm_env = {
                                "QASM_MAX_QUBITS": str(qasm_max),
                                "QASM_TIMEOUT_MS": str(payload.qasm_timeout_ms) if payload.qasm_timeout_ms is not None else None,
                                "QASM_MAX_MEM_MB": str(payload.qasm_max_mem_mb) if payload.qasm_max_mem_mb is not None else None,
                                "QASM_INCLUDE_LARGE": "1" if payload.qasm_include_large else None,
                                "QASM_SIMULATE_LIMIT": str(payload.qasm_simulate_limit) if payload.qasm_simulate_limit is not None else None,
                            }
                            with _temp_environ(qasm_env):
                                mlxq_bench.run_qasm_suite(
                                    csv_out=str(BENCH_DIR / "qasm_MLX_python.csv"),
                                    json_out=str(BENCH_DIR / "qasm_MLX_python.json"),
                                )
                        else:
                            mlxq_bench.run_scaling_benchmark(
                                cfg.name,
                                qubits,
                                simulate_cap=cfg.simulate_cap,
                                out_prefix=str(BENCH_DIR),
                            )
                except Exception:
                    log_file.write("\nBenchmark execution failed:\n")
                    log_file.write(traceback.format_exc())
                    exit_code = 1
                    break

            if exit_code == 0 and payload.save_plots:
                log_file.write("\n=== Post-processing (aggregate plots & reports) ===\n")
                log_file.flush()
                # Serialize post-processing: _pushd/_run_script mutate
                # global cwd and sys.argv which are not thread-safe.
                with _JOB_LOCK, _pushd(MLX_ROOT):
                    _run_script(MLX_ROOT / "src" / "benchmark" / "aggregate_plots.py", log_file)
                    _run_script(
                        MLX_ROOT / "tools" / "mps_report.py",
                        log_file,
                        argv=["mps_report.py", "--bench", str(BENCH_DIR)],
                    )
                    if payload.benchpress:
                        _run_script(MLX_ROOT / "src" / "benchmark" / "benchpress_replica.py", log_file)
                    _generate_ghz_distributions(log_file)

    bench_names = [cfg.name for cfg in payload.benchmark_configs]
    outputs = _collect_outputs(existing, start_ts, bench_names)
    ended_at = _now_iso()
    with RUNS_LOCK:
        RUNS[run_id]["status"] = "failed" if exit_code != 0 else "completed"
        RUNS[run_id]["ended_at"] = ended_at
        RUNS[run_id]["exit_code"] = exit_code
        RUNS[run_id]["outputs"] = outputs
    _persist_run(run_id)


app = FastAPI(title="QuantumStudio API", version="0.1.0")

# CORS: Allow localhost origins for the bundled Flutter app
# Note: Using allow_origin_regex for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    token = REQUEST_ID_CTX.set(request_id)
    started_at = time.perf_counter()
    response = None
    status_code = 500

    try:
        if _is_auth_required(request) or _is_static_auth_required(request):
            auth_token = _extract_auth_token(request)
            if auth_token != API_AUTH_TOKEN:
                _increment_metric("auth_rejected")
                status_code = 401
                response = JSONResponse(
                    status_code=401,
                    content={
                        "detail": "Unauthorized",
                        "request_id": request_id,
                    },
                    headers={
                        "www-authenticate": "Bearer",
                        "x-request-id": request_id,
                    },
                )
                if request.url.path.startswith("/api/"):
                    _apply_api_response_headers(response)
                return response

        if RATE_LIMIT_ENABLED and request.url.path.startswith("/api/"):
            max_requests, window_seconds = _rate_limit_policy_for(request)
            limit_key = (
                f"{request.method.upper()}:{request.url.path}:{_client_identifier(request)}"
            )
            allowed, retry_after = RATE_LIMITER.allow(
                limit_key,
                max_requests=max_requests,
                window_seconds=window_seconds,
            )
            if not allowed:
                _increment_metric("rate_limited")
                status_code = 429
                response = JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "request_id": request_id,
                        "retry_after_seconds": retry_after,
                    },
                    headers={
                        "retry-after": str(retry_after),
                        "x-request-id": request_id,
                    },
                )
                _apply_api_response_headers(response)
                return response

        response = await call_next(request)
        status_code = response.status_code
        response.headers["x-request-id"] = request_id
        if request.url.path.startswith("/api/"):
            _apply_api_response_headers(response)
        return response
    finally:
        _observe_status(status_code)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        logger.info(
            "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code if response is None else response.status_code,
            elapsed_ms,
        )
        REQUEST_ID_CTX.reset(token)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", _current_request_id())
    detail_text = _http_detail_text(exc.detail)
    logger.warning(
        "request_id=%s http_error status=%d path=%s detail=%s",
        request_id,
        exc.status_code,
        request.url.path,
        detail_text,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": detail_text,
            "request_id": request_id,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", _current_request_id())
    details = jsonable_encoder(exc.errors())
    logger.warning(
        "request_id=%s validation_error path=%s errors=%s",
        request_id,
        request.url.path,
        _truncate_for_log(str(details)),
    )
    return JSONResponse(
        status_code=422,
        content={
            "detail": details,
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", _current_request_id())
    logger.exception(
        "request_id=%s unhandled_exception path=%s",
        request_id,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id,
        },
    )


_ensure_paths()
_load_persisted_runs()
def _resolve_bench_asset(asset_path: str) -> Optional[Path]:
    rel = asset_path.strip().lstrip("/")
    if not rel:
        return None
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        return None

    roots = [BENCH_DIR, MLX_ROOT / "bench"]
    for root in roots:
        try:
            candidate = (root / rel_path).resolve()
            if candidate.exists() and candidate.is_file():
                try:
                    candidate.relative_to(root.resolve())
                except ValueError:
                    continue
                return candidate
        except Exception:
            continue

    # Legacy compatibility: UI may request flattened names, so search by basename.
    if "/" not in rel and "\\" not in rel:
        best: Optional[Path] = None
        for root in roots:
            if not root.exists():
                continue
            try:
                direct = root / rel
                if direct.exists() and direct.is_file():
                    if best is None or direct.stat().st_mtime > best.stat().st_mtime:
                        best = direct
                runs_dir = root / "runs"
                if runs_dir.exists() and runs_dir.is_dir():
                    iterator = runs_dir.glob(f"**/{rel}")
                else:
                    continue
            except Exception:
                continue
            for hit in iterator:
                try:
                    if not hit.is_file():
                        continue
                    if best is None or hit.stat().st_mtime > best.stat().st_mtime:
                        best = hit
                except Exception:
                    continue
        return best

    return None


@app.get("/bench/{asset_path:path}")
async def serve_bench_asset(asset_path: str):
    try:
        resolved = _resolve_bench_asset(asset_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="Bench asset not found")
        return FileResponse(str(resolved))
    except HTTPException:
        raise
    except Exception:
        logger.exception("bench_asset_serve_failed path=%s", asset_path)
        raise HTTPException(status_code=404, detail="Bench asset not found")
logger.info("QuantumStudio backend initialized (loaded %d persisted runs)", len(RUNS))
logger.info(
    "Runtime controls: auth_enabled=%s rate_limit_enabled=%s default_limit=%d/%ds heavy_limit=%d/%ds run_limit=%d/%ds",
    bool(API_AUTH_TOKEN),
    RATE_LIMIT_ENABLED,
    DEFAULT_RATE_LIMIT_MAX_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SEC,
    HEAVY_RATE_LIMIT_MAX_REQUESTS,
    HEAVY_RATE_LIMIT_WINDOW_SEC,
    RUN_CREATE_RATE_LIMIT_MAX_REQUESTS,
    RUN_CREATE_RATE_LIMIT_WINDOW_SEC,
)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    stopped = JOB_QUEUE.stop_all()
    logger.info("Shutdown complete: stop_all_jobs=%d", stopped)


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    metrics = _metrics_snapshot()
    return {
        "status": "ok",
        "service": "quantumstudio",
        "uptime_seconds": metrics["uptime_seconds"],
        "auth_enabled": metrics["auth_enabled"],
        "rate_limit_enabled": metrics["rate_limit_enabled"],
    }


@app.get("/api/system/info")
async def system_info() -> Dict[str, Any]:
    import platform
    import sys

    info: Dict[str, Any] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.mac_ver()[0] if platform.system() == "Darwin" else platform.version(),
        "arch": platform.machine(),
        "mlx_root": str(MLX_ROOT),
        "studio_root": str(STUDIO_ROOT),
    }

    if platform.system() == "Darwin":
        def _sysctl(key: str) -> str:
            try:
                return subprocess.check_output(
                    ["sysctl", "-n", key], stderr=subprocess.DEVNULL
                ).decode("utf-8", "ignore").strip()
            except Exception:
                return ""

        chip_brand = _sysctl("machdep.cpu.brand_string")
        info["chip"] = chip_brand or "Unknown"

        hw_prefix, hw_label = _detect_hw_label()
        info["chip_label"] = hw_label
        info["hw_prefix"] = hw_prefix

        # CPU
        perf_cores = _sysctl("hw.perflevel0.logicalcpu")
        eff_cores = _sysctl("hw.perflevel1.logicalcpu")
        total_cores = _sysctl("hw.logicalcpu_max")
        info["cpu_cores"] = total_cores
        info["cpu_perf_cores"] = perf_cores
        info["cpu_eff_cores"] = eff_cores

        # Memory
        mem_bytes = _sysctl("hw.memsize")
        if mem_bytes.isdigit():
            mem_gb = int(mem_bytes) / (1024 ** 3)
            info["memory_gb"] = f"{mem_gb:.0f}"
        else:
            info["memory_gb"] = "—"

        # GPU
        gpu_cores = _sysctl("machdep.gpu.core_count") or _sysctl("gpu.core_count")
        if not gpu_cores:
            # Lookup known Apple Silicon GPU core counts
            _gpu_map = {
                "M1 Base": "8", "M1 Pro": "16", "M1 Max": "32", "M1 Ultra": "64",
                "M2 Base": "10", "M2 Pro": "19", "M2 Max": "38", "M2 Ultra": "76",
                "M3 Base": "10", "M3 Pro": "18", "M3 Max": "40", "M3 Ultra": "80",
                "M4 Base": "10", "M4 Pro": "20", "M4 Max": "40", "M4 Ultra": "80",
            }
            gpu_cores = _gpu_map.get(hw_label, "")
        info["gpu_cores"] = gpu_cores or "—"

        # Neural Engine
        info["neural_engine_cores"] = "16"  # All M-series have 16-core NE

        # MLX
        try:
            import mlx.core as mx
            info["mlx_version"] = mx.__version__
            info["mlx_backend"] = str(mx.default_device())
        except Exception:
            info["mlx_version"] = "—"
            info["mlx_backend"] = "—"

    return info


@app.get("/api/system/stats")
async def system_stats() -> Dict[str, Any]:
    """Get live system stats (CPU, RAM usage)."""
    import psutil

    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)

    # Memory usage
    mem = psutil.virtual_memory()
    ram_used_gb = mem.used / (1024 ** 3)
    ram_total_gb = mem.total / (1024 ** 3)
    ram_percent = mem.percent

    stats: Dict[str, Any] = {
        "cpu_percent": round(cpu_percent, 1),
        "ram_used_gb": round(ram_used_gb, 2),
        "ram_total_gb": round(ram_total_gb, 2),
        "ram_percent": round(ram_percent, 1),
    }

    # Queue stats
    with RUNS_LOCK:
        running_count = sum(1 for r in RUNS.values() if r.get("status") == "running")
        queued_count = sum(1 for r in RUNS.values() if r.get("status") == "queued")

    stats["jobs_running"] = running_count
    stats["jobs_queued"] = queued_count
    stats["max_concurrent"] = SETTINGS.get("max_concurrent_jobs", 2)

    return stats


@app.get("/api/logs/system", response_class=PlainTextResponse)
async def system_logs() -> str:
    backend_log = LOG_DIR / "backend.log"
    if not backend_log.exists():
        return "No backend log file available.\n"
    return _read_log_text(backend_log)


@app.get("/api/metrics")
async def metrics() -> Dict[str, Any]:
    return _metrics_snapshot()


@app.get("/api/benchmarks")
async def list_benchmarks() -> Dict[str, Any]:
    return {"benchmarks": BENCHMARKS}


@app.post("/api/runs")
async def start_run(payload: RunRequest) -> Dict[str, Any]:
    available = {b["name"] for b in BENCHMARKS}
    invalid = [cfg.name for cfg in payload.benchmark_configs if cfg.name not in available]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown benchmarks: {', '.join(invalid)}")

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    existing_files = {p.relative_to(BENCH_DIR).as_posix() for p in BENCH_DIR.rglob("*") if p.is_file()}

    with RUNS_LOCK:
        RUNS[run_id] = {
            "id": run_id,
            "benchmarks": [cfg.name for cfg in payload.benchmark_configs],
            "benchmark_configs": [cfg.model_dump() for cfg in payload.benchmark_configs],
            "status": "queued",
            "queue_position": None,  # Will be set by queue manager
            "can_stop": False,
            "can_cancel": True,
            "started_at": None,
            "ended_at": None,
            "exit_code": None,
            "outputs": {"images": [], "data": []},
            # Internal fields for job execution
            "_payload": payload.model_dump(),
            "_existing_files": existing_files,
            "_start_ts": time.time(),
        }

    # Submit to job queue
    JOB_QUEUE.submit(run_id, payload)

    with RUNS_LOCK:
        snapshot = _sanitize_run(RUNS[run_id])

    return snapshot


def _sanitize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal fields before returning to API."""
    clean = {k: v for k, v in run.items() if not k.startswith("_")}
    clean["outputs"] = _normalize_run_outputs(clean.get("outputs"))
    return clean


def _normalize_run_outputs(outputs: Any) -> Dict[str, List[Dict[str, str]]]:
    if not isinstance(outputs, dict):
        return {"images": [], "data": []}

    def _normalize_items(items: Any) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if not isinstance(items, list):
            return normalized
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            raw_url = str(item.get("url", "")).strip()
            if not name:
                continue
            rel = raw_url
            if rel.startswith("/bench/"):
                rel = rel[len("/bench/"):]
            elif rel.startswith("/"):
                rel = rel[1:]
            rel = rel.strip("/")

            resolved = _resolve_bench_asset(rel or name)
            if resolved is None:
                resolved = _resolve_bench_asset(name)
            if resolved is None:
                continue

            # Always emit URL relative to the canonical bench root for stable serving.
            try:
                canonical_rel = resolved.relative_to(BENCH_DIR).as_posix()
            except Exception:
                try:
                    canonical_rel = resolved.relative_to(MLX_ROOT / "bench").as_posix()
                except Exception:
                    parts = resolved.as_posix().split("/")
                    if "runs" in parts:
                        idx = parts.index("runs")
                        canonical_rel = "/".join(parts[idx:])
                    else:
                        canonical_rel = resolved.name
            url = f"/bench/{canonical_rel}"
            key = f"{name}|{url}"
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"name": name, "url": url})
        return normalized

    return {
        "images": _normalize_items(outputs.get("images")),
        "data": _normalize_items(outputs.get("data")),
    }


def _update_run_flags(run: Dict[str, Any]) -> Dict[str, Any]:
    """Update can_stop and can_cancel based on status."""
    status = run.get("status", "")
    run["can_stop"] = status == "running"
    run["can_cancel"] = status == "queued"
    return run


@app.get("/api/runs")
async def list_runs() -> Dict[str, Any]:
    with RUNS_LOCK:
        runs = sorted(RUNS.values(), key=lambda r: r.get("id", ""), reverse=True)
        runs = [_sanitize_run(_update_run_flags(dict(r))) for r in runs]
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return _sanitize_run(_update_run_flags(dict(run)))


@app.get("/api/runs/{run_id}/log", response_class=PlainTextResponse)
async def get_run_log(run_id: str) -> str:
    log_path = RUNS_DIR / f"{run_id}.log"
    if not log_path.exists():
        return ""
    return _read_log_text(log_path)


# ---------------------------------------------------------------------------
# Job Control Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> Dict[str, Any]:
    """Stop a running job."""
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("status") != "running":
            raise HTTPException(status_code=400, detail="Job is not running")

    success = JOB_QUEUE.stop_job(run_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop job")

    with RUNS_LOCK:
        return _sanitize_run(_update_run_flags(dict(RUNS[run_id])))


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Dict[str, Any]:
    """Cancel a queued job."""
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("status") != "queued":
            raise HTTPException(status_code=400, detail="Job is not queued")

    success = JOB_QUEUE.cancel_job(run_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel job")

    with RUNS_LOCK:
        return _sanitize_run(_update_run_flags(dict(RUNS[run_id])))


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str) -> Dict[str, str]:
    """Delete a completed/failed/stopped/cancelled run from history."""
    with RUNS_LOCK:
        run = RUNS.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.get("status") in ("queued", "running"):
            raise HTTPException(status_code=400, detail="Cannot delete active job")
        del RUNS[run_id]

    # Delete associated files
    log_path = RUNS_DIR / f"{run_id}.log"
    meta_path = RUNS_DIR / f"{run_id}.json"
    _safe_remove_file(log_path)
    _safe_remove_file(meta_path)

    return {"status": "deleted", "id": run_id}


# ---------------------------------------------------------------------------
# Queue Status Endpoint
# ---------------------------------------------------------------------------

@app.get("/api/queue")
async def get_queue() -> Dict[str, Any]:
    """Get current queue status."""
    return JOB_QUEUE.get_queue_status()


@app.post("/api/queue/stop-all")
async def stop_all_jobs() -> Dict[str, Any]:
    """Stop all running jobs and clear queue."""
    stopped = JOB_QUEUE.stop_all()
    return {"stopped_count": stopped, "status": "ok"}


# ---------------------------------------------------------------------------
# Settings Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/settings")
async def get_settings() -> Dict[str, Any]:
    """Get current settings."""
    return {
        **SETTINGS,
        "default_max_concurrent": _get_default_max_concurrent(),
    }


@app.put("/api/settings")
async def update_settings(update: SettingsUpdate) -> Dict[str, Any]:
    """Update settings."""
    global SETTINGS
    if update.max_concurrent_jobs is not None:
        SETTINGS["max_concurrent_jobs"] = update.max_concurrent_jobs
    _save_settings(SETTINGS)
    return {
        **SETTINGS,
        "default_max_concurrent": _get_default_max_concurrent(),
    }


# ---------------------------------------------------------------------------
# QASM Visualization Endpoints
# ---------------------------------------------------------------------------

QASM_DIR = MLX_ROOT / "datasets" / "qasm" / "local"


class QasmParseRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=MAX_QASM_CONTENT_BYTES)
    filename: Optional[str] = None


class QasmVisualizeRequest(BaseModel):
    content: Optional[str] = Field(None, max_length=MAX_QASM_CONTENT_BYTES)
    filename: Optional[str] = None
    theme: str = "apple"


def _validate_qasm_content_size(content: str) -> None:
    content_bytes = len(content.encode("utf-8", errors="ignore"))
    if content_bytes > MAX_QASM_CONTENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "QASM content too large "
                f"({content_bytes} bytes, max {MAX_QASM_CONTENT_BYTES} bytes)"
            ),
        )


def _resolve_qasm_path(filename: str) -> Path:
    """Resolve a QASM filename safely inside QASM_DIR."""
    candidate = Path(filename)
    if candidate.name != filename or candidate.suffix.lower() != ".qasm":
        raise HTTPException(status_code=400, detail="Invalid QASM filename")

    qasm_root = QASM_DIR.resolve()
    qasm_path = (QASM_DIR / candidate.name).resolve()

    try:
        qasm_path.relative_to(qasm_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid QASM filename")

    return qasm_path


@app.get("/api/qasm/files")
async def list_qasm_files() -> Dict[str, Any]:
    """List available QASM files from the datasets directory."""
    files: List[Dict[str, Any]] = []
    if QASM_DIR.exists():
        for qasm_file in sorted(QASM_DIR.glob("*.qasm")):
            try:
                stat = qasm_file.stat()
                files.append({
                    "name": qasm_file.name,
                    "path": str(qasm_file),
                    "size": stat.st_size,
                })
            except Exception:
                pass
    return {"files": files, "directory": str(QASM_DIR)}


@app.get("/api/qasm/files/{filename}")
async def get_qasm_file(filename: str) -> Dict[str, Any]:
    """Get the content of a specific QASM file."""
    qasm_path = _resolve_qasm_path(filename)
    if not qasm_path.exists():
        raise HTTPException(status_code=404, detail="QASM file not found")
    try:
        content = qasm_path.read_text(encoding="utf-8")
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")


@app.post("/api/qasm/parse")
async def parse_qasm(request: QasmParseRequest) -> Dict[str, Any]:
    """Parse QASM content and return the circuit structure."""
    temp_path: Optional[str] = None
    try:
        from mlxq.qasm import parse_qasm_file
        import tempfile

        _validate_qasm_content_size(request.content)
        # Write content to temp file for parsing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False, encoding='utf-8') as f:
            f.write(request.content)
            temp_path = f.name

        n_qubits, ops = parse_qasm_file(temp_path)

        return {
            "n_qubits": n_qubits,
            "n_ops": len(ops),
            "ops": ops,
            "filename": request.filename,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {e}")
    finally:
        if temp_path:
            _safe_remove_file(Path(temp_path))


@app.post("/api/qasm/visualize/ascii")
async def visualize_qasm_ascii(request: QasmVisualizeRequest) -> Dict[str, Any]:
    """Generate ASCII circuit visualization from QASM."""
    temp_path: Optional[str] = None
    try:
        from mlxq.qasm import parse_qasm_file
        from mlxq.draw import circuit_ascii
        import tempfile

        # Get content
        if request.content:
            content = request.content
        elif request.filename:
            qasm_path = _resolve_qasm_path(request.filename)
            if not qasm_path.exists():
                raise HTTPException(status_code=404, detail="QASM file not found")
            content = qasm_path.read_text(encoding="utf-8")
        else:
            raise HTTPException(status_code=400, detail="Provide content or filename")
        _validate_qasm_content_size(content)

        # Write to temp file for parsing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        n_qubits, ops = parse_qasm_file(temp_path)

        ascii_diagram = circuit_ascii(n_qubits, ops)
        return {"ascii": ascii_diagram, "n_qubits": n_qubits, "n_ops": len(ops)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visualization error: {e}")
    finally:
        if temp_path:
            _safe_remove_file(Path(temp_path))


@app.post("/api/qasm/visualize/image")
async def visualize_qasm_image(request: QasmVisualizeRequest) -> Dict[str, Any]:
    """Generate PNG circuit visualization from QASM, return as base64."""
    temp_path: Optional[str] = None
    try:
        from mlxq.qasm import parse_qasm_file
        from mlxq.draw import circuit_mpl
        import tempfile
        import base64
        import io

        os.environ.setdefault("MPLBACKEND", "Agg")

        # Get content
        if request.content:
            content = request.content
            title = request.filename or "Circuit"
        elif request.filename:
            qasm_path = _resolve_qasm_path(request.filename)
            if not qasm_path.exists():
                raise HTTPException(status_code=404, detail="QASM file not found")
            content = qasm_path.read_text(encoding="utf-8")
            title = request.filename
        else:
            raise HTTPException(status_code=400, detail="Provide content or filename")
        _validate_qasm_content_size(content)

        # Write to temp file for parsing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.qasm', delete=False, encoding='utf-8') as f:
            f.write(content)
            temp_path = f.name

        n_qubits, ops = parse_qasm_file(temp_path)

        # Generate matplotlib figure
        result = circuit_mpl(n_qubits, ops, title=title, theme=request.theme, badge=True)
        if result is None:
            raise HTTPException(status_code=500, detail="Matplotlib not available")

        fig, ax = result

        # Convert to base64 PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        import matplotlib.pyplot as plt
        plt.close(fig)

        b64_image = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "image_base64": b64_image,
            "n_qubits": n_qubits,
            "n_ops": len(ops),
            "filename": request.filename,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visualization error: {e}")
    finally:
        if temp_path:
            _safe_remove_file(Path(temp_path))
