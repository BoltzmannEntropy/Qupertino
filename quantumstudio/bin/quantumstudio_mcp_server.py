#!/usr/bin/env python3
"""
QuantumStudio MCP Server

Model Context Protocol (MCP) server for Claude Code integration.
Exposes QuantumStudio quantum benchmarking capabilities via MCP tools.

Usage:
    python3 quantumstudio_mcp_server.py [--port PORT] [--host HOST]

Default: http://127.0.0.1:8087

Configuration for Claude Code (~/.claude/settings.json):
{
  "mcpServers": {
    "quantumstudio": {
      "command": "python3",
      "args": ["/path/to/bin/quantumstudio_mcp_server.py"],
      "env": {
        "QUANTUMSTUDIO_BACKEND_URL": "http://localhost:8127"
      }
    }
  }
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import urlopen, Request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8087
BACKEND_URL = os.environ.get("QUANTUMSTUDIO_BACKEND_URL", "http://localhost:8127")
API_TOKEN = os.environ.get("QUANTUMSTUDIO_API_TOKEN", "").strip()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def _resolve_log_dir() -> Path:
    candidates: List[Path] = []

    explicit = os.environ.get("QUANTUMSTUDIO_MCP_LOG_DIR", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    home = os.environ.get("HOME", "").strip()
    if home:
        candidates.append(Path(home) / "Library" / "Logs" / "QuantumStudio")

    candidates.append(ROOT_DIR / "runs" / "logs")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue

    raise RuntimeError("Could not create MCP log directory")


LOG_DIR = _resolve_log_dir()

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

log_handler = RotatingFileHandler(
    LOG_DIR / "quantumstudio_mcp_server.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
    encoding="utf-8",
)
log_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logger = logging.getLogger("quantumstudio_mcp")
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logger.addHandler(console)

# ---------------------------------------------------------------------------
# MCP Tool Definitions
# ---------------------------------------------------------------------------

MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "quantum_health_check",
        "description": "Check if the QuantumStudio backend is running and healthy.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "quantum_system_info",
        "description": "Get system information including Apple Silicon chip, GPU cores, memory, and MLX version.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "quantum_list_benchmarks",
        "description": "List all available quantum benchmarks (QFT, Grover, VQE, QAOA, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "quantum_run_benchmark",
        "description": "Run a quantum benchmark simulation. Returns run ID for tracking progress.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "benchmark": {
                    "type": "string",
                    "description": "Name of the benchmark to run (e.g., 'qft', 'grover', 'vqe', 'qaoa', 'random_circuit')"
                },
                "qubits": {
                    "type": "string",
                    "description": "Qubit range specification (e.g., '4,6,8,10' or '4-12' or 'default')",
                    "default": "default"
                },
                "backend": {
                    "type": "string",
                    "enum": ["sv", "mps"],
                    "description": "Simulation backend: 'sv' (state vector) or 'mps' (matrix product state)",
                    "default": "sv"
                }
            },
            "required": ["benchmark"]
        }
    },
    {
        "name": "quantum_list_runs",
        "description": "List all benchmark runs with their status (running, completed, failed, queued).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of runs to return",
                    "default": 10
                }
            },
            "required": []
        }
    },
    {
        "name": "quantum_get_run_status",
        "description": "Get detailed status of a specific benchmark run including outputs and logs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run ID to get status for"
                }
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "quantum_get_run_log",
        "description": "Get the log output of a benchmark run for debugging or analysis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run ID to get logs for"
                },
                "tail": {
                    "type": "integer",
                    "description": "Number of lines from the end to return (default: all)",
                    "default": 100
                }
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "quantum_stop_run",
        "description": "Stop a running benchmark.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "The run ID to stop"
                }
            },
            "required": ["run_id"]
        }
    },
    {
        "name": "quantum_queue_status",
        "description": "Get the current job queue status including running and queued jobs.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "quantum_system_stats",
        "description": "Get live system statistics (CPU usage, RAM usage, active jobs).",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

# ---------------------------------------------------------------------------
# Backend Communication
# ---------------------------------------------------------------------------

def _call_backend(method: str, path: str, data: Optional[Dict] = None, timeout: int = 30) -> Dict[str, Any]:
    """Call the QuantumStudio backend API."""
    url = f"{BACKEND_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"

    body = json.dumps(data).encode("utf-8") if data else None
    req = Request(url, data=body, headers=headers, method=method)

    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        raise RuntimeError(f"Backend request failed: {e}")

# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

def handle_health_check(args: Dict[str, Any]) -> str:
    """Check backend health."""
    try:
        result = _call_backend("GET", "/api/health")
        return json.dumps({
            "status": "healthy",
            "backend": BACKEND_URL,
            "response": result
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "unhealthy",
            "backend": BACKEND_URL,
            "error": str(e)
        }, indent=2)

def handle_system_info(args: Dict[str, Any]) -> str:
    """Get system information."""
    result = _call_backend("GET", "/api/system/info")
    return json.dumps(result, indent=2)

def handle_list_benchmarks(args: Dict[str, Any]) -> str:
    """List available benchmarks."""
    result = _call_backend("GET", "/api/benchmarks")
    return json.dumps(result, indent=2)

def handle_run_benchmark(args: Dict[str, Any]) -> str:
    """Run a benchmark."""
    benchmark = args.get("benchmark", "").lower()
    qubits = args.get("qubits", "default")
    backend = args.get("backend", "sv")

    payload = {
        "benchmark_configs": [{
            "name": benchmark,
            "qubits_spec": qubits,
            "backend": backend
        }],
        "save_plots": True
    }

    result = _call_backend("POST", "/api/runs", data=payload)
    return json.dumps({
        "message": f"Benchmark '{benchmark}' started",
        "run_id": result.get("id"),
        "status": result.get("status"),
        "benchmarks": result.get("benchmarks")
    }, indent=2)

def handle_list_runs(args: Dict[str, Any]) -> str:
    """List benchmark runs."""
    limit = args.get("limit", 10)
    result = _call_backend("GET", "/api/runs")
    runs = result.get("runs", [])[:limit]

    # Summarize for readability
    summary = []
    for run in runs:
        summary.append({
            "id": run.get("id"),
            "benchmarks": run.get("benchmarks"),
            "status": run.get("status"),
            "started_at": run.get("started_at"),
            "ended_at": run.get("ended_at")
        })

    return json.dumps({"runs": summary, "total_shown": len(summary)}, indent=2)

def handle_get_run_status(args: Dict[str, Any]) -> str:
    """Get run status."""
    run_id = args.get("run_id")
    if not run_id:
        return json.dumps({"error": "run_id is required"})

    result = _call_backend("GET", f"/api/runs/{run_id}")
    return json.dumps(result, indent=2)

def handle_get_run_log(args: Dict[str, Any]) -> str:
    """Get run logs."""
    run_id = args.get("run_id")
    tail = args.get("tail", 100)

    if not run_id:
        return json.dumps({"error": "run_id is required"})

    try:
        url = f"{BACKEND_URL}/api/runs/{run_id}/log"
        headers: Dict[str, str] = {}
        if API_TOKEN:
            headers["Authorization"] = f"Bearer {API_TOKEN}"
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=10) as resp:
            log_content = resp.read().decode("utf-8")

        # Get last N lines
        lines = log_content.strip().split("\n")
        if tail and len(lines) > tail:
            lines = lines[-tail:]

        return json.dumps({
            "run_id": run_id,
            "lines_shown": len(lines),
            "log": "\n".join(lines)
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

def handle_stop_run(args: Dict[str, Any]) -> str:
    """Stop a running benchmark."""
    run_id = args.get("run_id")
    if not run_id:
        return json.dumps({"error": "run_id is required"})

    result = _call_backend("POST", f"/api/runs/{run_id}/stop")
    return json.dumps(result, indent=2)

def handle_queue_status(args: Dict[str, Any]) -> str:
    """Get queue status."""
    result = _call_backend("GET", "/api/queue")
    return json.dumps(result, indent=2)

def handle_system_stats(args: Dict[str, Any]) -> str:
    """Get system stats."""
    result = _call_backend("GET", "/api/system/stats")
    return json.dumps(result, indent=2)

# Tool dispatch table
TOOL_HANDLERS = {
    "quantum_health_check": handle_health_check,
    "quantum_system_info": handle_system_info,
    "quantum_list_benchmarks": handle_list_benchmarks,
    "quantum_run_benchmark": handle_run_benchmark,
    "quantum_list_runs": handle_list_runs,
    "quantum_get_run_status": handle_get_run_status,
    "quantum_get_run_log": handle_get_run_log,
    "quantum_stop_run": handle_stop_run,
    "quantum_queue_status": handle_queue_status,
    "quantum_system_stats": handle_system_stats,
}

# ---------------------------------------------------------------------------
# MCP Server Handler
# ---------------------------------------------------------------------------

class MCPHandler(BaseHTTPRequestHandler):
    """HTTP handler for JSON-RPC MCP protocol."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format % args)

    def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _jsonrpc_response(self, id: Any, result: Any = None, error: Any = None) -> Dict:
        resp = {"jsonrpc": "2.0", "id": id}
        if error:
            resp["error"] = error
        else:
            resp["result"] = result
        return resp

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")

        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_json(self._jsonrpc_response(None, error={
                "code": -32700,
                "message": f"Parse error: {e}"
            }), 400)
            return

        req_id = req.get("id")
        method = req.get("method", "")
        params = req.get("params", {})

        logger.info(f"MCP request: {method}")

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "quantumstudio-mcp",
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {}
                    }
                }
                self._send_json(self._jsonrpc_response(req_id, result))

            elif method in ("tools/list", "tools.list"):
                self._send_json(self._jsonrpc_response(req_id, {"tools": MCP_TOOLS}))

            elif method in ("tools/call", "tools.call"):
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})

                handler = TOOL_HANDLERS.get(tool_name)
                if not handler:
                    self._send_json(self._jsonrpc_response(req_id, error={
                        "code": -32601,
                        "message": f"Unknown tool: {tool_name}"
                    }))
                    return

                try:
                    result_text = handler(tool_args)
                    self._send_json(self._jsonrpc_response(req_id, {
                        "content": [{"type": "text", "text": result_text}]
                    }))
                except Exception as e:
                    logger.exception(f"Tool {tool_name} failed")
                    self._send_json(self._jsonrpc_response(req_id, {
                        "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                        "isError": True
                    }))

            else:
                self._send_json(self._jsonrpc_response(req_id, error={
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                }))

        except Exception as e:
            logger.exception("MCP handler error")
            self._send_json(self._jsonrpc_response(req_id, error={
                "code": -32603,
                "message": f"Internal error: {e}"
            }), 500)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="QuantumStudio MCP Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MCPHandler)
    logger.info(f"QuantumStudio MCP Server starting on http://{args.host}:{args.port}")
    logger.info(f"Backend URL: {BACKEND_URL}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down")
        server.shutdown()

if __name__ == "__main__":
    main()
