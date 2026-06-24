import importlib.util
import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_MAIN = ROOT_DIR / "backend" / "main.py"

spec = importlib.util.spec_from_file_location("quantumstudio_backend_main", BACKEND_MAIN)
assert spec is not None and spec.loader is not None
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class BackendApiHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(backend.app)

    def setUp(self):
        backend.RATE_LIMITER.clear()
        backend.API_AUTH_TOKEN = ""

    def test_health_contains_runtime_controls(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertIn("uptime_seconds", body)
        self.assertIn("auth_enabled", body)
        self.assertIn("rate_limit_enabled", body)

    def test_metrics_endpoint_tracks_requests(self):
        before = self.client.get("/api/metrics").json()["total_requests"]
        self.client.get("/api/benchmarks")
        after = self.client.get("/api/metrics").json()["total_requests"]
        self.assertGreaterEqual(after, before + 1)

    def test_api_security_headers_present(self):
        response = self.client.get("/api/benchmarks")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
        self.assertEqual(response.headers.get("x-frame-options"), "DENY")
        self.assertEqual(response.headers.get("referrer-policy"), "no-referrer")

    def test_auth_token_enforced_when_enabled(self):
        backend.API_AUTH_TOKEN = "secret-token"
        try:
            unauth = self.client.get("/api/benchmarks")
            self.assertEqual(unauth.status_code, 401)
            self.assertIn("request_id", unauth.json())

            auth = self.client.get(
                "/api/benchmarks",
                headers={"Authorization": "Bearer secret-token"},
            )
            self.assertEqual(auth.status_code, 200)
        finally:
            backend.API_AUTH_TOKEN = ""

    def test_rate_limit_returns_429(self):
        old_enabled = backend.RATE_LIMIT_ENABLED
        old_max = backend.DEFAULT_RATE_LIMIT_MAX_REQUESTS
        old_window = backend.DEFAULT_RATE_LIMIT_WINDOW_SEC
        try:
            backend.RATE_LIMIT_ENABLED = True
            backend.DEFAULT_RATE_LIMIT_MAX_REQUESTS = 1
            backend.DEFAULT_RATE_LIMIT_WINDOW_SEC = 120

            first = self.client.get("/api/benchmarks")
            second = self.client.get("/api/benchmarks")
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 429)
            self.assertIsNotNone(second.headers.get("retry-after"))
            self.assertIn("request_id", second.json())
        finally:
            backend.DEFAULT_RATE_LIMIT_MAX_REQUESTS = old_max
            backend.DEFAULT_RATE_LIMIT_WINDOW_SEC = old_window
            backend.RATE_LIMIT_ENABLED = old_enabled
            backend.RATE_LIMITER.clear()

    def test_http_error_includes_request_id(self):
        response = self.client.get("/api/runs/does-not-exist")
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertIn("request_id", body)
        self.assertEqual(body["detail"], "Run not found")
        self.assertEqual(response.headers.get("x-request-id"), body["request_id"])

    def test_invalid_env_override_is_rejected(self):
        response = self.client.post(
            "/api/runs",
            json={
                "benchmark_configs": [{"name": "qasm", "qubits_spec": "default", "backend": "sv"}],
                "env_overrides": {"bad-key": "1"},
            },
        )
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertIn("request_id", body)
        self.assertEqual(response.headers.get("x-request-id"), body["request_id"])

    def test_large_logs_are_truncated(self):
        run_id = "run_large_log_fixture"
        log_path = backend.RUNS_DIR / f"{run_id}.log"
        try:
            chunk = ("x" * 1024).encode("utf-8")
            with log_path.open("wb") as fh:
                for _ in range((backend.MAX_LOG_READ_BYTES // 1024) + 64):
                    fh.write(chunk)
                fh.write(b"\nTAIL_MARKER\n")

            response = self.client.get(f"/api/runs/{run_id}/log")
            self.assertEqual(response.status_code, 200)
            text = response.text
            self.assertIn("log truncated to last", text)
            self.assertIn("TAIL_MARKER", text)
            self.assertLessEqual(len(text.encode("utf-8")), backend.MAX_LOG_READ_BYTES + 128)
        finally:
            if log_path.exists():
                os.unlink(log_path)


class RunRequestValidationTests(unittest.TestCase):
    def test_env_override_limit(self):
        overrides = {f"KEY_{i}": "value" for i in range(backend.MAX_ENV_OVERRIDES + 1)}
        with self.assertRaises(ValidationError):
            backend.RunRequest(
                benchmark_configs=[backend.BenchmarkConfig(name="qasm", backend="sv")],
                env_overrides=overrides,
            )

    def test_env_override_names_are_normalized(self):
        request = backend.RunRequest(
            benchmark_configs=[backend.BenchmarkConfig(name="qasm", backend="sv")],
            env_overrides={"  TEST_FLAG  ": "1"},
        )
        self.assertIn("TEST_FLAG", request.env_overrides)
        self.assertEqual(request.env_overrides["TEST_FLAG"], "1")


if __name__ == "__main__":
    unittest.main()
