import importlib.util
import json
import threading
import unittest
from pathlib import Path
from urllib import request


ROOT_DIR = Path(__file__).resolve().parents[1]
MCP_PATH = ROOT_DIR / "bin" / "quantumstudio_mcp_server.py"


spec = importlib.util.spec_from_file_location("quantumstudio_mcp_server", MCP_PATH)
assert spec is not None and spec.loader is not None
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


class MCPToolSchemaTests(unittest.TestCase):
    def test_all_tools_have_required_fields(self):
        for tool in mcp.MCP_TOOLS:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("inputSchema", tool)
            self.assertTrue(tool["name"])
            self.assertTrue(tool["description"])
            schema = tool["inputSchema"]
            self.assertIsInstance(schema, dict)
            self.assertEqual(schema.get("type"), "object")

    def test_tool_names_are_unique(self):
        names = [tool["name"] for tool in mcp.MCP_TOOLS]
        self.assertEqual(len(names), len(set(names)))

    def test_required_tool_coverage(self):
        names = {tool["name"] for tool in mcp.MCP_TOOLS}
        self.assertIn("quantum_health_check", names)
        self.assertIn("quantum_system_info", names)
        self.assertIn("quantum_list_benchmarks", names)
        self.assertIn("quantum_run_benchmark", names)


class MCPJsonRpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = mcp.HTTPServer(("127.0.0.1", 0), mcp.MCPHandler)
        cls.host, cls.port = cls.server.server_address
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def _rpc(self, payload):
        url = f"http://{self.host}:{self.port}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8")
        return json.loads(data)

    def test_initialize(self):
        response = self._rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(response["jsonrpc"], "2.0")
        self.assertEqual(response["id"], 1)
        self.assertIn("result", response)
        self.assertIn("capabilities", response["result"])

    def test_tools_list(self):
        response = self._rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertIn("result", response)
        self.assertIn("tools", response["result"])
        self.assertGreater(len(response["result"]["tools"]), 0)

    def test_unknown_tool_returns_error(self):
        response = self._rpc(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "does_not_exist", "arguments": {}},
            }
        )
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601)

    def test_unknown_method_returns_error(self):
        response = self._rpc({"jsonrpc": "2.0", "id": 4, "method": "unknown/method"})
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
