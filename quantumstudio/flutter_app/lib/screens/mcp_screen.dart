import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as path;
import '../services/log_service.dart';

class McpScreen extends StatefulWidget {
  const McpScreen({super.key});

  @override
  State<McpScreen> createState() => _McpScreenState();
}

class _McpScreenState extends State<McpScreen> {
  final TextEditingController _hostController = TextEditingController(
    text: '127.0.0.1',
  );
  final TextEditingController _portController = TextEditingController(
    text: '8087',
  );

  bool _serverRunning = false;
  bool _startingServer = false;
  bool _checking = false;
  List<Map<String, dynamic>> _tools = [];
  int _requestCount = 0;
  DateTime? _lastCheck;
  Timer? _pollTimer;
  Process? _serverProcess;

  @override
  void initState() {
    super.initState();
    _checkServerStatus();
    // Poll every 5 seconds
    _pollTimer = Timer.periodic(const Duration(seconds: 5), (_) {
      if (mounted) _checkServerStatus();
    });
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _serverProcess?.kill(ProcessSignal.sigterm);
    _hostController.dispose();
    _portController.dispose();
    super.dispose();
  }

  String get _mcpUrl =>
      'http://${_hostController.text}:${_portController.text}';

  String? _bundledResourcesDir() {
    if (!Platform.isMacOS) return null;
    final execPath = Platform.resolvedExecutable;
    final macosDir = path.dirname(execPath);
    final contentsDir = path.dirname(macosDir);
    final resourcesDir = path.join(contentsDir, 'Resources');
    if (Directory(resourcesDir).existsSync()) {
      return resourcesDir;
    }
    return null;
  }

  String? _resolveMcpScriptPath() {
    final envPath = const String.fromEnvironment(
      'QS_MCP_SCRIPT_PATH',
      defaultValue: '',
    ).trim();

    final candidates = <String>[
      if (envPath.isNotEmpty) envPath,
      path.join(Directory.current.path, 'bin', 'quantumstudio_mcp_server.py'),
      if (_bundledResourcesDir() != null)
        path.join(
          _bundledResourcesDir()!,
          'backend',
          'quantumstudio_mcp_server.py',
        ),
    ];

    for (final candidate in candidates) {
      if (File(candidate).existsSync()) {
        return candidate;
      }
    }
    return null;
  }

  String _resolvePythonCommand() {
    final resourcesDir = _bundledResourcesDir();
    if (resourcesDir != null) {
      final bundledPython = path.join(resourcesDir, 'python', 'bin', 'python3');
      if (File(bundledPython).existsSync()) {
        return bundledPython;
      }
    }
    return 'python3';
  }

  String _defaultLogDir() {
    final home = Platform.environment['HOME'];
    if (home != null && home.isNotEmpty) {
      return path.join(home, 'Library', 'Logs', 'QuantumStudio');
    }
    return path.join(Directory.systemTemp.path, 'QuantumStudio', 'logs');
  }

  String _suggestedScriptPath() {
    return _resolveMcpScriptPath() ??
        '/absolute/path/to/bin/quantumstudio_mcp_server.py';
  }

  Future<void> _checkServerStatus() async {
    if (_checking) return;
    setState(() => _checking = true);

    try {
      final response = await http
          .post(
            Uri.parse(_mcpUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'jsonrpc': '2.0',
              'method': 'tools/list',
              'params': {},
              'id': 1,
            }),
          )
          .timeout(const Duration(seconds: 3));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final tools =
            (data['result']?['tools'] as List<dynamic>?)
                ?.map((t) => t as Map<String, dynamic>)
                .toList() ??
            [];

        if (mounted) {
          setState(() {
            _serverRunning = true;
            _tools = tools;
            _lastCheck = DateTime.now();
            _requestCount++;
          });
        }
      } else {
        if (mounted) {
          setState(() {
            _serverRunning = false;
            _tools = [];
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _serverRunning = false;
          _tools = [];
        });
      }
    } finally {
      if (mounted) setState(() => _checking = false);
    }
  }

  Future<void> _startServer() async {
    if (_startingServer || _serverRunning) return;

    final host = _hostController.text.trim();
    final port = int.tryParse(_portController.text.trim());
    if (host.isEmpty || port == null || port < 1 || port > 65535) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Enter a valid host and port first.')),
      );
      return;
    }

    final scriptPath = _resolveMcpScriptPath();
    if (scriptPath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('MCP server script not found in this environment.'),
        ),
      );
      return;
    }

    setState(() => _startingServer = true);
    try {
      final process = await Process.start(
        _resolvePythonCommand(),
        [scriptPath, '--host', host, '--port', port.toString()],
        environment: {
          ...Platform.environment,
          'QUANTUMSTUDIO_BACKEND_URL': 'http://127.0.0.1:8127',
          'QUANTUMSTUDIO_MCP_LOG_DIR': _defaultLogDir(),
        },
      );
      _serverProcess = process;

      process.stdout.transform(utf8.decoder).listen((line) {
        final text = line.trim();
        if (text.isNotEmpty) {
          logger.info('MCP', text);
        }
      });
      process.stderr.transform(utf8.decoder).listen((line) {
        final text = line.trim();
        if (text.isNotEmpty) {
          logger.warning('MCP', text);
        }
      });

      unawaited(
        process.exitCode.then((_) async {
          if (!mounted) return;
          setState(() {
            _serverProcess = null;
            _serverRunning = false;
            _tools = [];
          });
          await _checkServerStatus();
        }),
      );

      await Future.delayed(const Duration(milliseconds: 700));
      await _checkServerStatus();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to start MCP server: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _startingServer = false);
    }
  }

  Future<void> _stopServer() async {
    final process = _serverProcess;
    if (process != null) {
      process.kill(ProcessSignal.sigterm);
      await Future.delayed(const Duration(milliseconds: 500));
      _serverProcess = null;
    }

    if (!mounted) return;
    setState(() {
      _serverRunning = false;
      _tools = [];
    });
  }

  Future<Map<String, dynamic>?> _callTool(
    String name,
    Map<String, dynamic> args,
  ) async {
    try {
      final response = await http
          .post(
            Uri.parse(_mcpUrl),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'jsonrpc': '2.0',
              'method': 'tools/call',
              'params': {'name': name, 'arguments': args},
              'id': DateTime.now().millisecondsSinceEpoch,
            }),
          )
          .timeout(const Duration(seconds: 10));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final content = data['result']?['content'] as List<dynamic>?;
        if (content != null && content.isNotEmpty) {
          final text = content[0]['text'] as String?;
          if (text != null) {
            return jsonDecode(text) as Map<String, dynamic>;
          }
        }
      }
    } catch (e) {
      logger.error('MCP', 'Tool call failed: $e');
    }
    return null;
  }

  void _copyConfig() {
    final config =
        '''
{
  "mcpServers": {
    "quantumstudio": {
      "command": "python3",
      "args": ["${_suggestedScriptPath()}"],
      "env": {
        "QUANTUMSTUDIO_BACKEND_URL": "http://localhost:8127"
      }
    }
  }
}''';
    Clipboard.setData(ClipboardData(text: config));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Configuration copied to clipboard')),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colorScheme = theme.colorScheme;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 900),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Row(
                children: [
                  Icon(Icons.hub_rounded, size: 28, color: colorScheme.primary),
                  const SizedBox(width: 12),
                  Text(
                    'MCP Integration',
                    style: theme.textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Chip(
                    label: Text(_serverRunning ? 'Connected' : 'Disconnected'),
                    backgroundColor: _serverRunning
                        ? Colors.green.shade100
                        : Colors.red.shade100,
                    labelStyle: TextStyle(
                      color: _serverRunning
                          ? Colors.green.shade800
                          : Colors.red.shade800,
                      fontWeight: FontWeight.bold,
                    ),
                    avatar: Icon(
                      _serverRunning ? Icons.check_circle : Icons.cancel,
                      size: 18,
                      color: _serverRunning
                          ? Colors.green.shade800
                          : Colors.red.shade800,
                    ),
                  ),
                  if (_checking)
                    const Padding(
                      padding: EdgeInsets.only(left: 8),
                      child: SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
                ],
              ),
              const SizedBox(height: 8),
              Text(
                'Connect QuantumStudio to Claude Code via Model Context Protocol',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 24),

              // Server Status Card
              _buildServerStatusCard(theme, colorScheme),
              const SizedBox(height: 24),

              // Available Tools Card
              _buildToolsCard(theme, colorScheme),
              const SizedBox(height: 24),

              // Claude Code Setup Card
              _buildClaudeCodeSetupCard(theme, colorScheme),
              const SizedBox(height: 24),

              // Quick Actions Card
              _buildQuickActionsCard(theme, colorScheme),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildServerStatusCard(ThemeData theme, ColorScheme colorScheme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.dns_rounded, color: colorScheme.primary),
                const SizedBox(width: 12),
                Text(
                  'MCP Server',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Configuration Row
            Row(
              children: [
                Expanded(
                  flex: 2,
                  child: TextField(
                    controller: _hostController,
                    decoration: const InputDecoration(
                      labelText: 'Host',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: TextField(
                    controller: _portController,
                    decoration: const InputDecoration(
                      labelText: 'Port',
                      border: OutlineInputBorder(),
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 8,
                      ),
                    ),
                    keyboardType: TextInputType.number,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // Status Info
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: colorScheme.surfaceContainerHighest.withValues(
                  alpha: 0.5,
                ),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'URL',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: colorScheme.onSurfaceVariant,
                          ),
                        ),
                        Text(
                          _mcpUrl,
                          style: theme.textTheme.bodyMedium?.copyWith(
                            fontFamily: 'monospace',
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Tools',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: colorScheme.onSurfaceVariant,
                          ),
                        ),
                        Text(
                          '${_tools.length} available',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Last Check',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: colorScheme.onSurfaceVariant,
                          ),
                        ),
                        Text(
                          _lastCheck != null
                              ? '${_lastCheck!.hour.toString().padLeft(2, '0')}:${_lastCheck!.minute.toString().padLeft(2, '0')}:${_lastCheck!.second.toString().padLeft(2, '0')}'
                              : 'Never',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Requests',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: colorScheme.onSurfaceVariant,
                          ),
                        ),
                        Text(
                          '$_requestCount',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Action Buttons
            Row(
              children: [
                FilledButton.icon(
                  onPressed: (_checking || _startingServer)
                      ? null
                      : _checkServerStatus,
                  icon: const Icon(Icons.refresh),
                  label: const Text('Refresh'),
                ),
                const SizedBox(width: 8),
                if (!_serverRunning)
                  OutlinedButton.icon(
                    onPressed: _startingServer ? null : _startServer,
                    icon: _startingServer
                        ? const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.play_arrow),
                    label: Text(
                      _startingServer ? 'Starting...' : 'Start Server',
                    ),
                  ),
                if (_serverRunning) ...[
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: _stopServer,
                    icon: const Icon(Icons.stop),
                    label: const Text('Stop Server'),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildToolsCard(ThemeData theme, ColorScheme colorScheme) {
    // Group tools by category
    final categories = <String, List<Map<String, dynamic>>>{
      'Health & System': _tools
          .where(
            (t) =>
                t['name'].toString().contains('health') ||
                t['name'].toString().contains('system'),
          )
          .toList(),
      'Benchmarks': _tools
          .where(
            (t) =>
                t['name'].toString().contains('benchmark') ||
                t['name'].toString().contains('run'),
          )
          .toList(),
      'Queue': _tools
          .where((t) => t['name'].toString().contains('queue'))
          .toList(),
    };

    // Add remaining tools to "Other"
    final categorized = categories.values.expand((e) => e).toList();
    final other = _tools.where((t) => !categorized.contains(t)).toList();
    if (other.isNotEmpty) {
      categories['Other'] = other;
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.build_rounded, color: colorScheme.primary),
                const SizedBox(width: 12),
                Text(
                  'Available Tools',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 12),
                Chip(
                  label: Text('${_tools.length} tools'),
                  backgroundColor: colorScheme.primaryContainer,
                  labelStyle: TextStyle(
                    color: colorScheme.onPrimaryContainer,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            if (_tools.isEmpty)
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: colorScheme.surfaceContainerHighest.withValues(
                    alpha: 0.3,
                  ),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Center(
                  child: Column(
                    children: [
                      Icon(
                        Icons.cloud_off,
                        size: 48,
                        color: colorScheme.onSurfaceVariant.withValues(
                          alpha: 0.5,
                        ),
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'No tools available. Start the MCP server to see available tools.',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else
              ...categories.entries.where((e) => e.value.isNotEmpty).map((
                entry,
              ) {
                return Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      child: Text(
                        entry.key,
                        style: theme.textTheme.labelLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: colorScheme.primary,
                        ),
                      ),
                    ),
                    ...entry.value.map(
                      (tool) => _buildToolTile(tool, theme, colorScheme),
                    ),
                    const Divider(),
                  ],
                );
              }),
          ],
        ),
      ),
    );
  }

  Widget _buildToolTile(
    Map<String, dynamic> tool,
    ThemeData theme,
    ColorScheme colorScheme,
  ) {
    final name = tool['name'] as String? ?? 'unknown';
    final description = tool['description'] as String? ?? '';
    final inputSchema = tool['inputSchema'] as Map<String, dynamic>? ?? {};
    final properties = inputSchema['properties'] as Map<String, dynamic>? ?? {};
    final required =
        (inputSchema['required'] as List<dynamic>?)?.cast<String>() ?? [];

    return ExpansionTile(
      tilePadding: EdgeInsets.zero,
      title: Text(
        name,
        style: theme.textTheme.bodyMedium?.copyWith(
          fontFamily: 'monospace',
          fontWeight: FontWeight.w600,
        ),
      ),
      subtitle: Text(
        description,
        style: theme.textTheme.bodySmall?.copyWith(
          color: colorScheme.onSurfaceVariant,
        ),
        maxLines: 1,
        overflow: TextOverflow.ellipsis,
      ),
      children: [
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          margin: const EdgeInsets.only(bottom: 8),
          decoration: BoxDecoration(
            color: colorScheme.surfaceContainerHighest.withValues(alpha: 0.3),
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(description, style: theme.textTheme.bodySmall),
              if (properties.isNotEmpty) ...[
                const SizedBox(height: 12),
                Text(
                  'Parameters:',
                  style: theme.textTheme.labelMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                ...properties.entries.map((param) {
                  final paramInfo = param.value as Map<String, dynamic>;
                  final isRequired = required.contains(param.key);
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          param.key,
                          style: theme.textTheme.bodySmall?.copyWith(
                            fontFamily: 'monospace',
                            fontWeight: FontWeight.bold,
                            color: isRequired
                                ? colorScheme.error
                                : colorScheme.onSurface,
                          ),
                        ),
                        if (isRequired)
                          Text('*', style: TextStyle(color: colorScheme.error)),
                        Text(
                          ': ${paramInfo['type'] ?? 'any'}',
                          style: theme.textTheme.bodySmall?.copyWith(
                            fontFamily: 'monospace',
                            color: colorScheme.onSurfaceVariant,
                          ),
                        ),
                        if (paramInfo['description'] != null)
                          Expanded(
                            child: Text(
                              ' - ${paramInfo['description']}',
                              style: theme.textTheme.bodySmall?.copyWith(
                                color: colorScheme.onSurfaceVariant,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                      ],
                    ),
                  );
                }),
              ],
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildClaudeCodeSetupCard(ThemeData theme, ColorScheme colorScheme) {
    final config =
        '''
{
  "mcpServers": {
    "quantumstudio": {
      "command": "python3",
      "args": ["${_suggestedScriptPath()}"],
      "env": {
        "QUANTUMSTUDIO_BACKEND_URL": "http://localhost:8127"
      }
    }
  }
}''';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.terminal_rounded, color: colorScheme.primary),
                const SizedBox(width: 12),
                Text(
                  'Claude Code Setup',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            Text(
              'Add this configuration to your Claude Code settings:',
              style: theme.textTheme.bodyMedium,
            ),
            const SizedBox(height: 12),

            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: colorScheme.outlineVariant),
              ),
              child: SelectableText(
                config,
                style: theme.textTheme.bodySmall?.copyWith(
                  fontFamily: 'monospace',
                  fontSize: 12,
                ),
              ),
            ),
            const SizedBox(height: 12),

            Row(
              children: [
                FilledButton.icon(
                  onPressed: _copyConfig,
                  icon: const Icon(Icons.copy),
                  label: const Text('Copy Configuration'),
                ),
              ],
            ),

            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue.withValues(alpha: 0.3)),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    color: Colors.blue.shade800,
                    size: 20,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Use Start Server on this screen, or run the same script path in Claude Code settings.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: Colors.blue.shade900,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQuickActionsCard(ThemeData theme, ColorScheme colorScheme) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.bolt_rounded, color: colorScheme.primary),
                const SizedBox(width: 12),
                Text(
                  'Quick Actions',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),

            if (!_serverRunning)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: colorScheme.errorContainer.withValues(alpha: 0.3),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  'Connect to MCP server to use quick actions',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: colorScheme.onErrorContainer,
                  ),
                ),
              )
            else
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: [
                  OutlinedButton.icon(
                    onPressed: () async {
                      final result = await _callTool(
                        'quantum_health_check',
                        {},
                      );
                      if (result != null && mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(
                              'Health: ${result['status']} - Version: ${result['version']}',
                            ),
                          ),
                        );
                      }
                    },
                    icon: const Icon(Icons.favorite),
                    label: const Text('Health Check'),
                  ),
                  OutlinedButton.icon(
                    onPressed: () async {
                      final result = await _callTool('quantum_system_info', {});
                      if (result != null && mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(
                              'Chip: ${result['chip']} - MLX: ${result['mlx_version']}',
                            ),
                            duration: const Duration(seconds: 5),
                          ),
                        );
                      }
                    },
                    icon: const Icon(Icons.memory),
                    label: const Text('System Info'),
                  ),
                  OutlinedButton.icon(
                    onPressed: () async {
                      final result = await _callTool(
                        'quantum_queue_status',
                        {},
                      );
                      if (result != null && mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(
                              'Queue: ${result['running_jobs']} running, ${result['queued_jobs']} queued',
                            ),
                          ),
                        );
                      }
                    },
                    icon: const Icon(Icons.queue),
                    label: const Text('Queue Status'),
                  ),
                  OutlinedButton.icon(
                    onPressed: () async {
                      final result = await _callTool(
                        'quantum_list_benchmarks',
                        {},
                      );
                      if (result != null && mounted) {
                        final benchmarks =
                            result['benchmarks'] as List<dynamic>? ?? [];
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(
                              '${benchmarks.length} benchmarks available',
                            ),
                          ),
                        );
                      }
                    },
                    icon: const Icon(Icons.science),
                    label: const Text('List Benchmarks'),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
