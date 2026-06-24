import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../services/api_service.dart';
import '../services/log_service.dart';
import '../services/theme_service.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  static const String _kApiBaseUrl = 'api_base_url';
  final _apiUrlController = TextEditingController(text: ApiService.baseUrl);
  bool _isConnected = false;
  bool _isTestingConnection = false;
  bool _autoReconnect = true;

  int _maxConcurrentJobs = 4;
  String _queueBehavior = 'wait';

  String _defaultQubitRange = '4,6,8,10';
  String _defaultBackend = 'state_vector';
  bool _defaultSimulateCap = false;
  int _defaultSimulateCapValue = 1000;

  int _maxLogEntries = 500;
  String _logLevel = 'debug';
  bool _autoClearLogs = false;
  String _exportFormat = 'txt';

  String _exportDirectory = '';
  bool _includeTimestamps = true;
  bool _autoExportRuns = false;

  bool _isSaving = false;
  bool _exportingDiagnostics = false;

  @override
  void initState() {
    super.initState();
    _initializeScreen();
  }

  @override
  void dispose() {
    _apiUrlController.dispose();
    super.dispose();
  }

  Future<void> _initializeScreen() async {
    await _loadLocalApiBaseUrl();
    await _loadSettings();
    await _testConnection();
  }

  Future<void> _loadLocalApiBaseUrl() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final savedUrl = prefs.getString(_kApiBaseUrl);
      if (savedUrl != null && savedUrl.trim().isNotEmpty) {
        ApiService.setRuntimeBaseUrl(savedUrl);
        _apiUrlController.text = savedUrl;
      }
    } catch (e) {
      logger.error('settings', 'Failed to load local API URL: $e');
    }
  }

  Future<void> _loadSettings() async {
    try {
      final settings = await ApiService().getSettings();
      if (mounted) {
        setState(() {
          _maxConcurrentJobs = settings['max_concurrent_jobs'] ?? 4;
          _defaultQubitRange = settings['default_qubit_range'] ?? '4,6,8,10';
          _defaultBackend = settings['default_backend'] ?? 'state_vector';
        });
      }
    } catch (e) {
      logger.error('settings', 'Failed to load settings: $e');
    }
  }

  Future<void> _testConnection() async {
    ApiService.setRuntimeBaseUrl(_apiUrlController.text.trim());
    setState(() => _isTestingConnection = true);
    try {
      final healthy = await ApiService().checkHealth();
      if (mounted) {
        setState(() {
          _isConnected = healthy;
          _isTestingConnection = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isConnected = false;
          _isTestingConnection = false;
        });
      }
    }
  }

  Future<void> _saveSettings() async {
    setState(() => _isSaving = true);
    try {
      final prefs = await SharedPreferences.getInstance();
      final apiBaseUrl = _apiUrlController.text.trim();
      if (apiBaseUrl.isNotEmpty) {
        await prefs.setString(_kApiBaseUrl, apiBaseUrl);
        ApiService.setRuntimeBaseUrl(apiBaseUrl);
      }

      await ApiService().updateSettings(maxConcurrentJobs: _maxConcurrentJobs);

      logger.maxLogs = _maxLogEntries;
      logger.minLevel = LogLevel.values.firstWhere((l) => l.name == _logLevel);

      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Settings saved')));
      }
      logger.info('settings', 'Settings saved successfully');
      await _testConnection();
    } catch (e) {
      logger.error('settings', 'Failed to save settings: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to save: $e')));
      }
    } finally {
      if (mounted) {
        setState(() => _isSaving = false);
      }
    }
  }

  Future<void> _exportDiagnosticLogs() async {
    setState(() => _exportingDiagnostics = true);
    final path = await logger.export();
    if (!mounted) return;
    setState(() => _exportingDiagnostics = false);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          path == null
              ? 'Failed to export logs'
              : 'Diagnostic logs exported to $path',
        ),
      ),
    );
  }

  void _resetToDefaults() {
    setState(() {
      _maxConcurrentJobs = 4;
      _queueBehavior = 'wait';
      _defaultQubitRange = '4,6,8,10';
      _defaultBackend = 'state_vector';
      _defaultSimulateCap = false;
      _defaultSimulateCapValue = 1000;
      _maxLogEntries = 500;
      _logLevel = 'debug';
      _autoClearLogs = false;
      _exportFormat = 'txt';
      _includeTimestamps = true;
      _autoExportRuns = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 700),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Text(
                'Settings',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 24),

              _buildSection(
                context: context,
                icon: Icons.palette_outlined,
                title: 'Appearance',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Theme', style: TextStyle(fontSize: 13)),
                    const SizedBox(height: 8),
                    ListenableBuilder(
                      listenable: themeService,
                      builder: (context, _) {
                        return SegmentedButton<ThemeMode>(
                          segments: const [
                            ButtonSegment(
                              value: ThemeMode.system,
                              label: Text('System'),
                              icon: Icon(Icons.brightness_auto, size: 18),
                            ),
                            ButtonSegment(
                              value: ThemeMode.light,
                              label: Text('Light'),
                              icon: Icon(Icons.light_mode, size: 18),
                            ),
                            ButtonSegment(
                              value: ThemeMode.dark,
                              label: Text('Dark'),
                              icon: Icon(Icons.dark_mode, size: 18),
                            ),
                          ],
                          selected: {themeService.themeMode},
                          onSelectionChanged: (Set<ThemeMode> selection) {
                            themeService.setThemeMode(selection.first);
                          },
                        );
                      },
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Choose how QuantumStudio appears. System follows your OS preference.',
                      style: TextStyle(
                        fontSize: 11,
                        color: Theme.of(context).colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              _buildSection(
                context: context,
                icon: Icons.cloud,
                title: 'Backend Connection',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: TextField(
                            controller: _apiUrlController,
                            decoration: const InputDecoration(
                              labelText: 'API URL',
                              border: OutlineInputBorder(),
                              isDense: true,
                            ),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Container(
                          width: 12,
                          height: 12,
                          decoration: BoxDecoration(
                            color: _isConnected ? Colors.green : Colors.red,
                            shape: BoxShape.circle,
                            boxShadow: [
                              BoxShadow(
                                color:
                                    (_isConnected ? Colors.green : Colors.red)
                                        .withValues(alpha: 0.4),
                                blurRadius: 6,
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(width: 8),
                        Text(
                          _isConnected ? 'Connected' : 'Disconnected',
                          style: TextStyle(
                            fontSize: 12,
                            color: _isConnected ? Colors.green : Colors.red,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        OutlinedButton.icon(
                          onPressed: _isTestingConnection
                              ? null
                              : _testConnection,
                          icon: _isTestingConnection
                              ? const SizedBox(
                                  width: 14,
                                  height: 14,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Icon(Icons.refresh, size: 14),
                          label: const Text('Test Connection'),
                        ),
                        const SizedBox(width: 16),
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Checkbox(
                              value: _autoReconnect,
                              onChanged: (v) =>
                                  setState(() => _autoReconnect = v ?? true),
                            ),
                            const Text(
                              'Auto-reconnect',
                              style: TextStyle(fontSize: 13),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              _buildSection(
                context: context,
                icon: Icons.queue,
                title: 'Job Queue',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Max Concurrent Jobs: $_maxConcurrentJobs',
                      style: const TextStyle(fontSize: 13),
                    ),
                    Slider(
                      value: _maxConcurrentJobs.toDouble(),
                      min: 1,
                      max: 16,
                      divisions: 15,
                      label: '$_maxConcurrentJobs',
                      onChanged: (v) =>
                          setState(() => _maxConcurrentJobs = v.round()),
                    ),
                    const SizedBox(height: 8),
                    DropdownButtonFormField<String>(
                      initialValue: _queueBehavior,
                      decoration: const InputDecoration(
                        labelText: 'Queue Behavior',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      items: const [
                        DropdownMenuItem(
                          value: 'wait',
                          child: Text('Wait for slot'),
                        ),
                        DropdownMenuItem(
                          value: 'replace',
                          child: Text('Replace oldest'),
                        ),
                      ],
                      onChanged: (v) =>
                          setState(() => _queueBehavior = v ?? 'wait'),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              _buildSection(
                context: context,
                icon: Icons.science,
                title: 'Benchmark Defaults',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    TextField(
                      decoration: const InputDecoration(
                        labelText: 'Default Qubit Range',
                        hintText: 'e.g., 4,6,8,10 or 4-12',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      controller: TextEditingController(
                        text: _defaultQubitRange,
                      ),
                      onChanged: (v) => _defaultQubitRange = v,
                    ),
                    const SizedBox(height: 12),
                    DropdownButtonFormField<String>(
                      initialValue: _defaultBackend,
                      decoration: const InputDecoration(
                        labelText: 'Default Backend',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      items: const [
                        DropdownMenuItem(
                          value: 'state_vector',
                          child: Text('State Vector'),
                        ),
                        DropdownMenuItem(
                          value: 'mps',
                          child: Text('Matrix Product State'),
                        ),
                      ],
                      onChanged: (v) =>
                          setState(() => _defaultBackend = v ?? 'state_vector'),
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        Checkbox(
                          value: _defaultSimulateCap,
                          onChanged: (v) =>
                              setState(() => _defaultSimulateCap = v ?? false),
                        ),
                        const Text(
                          'Enable Simulate Cap',
                          style: TextStyle(fontSize: 13),
                        ),
                        const SizedBox(width: 16),
                        if (_defaultSimulateCap)
                          SizedBox(
                            width: 100,
                            child: TextField(
                              decoration: const InputDecoration(
                                border: OutlineInputBorder(),
                                isDense: true,
                              ),
                              keyboardType: TextInputType.number,
                              controller: TextEditingController(
                                text: '$_defaultSimulateCapValue',
                              ),
                              onChanged: (v) => _defaultSimulateCapValue =
                                  int.tryParse(v) ?? 1000,
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              _buildSection(
                context: context,
                icon: Icons.terminal,
                title: 'Logging',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Max Log Entries: $_maxLogEntries',
                      style: const TextStyle(fontSize: 13),
                    ),
                    Slider(
                      value: _maxLogEntries.toDouble(),
                      min: 100,
                      max: 2000,
                      divisions: 19,
                      label: '$_maxLogEntries',
                      onChanged: (v) =>
                          setState(() => _maxLogEntries = v.round()),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Expanded(
                          child: DropdownButtonFormField<String>(
                            initialValue: _logLevel,
                            decoration: const InputDecoration(
                              labelText: 'Log Level',
                              border: OutlineInputBorder(),
                              isDense: true,
                            ),
                            items: LogLevel.values
                                .map(
                                  (l) => DropdownMenuItem(
                                    value: l.name,
                                    child: Text(l.name.toUpperCase()),
                                  ),
                                )
                                .toList(),
                            onChanged: (v) =>
                                setState(() => _logLevel = v ?? 'debug'),
                          ),
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: DropdownButtonFormField<String>(
                            initialValue: _exportFormat,
                            decoration: const InputDecoration(
                              labelText: 'Export Format',
                              border: OutlineInputBorder(),
                              isDense: true,
                            ),
                            items: const [
                              DropdownMenuItem(
                                value: 'txt',
                                child: Text('Plain Text'),
                              ),
                              DropdownMenuItem(
                                value: 'json',
                                child: Text('JSON'),
                              ),
                            ],
                            onChanged: (v) =>
                                setState(() => _exportFormat = v ?? 'txt'),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Checkbox(
                          value: _autoClearLogs,
                          onChanged: (v) =>
                              setState(() => _autoClearLogs = v ?? false),
                        ),
                        const Text(
                          'Auto-clear on restart',
                          style: TextStyle(fontSize: 13),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              _buildSection(
                context: context,
                icon: Icons.folder,
                title: 'Export Preferences',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    TextField(
                      decoration: const InputDecoration(
                        labelText: 'Export Directory',
                        hintText: 'Leave empty for default',
                        border: OutlineInputBorder(),
                        isDense: true,
                      ),
                      controller: TextEditingController(text: _exportDirectory),
                      onChanged: (v) => _exportDirectory = v,
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Checkbox(
                          value: _includeTimestamps,
                          onChanged: (v) =>
                              setState(() => _includeTimestamps = v ?? true),
                        ),
                        const Text(
                          'Include timestamps in filenames',
                          style: TextStyle(fontSize: 13),
                        ),
                      ],
                    ),
                    Row(
                      children: [
                        Checkbox(
                          value: _autoExportRuns,
                          onChanged: (v) =>
                              setState(() => _autoExportRuns = v ?? false),
                        ),
                        const Text(
                          'Auto-export completed runs',
                          style: TextStyle(fontSize: 13),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              _buildSection(
                context: context,
                icon: Icons.bug_report_outlined,
                title: 'Diagnostics',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Export system logs for troubleshooting and support.',
                      style: TextStyle(fontSize: 13),
                    ),
                    const SizedBox(height: 10),
                    FilledButton.tonalIcon(
                      onPressed: _exportingDiagnostics
                          ? null
                          : _exportDiagnosticLogs,
                      icon: _exportingDiagnostics
                          ? const SizedBox(
                              width: 14,
                              height: 14,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.manage_search_rounded),
                      label: Text(
                        _exportingDiagnostics
                            ? 'Exporting...'
                            : 'Export Diagnostic Logs',
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              const SizedBox(height: 24),

              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton(
                    onPressed: _resetToDefaults,
                    child: const Text('Reset All to Defaults'),
                  ),
                  const SizedBox(width: 16),
                  FilledButton.icon(
                    onPressed: _isSaving ? null : _saveSettings,
                    icon: _isSaving
                        ? SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Theme.of(context).colorScheme.onPrimary,
                            ),
                          )
                        : const Icon(Icons.save),
                    label: const Text('Save Settings'),
                  ),
                ],
              ),
              const SizedBox(height: 48),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSection({
    required BuildContext context,
    required IconData icon,
    required String title,
    required Widget child,
  }) {
    final colorScheme = Theme.of(context).colorScheme;

    return Card(
      elevation: 0,
      color: colorScheme.surfaceContainerHigh,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 18, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            child,
          ],
        ),
      ),
    );
  }
}
