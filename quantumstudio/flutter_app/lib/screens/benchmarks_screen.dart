import 'dart:async';
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import 'package:csv/csv.dart';
import '../services/api_service.dart';

// Semantic status colors (kept for status indicators)
const _statusSuccess = Color(0xFF30D158);
const _statusWarning = Color(0xFFFF9F0A);
// Log panel colors (dark terminal aesthetic)
const _logBackgroundColor = Color(0xFF0F1115);
const _logTextColor = Color(0xFF9BD3FF);
const _logHeaderColor = Color(0xFF6CB4EE);
const List<Map<String, dynamic>> _fallbackBenchmarks = [
  {
    'name': 'hamiltonian_simulation',
    'label': 'Hamiltonian Simulation',
    'max_qubits': 30,
  },
  {'name': 'time_evolution', 'label': 'Time Evolution', 'max_qubits': 30},
  {'name': 'trotter', 'label': 'Trotter', 'max_qubits': 30},
  {'name': 'heisenberg', 'label': 'Heisenberg', 'max_qubits': 30},
  {'name': 'heisenberg_xxz', 'label': 'Heisenberg XXZ', 'max_qubits': 25},
  {
    'name': 'heisenberg_random_field',
    'label': 'Heisenberg Random Field',
    'max_qubits': 25,
  },
  {'name': 'tfim', 'label': 'TFIM', 'max_qubits': 25},
  {'name': 'tfim_trotter2', 'label': 'TFIM Trotter2', 'max_qubits': 25},
  {'name': 'tfim_random_field', 'label': 'TFIM Random Field', 'max_qubits': 25},
  {'name': 'long_range_ising', 'label': 'Long-Range Ising', 'max_qubits': 25},
  {'name': 'ladder_heisenberg', 'label': 'Ladder Heisenberg', 'max_qubits': 25},
  {'name': 'steady_state', 'label': 'Steady State', 'max_qubits': 12},
  {'name': 'random_circuit', 'label': 'Random Circuit', 'max_qubits': 25},
  {'name': 'qcbm', 'label': 'QCBM', 'max_qubits': 25},
  {'name': 'phase_estimation', 'label': 'Phase Estimation', 'max_qubits': 12},
  {'name': 'qft', 'label': 'QFT', 'max_qubits': 12},
  {'name': 'qaoa', 'label': 'QAOA', 'max_qubits': 25},
  {'name': 'vqe', 'label': 'VQE', 'max_qubits': 15},
  {
    'name': 'variational_circuit',
    'label': 'Variational Circuit',
    'max_qubits': 25,
  },
  {'name': 'grover', 'label': 'Grover', 'max_qubits': 25},
  {'name': 'ghz', 'label': 'GHZ', 'max_qubits': 25},
  {'name': 'qasm', 'label': 'QASM', 'max_qubits': 18},
];

class BenchmarksScreen extends StatefulWidget {
  const BenchmarksScreen({super.key, required this.backendReady});

  final bool backendReady;

  @override
  State<BenchmarksScreen> createState() => _BenchmarksScreenState();
}

class _BenchmarksScreenState extends State<BenchmarksScreen> {
  final ApiService _api = ApiService();
  bool _backendReady = false;
  bool _loading = true;
  List<Map<String, dynamic>> _benchmarks = _fallbackBenchmarks
      .map((e) => Map<String, dynamic>.from(e))
      .toList();
  final Map<String, bool> _selected = {};
  final TextEditingController _qubitsController = TextEditingController(
    text: '1-12',
  );
  String _backend = 'sv';
  final TextEditingController _simulateCapController = TextEditingController();
  final TextEditingController _maxQubitsController = TextEditingController();
  final TextEditingController _qasmMaxQubitsController =
      TextEditingController();
  final TextEditingController _qasmTimeoutController = TextEditingController();
  final TextEditingController _qasmMemController = TextEditingController();
  final TextEditingController _qasmSimLimitController = TextEditingController();
  final TextEditingController _benchmarkWarmupsController =
      TextEditingController(text: '0');
  final TextEditingController _benchmarkRepeatsController =
      TextEditingController(text: '1');
  final TextEditingController _envOverridesController = TextEditingController();
  bool _useDefaultQubits = false;
  bool _qasmIncludeLarge = false;
  bool _benchpress = false;

  // System info
  Map<String, dynamic> _systemInfo = {};

  // Run history
  List<Map<String, dynamic>> _runs = [];
  String? _selectedRunId;
  Map<String, dynamic>? _selectedRun;
  String _logText = '';
  Timer? _poller;
  Timer? _benchmarksRetryTimer;
  bool _bootstrapRefreshScheduled = false;
  final ScrollController _logScrollController = ScrollController();

  // Queue status
  Map<String, dynamic> _queueStatus = {};

  // Get only active (running/queued) jobs
  List<Map<String, dynamic>> get _activeJobs =>
      _runs.where((r) => ['running', 'queued'].contains(r['status'])).toList();

  @override
  void initState() {
    super.initState();
    for (final item in _benchmarks) {
      final name = item['name'] as String?;
      if (name != null) {
        _selected[name] = true;
      }
    }
    if (widget.backendReady) {
      _initialize();
    } else {
      _loading = false;
    }
  }

  @override
  void didUpdateWidget(covariant BenchmarksScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!oldWidget.backendReady && widget.backendReady) {
      _initialize();
    }
  }

  @override
  void dispose() {
    _poller?.cancel();
    _benchmarksRetryTimer?.cancel();
    _logScrollController.dispose();
    _qubitsController.dispose();
    _simulateCapController.dispose();
    _maxQubitsController.dispose();
    _qasmMaxQubitsController.dispose();
    _qasmTimeoutController.dispose();
    _qasmMemController.dispose();
    _qasmSimLimitController.dispose();
    _benchmarkWarmupsController.dispose();
    _benchmarkRepeatsController.dispose();
    _envOverridesController.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    setState(() => _loading = true);
    _backendReady = widget.backendReady;
    if (_backendReady) {
      await Future.wait([_loadBenchmarks(), _loadRuns(), _loadSystemInfo()]);
    }
    _scheduleBenchmarksRetryIfNeeded();
    if (!mounted) return;
    setState(() => _loading = false);
  }

  Future<void> _loadBenchmarks() async {
    try {
      final list = await _api.getBenchmarks();
      final effectiveList = list.isEmpty
          ? _fallbackBenchmarks
                .map((e) => Map<String, dynamic>.from(e))
                .toList()
          : list;
      if (!mounted) return;
      final nextSelected = <String, bool>{};
      for (final item in effectiveList) {
        final name = item['name'] as String?;
        if (name != null) {
          nextSelected[name] = _selected[name] ?? true;
        }
      }
      setState(() {
        _benchmarks = effectiveList;
        _selected
          ..clear()
          ..addAll(nextSelected);
      });
    } catch (_) {
      if (!mounted) return;
      final nextSelected = <String, bool>{};
      final fallback = _fallbackBenchmarks
          .map((e) => Map<String, dynamic>.from(e))
          .toList();
      for (final item in fallback) {
        final name = item['name'] as String?;
        if (name != null) {
          nextSelected[name] = _selected[name] ?? true;
        }
      }
      setState(() {
        _benchmarks = fallback;
        _selected
          ..clear()
          ..addAll(nextSelected);
      });
      _scheduleBenchmarksRetryIfNeeded();
    }
  }

  void _scheduleBenchmarksRetryIfNeeded() {
    _benchmarksRetryTimer?.cancel();
    if (!_backendReady || _benchmarks.isNotEmpty) {
      return;
    }
    _benchmarksRetryTimer = Timer.periodic(const Duration(seconds: 2), (
      timer,
    ) async {
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (_benchmarks.isNotEmpty || !_backendReady) {
        timer.cancel();
        return;
      }
      await _loadBenchmarks();
      if (!mounted) {
        timer.cancel();
        return;
      }
      if (_benchmarks.isNotEmpty) {
        timer.cancel();
      }
    });
  }

  void _ensureBenchmarksLoadedAfterFirstFrame() {
    if (_bootstrapRefreshScheduled ||
        !_backendReady ||
        _benchmarks.isNotEmpty) {
      return;
    }
    _bootstrapRefreshScheduled = true;
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      if (!mounted) return;
      await _loadBenchmarks();
      _scheduleBenchmarksRetryIfNeeded();
    });
  }

  Future<void> _loadRuns() async {
    try {
      final results = await Future.wait([
        _api.listRuns(),
        _api.getQueueStatus(),
      ]);
      if (!mounted) return;
      setState(() {
        _runs = results[0] as List<Map<String, dynamic>>;
        _queueStatus = results[1] as Map<String, dynamic>;
      });
    } catch (_) {}
  }

  Future<void> _stopJob(String runId) async {
    try {
      await _api.stopRun(runId);
      await _loadRuns();
    } catch (e) {
      _showSnack('Failed to stop job: $e');
    }
  }

  Future<void> _cancelJob(String runId) async {
    try {
      await _api.cancelRun(runId);
      await _loadRuns();
    } catch (e) {
      _showSnack('Failed to cancel job: $e');
    }
  }

  Future<void> _stopAllJobs() async {
    try {
      await _api.stopAllJobs();
      await _loadRuns();
      _showSnack('All jobs stopped');
    } catch (e) {
      _showSnack('Failed to stop jobs: $e');
    }
  }

  Future<void> _loadSystemInfo() async {
    try {
      final info = await _api.getSystemInfo();
      if (!mounted) return;
      setState(() => _systemInfo = info);
    } catch (_) {}
  }

  void _selectAll(bool value) {
    setState(() {
      for (final key in _selected.keys) {
        _selected[key] = value;
      }
    });
  }

  List<String> _selectedBenchmarks() {
    return _selected.entries.where((e) => e.value).map((e) => e.key).toList();
  }

  Future<void> _startRun() async {
    final selected = _selectedBenchmarks();
    if (selected.isEmpty) {
      _showSnack('Select at least one benchmark.');
      return;
    }
    final qubitsSpec = _useDefaultQubits
        ? 'default'
        : _qubitsController.text.trim();
    if (!_useDefaultQubits && qubitsSpec.isEmpty) {
      _showSnack('Enter a qubit range or enable defaults.');
      return;
    }
    final simulateCapText = _simulateCapController.text.trim();
    int? simulateCap;
    if (simulateCapText.isNotEmpty) {
      simulateCap = int.tryParse(simulateCapText);
      if (simulateCap == null) {
        _showSnack('Simulate cap must be a whole number.');
        return;
      }
    }
    int? maxQubits;
    if (_maxQubitsController.text.trim().isNotEmpty) {
      maxQubits = int.tryParse(_maxQubitsController.text.trim());
      if (maxQubits == null) {
        _showSnack('Max qubits must be a whole number.');
        return;
      }
    }
    int? qasmMax;
    if (_qasmMaxQubitsController.text.trim().isNotEmpty) {
      qasmMax = int.tryParse(_qasmMaxQubitsController.text.trim());
      if (qasmMax == null) {
        _showSnack('QASM max qubits must be a whole number.');
        return;
      }
    }
    int? qasmTimeout;
    if (_qasmTimeoutController.text.trim().isNotEmpty) {
      qasmTimeout = int.tryParse(_qasmTimeoutController.text.trim());
      if (qasmTimeout == null) {
        _showSnack('QASM timeout must be a whole number (ms).');
        return;
      }
    }
    int? qasmMem;
    if (_qasmMemController.text.trim().isNotEmpty) {
      qasmMem = int.tryParse(_qasmMemController.text.trim());
      if (qasmMem == null) {
        _showSnack('QASM max memory must be a whole number (MB).');
        return;
      }
    }
    int? qasmSimLimit;
    if (_qasmSimLimitController.text.trim().isNotEmpty) {
      qasmSimLimit = int.tryParse(_qasmSimLimitController.text.trim());
      if (qasmSimLimit == null) {
        _showSnack('QASM simulate limit must be a whole number.');
        return;
      }
    }
    var benchmarkWarmups = 0;
    if (_benchmarkWarmupsController.text.trim().isNotEmpty) {
      final parsed = int.tryParse(_benchmarkWarmupsController.text.trim());
      if (parsed == null || parsed < 0) {
        _showSnack('Benchmark warmups must be zero or a whole number.');
        return;
      }
      benchmarkWarmups = parsed;
    }
    var benchmarkRepeats = 1;
    if (_benchmarkRepeatsController.text.trim().isNotEmpty) {
      final parsed = int.tryParse(_benchmarkRepeatsController.text.trim());
      if (parsed == null || parsed < 1) {
        _showSnack('Benchmark repeats must be at least 1.');
        return;
      }
      benchmarkRepeats = parsed;
    }
    final envOverrides = _parseEnvOverrides(_envOverridesController.text);
    if (envOverrides == null) {
      _showSnack('Env overrides must be KEY=VALUE per line.');
      return;
    }
    setState(() {
      _selectedRun = null;
      _selectedRunId = null;
      _logText = '';
    });
    try {
      final benchmarkConfigs = selected
          .map(
            (name) => {
              'name': name,
              'qubits_spec': qubitsSpec,
              'backend': _backend,
              if (simulateCap != null) 'simulate_cap': simulateCap,
            },
          )
          .toList();
      final run = await _api.startRun(
        benchmarkConfigs: benchmarkConfigs,
        maxQubits: maxQubits,
        qasmMaxQubits: qasmMax,
        qasmTimeoutMs: qasmTimeout,
        qasmMaxMemMb: qasmMem,
        qasmIncludeLarge: _qasmIncludeLarge,
        qasmSimulateLimit: qasmSimLimit,
        benchmarkWarmups: benchmarkWarmups,
        benchmarkRepeats: benchmarkRepeats,
        benchpress: _benchpress,
        envOverrides: envOverrides,
      );
      if (!mounted) return;
      final runId = run['id'] as String;
      setState(() {
        _selectedRun = run;
        _selectedRunId = runId;
        // Add to run list at top
        _runs.insert(0, run);
      });
      _startPolling(runId);
    } catch (e) {
      _showSnack('Failed to start run: $e');
    }
  }

  void _selectRun(String runId) {
    _poller?.cancel();
    setState(() {
      _selectedRunId = runId;
      _logText = '';
    });
    _refreshSelectedRun(runId);
    // If the run is still active, start polling
    final run = _runs.firstWhere((r) => r['id'] == runId, orElse: () => {});
    final status = run['status'] as String? ?? '';
    if (status == 'running' || status == 'queued') {
      _startPolling(runId);
    }
  }

  void _startPolling(String runId) {
    _poller?.cancel();
    _refreshSelectedRun(runId);
    _poller = Timer.periodic(const Duration(seconds: 1), (_) async {
      await _refreshSelectedRun(runId);
    });
  }

  Future<void> _refreshSelectedRun(String runId) async {
    try {
      final run = await _api.getRun(runId);
      final log = await _api.getRunLog(runId);
      if (!mounted) return;
      final logChanged = log != _logText;
      setState(() {
        _selectedRun = run;
        _logText = log;
        // Update in the runs list too
        final idx = _runs.indexWhere((r) => r['id'] == runId);
        if (idx >= 0) {
          _runs[idx] = run;
        }
      });
      if (logChanged) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (_logScrollController.hasClients) {
            _logScrollController.animateTo(
              _logScrollController.position.maxScrollExtent,
              duration: const Duration(milliseconds: 200),
              curve: Curves.easeOut,
            );
          }
        });
      }
      final status = run['status'] as String?;
      if (status != 'running' && status != 'queued') {
        _poller?.cancel();
        // Reload full run list to get final outputs
        await _loadRuns();
      }
    } catch (e) {
      if (e is ApiException && e.statusCode == 404) {
        _poller?.cancel();
      }
    }
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(SnackBar(content: Text(message)));
  }

  Map<String, String>? _parseEnvOverrides(String raw) {
    if (raw.trim().isEmpty) {
      return {};
    }
    final lines = raw.split('\n');
    final Map<String, String> env = {};
    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isEmpty) continue;
      final idx = trimmed.indexOf('=');
      if (idx <= 0 || idx == trimmed.length - 1) {
        return null;
      }
      final key = trimmed.substring(0, idx).trim();
      final value = trimmed.substring(idx + 1).trim();
      if (key.isEmpty || value.isEmpty) {
        return null;
      }
      env[key] = value;
    }
    return env;
  }

  // --- Helpers ---

  String _runLabel(Map<String, dynamic> run) {
    final benchmarks =
        (run['benchmarks'] as List<dynamic>?)?.cast<String>() ?? [];
    if (benchmarks.isEmpty) return 'Unknown';
    if (benchmarks.length == 1) return _benchLabel(benchmarks.first);
    return '${_benchLabel(benchmarks.first)} +${benchmarks.length - 1} more';
  }

  String _benchLabel(String name) {
    // Convert snake_case to Title Case
    return name
        .split('_')
        .map((w) => w.isEmpty ? '' : '${w[0].toUpperCase()}${w.substring(1)}')
        .join(' ');
  }

  String _runTime(Map<String, dynamic> run) {
    final started = run['started_at'] as String?;
    if (started == null) return '—';
    try {
      final dt = DateTime.parse(started);
      final h = dt.hour.toString().padLeft(2, '0');
      final m = dt.minute.toString().padLeft(2, '0');
      final mon = dt.month.toString().padLeft(2, '0');
      final d = dt.day.toString().padLeft(2, '0');
      return '$mon-$d $h:$m';
    } catch (_) {
      return started;
    }
  }

  String _runDuration(Map<String, dynamic> run) {
    final started = run['started_at'] as String?;
    final ended = run['ended_at'] as String?;
    if (started == null || ended == null) return '';
    try {
      final s = DateTime.parse(started);
      final e = DateTime.parse(ended);
      final dur = e.difference(s);
      if (dur.inMinutes > 0) {
        return '${dur.inMinutes}m ${dur.inSeconds % 60}s';
      }
      return '${dur.inSeconds}s';
    } catch (_) {
      return '';
    }
  }

  Color _statusColor(String status) {
    final colorScheme = Theme.of(context).colorScheme;
    return switch (status) {
      'running' => colorScheme.primary,
      'queued' => _statusWarning,
      'completed' => _statusSuccess,
      'failed' => colorScheme.error,
      'stopped' => _statusWarning,
      'cancelled' => colorScheme.outline,
      _ => colorScheme.outline,
    };
  }

  // --- Build methods ---

  @override
  Widget build(BuildContext context) {
    _ensureBenchmarksLoadedAfterFirstFrame();
    return _loading
        ? const Center(child: CircularProgressIndicator())
        : Column(
            children: [
              // Hardware info bar
              _buildHardwareBar(),
              // Main content
              Expanded(
                child: Row(
                  children: [
                    _buildSidebar(),
                    Expanded(
                      child: SingleChildScrollView(
                        padding: const EdgeInsets.all(24),
                        child: _buildMainContent(),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          );
  }

  // --- Hardware info bar ---

  Widget _buildHardwareBar() {
    if (_systemInfo.isEmpty) return const SizedBox.shrink();

    final colorScheme = Theme.of(context).colorScheme;
    final chip = _systemInfo['chip'] as String? ?? '—';
    final memGb = _systemInfo['memory_gb'] as String? ?? '—';
    final cpuCores = _systemInfo['cpu_cores'] as String? ?? '—';
    final perfCores = _systemInfo['cpu_perf_cores'] as String? ?? '';
    final effCores = _systemInfo['cpu_eff_cores'] as String? ?? '';
    final gpuCores = _systemInfo['gpu_cores'] as String? ?? '—';
    final mlxVersion = _systemInfo['mlx_version'] as String? ?? '—';
    final mlxBackend = _systemInfo['mlx_backend'] as String? ?? '';

    String cpuDetail = '$cpuCores cores';
    if (perfCores.isNotEmpty && effCores.isNotEmpty) {
      cpuDetail += ' (${perfCores}P + ${effCores}E)';
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHigh,
        border: Border(bottom: BorderSide(color: colorScheme.outlineVariant)),
      ),
      child: Row(
        children: [
          _hwChip(Icons.memory, chip),
          _hwDivider(),
          _hwChip(Icons.storage_rounded, '$memGb GB Unified'),
          _hwDivider(),
          _hwChip(Icons.grid_view_rounded, 'CPU $cpuDetail'),
          _hwDivider(),
          _hwChip(Icons.auto_awesome, 'GPU $gpuCores cores'),
          _hwDivider(),
          _hwChip(
            CupertinoIcons.cube_fill,
            'MLX $mlxVersion${mlxBackend.isNotEmpty ? " ($mlxBackend)" : ""}',
          ),
          const Spacer(),
        ],
      ),
    );
  }

  Widget _hwChip(IconData icon, String label) {
    final colorScheme = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 17, color: colorScheme.onSurfaceVariant),
        const SizedBox(width: 6),
        Text(
          label,
          style: TextStyle(
            fontSize: 14,
            color: colorScheme.onSurface,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _hwDivider() {
    final colorScheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Container(width: 1, height: 18, color: colorScheme.outlineVariant),
    );
  }

  // --- Sidebar: vertical benchmark list ---

  Widget _buildSidebar() {
    final colorScheme = Theme.of(context).colorScheme;
    final selectedCount = _selected.values.where((v) => v).length;
    return Container(
      width: 260,
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        border: Border(right: BorderSide(color: colorScheme.outlineVariant)),
      ),
      child: Column(
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 12, 0),
            child: Row(
              children: [
                const Text(
                  'Benchmarks',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700),
                ),
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 7,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: colorScheme.primary.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text(
                    '$selectedCount',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w700,
                      color: colorScheme.primary,
                    ),
                  ),
                ),
              ],
            ),
          ),
          // Select All / Clear
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Row(
              children: [
                _sidebarAction('Select All', () => _selectAll(true)),
                const SizedBox(width: 2),
                _sidebarAction('Clear', () => _selectAll(false)),
              ],
            ),
          ),
          const Divider(height: 1),
          // Benchmark list
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 4),
              itemCount: _benchmarks.length,
              itemBuilder: (context, index) =>
                  _buildBenchmarkRow(_benchmarks[index]),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sidebarAction(String label, VoidCallback onTap) {
    return TextButton(
      onPressed: onTap,
      style: TextButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        minimumSize: Size.zero,
        tapTargetSize: MaterialTapTargetSize.shrinkWrap,
        textStyle: const TextStyle(fontSize: 12),
      ),
      child: Text(label),
    );
  }

  Widget _buildBenchmarkRow(Map<String, dynamic> bench) {
    final colorScheme = Theme.of(context).colorScheme;
    final name = bench['name'] as String? ?? 'unknown';
    final label = bench['label'] as String? ?? name;
    final maxQubits = bench['max_qubits']?.toString() ?? '—';
    final selected = _selected[name] ?? false;

    return InkWell(
      onTap: () => setState(() => _selected[name] = !selected),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 140),
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 9),
        decoration: BoxDecoration(
          color: selected ? colorScheme.primary.withValues(alpha: 0.10) : null,
          border: Border(
            left: BorderSide(
              color: selected ? colorScheme.primary : Colors.transparent,
              width: 3,
            ),
          ),
        ),
        child: Row(
          children: [
            // Checkbox indicator
            AnimatedContainer(
              duration: const Duration(milliseconds: 140),
              width: 18,
              height: 18,
              decoration: BoxDecoration(
                color: selected ? colorScheme.primary : Colors.transparent,
                borderRadius: BorderRadius.circular(5),
                border: Border.all(
                  color: selected ? colorScheme.primary : colorScheme.outline,
                  width: 1.5,
                ),
              ),
              child: selected
                  ? Icon(Icons.check, size: 13, color: colorScheme.onPrimary)
                  : null,
            ),
            const SizedBox(width: 10),
            // Label
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: selected ? FontWeight.w600 : FontWeight.w400,
                  color: selected ? colorScheme.primary : colorScheme.onSurface,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            // Max qubits badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: selected
                    ? colorScheme.primary.withValues(alpha: 0.12)
                    : colorScheme.surfaceContainerHigh,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                maxQubits,
                style: TextStyle(
                  fontSize: 10,
                  fontWeight: FontWeight.w600,
                  color: selected
                      ? colorScheme.primary
                      : colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSettingsCard() {
    final colorScheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Run Settings',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Qubits (CSV or range)',
                hintText: '1,2,5 or 1-12 (ignored when using defaults)',
              ),
              controller: _qubitsController,
              enabled: !_useDefaultQubits,
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Switch(
                  value: _useDefaultQubits,
                  onChanged: (value) =>
                      setState(() => _useDefaultQubits = value),
                ),
                const SizedBox(width: 6),
                const Expanded(
                  child: Text(
                    'Use default qubit lists (bench.sh parity)',
                    maxLines: 2,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            LayoutBuilder(
              builder: (context, constraints) {
                final compact = constraints.maxWidth < 520;
                final children = [
                  DropdownButtonFormField<String>(
                    initialValue: _backend,
                    isExpanded: true,
                    items: const [
                      DropdownMenuItem(
                        value: 'sv',
                        child: Text('State Vector (sv)'),
                      ),
                      DropdownMenuItem(
                        value: 'mps',
                        child: Text('Matrix Product State (mps)'),
                      ),
                    ],
                    onChanged: (val) {
                      if (val != null) {
                        setState(() => _backend = val);
                      }
                    },
                    decoration: const InputDecoration(labelText: 'Backend'),
                  ),
                  TextField(
                    decoration: const InputDecoration(
                      labelText: 'Simulate cap (optional)',
                    ),
                    controller: _simulateCapController,
                    keyboardType: TextInputType.number,
                  ),
                ];

                if (compact) {
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      children[0],
                      const SizedBox(height: 12),
                      children[1],
                    ],
                  );
                }

                return Row(
                  children: [
                    Expanded(child: children[0]),
                    const SizedBox(width: 12),
                    Expanded(child: children[1]),
                  ],
                );
              },
            ),
            const SizedBox(height: 16),
            Align(
              alignment: Alignment.centerRight,
              child: FilledButton.icon(
                onPressed: _backendReady ? _startRun : null,
                icon: const Icon(CupertinoIcons.play_fill),
                label: const Text('Run Selected'),
              ),
            ),
            const SizedBox(height: 8),
            ExpansionTile(
              tilePadding: EdgeInsets.zero,
              title: const Text(
                'Advanced Options',
                style: TextStyle(fontWeight: FontWeight.w600),
              ),
              children: [
                TextField(
                  decoration: const InputDecoration(
                    labelText: 'Global max qubits (optional)',
                    hintText: '25',
                  ),
                  controller: _maxQubitsController,
                  keyboardType: TextInputType.number,
                ),
                const SizedBox(height: 12),
                const Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'Benchmark protocol',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                ),
                const SizedBox(height: 8),
                LayoutBuilder(
                  builder: (context, constraints) {
                    final compact = constraints.maxWidth < 520;
                    final fields = [
                      TextField(
                        decoration: const InputDecoration(
                          labelText: 'Warmups',
                          hintText: '0',
                        ),
                        controller: _benchmarkWarmupsController,
                        keyboardType: TextInputType.number,
                      ),
                      TextField(
                        decoration: const InputDecoration(
                          labelText: 'Repeats',
                          hintText: '1',
                        ),
                        controller: _benchmarkRepeatsController,
                        keyboardType: TextInputType.number,
                      ),
                    ];
                    if (compact) {
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          fields[0],
                          const SizedBox(height: 8),
                          fields[1],
                        ],
                      );
                    }
                    return Row(
                      children: [
                        Expanded(child: fields[0]),
                        const SizedBox(width: 12),
                        Expanded(child: fields[1]),
                      ],
                    );
                  },
                ),
                const SizedBox(height: 12),
                const Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    'QASM options',
                    style: TextStyle(fontWeight: FontWeight.w600),
                  ),
                ),
                const SizedBox(height: 8),
                LayoutBuilder(
                  builder: (context, constraints) {
                    final compact = constraints.maxWidth < 520;
                    final fields = [
                      TextField(
                        decoration: const InputDecoration(
                          labelText: 'QASM max qubits',
                          hintText: '18',
                        ),
                        controller: _qasmMaxQubitsController,
                        keyboardType: TextInputType.number,
                      ),
                      TextField(
                        decoration: const InputDecoration(
                          labelText: 'QASM timeout (ms)',
                          hintText: '0 = no timeout',
                        ),
                        controller: _qasmTimeoutController,
                        keyboardType: TextInputType.number,
                      ),
                      TextField(
                        decoration: const InputDecoration(
                          labelText: 'QASM max mem (MB)',
                          hintText: '4096',
                        ),
                        controller: _qasmMemController,
                        keyboardType: TextInputType.number,
                      ),
                      TextField(
                        decoration: const InputDecoration(
                          labelText: 'QASM simulate limit',
                          hintText: '0 = no limit',
                        ),
                        controller: _qasmSimLimitController,
                        keyboardType: TextInputType.number,
                      ),
                    ];
                    if (compact) {
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          fields[0],
                          const SizedBox(height: 8),
                          fields[1],
                          const SizedBox(height: 8),
                          fields[2],
                          const SizedBox(height: 8),
                          fields[3],
                        ],
                      );
                    }
                    return Column(
                      children: [
                        Row(
                          children: [
                            Expanded(child: fields[0]),
                            const SizedBox(width: 12),
                            Expanded(child: fields[1]),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            Expanded(child: fields[2]),
                            const SizedBox(width: 12),
                            Expanded(child: fields[3]),
                          ],
                        ),
                      ],
                    );
                  },
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Switch(
                      value: _qasmIncludeLarge,
                      onChanged: (value) =>
                          setState(() => _qasmIncludeLarge = value),
                    ),
                    const SizedBox(width: 6),
                    const Text('Include large QASM group'),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Switch(
                      value: _benchpress,
                      onChanged: (value) => setState(() => _benchpress = value),
                    ),
                    const SizedBox(width: 6),
                    const Text('Generate Benchpress-style figures'),
                  ],
                ),
                const SizedBox(height: 8),
                TextField(
                  decoration: const InputDecoration(
                    labelText: 'Env overrides (KEY=VALUE per line)',
                    hintText: 'MLXQ_MPS_DMAX=64',
                  ),
                  controller: _envOverridesController,
                  maxLines: 4,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // --- Main content area ---

  Widget _buildMainContent() {
    final screenWidth = MediaQuery.of(context).size.width;
    final isWide = screenWidth > 1200;

    if (isWide) {
      return Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Left: settings + run history
          SizedBox(
            width: 380,
            child: Column(
              children: [
                _buildSettingsCard(),
                const SizedBox(height: 20),
                _buildRunHistory(),
              ],
            ),
          ),
          const SizedBox(width: 24),
          // Right: selected run detail
          Expanded(
            child: _selectedRun != null
                ? _buildSelectedRunDetail()
                : _buildEmptyRunPlaceholder(),
          ),
        ],
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildSettingsCard(),
        const SizedBox(height: 20),
        _buildRunHistory(),
        if (_selectedRun != null) ...[
          const SizedBox(height: 20),
          _buildSelectedRunDetail(),
        ],
      ],
    );
  }

  Widget _buildEmptyRunPlaceholder() {
    final colorScheme = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 80),
      alignment: Alignment.center,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            CupertinoIcons.graph_square,
            size: 48,
            color: colorScheme.onSurface.withValues(alpha: 0.12),
          ),
          const SizedBox(height: 12),
          Text(
            'Select a run to view results',
            style: TextStyle(
              color: colorScheme.onSurface.withValues(alpha: 0.3),
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildQueueStatus() {
    final colorScheme = Theme.of(context).colorScheme;
    final runningCount = _queueStatus['running_count'] ?? 0;
    final queuedCount = _queueStatus['queued_count'] ?? 0;
    final maxConcurrent = _queueStatus['max_concurrent'] ?? 2;
    final hasActive = runningCount > 0 || queuedCount > 0;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: hasActive
            ? colorScheme.primary.withValues(alpha: 0.08)
            : colorScheme.outline.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: hasActive
              ? colorScheme.primary.withValues(alpha: 0.3)
              : colorScheme.outline.withValues(alpha: 0.3),
        ),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          final compact = constraints.maxWidth < 560;
          final queueText =
              'Queue: $runningCount running / $queuedCount queued (max $maxConcurrent)';

          if (compact) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(
                      Icons.queue_outlined,
                      size: 18,
                      color: hasActive
                          ? colorScheme.primary
                          : colorScheme.outline,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        queueText,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w500,
                          color: hasActive
                              ? colorScheme.primary
                              : colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ),
                  ],
                ),
                if (hasActive) ...[
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton(
                      onPressed: _stopAllJobs,
                      style: TextButton.styleFrom(
                        foregroundColor: colorScheme.error,
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        minimumSize: Size.zero,
                      ),
                      child: const Text('Stop All'),
                    ),
                  ),
                ],
              ],
            );
          }

          return Row(
            children: [
              Icon(
                Icons.queue_outlined,
                size: 18,
                color: hasActive ? colorScheme.primary : colorScheme.outline,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  queueText,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: hasActive
                        ? colorScheme.primary
                        : colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
              if (hasActive) ...[
                const SizedBox(width: 8),
                TextButton(
                  onPressed: _stopAllJobs,
                  style: TextButton.styleFrom(
                    foregroundColor: colorScheme.error,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    minimumSize: Size.zero,
                  ),
                  child: const Text('Stop All'),
                ),
              ],
            ],
          );
        },
      ),
    );
  }

  Widget _buildRunHistory() {
    final colorScheme = Theme.of(context).colorScheme;
    final activeJobs = _activeJobs;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        LayoutBuilder(
          builder: (context, constraints) {
            final compact = constraints.maxWidth < 560;
            final titleRow = Row(
              children: [
                const Text(
                  'Active Jobs',
                  style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 12),
                if (activeJobs.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 8,
                      vertical: 2,
                    ),
                    decoration: BoxDecoration(
                      color: colorScheme.primary.withValues(alpha: 0.12),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '${activeJobs.length}',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: colorScheme.primary,
                      ),
                    ),
                  ),
              ],
            );

            final refreshButton = TextButton.icon(
              onPressed: _loadRuns,
              icon: const Icon(CupertinoIcons.refresh, size: 14),
              label: const Text('Refresh'),
            );

            if (compact) {
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  titleRow,
                  const SizedBox(height: 8),
                  Align(alignment: Alignment.centerRight, child: refreshButton),
                ],
              );
            }

            return Row(
              children: [
                Expanded(child: titleRow),
                refreshButton,
              ],
            );
          },
        ),
        const SizedBox(height: 12),
        _buildQueueStatus(),
        const SizedBox(height: 12),
        Card(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(color: colorScheme.outlineVariant),
          ),
          child: activeJobs.isEmpty
              ? Padding(
                  padding: const EdgeInsets.all(24),
                  child: Center(
                    child: Text(
                      'No active jobs. Select benchmarks and click Run.',
                      style: TextStyle(color: colorScheme.onSurfaceVariant),
                    ),
                  ),
                )
              : SizedBox(
                  height: math.min(300.0, activeJobs.length * 62.0 + 8),
                  child: ListView.separated(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    itemCount: activeJobs.length,
                    separatorBuilder: (context, index) =>
                        const Divider(height: 1, indent: 16, endIndent: 16),
                    itemBuilder: (context, index) {
                      final run = activeJobs[index];
                      return _buildRunHistoryItem(run);
                    },
                  ),
                ),
        ),
      ],
    );
  }

  Widget _buildRunHistoryItem(Map<String, dynamic> run) {
    final colorScheme = Theme.of(context).colorScheme;
    final runId = run['id'] as String? ?? '';
    final status = run['status'] as String? ?? 'idle';
    final isSelected = runId == _selectedRunId;
    final isRunning = status == 'running';
    final isQueued = status == 'queued';
    final isActive = isRunning || isQueued;
    final color = _statusColor(status);
    final benchmarks =
        (run['benchmarks'] as List<dynamic>?)?.cast<String>() ?? [];
    final duration = _runDuration(run);
    final queuePosition = run['queue_position'];

    return InkWell(
      onTap: () => _selectRun(runId),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: isSelected
              ? colorScheme.primary.withValues(alpha: 0.08)
              : null,
          border: isSelected
              ? Border(left: BorderSide(color: colorScheme.primary, width: 3))
              : null,
        ),
        child: Row(
          children: [
            // Status indicator
            if (isRunning)
              SizedBox(
                width: 14,
                height: 14,
                child: CircularProgressIndicator(strokeWidth: 2, color: color),
              )
            else if (isQueued)
              Container(
                width: 14,
                height: 14,
                decoration: BoxDecoration(
                  color: _statusWarning.withValues(alpha: 0.2),
                  shape: BoxShape.circle,
                  border: Border.all(color: _statusWarning, width: 2),
                ),
              )
            else
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(color: color, shape: BoxShape.circle),
              ),
            const SizedBox(width: 10),
            // Benchmark names
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    _runLabel(run),
                    style: TextStyle(
                      fontWeight: isSelected
                          ? FontWeight.w700
                          : FontWeight.w500,
                      fontSize: 13,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 2),
                  Text(
                    isQueued && queuePosition != null
                        ? 'Queued #$queuePosition'
                        : (benchmarks.length > 1
                              ? benchmarks.map(_benchLabel).join(', ')
                              : status),
                    style: TextStyle(
                      fontSize: 11,
                      color: colorScheme.onSurfaceVariant,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ],
              ),
            ),
            // Time and duration
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Text(
                  _runTime(run),
                  style: TextStyle(
                    fontSize: 11,
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
                if (duration.isNotEmpty)
                  Text(duration, style: TextStyle(fontSize: 10, color: color)),
              ],
            ),
            // Stop/Cancel buttons for active jobs
            if (isActive) ...[
              const SizedBox(width: 8),
              if (isRunning)
                IconButton(
                  onPressed: () => _stopJob(runId),
                  icon: const Icon(Icons.stop, size: 20),
                  color: colorScheme.error,
                  tooltip: 'Stop',
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(
                    minWidth: 32,
                    minHeight: 32,
                  ),
                )
              else if (isQueued)
                IconButton(
                  onPressed: () => _cancelJob(runId),
                  icon: const Icon(Icons.cancel_outlined, size: 20),
                  color: colorScheme.onSurfaceVariant,
                  tooltip: 'Cancel',
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(
                    minWidth: 32,
                    minHeight: 32,
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildSelectedRunDetail() {
    final colorScheme = Theme.of(context).colorScheme;
    final run = _selectedRun!;
    final status = run['status'] as String? ?? 'idle';
    final runId = run['id'] as String? ?? '';
    final benchmarks =
        (run['benchmarks'] as List<dynamic>?)?.cast<String>() ?? [];
    final isActive = status == 'running' || status == 'queued';
    final color = _statusColor(status);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Run header
        Card(
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(color: colorScheme.outlineVariant),
          ),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      _runLabel(run),
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(width: 10),
                    if (isActive)
                      SizedBox(
                        width: 14,
                        height: 14,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: color,
                        ),
                      ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 10,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: color.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        status.toUpperCase(),
                        style: TextStyle(
                          color: color,
                          fontWeight: FontWeight.w700,
                          fontSize: 11,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                if (benchmarks.length > 1)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 4),
                    child: Wrap(
                      spacing: 6,
                      runSpacing: 4,
                      children: benchmarks
                          .map(
                            (b) => Chip(
                              label: Text(
                                _benchLabel(b),
                                style: const TextStyle(fontSize: 11),
                              ),
                              materialTapTargetSize:
                                  MaterialTapTargetSize.shrinkWrap,
                              visualDensity: VisualDensity.compact,
                            ),
                          )
                          .toList(),
                    ),
                  ),
                Text(
                  'ID: $runId',
                  style: TextStyle(
                    fontSize: 11,
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
                Text(
                  'Started: ${run['started_at'] ?? '—'}',
                  style: TextStyle(
                    fontSize: 11,
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
                Text(
                  'Ended: ${run['ended_at'] ?? '—'}',
                  style: TextStyle(
                    fontSize: 11,
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 16),
        // Log
        _buildLogCard(),
        const SizedBox(height: 16),
        // Outputs
        _buildOutputsCard(),
      ],
    );
  }

  Widget _buildOutputsCard() {
    final colorScheme = Theme.of(context).colorScheme;
    final outputs = _selectedRun?['outputs'] as Map<String, dynamic>?;
    final images = _normalizedImageOutputs(
      outputs?['images'] as List<dynamic>?,
    );
    final data = outputs?['data'] as List<dynamic>? ?? [];

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Outputs',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            ExpansionTile(
              tilePadding: EdgeInsets.zero,
              initiallyExpanded: images.isNotEmpty,
              title: Text(
                'Images (${images.length})',
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              children: [
                if (images.isEmpty)
                  const Align(
                    alignment: Alignment.centerLeft,
                    child: Text('No images yet.'),
                  )
                else
                  SizedBox(
                    height: 240,
                    child: SingleChildScrollView(
                      child: Wrap(
                        spacing: 12,
                        runSpacing: 12,
                        children: images.map((img) {
                          final name = img['name'] as String? ?? 'image';
                          final url = _api.resolveUrl(
                            img['url'] as String? ?? '',
                          );
                          return SizedBox(
                            width: 220,
                            child: GestureDetector(
                              onTap: () => _showImagePreview(name, url),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  ClipRRect(
                                    borderRadius: BorderRadius.circular(12),
                                    child: Image.network(
                                      url,
                                      fit: BoxFit.cover,
                                      errorBuilder:
                                          (context, error, stackTrace) =>
                                              Container(
                                                height: 120,
                                                color: colorScheme
                                                    .surfaceContainerHighest,
                                                alignment: Alignment.center,
                                                child: const Text(
                                                  'Image unavailable',
                                                  style: TextStyle(
                                                    fontSize: 12,
                                                  ),
                                                ),
                                              ),
                                    ),
                                  ),
                                  const SizedBox(height: 6),
                                  Text(
                                    name,
                                    style: const TextStyle(fontSize: 12),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                ],
                              ),
                            ),
                          );
                        }).toList(),
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            ExpansionTile(
              tilePadding: EdgeInsets.zero,
              initiallyExpanded: data.isNotEmpty,
              title: Text(
                'Data files (${data.length})',
                style: const TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                ),
              ),
              children: [
                if (data.isEmpty)
                  const Align(
                    alignment: Alignment.centerLeft,
                    child: Text('No data yet.'),
                  )
                else
                  Column(
                    children: data.map((item) {
                      final name = item['name'] as String? ?? '';
                      final url = item['url'] as String? ?? '';
                      return Container(
                        margin: const EdgeInsets.only(bottom: 8),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 10,
                        ),
                        decoration: BoxDecoration(
                          color: colorScheme.surface,
                          borderRadius: BorderRadius.circular(12),
                          border: Border.all(color: colorScheme.outlineVariant),
                        ),
                        child: Row(
                          children: [
                            Expanded(
                              child: Text(
                                name,
                                style: const TextStyle(fontSize: 12),
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            TextButton(
                              onPressed: () => _showDataPreview(name, url),
                              child: const Text('View'),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _showImagePreview(String name, String url) {
    showDialog<void>(
      context: context,
      builder: (context) => Dialog(
        insetPadding: const EdgeInsets.all(24),
        child: Container(
          padding: const EdgeInsets.all(16),
          constraints: const BoxConstraints(maxWidth: 900, maxHeight: 700),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(name, style: const TextStyle(fontWeight: FontWeight.w600)),
              const SizedBox(height: 12),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: InteractiveViewer(
                    minScale: 0.8,
                    maxScale: 4,
                    child: Image.network(
                      url,
                      fit: BoxFit.contain,
                      errorBuilder: (context, error, stackTrace) =>
                          const Center(child: Text('Image unavailable')),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  List<Map<String, dynamic>> _normalizedImageOutputs(List<dynamic>? raw) {
    final items = (raw ?? const <dynamic>[])
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e))
        .toList();
    final byName = <String, Map<String, dynamic>>{};
    for (final item in items) {
      final name = (item['name'] ?? '').toString();
      if (name.isEmpty) continue;
      final url = (item['url'] ?? '').toString();
      final prev = byName[name];
      if (prev == null) {
        byName[name] = item;
        continue;
      }
      final prevUrl = (prev['url'] ?? '').toString();
      final currentIsRunPath = url.contains('/runs/');
      final prevIsRunPath = prevUrl.contains('/runs/');
      if (currentIsRunPath && !prevIsRunPath) {
        byName[name] = item;
      }
    }
    return byName.values.toList();
  }

  void _showDataPreview(String name, String url) {
    showDialog<void>(
      context: context,
      builder: (context) => Dialog(
        insetPadding: const EdgeInsets.all(24),
        child: Container(
          padding: const EdgeInsets.all(16),
          constraints: const BoxConstraints(maxWidth: 920, maxHeight: 700),
          child: FutureBuilder<String>(
            future: _api.fetchText(url),
            builder: (context, snapshot) {
              if (snapshot.connectionState != ConnectionState.done) {
                return const Center(child: CircularProgressIndicator());
              }
              if (snapshot.hasError) {
                return Text('Failed to load $name');
              }
              final text = snapshot.data ?? '';
              final ext = name.split('.').last.toLowerCase();
              Widget body;
              if (ext == 'json') {
                body = _buildJsonPreview(text);
              } else if (ext == 'csv') {
                body = _buildCsvPreview(text);
              } else if (ext == 'md') {
                body = Markdown(data: text);
              } else {
                body = SelectableText(
                  text,
                  style: const TextStyle(fontFamily: 'Menlo', fontSize: 12),
                );
              }
              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  const SizedBox(height: 12),
                  Expanded(child: SingleChildScrollView(child: body)),
                ],
              );
            },
          ),
        ),
      ),
    );
  }

  Widget _buildJsonPreview(String text) {
    try {
      final obj = json.decode(text);
      final pretty = const JsonEncoder.withIndent('  ').convert(obj);
      return SelectableText(
        pretty,
        style: const TextStyle(fontFamily: 'Menlo', fontSize: 12),
      );
    } catch (_) {
      return SelectableText(
        text,
        style: const TextStyle(fontFamily: 'Menlo', fontSize: 12),
      );
    }
  }

  Widget _buildCsvPreview(String text) {
    final normalized = text.replaceAll('\r\n', '\n').replaceAll('\r', '\n');
    List<List<dynamic>> rows = const CsvToListConverter(
      eol: '\n',
    ).convert(normalized);
    if (rows.isEmpty) {
      return const Text('No rows found.');
    }
    final header = rows.first.map((e) => e.toString()).toList();
    final bodyRows = rows.skip(1).take(60).toList();
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: DataTable(
        columns: header.map((h) => DataColumn(label: Text(h))).toList(),
        rows: bodyRows.map((row) {
          final cells = <DataCell>[];
          for (var i = 0; i < header.length; i++) {
            final value = i < row.length ? row[i] : '';
            cells.add(DataCell(Text(value.toString())));
          }
          return DataRow(cells: cells);
        }).toList(),
      ),
    );
  }

  Widget _buildLogCard() {
    final isActive =
        (_selectedRun?['status'] as String?) == 'running' ||
        (_selectedRun?['status'] as String?) == 'queued';
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _logBackgroundColor,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text(
                'Live Log',
                style: TextStyle(
                  color: _logHeaderColor,
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                ),
              ),
              if (isActive) ...[
                const SizedBox(width: 8),
                Container(
                  width: 6,
                  height: 6,
                  decoration: const BoxDecoration(
                    color: _statusSuccess,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 4),
                const Text(
                  'streaming',
                  style: TextStyle(color: _statusSuccess, fontSize: 11),
                ),
              ],
            ],
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 360,
            child: SingleChildScrollView(
              controller: _logScrollController,
              child: SelectableText(
                _logText.isEmpty ? 'Logs will appear here.' : _logText,
                style: const TextStyle(
                  color: _logTextColor,
                  fontFamily: 'Menlo',
                  fontSize: 12,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
