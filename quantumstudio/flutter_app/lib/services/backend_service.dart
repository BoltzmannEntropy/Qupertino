import 'dart:io';
import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as path;

class BackendService {
  static final BackendService _instance = BackendService._internal();
  factory BackendService() => _instance;
  BackendService._internal();

  Process? _backendProcess;
  bool _isStarting = false;
  String? _lastStartupError;

  static const int _port = 8127;
  String? get lastStartupError => _lastStartupError;

  String? get bundledBackendPath {
    if (!Platform.isMacOS) return null;

    final execPath = Platform.resolvedExecutable;
    final macosDir = path.dirname(execPath);
    final contentsDir = path.dirname(macosDir);
    final resourcesDir = path.join(contentsDir, 'Resources');
    final backendDir = path.join(resourcesDir, 'backend');

    if (Directory(backendDir).existsSync()) {
      return backendDir;
    }
    return null;
  }

  String? get localBackendPath {
    // Dev/source-run fallback: locate quantumstudio/backend near cwd.
    final candidates = <String>[
      path.join(Directory.current.path, 'backend'),
      path.join(Directory.current.path, '..', 'backend'),
      path.join(Directory.current.path, '..', '..', 'backend'),
    ];

    for (final candidate in candidates) {
      final mainPy = path.join(candidate, 'main.py');
      if (Directory(candidate).existsSync() && File(mainPy).existsSync()) {
        return path.normalize(candidate);
      }
    }
    return null;
  }

  String? get bundledMlxRoot {
    final backendPath = bundledBackendPath;
    if (backendPath == null) return null;

    final resourcesDir = path.dirname(backendPath);
    final mlxDir = path.join(resourcesDir, 'mlx');
    if (Directory(mlxDir).existsSync()) {
      return mlxDir;
    }

    return null;
  }

  Future<bool> isBackendRunning() async {
    try {
      final socket = await Socket.connect(
        '127.0.0.1',
        _port,
        timeout: const Duration(milliseconds: 500),
      );
      socket.destroy();
      return true;
    } catch (_) {
      return false;
    }
  }

  Future<bool> isBackendHealthy() async {
    HttpClient? client;
    try {
      client = HttpClient()..connectionTimeout = const Duration(seconds: 2);
      final request = await client.getUrl(
        Uri.parse('http://127.0.0.1:$_port/api/health'),
      );
      request.headers.set(HttpHeaders.acceptHeader, 'application/json');
      final response = await request.close().timeout(
        const Duration(seconds: 2),
      );
      if (response.statusCode != 200) {
        return false;
      }

      final body = await response.transform(utf8.decoder).join();
      return body.contains('"status"');
    } catch (_) {
      return false;
    } finally {
      client?.close(force: true);
    }
  }

  Future<bool> ensureStarted() async {
    if (await isBackendHealthy()) {
      _lastStartupError = null;
      return true;
    }

    final portInUse = await isBackendRunning();
    if (portInUse) {
      _lastStartupError =
          'Port $_port is already in use by another local process.';
      return false;
    }

    final backendPath = bundledBackendPath ?? localBackendPath;
    if (backendPath == null) {
      _lastStartupError =
          'Bundled backend was not found in app resources and no local backend directory was detected.';
      return false;
    }

    if (_isStarting) {
      return _waitForBackend(timeout: const Duration(seconds: 10));
    }

    _lastStartupError = null;
    _isStarting = true;
    try {
      final launcherScript = path.join(backendPath, 'run_backend.sh');
      if (File(launcherScript).existsSync()) {
        final launcherEnv = <String, String>{
          ...Platform.environment,
          'QUANTUMSTUDIO_BACKEND_PORT': '$_port',
        };
        final process = await Process.start(
          launcherScript,
          const [],
          workingDirectory: backendPath,
          environment: launcherEnv,
        );
        _backendProcess = process;

        process.stdout.transform(const SystemEncoding().decoder).listen((data) {
          debugPrint('[QuantumStudio backend] $data');
        });
        process.stderr.transform(const SystemEncoding().decoder).listen((data) {
          debugPrint('[QuantumStudio backend error] $data');
        });

        final started = await _waitForBackend(
          timeout: const Duration(minutes: 2),
        );
        if (!started) {
          _lastStartupError =
              'Backend launcher script failed health checks on port $_port.';
        }
        return started;
      }

      final pythonBin = _resolvePython(backendPath);
      if (pythonBin == null) {
        _lastStartupError = 'No usable Python runtime was found for backend.';
        return false;
      }

      final sitePackagesDir = _resolveBundledSitePackages(backendPath);
      final env = _buildBackendEnvironment(
        backendPath,
        mlxRoot: bundledMlxRoot,
        sitePackagesDir: sitePackagesDir,
      );

      final process = await Process.start(
        pythonBin,
        [
          '-m',
          'uvicorn',
          'main:app',
          '--host',
          '127.0.0.1',
          '--port',
          '$_port',
        ],
        workingDirectory: backendPath,
        environment: env,
      );
      _backendProcess = process;

      process.stdout.transform(const SystemEncoding().decoder).listen((data) {
        debugPrint('[QuantumStudio backend] $data');
      });
      process.stderr.transform(const SystemEncoding().decoder).listen((data) {
        debugPrint('[QuantumStudio backend error] $data');
      });

      final started = await _waitForBackend(
        timeout: const Duration(minutes: 2),
      );
      if (!started) {
        _lastStartupError =
            'Backend startup timed out or failed health checks on port $_port.';
      }
      return started;
    } catch (e) {
      _lastStartupError = 'Failed to start bundled backend: $e';
      debugPrint('[QuantumStudio backend start error] $e');
      return false;
    } finally {
      _isStarting = false;
    }
  }

  Future<void> stopBackend({
    Duration timeout = const Duration(seconds: 6),
  }) async {
    final process = _backendProcess;
    if (process != null) {
      process.kill(ProcessSignal.sigterm);
      try {
        await process.exitCode.timeout(timeout);
      } catch (_) {
        process.kill(ProcessSignal.sigkill);
      }
      _backendProcess = null;
    }
  }

  String? _resolvePython(String backendPath) {
    final envPython = Platform.environment['QUANTUMSTUDIO_PYTHON'];
    if (envPython != null &&
        envPython.isNotEmpty &&
        File(envPython).existsSync() &&
        _pythonSupportsBackend(envPython)) {
      return envPython;
    }

    final resourcesDir = path.dirname(backendPath);
    final projectRoot = path.dirname(backendPath);
    final workspaceRoot = path.dirname(projectRoot);
    final candidates = <String>[
      path.join(projectRoot, '.runtime-venv', 'bin', 'python3'),
      path.join(projectRoot, '.runtime-venv', 'bin', 'python'),
      path.join(workspaceRoot, '.runtime-venv', 'bin', 'python3'),
      path.join(workspaceRoot, '.runtime-venv', 'bin', 'python'),
      path.join(resourcesDir, 'python', 'bin', 'python3'),
      path.join(resourcesDir, 'python', 'bin', 'python'),
      path.join(backendPath, 'venv', 'bin', 'python3'),
      path.join(backendPath, 'venv', 'bin', 'python'),
      '/Users/sol/.pyenv/shims/python3',
      '/opt/homebrew/bin/python3',
    ];
    for (final candidate in candidates) {
      if (File(candidate).existsSync() && _pythonSupportsBackend(candidate)) {
        return candidate;
      }
    }
    return null;
  }

  bool _pythonSupportsBackend(String pythonBin) {
    try {
      final probe = Process.runSync(pythonBin, const [
        '-c',
        'import uvicorn; print("ok")',
      ]);
      return probe.exitCode == 0;
    } catch (_) {
      return false;
    }
  }

  String? _resolveBundledSitePackages(String backendPath) {
    final libDir = Directory(path.join(backendPath, 'venv', 'lib'));
    if (!libDir.existsSync()) return null;

    try {
      for (final entity in libDir.listSync()) {
        if (entity is! Directory) continue;
        final name = path.basename(entity.path);
        if (!name.startsWith('python')) continue;
        final sitePackages = path.join(entity.path, 'site-packages');
        if (Directory(sitePackages).existsSync()) {
          return sitePackages;
        }
      }
    } catch (_) {
      return null;
    }
    return null;
  }

  Map<String, String> _buildBackendEnvironment(
    String backendPath, {
    String? mlxRoot,
    String? sitePackagesDir,
  }) {
    final home = Platform.environment['HOME'] ?? '';
    final appSupportDir = home.isNotEmpty
        ? path.join(home, 'Library', 'Application Support', 'QuantumStudio')
        : '/tmp/QuantumStudio';
    final appCacheDir = home.isNotEmpty
        ? path.join(home, 'Library', 'Caches', 'QuantumStudio')
        : '/tmp/QuantumStudio/cache';
    final appLogDir = home.isNotEmpty
        ? path.join(home, 'Library', 'Logs', 'QuantumStudio')
        : '/tmp/QuantumStudio/logs';

    final runsDir = path.join(appSupportDir, 'runs');
    final benchDir = path.join(appSupportDir, 'bench');
    final settingsFile = path.join(appSupportDir, 'settings.json');
    final tmpDir = path.join(appCacheDir, 'tmp');
    final hfHome = path.join(appCacheDir, 'huggingface');
    final hfHubCache = path.join(hfHome, 'hub');
    final transformersCache = path.join(hfHome, 'transformers');

    for (final dir in [
      appSupportDir,
      appCacheDir,
      appLogDir,
      runsDir,
      benchDir,
      tmpDir,
      hfHome,
      hfHubCache,
      transformersCache,
    ]) {
      try {
        Directory(dir).createSync(recursive: true);
      } catch (_) {
        // Ignore and let backend fallback.
      }
    }

    final existingPythonPath = Platform.environment['PYTHONPATH'] ?? '';
    final pythonPathParts = <String>[backendPath];
    if (sitePackagesDir != null && sitePackagesDir.isNotEmpty) {
      pythonPathParts.add(sitePackagesDir);
    }

    if (mlxRoot != null) {
      final mlxPython = path.join(mlxRoot, 'src');
      if (Directory(mlxPython).existsSync()) {
        pythonPathParts.add(mlxPython);
      }
    }

    if (existingPythonPath.isNotEmpty) {
      pythonPathParts.add(existingPythonPath);
    }

    return {
      ...Platform.environment,
      'PYTHONUNBUFFERED': '1',
      'PYTHONPATH': pythonPathParts.join(':'),
      'PYTHONPYCACHEPREFIX': path.join(appCacheDir, 'pycache'),
      'XDG_CACHE_HOME': appCacheDir,
      'TMPDIR': tmpDir,
      'QUANTUMSTUDIO_RUNTIME_HOME': appSupportDir,
      'QUANTUMSTUDIO_LOG_DIR': appLogDir,
      'QUANTUMSTUDIO_RUNS_DIR': runsDir,
      'QUANTUMSTUDIO_SETTINGS_FILE': settingsFile,
      'QUANTUMSTUDIO_BENCH_DIR': benchDir,
      'HF_HOME': hfHome,
      'HUGGINGFACE_HUB_CACHE': hfHubCache,
      'TRANSFORMERS_CACHE': transformersCache,
      if (mlxRoot != null) 'QUANTUMSTUDIO_MLX_ROOT': mlxRoot,
      if (mlxRoot != null)
        'QUANTUMSTUDIO_MLX_PYTHON': path.join(mlxRoot, 'src'),
    };
  }

  Future<bool> _waitForBackend({
    Duration timeout = const Duration(seconds: 30),
  }) async {
    final stopwatch = Stopwatch()..start();
    while (stopwatch.elapsed < timeout) {
      if (await isBackendHealthy()) {
        return true;
      }
      await Future.delayed(const Duration(milliseconds: 500));
    }
    return false;
  }
}
