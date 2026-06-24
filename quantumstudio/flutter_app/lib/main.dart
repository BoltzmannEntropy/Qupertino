import 'dart:async';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'screens/benchmarks_screen.dart';
import 'screens/circuits_screen.dart';
import 'screens/history_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/mcp_screen.dart';
import 'screens/about_screen.dart';
import 'widgets/status_bar.dart';
import 'widgets/logs_panel.dart';
import 'services/log_service.dart';
import 'services/api_service.dart';
import 'services/backend_service.dart';
import 'services/theme_service.dart';

Future<void> main() async {
  runZonedGuarded(
    () {
      WidgetsFlutterBinding.ensureInitialized();
      _configureGlobalErrorHandling();
      runApp(const QuantumStudioApp());
    },
    (error, stackTrace) {
      logger.error('zone', 'Unhandled async exception: $error\n$stackTrace');
    },
  );
}

void _configureGlobalErrorHandling() {
  FlutterError.onError = (details) {
    FlutterError.presentError(details);
    logger.error(
      'flutter_error',
      'Framework exception: ${details.exceptionAsString()}\n${details.stack ?? ''}',
    );
  };

  PlatformDispatcher.instance.onError = (error, stackTrace) {
    logger.error(
      'platform_error',
      'Unhandled platform exception: $error\n$stackTrace',
    );
    return true;
  };

  ErrorWidget.builder = (details) {
    logger.error(
      'ui_error',
      'Widget build failure: ${details.exceptionAsString()}',
    );
    return _AppFailureFallback(details: details);
  };
}

class QuantumStudioApp extends StatefulWidget {
  const QuantumStudioApp({super.key});

  @override
  State<QuantumStudioApp> createState() => _QuantumStudioAppState();
}

class _QuantumStudioAppState extends State<QuantumStudioApp> {
  @override
  void initState() {
    super.initState();
    themeService.addListener(_onThemeChanged);
  }

  @override
  void dispose() {
    themeService.removeListener(_onThemeChanged);
    super.dispose();
  }

  void _onThemeChanged() {
    if (!mounted) return;
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'QuantumStudio',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3F51B5), // Indigo
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF3F51B5), // Indigo
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      themeMode: themeService.themeMode,
      home: const SafeScreen(screenName: 'MainShell', child: MainShell()),
    );
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  AppLifecycleListener? _appLifecycleListener;
  bool _allowImmediateExit = false;
  bool _isShuttingDown = false;

  // Logs panel state
  bool _logsCollapsed = false;
  double _logsHeight = 150;

  // System info state
  Map<String, dynamic>? _systemInfo;
  bool _backendStarting = true;
  bool _backendReady = false;
  String? _backendError;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 6, vsync: this);
    _appLifecycleListener = AppLifecycleListener(
      onExitRequested: _handleExitRequested,
    );
    unawaited(_initializeBackend());
    logger.info('main', 'QuantumStudio started');
  }

  @override
  void dispose() {
    _appLifecycleListener?.dispose();
    _tabController.dispose();
    super.dispose();
  }

  Future<AppExitResponse> _handleExitRequested() async {
    if (_allowImmediateExit) {
      return AppExitResponse.exit;
    }
    if (_isShuttingDown) {
      return AppExitResponse.cancel;
    }

    _isShuttingDown = true;
    final navigator = Navigator.of(context, rootNavigator: true);
    if (mounted) {
      unawaited(
        showDialog<void>(
          context: context,
          barrierDismissible: false,
          builder: (_) => const AlertDialog(
            content: Row(
              children: [
                SizedBox(
                  width: 20,
                  height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
                SizedBox(width: 12),
                Expanded(child: Text('Stopping backend before exit...')),
              ],
            ),
          ),
        ),
      );
      await Future<void>.delayed(const Duration(milliseconds: 120));
    }

    try {
      await BackendService().stopBackend();
      _allowImmediateExit = true;
      return AppExitResponse.exit;
    } catch (e, stackTrace) {
      logger.error(
        'main',
        'Backend shutdown during exit failed: $e\n$stackTrace',
      );
      _allowImmediateExit = true;
      return AppExitResponse.exit;
    } finally {
      if (mounted && navigator.canPop()) {
        navigator.pop();
      }
      _isShuttingDown = false;
    }
  }

  Future<void> _loadSystemInfo() async {
    try {
      final info = await ApiService().getSystemInfo();
      if (mounted) {
        setState(() => _systemInfo = info);
        logger.info('main', 'System info loaded: ${info['chip']}');
      }
    } catch (e) {
      logger.error('main', 'Failed to load system info: $e');
    }
  }

  Future<void> _initializeBackend() async {
    try {
      if (mounted) {
        setState(() {
          _backendStarting = true;
          _backendReady = false;
          _backendError = null;
        });
      }

      final backendService = BackendService();
      final started = await backendService.ensureStarted();
      if (!mounted) return;

      if (started) {
        setState(() {
          _backendStarting = false;
          _backendReady = true;
          _backendError = null;
        });
        await _loadSystemInfo();
        return;
      }

      setState(() {
        _backendStarting = false;
        _backendReady = false;
        _backendError =
            backendService.lastStartupError ??
            'Could not connect to the local backend service.';
      });
    } catch (e, stackTrace) {
      logger.error('main', 'Backend initialization failed: $e\n$stackTrace');
      if (!mounted) return;
      setState(() {
        _backendStarting = false;
        _backendReady = false;
        _backendError = 'Unexpected backend startup error. Check logs.';
      });
    }
  }

  Widget _buildBackendBanner(BuildContext context) {
    if (_backendReady) {
      return const SizedBox.shrink();
    }

    final colorScheme = Theme.of(context).colorScheme;
    final isStarting = _backendStarting;
    final tone = isStarting
        ? colorScheme.secondaryContainer
        : colorScheme.errorContainer;
    final iconColor = isStarting
        ? colorScheme.onSecondaryContainer
        : colorScheme.onErrorContainer;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      color: tone,
      child: Row(
        children: [
          if (isStarting)
            SizedBox(
              width: 16,
              height: 16,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: iconColor,
              ),
            )
          else
            Icon(Icons.warning_amber_rounded, color: iconColor, size: 18),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              isStarting
                  ? 'Starting local backend service...'
                  : (_backendError ?? 'Backend is unavailable.'),
              style: TextStyle(
                color: iconColor,
                fontWeight: FontWeight.w600,
                fontSize: 12,
              ),
            ),
          ),
          if (!isStarting)
            TextButton(
              onPressed: _initializeBackend,
              child: const Text('Retry'),
            ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          _buildBackendBanner(context),
          // Tab bar at the top
          Container(
            color: Theme.of(context).scaffoldBackgroundColor,
            child: SafeArea(
              bottom: false,
              child: TabBar(
                controller: _tabController,
                tabs: const [
                  Tab(icon: Icon(Icons.science_outlined), text: 'Benchmarks'),
                  Tab(
                    icon: Icon(Icons.account_tree_outlined),
                    text: 'Circuits',
                  ),
                  Tab(icon: Icon(Icons.history), text: 'History'),
                  Tab(icon: Icon(Icons.settings_outlined), text: 'Settings'),
                  Tab(icon: Icon(Icons.hub_outlined), text: 'MCP'),
                  Tab(icon: Icon(Icons.info_outline), text: 'About'),
                ],
              ),
            ),
          ),
          // Status bar with hardware info, system stats, and jobs
          StatusBar(systemInfo: _systemInfo),
          // Tab content
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                SafeScreen(
                  screenName: 'Benchmarks',
                  child: BenchmarksScreen(backendReady: _backendReady),
                ),
                SafeScreen(screenName: 'Circuits', child: CircuitsScreen()),
                SafeScreen(screenName: 'History', child: HistoryScreen()),
                SafeScreen(screenName: 'Settings', child: SettingsScreen()),
                SafeScreen(screenName: 'MCP', child: McpScreen()),
                SafeScreen(screenName: 'About', child: AboutScreen()),
              ],
            ),
          ),
          // Logs Panel at the bottom
          LogsPanel(
            isCollapsed: _logsCollapsed,
            onToggleCollapse: () {
              setState(() => _logsCollapsed = !_logsCollapsed);
            },
            height: _logsHeight,
            onHeightChanged: (height) {
              setState(() => _logsHeight = height);
            },
          ),
        ],
      ),
    );
  }
}

class SafeScreen extends StatelessWidget {
  const SafeScreen({super.key, required this.screenName, required this.child});

  final String screenName;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    try {
      return child;
    } catch (error, stackTrace) {
      logger.error(
        'safe_screen',
        'Screen $screenName failed to render: $error\n$stackTrace',
      );
      return _ScreenFailureCard(screenName: screenName);
    }
  }
}

class _AppFailureFallback extends StatelessWidget {
  const _AppFailureFallback({required this.details});

  final FlutterErrorDetails details;

  @override
  Widget build(BuildContext context) {
    return Material(
      color: const Color(0xFFFDF2F2),
      child: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 520),
            child: _ScreenFailureCard(
              screenName: 'Application',
              detail: details.exceptionAsString(),
            ),
          ),
        ),
      ),
    );
  }
}

class _ScreenFailureCard extends StatelessWidget {
  const _ScreenFailureCard({required this.screenName, this.detail});

  final String screenName;
  final String? detail;

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final onSurface = colorScheme.onSurface;
    return Container(
      decoration: BoxDecoration(
        color: colorScheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: colorScheme.error.withValues(alpha: 0.35)),
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.error_outline, color: colorScheme.error),
              const SizedBox(width: 10),
              Text(
                '$screenName crashed',
                style: TextStyle(
                  color: onSurface,
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'An internal UI error was caught to keep the app responsive. '
            'Please open the logs panel for details.',
            style: TextStyle(color: onSurface.withValues(alpha: 0.8)),
          ),
          if (detail != null) ...[
            const SizedBox(height: 10),
            SelectableText(
              detail!,
              style: TextStyle(color: colorScheme.error, fontSize: 12),
            ),
          ],
        ],
      ),
    );
  }
}
