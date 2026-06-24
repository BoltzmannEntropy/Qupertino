import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

enum LogLevel { debug, info, warning, error }

class LogEntry {
  final DateTime timestamp;
  final LogLevel level;
  final String source;
  final String message;

  LogEntry({
    required this.timestamp,
    required this.level,
    required this.source,
    required this.message,
  });

  String get levelName {
    switch (level) {
      case LogLevel.debug:
        return 'DEBUG';
      case LogLevel.info:
        return 'INFO';
      case LogLevel.warning:
        return 'WARN';
      case LogLevel.error:
        return 'ERROR';
    }
  }

  String toLogLine() {
    final time = '${timestamp.hour.toString().padLeft(2, '0')}:'
        '${timestamp.minute.toString().padLeft(2, '0')}:'
        '${timestamp.second.toString().padLeft(2, '0')}';
    return '$time | $levelName | $source | $message';
  }
}

class LogService extends ChangeNotifier {
  static final LogService _instance = LogService._internal();
  static LogService get instance => _instance;
  factory LogService() => _instance;
  LogService._internal();

  final List<LogEntry> _logs = [];
  List<LogEntry> get logs => List.unmodifiable(_logs);

  int maxLogs = 500;
  LogLevel minLevel = LogLevel.debug;

  void log(LogLevel level, String source, String message) {
    if (level.index < minLevel.index) return;

    final entry = LogEntry(
      timestamp: DateTime.now(),
      level: level,
      source: source,
      message: message,
    );
    _logs.add(entry);

    // Trim oldest logs if over limit
    while (_logs.length > maxLogs) {
      _logs.removeAt(0);
    }

    notifyListeners();
  }

  void debug(String source, String message) => log(LogLevel.debug, source, message);
  void info(String source, String message) => log(LogLevel.info, source, message);
  void warning(String source, String message) => log(LogLevel.warning, source, message);
  void error(String source, String message) => log(LogLevel.error, source, message);

  void clear() {
    _logs.clear();
    notifyListeners();
  }

  Future<String?> export() async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
      final file = File('${directory.path}/quantumstudio_logs_$timestamp.txt');
      final content = _logs.map((e) => e.toLogLine()).join('\n');
      await file.writeAsString(content);
      info('log_service', 'Logs exported to ${file.path}');
      return file.path;
    } catch (e) {
      error('log_service', 'Failed to export logs: $e');
      return null;
    }
  }

  List<LogEntry> filtered({LogLevel? level}) {
    if (level == null) return logs;
    return _logs.where((e) => e.level.index >= level.index).toList();
  }
}

// Global logger instance
final logger = LogService.instance;
