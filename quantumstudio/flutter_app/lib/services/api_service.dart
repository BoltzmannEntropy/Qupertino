import 'dart:convert';

import 'package:http/http.dart' as http;

class ApiException implements Exception {
  ApiException({required this.message, this.statusCode, this.body, this.path});

  final String message;
  final int? statusCode;
  final String? body;
  final String? path;

  @override
  String toString() {
    final parts = <String>[message];
    if (statusCode != null) {
      parts.add('status=$statusCode');
    }
    if (path != null) {
      parts.add('path=$path');
    }
    return 'ApiException(${parts.join(', ')})';
  }
}

class ApiService {
  static const String _configuredBaseUrl = String.fromEnvironment(
    'QS_BACKEND_URL',
    defaultValue: '',
  );
  static const String _host = String.fromEnvironment(
    'QS_BACKEND_HOST',
    defaultValue: '127.0.0.1',
  );
  static const String _port = String.fromEnvironment(
    'QS_BACKEND_PORT',
    defaultValue: '8127',
  );
  static String _runtimeBaseUrl = '';

  static String get baseUrl {
    if (_runtimeBaseUrl.isNotEmpty) {
      return _runtimeBaseUrl;
    }
    if (_configuredBaseUrl.isNotEmpty) {
      return _configuredBaseUrl;
    }
    return 'http://$_host:$_port';
  }

  static void setRuntimeBaseUrl(String url) {
    _runtimeBaseUrl = url.trim().replaceAll(RegExp(r'/$'), '');
  }

  static const Duration _defaultTimeout = Duration(seconds: 30);
  static const Duration _healthTimeout = Duration(seconds: 5);
  static const Duration _uploadTimeout = Duration(seconds: 120);

  Future<bool> checkHealth() async {
    try {
      final response = await _request(
        method: 'GET',
        path: '/api/health',
        timeout: _healthTimeout,
      );
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  Future<Map<String, dynamic>> getSystemInfo() async {
    final response = await _request(
      method: 'GET',
      path: '/api/system/info',
      timeout: _defaultTimeout,
    );
    return _decodeJsonObject(response, '/api/system/info');
  }

  Future<Map<String, dynamic>?> getSystemStats() async {
    try {
      final response = await _request(
        method: 'GET',
        path: '/api/system/stats',
        timeout: _healthTimeout,
      );
      if (response.statusCode != 200) {
        return null;
      }
      return _decodeJsonObject(response, '/api/system/stats');
    } catch (_) {
      return null;
    }
  }

  Future<List<Map<String, dynamic>>> getBenchmarks() async {
    final response = await _request(
      method: 'GET',
      path: '/api/benchmarks',
      timeout: _defaultTimeout,
    );
    final data = _decodeJsonObject(response, '/api/benchmarks');
    return _decodeJsonList(data, key: 'benchmarks', path: '/api/benchmarks');
  }

  Future<Map<String, dynamic>> startRun({
    required List<Map<String, dynamic>> benchmarkConfigs,
    bool savePlots = true,
    int? maxQubits,
    int? qasmMaxQubits,
    int? qasmTimeoutMs,
    int? qasmMaxMemMb,
    bool qasmIncludeLarge = false,
    int? qasmSimulateLimit,
    bool benchpress = false,
    Map<String, String> envOverrides = const {},
  }) async {
    final response = await _request(
      method: 'POST',
      path: '/api/runs',
      timeout: _defaultTimeout,
      body: {
        'benchmark_configs': benchmarkConfigs,
        'save_plots': savePlots,
        if (maxQubits != null) 'max_qubits': maxQubits,
        if (qasmMaxQubits != null) 'qasm_max_qubits': qasmMaxQubits,
        if (qasmTimeoutMs != null) 'qasm_timeout_ms': qasmTimeoutMs,
        if (qasmMaxMemMb != null) 'qasm_max_mem_mb': qasmMaxMemMb,
        'qasm_include_large': qasmIncludeLarge,
        if (qasmSimulateLimit != null) 'qasm_simulate_limit': qasmSimulateLimit,
        'benchpress': benchpress,
        if (envOverrides.isNotEmpty) 'env_overrides': envOverrides,
      },
    );
    return _decodeJsonObject(response, '/api/runs');
  }

  Future<List<Map<String, dynamic>>> listRuns() async {
    final response = await _request(
      method: 'GET',
      path: '/api/runs',
      timeout: _defaultTimeout,
    );
    final data = _decodeJsonObject(response, '/api/runs');
    return _decodeJsonList(data, key: 'runs', path: '/api/runs');
  }

  Future<Map<String, dynamic>> getRun(String runId) async {
    final path = '/api/runs/$runId';
    final response = await _request(
      method: 'GET',
      path: path,
      timeout: _defaultTimeout,
    );
    return _decodeJsonObject(response, path);
  }

  Future<String> getRunLog(String runId) async {
    final path = '/api/runs/$runId/log';
    final response = await _request(
      method: 'GET',
      path: path,
      timeout: _defaultTimeout,
    );
    if (response.statusCode == 200) {
      return response.body;
    }
    throw _httpFailure(path: path, response: response);
  }

  Future<String> fetchText(String url) async {
    final resolved = resolveUrl(url);
    final response = await _requestAbsolute(
      method: 'GET',
      url: resolved,
      timeout: _defaultTimeout,
    );
    if (response.statusCode == 200) {
      return response.body;
    }
    throw _httpFailure(path: resolved, response: response);
  }

  String resolveUrl(String path) {
    if (path.startsWith('http')) {
      return path;
    }
    return '$baseUrl$path';
  }

  Future<Map<String, dynamic>> stopRun(String runId) async {
    final path = '/api/runs/$runId/stop';
    final response = await _request(
      method: 'POST',
      path: path,
      timeout: _defaultTimeout,
    );
    return _decodeJsonObject(response, path);
  }

  Future<Map<String, dynamic>> cancelRun(String runId) async {
    final path = '/api/runs/$runId/cancel';
    final response = await _request(
      method: 'POST',
      path: path,
      timeout: _defaultTimeout,
    );
    return _decodeJsonObject(response, path);
  }

  Future<void> deleteRun(String runId) async {
    final path = '/api/runs/$runId';
    final response = await _request(
      method: 'DELETE',
      path: path,
      timeout: _defaultTimeout,
    );
    if (response.statusCode != 200) {
      throw _httpFailure(path: path, response: response);
    }
  }

  Future<Map<String, dynamic>> getQueueStatus() async {
    final path = '/api/queue';
    final response = await _request(
      method: 'GET',
      path: path,
      timeout: _healthTimeout,
    );
    return _decodeJsonObject(response, path);
  }

  Future<Map<String, dynamic>> stopAllJobs() async {
    final path = '/api/queue/stop-all';
    final response = await _request(
      method: 'POST',
      path: path,
      timeout: _defaultTimeout,
    );
    return _decodeJsonObject(response, path);
  }

  Future<Map<String, dynamic>> getSettings() async {
    final path = '/api/settings';
    final response = await _request(
      method: 'GET',
      path: path,
      timeout: _defaultTimeout,
    );
    return _decodeJsonObject(response, path);
  }

  Future<Map<String, dynamic>> updateSettings({int? maxConcurrentJobs}) async {
    final path = '/api/settings';
    final response = await _request(
      method: 'PUT',
      path: path,
      timeout: _defaultTimeout,
      body: {
        if (maxConcurrentJobs != null) 'max_concurrent_jobs': maxConcurrentJobs,
      },
    );
    return _decodeJsonObject(response, path);
  }

  Future<List<Map<String, dynamic>>> listQasmFiles() async {
    final path = '/api/qasm/files';
    final response = await _request(
      method: 'GET',
      path: path,
      timeout: _defaultTimeout,
    );
    final data = _decodeJsonObject(response, path);
    return _decodeJsonList(data, key: 'files', path: path);
  }

  Future<Map<String, dynamic>> getQasmFile(String filename) async {
    final path = '/api/qasm/files/$filename';
    final response = await _request(
      method: 'GET',
      path: path,
      timeout: _defaultTimeout,
    );
    return _decodeJsonObject(response, path);
  }

  Future<Map<String, dynamic>> parseQasm(
    String content, {
    String? filename,
  }) async {
    const path = '/api/qasm/parse';
    final response = await _request(
      method: 'POST',
      path: path,
      timeout: _defaultTimeout,
      body: {'content': content, if (filename != null) 'filename': filename},
    );
    return _decodeJsonObject(response, path);
  }

  Future<Map<String, dynamic>> visualizeQasmAscii({
    String? content,
    String? filename,
  }) async {
    const path = '/api/qasm/visualize/ascii';
    final response = await _request(
      method: 'POST',
      path: path,
      timeout: _defaultTimeout,
      body: {
        if (content != null) 'content': content,
        if (filename != null) 'filename': filename,
      },
    );
    return _decodeJsonObject(response, path);
  }

  Future<Map<String, dynamic>> visualizeQasmImage({
    String? content,
    String? filename,
    String theme = 'apple',
  }) async {
    const path = '/api/qasm/visualize/image';
    final response = await _request(
      method: 'POST',
      path: path,
      timeout: _uploadTimeout,
      body: {
        if (content != null) 'content': content,
        if (filename != null) 'filename': filename,
        'theme': theme,
      },
    );
    return _decodeJsonObject(response, path);
  }

  Future<http.Response> _request({
    required String method,
    required String path,
    Duration timeout = _defaultTimeout,
    Map<String, dynamic>? body,
  }) {
    return _requestAbsolute(
      method: method,
      url: '$baseUrl$path',
      timeout: timeout,
      body: body,
      pathLabel: path,
    );
  }

  Future<http.Response> _requestAbsolute({
    required String method,
    required String url,
    Duration timeout = _defaultTimeout,
    Map<String, dynamic>? body,
    String? pathLabel,
  }) async {
    final uri = Uri.parse(url);
    final client = http.Client();
    final requestPath = pathLabel ?? uri.path;
    try {
      late final http.Response response;
      switch (method) {
        case 'GET':
          response = await client.get(uri).timeout(timeout);
          break;
        case 'POST':
          response = await client
              .post(
                uri,
                headers: {'Content-Type': 'application/json'},
                body: body == null ? null : json.encode(body),
              )
              .timeout(timeout);
          break;
        case 'PUT':
          response = await client
              .put(
                uri,
                headers: {'Content-Type': 'application/json'},
                body: body == null ? null : json.encode(body),
              )
              .timeout(timeout);
          break;
        case 'DELETE':
          response = await client.delete(uri).timeout(timeout);
          break;
        default:
          throw ApiException(
            message: 'Unsupported HTTP method: $method',
            path: requestPath,
          );
      }

      if (response.statusCode >= 400) {
        throw _httpFailure(path: requestPath, response: response);
      }
      return response;
    } on ApiException {
      rethrow;
    } catch (e) {
      throw ApiException(message: 'Request failed: $e', path: requestPath);
    } finally {
      client.close();
    }
  }

  ApiException _httpFailure({
    required String path,
    required http.Response response,
  }) {
    return ApiException(
      message:
          _extractServerError(response.body) ??
          'Request failed with status ${response.statusCode}',
      statusCode: response.statusCode,
      body: response.body,
      path: path,
    );
  }

  Map<String, dynamic> _decodeJsonObject(http.Response response, String path) {
    try {
      final dynamic data = json.decode(response.body);
      if (data is Map<String, dynamic>) {
        return data;
      }
      throw ApiException(
        message: 'Expected JSON object response',
        statusCode: response.statusCode,
        body: response.body,
        path: path,
      );
    } catch (e) {
      if (e is ApiException) {
        rethrow;
      }
      throw ApiException(
        message: 'Invalid JSON response: $e',
        statusCode: response.statusCode,
        body: response.body,
        path: path,
      );
    }
  }

  List<Map<String, dynamic>> _decodeJsonList(
    Map<String, dynamic> data, {
    required String key,
    required String path,
  }) {
    final dynamic raw = data[key];
    if (raw is! List) {
      throw ApiException(message: 'Expected list at key "$key"', path: path);
    }
    try {
      return raw.map((item) => Map<String, dynamic>.from(item as Map)).toList();
    } catch (e) {
      throw ApiException(
        message: 'Invalid list payload for "$key": $e',
        path: path,
      );
    }
  }

  String? _extractServerError(String body) {
    if (body.trim().isEmpty) {
      return null;
    }
    try {
      final decoded = json.decode(body);
      if (decoded is Map<String, dynamic>) {
        final detail = decoded['detail'];
        if (detail is String && detail.trim().isNotEmpty) {
          return detail.trim();
        }
        final message = decoded['message'];
        if (message is String && message.trim().isNotEmpty) {
          return message.trim();
        }
      }
    } catch (_) {
      // Fallback to raw body snippet.
    }
    if (body.length > 240) {
      return '${body.substring(0, 240)}...';
    }
    return body;
  }
}
