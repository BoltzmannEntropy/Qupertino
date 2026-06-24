import 'dart:convert';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import '../services/api_service.dart';
import '../services/backend_service.dart';

class CircuitsScreen extends StatefulWidget {
  const CircuitsScreen({super.key});

  @override
  State<CircuitsScreen> createState() => _CircuitsScreenState();
}

class _CircuitsScreenState extends State<CircuitsScreen> {
  final ApiService _api = ApiService();
  bool _loading = true;
  List<Map<String, dynamic>> _qasmFiles = [];
  String? _selectedFile;
  String _qasmContent = '';
  String _asciiDiagram = '';
  String? _imageBase64;
  int _nQubits = 0;
  int _nOps = 0;
  String? _error;
  bool _visualizing = false;
  bool _isRecoveringBackend = false;

  // View mode: 'ascii' or 'image'
  String _viewMode = 'image';

  @override
  void initState() {
    super.initState();
    _loadQasmFiles();
  }

  Future<void> _loadQasmFiles() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final files = await _api.listQasmFiles();
      if (!mounted) return;
      setState(() {
        _qasmFiles = files;
        _loading = false;
      });
    } catch (e) {
      if (_isBackendConnectionError(e)) {
        final recovered = await _recoverBackend();
        if (recovered) {
          try {
            final files = await _api.listQasmFiles();
            if (!mounted) return;
            setState(() {
              _qasmFiles = files;
              _loading = false;
              _error = null;
            });
            return;
          } catch (_) {
            // Fall through to show the original error context below.
          }
        }
      }
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load QASM files: $e';
        _loading = false;
      });
    }
  }

  Future<void> _selectFile(String filename, {int retryCount = 1}) async {
    setState(() {
      _selectedFile = filename;
      _error = null;
      _visualizing = true;
    });
    try {
      // Load content
      final fileData = await _api.getQasmFile(filename);
      final content = fileData['content'] as String;
      setState(() {
        _qasmContent = content;
      });

      // Visualize
      await _visualize(content: content, filename: filename);
    } catch (e) {
      if (_isBackendConnectionError(e) && retryCount > 0) {
        final recovered = await _recoverBackend();
        if (recovered) {
          await _selectFile(filename, retryCount: retryCount - 1);
          return;
        }
      }
      if (!mounted) return;
      setState(() {
        _error = 'Failed to load file: $e';
        _visualizing = false;
      });
    }
  }

  Future<void> _visualize({
    String? content,
    String? filename,
    int retryCount = 1,
  }) async {
    setState(() {
      _visualizing = true;
      _error = null;
    });
    try {
      // Get both ASCII and image visualizations
      final results = await Future.wait([
        _api.visualizeQasmAscii(
          content: content,
          filename: content == null ? filename : null,
        ),
        _api.visualizeQasmImage(
          content: content,
          filename: content == null ? filename : null,
          theme: 'apple',
        ),
      ]);

      if (!mounted) return;
      final asciiResult = results[0];
      final imageResult = results[1];

      setState(() {
        _asciiDiagram = asciiResult['ascii'] as String? ?? '';
        _nQubits = asciiResult['n_qubits'] as int? ?? 0;
        _nOps = asciiResult['n_ops'] as int? ?? 0;
        _imageBase64 = imageResult['image_base64'] as String?;
        _visualizing = false;
      });
    } catch (e) {
      if (_isBackendConnectionError(e) && retryCount > 0) {
        final recovered = await _recoverBackend();
        if (recovered) {
          await _visualize(
            content: content,
            filename: filename,
            retryCount: retryCount - 1,
          );
          return;
        }
      }
      if (!mounted) return;
      setState(() {
        _error = 'Visualization failed: $e';
        _visualizing = false;
      });
    }
  }

  Future<void> _visualizeCustom() async {
    if (_qasmContent.trim().isEmpty) {
      setState(() {
        _error = 'Enter QASM code to visualize';
      });
      return;
    }
    setState(() {
      _selectedFile = null;
    });
    await _visualize(content: _qasmContent);
  }

  bool _isBackendConnectionError(Object error) {
    final msg = error.toString().toLowerCase();
    return msg.contains('connection refused') ||
        msg.contains('socketexception') ||
        msg.contains('failed host lookup') ||
        msg.contains('connection reset by peer');
  }

  Future<bool> _recoverBackend() async {
    if (_isRecoveringBackend) return false;
    _isRecoveringBackend = true;
    try {
      final service = BackendService();
      final started = await service.ensureStarted();
      if (!started && mounted) {
        final detail = service.lastStartupError;
        setState(() {
          _error = detail == null
              ? 'Backend is unavailable. Start backend and try again.'
              : 'Backend is unavailable: $detail';
        });
      }
      return started;
    } finally {
      _isRecoveringBackend = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    return Row(
      children: [
        // Left sidebar: file list
        _buildFileSidebar(colorScheme),
        // Main content area
        Expanded(
          child: Column(
            children: [
              // QASM editor
              Expanded(
                flex: 2,
                child: _buildEditorPanel(colorScheme),
              ),
              const Divider(height: 1),
              // Visualization panel
              Expanded(
                flex: 3,
                child: _buildVisualizationPanel(colorScheme),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildFileSidebar(ColorScheme colorScheme) {
    return Container(
      width: 240,
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHighest,
        border: Border(right: BorderSide(color: colorScheme.outlineVariant)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 16, 12, 8),
            child: Row(
              children: [
                const Icon(CupertinoIcons.doc_text, size: 18),
                const SizedBox(width: 8),
                const Text('QASM Files',
                    style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700)),
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                  decoration: BoxDecoration(
                    color: colorScheme.primary.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Text('${_qasmFiles.length}',
                      style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          color: colorScheme.primary)),
                ),
              ],
            ),
          ),
          const Divider(height: 1),
          // File list
          Expanded(
            child: _qasmFiles.isEmpty
                ? Center(
                    child: Text(
                      'No QASM files found',
                      style: TextStyle(color: colorScheme.onSurfaceVariant),
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 4),
                    itemCount: _qasmFiles.length,
                    itemBuilder: (context, index) {
                      final file = _qasmFiles[index];
                      final name = file['name'] as String? ?? '';
                      final size = file['size'] as int? ?? 0;
                      final isSelected = name == _selectedFile;

                      return InkWell(
                        onTap: () => _selectFile(name),
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 14, vertical: 8),
                          decoration: BoxDecoration(
                            color: isSelected
                                ? colorScheme.primary.withValues(alpha: 0.10)
                                : null,
                            border: Border(
                              left: BorderSide(
                                color: isSelected
                                    ? colorScheme.primary
                                    : Colors.transparent,
                                width: 3,
                              ),
                            ),
                          ),
                          child: Row(
                            children: [
                              Icon(
                                CupertinoIcons.doc_text_fill,
                                size: 16,
                                color: isSelected
                                    ? colorScheme.primary
                                    : colorScheme.onSurfaceVariant,
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  name,
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: isSelected
                                        ? FontWeight.w600
                                        : FontWeight.w400,
                                    color: isSelected
                                        ? colorScheme.primary
                                        : colorScheme.onSurface,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                              Text(
                                _formatSize(size),
                                style: TextStyle(
                                  fontSize: 10,
                                  color: colorScheme.onSurfaceVariant,
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
          ),
          // Refresh button
          Padding(
            padding: const EdgeInsets.all(12),
            child: TextButton.icon(
              onPressed: _loadQasmFiles,
              icon: const Icon(CupertinoIcons.refresh, size: 14),
              label: const Text('Refresh'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEditorPanel(ColorScheme colorScheme) {
    return Container(
      color: colorScheme.surfaceContainerLowest,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Editor header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: colorScheme.surfaceContainer,
              border: Border(bottom: BorderSide(color: colorScheme.outlineVariant)),
            ),
            child: Row(
              children: [
                Icon(CupertinoIcons.chevron_left_slash_chevron_right,
                    size: 16, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  _selectedFile ?? 'QASM Editor',
                  style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
                ),
                if (_nQubits > 0) ...[
                  const SizedBox(width: 12),
                  _buildInfoChip('${_nQubits}q', colorScheme),
                  const SizedBox(width: 6),
                  _buildInfoChip('$_nOps ops', colorScheme),
                ],
                const Spacer(),
                FilledButton.icon(
                  onPressed: _visualizing ? null : _visualizeCustom,
                  icon: _visualizing
                      ? const SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(CupertinoIcons.play_fill, size: 14),
                  label: const Text('Visualize'),
                ),
              ],
            ),
          ),
          // Code editor
          Expanded(
            child: TextField(
              maxLines: null,
              expands: true,
              style: const TextStyle(
                fontFamily: 'Menlo',
                fontSize: 12,
                height: 1.4,
              ),
              decoration: const InputDecoration(
                contentPadding: EdgeInsets.all(12),
                border: InputBorder.none,
                hintText: 'Enter or paste QASM code here...',
              ),
              controller: TextEditingController(text: _qasmContent),
              onChanged: (value) {
                _qasmContent = value;
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildVisualizationPanel(ColorScheme colorScheme) {
    return Container(
      color: colorScheme.surface,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Visualization header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            decoration: BoxDecoration(
              color: colorScheme.surfaceContainer,
              border: Border(bottom: BorderSide(color: colorScheme.outlineVariant)),
            ),
            child: Row(
              children: [
                const Icon(CupertinoIcons.graph_circle, size: 16),
                const SizedBox(width: 8),
                const Text(
                  'Circuit Visualization',
                  style: TextStyle(fontWeight: FontWeight.w600, fontSize: 13),
                ),
                const Spacer(),
                // View mode toggle
                SegmentedButton<String>(
                  segments: const [
                    ButtonSegment(
                      value: 'image',
                      label: Text('Image'),
                      icon: Icon(CupertinoIcons.photo, size: 14),
                    ),
                    ButtonSegment(
                      value: 'ascii',
                      label: Text('ASCII'),
                      icon: Icon(CupertinoIcons.text_alignleft, size: 14),
                    ),
                  ],
                  selected: {_viewMode},
                  onSelectionChanged: (value) {
                    setState(() => _viewMode = value.first);
                  },
                  style: const ButtonStyle(
                    visualDensity: VisualDensity.compact,
                  ),
                ),
              ],
            ),
          ),
          // Visualization content
          Expanded(
            child: _buildVisualizationContent(colorScheme),
          ),
        ],
      ),
    );
  }

  Widget _buildVisualizationContent(ColorScheme colorScheme) {
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(CupertinoIcons.exclamationmark_triangle,
                size: 48, color: colorScheme.error),
            const SizedBox(height: 12),
            Text(
              _error!,
              style: TextStyle(color: colorScheme.error),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    if (_visualizing) {
      return const Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 12),
            Text('Generating visualization...'),
          ],
        ),
      );
    }

    if (_viewMode == 'image' && _imageBase64 != null) {
      return InteractiveViewer(
        minScale: 0.5,
        maxScale: 4,
        child: Center(
          child: Container(
            padding: const EdgeInsets.all(24),
            child: Image.memory(
              base64Decode(_imageBase64!),
              fit: BoxFit.contain,
            ),
          ),
        ),
      );
    }

    if (_viewMode == 'ascii' && _asciiDiagram.isNotEmpty) {
      return Container(
        padding: const EdgeInsets.all(16),
        color: const Color(0xFF1E1E1E),
        child: SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          child: SingleChildScrollView(
            child: SelectableText(
              _asciiDiagram,
              style: const TextStyle(
                fontFamily: 'Menlo',
                fontSize: 13,
                color: Color(0xFF9BD3FF),
                height: 1.3,
              ),
            ),
          ),
        ),
      );
    }

    // Empty state
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            CupertinoIcons.graph_square,
            size: 64,
            color: colorScheme.onSurface.withValues(alpha: 0.15),
          ),
          const SizedBox(height: 16),
          Text(
            'Select a QASM file or enter code\nto visualize the circuit',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: colorScheme.onSurface.withValues(alpha: 0.4),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoChip(String label, ColorScheme colorScheme) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: colorScheme.primary.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: colorScheme.primary,
        ),
      ),
    );
  }

  String _formatSize(int bytes) {
    if (bytes < 1024) return '${bytes}B';
    if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)}KB';
    return '${(bytes / (1024 * 1024)).toStringAsFixed(1)}MB';
  }
}
