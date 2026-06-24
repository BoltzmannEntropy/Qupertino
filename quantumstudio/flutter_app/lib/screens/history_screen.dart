import 'dart:convert';
import 'package:flutter/material.dart';
import '../services/api_service.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  final _api = ApiService();

  List<Map<String, dynamic>> _allRuns = [];
  List<Map<String, dynamic>> _filteredRuns = [];
  List<Map<String, dynamic>> _benchmarks = [];
  bool _loading = true;
  String? _error;

  // Filters
  String _statusFilter = 'all';
  String _benchmarkFilter = 'all';
  String _searchQuery = '';
  String _sortBy = 'newest';

  // Selected run for detail view
  Map<String, dynamic>? _selectedRun;
  String _selectedRunLog = '';
  bool _loadingDetail = false;

  // Selection for bulk operations
  final Set<String> _selectedIds = {};

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final results = await Future.wait([
        _api.listRuns(),
        _api.getBenchmarks(),
      ]);

      setState(() {
        _allRuns = results[0];
        _benchmarks = results[1];
        _applyFilters();
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  void _applyFilters() {
    var runs = List<Map<String, dynamic>>.from(_allRuns);

    // Filter out active jobs (queued/running) - they belong on the Benchmarks tab
    runs = runs.where((r) {
      final status = r['status'] as String? ?? '';
      return !['queued', 'running'].contains(status);
    }).toList();

    // Status filter
    if (_statusFilter != 'all') {
      runs = runs.where((r) => r['status'] == _statusFilter).toList();
    }

    // Benchmark filter
    if (_benchmarkFilter != 'all') {
      runs = runs.where((r) {
        final benchmarks = r['benchmarks'] as List<dynamic>? ?? [];
        return benchmarks.contains(_benchmarkFilter);
      }).toList();
    }

    // Search filter
    if (_searchQuery.isNotEmpty) {
      final query = _searchQuery.toLowerCase();
      runs = runs.where((r) {
        final id = (r['id'] as String? ?? '').toLowerCase();
        final benchmarks = (r['benchmarks'] as List<dynamic>? ?? [])
            .map((b) => b.toString().toLowerCase())
            .toList();
        return id.contains(query) || benchmarks.any((b) => b.contains(query));
      }).toList();
    }

    // Sorting
    runs.sort((a, b) {
      switch (_sortBy) {
        case 'oldest':
          return (a['id'] as String).compareTo(b['id'] as String);
        case 'duration_asc':
          return _getDuration(a).compareTo(_getDuration(b));
        case 'duration_desc':
          return _getDuration(b).compareTo(_getDuration(a));
        case 'newest':
        default:
          return (b['id'] as String).compareTo(a['id'] as String);
      }
    });

    _filteredRuns = runs;
  }

  Duration _getDuration(Map<String, dynamic> run) {
    final started = run['started_at'] as String?;
    final ended = run['ended_at'] as String?;
    if (started == null || ended == null) return Duration.zero;
    try {
      return DateTime.parse(ended).difference(DateTime.parse(started));
    } catch (_) {
      return Duration.zero;
    }
  }

  String _formatDuration(Duration d) {
    if (d == Duration.zero) return '-';
    final hours = d.inHours;
    final minutes = d.inMinutes % 60;
    final seconds = d.inSeconds % 60;
    if (hours > 0) return '${hours}h ${minutes}m';
    if (minutes > 0) return '${minutes}m ${seconds}s';
    return '${seconds}s';
  }

  String _formatDateTime(String? isoString) {
    if (isoString == null) return '-';
    try {
      final dt = DateTime.parse(isoString).toLocal();
      final now = DateTime.now();
      final diff = now.difference(dt);

      if (diff.inDays == 0) {
        return 'Today ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
      } else if (diff.inDays == 1) {
        return 'Yesterday ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
      } else {
        return '${dt.month}/${dt.day} ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
      }
    } catch (_) {
      return isoString;
    }
  }

  Widget _buildStatusBadge(String status, {ColorScheme? colorScheme}) {
    final scheme = colorScheme ?? ColorScheme.fromSeed(seedColor: Colors.blue);
    Color color;
    IconData icon;
    String label;

    switch (status) {
      case 'completed':
        color = Colors.green;
        icon = Icons.check_circle;
        label = 'Done';
        break;
      case 'failed':
        color = scheme.error;
        icon = Icons.error;
        label = 'Failed';
        break;
      case 'stopped':
        color = Colors.orange;
        icon = Icons.stop_circle;
        label = 'Stopped';
        break;
      case 'cancelled':
        color = Colors.orange;
        icon = Icons.cancel;
        label = 'Cancelled';
        break;
      default:
        color = scheme.outline;
        icon = Icons.help;
        label = status;
    }

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(width: 4),
        Text(
          label,
          style: TextStyle(color: color, fontWeight: FontWeight.w500),
        ),
      ],
    );
  }

  Future<void> _viewRun(Map<String, dynamic> run) async {
    setState(() {
      _selectedRun = run;
      _loadingDetail = true;
    });

    try {
      final log = await _api.getRunLog(run['id'] as String);
      setState(() {
        _selectedRunLog = log;
        _loadingDetail = false;
      });
    } catch (e) {
      setState(() {
        _selectedRunLog = 'Failed to load log: $e';
        _loadingDetail = false;
      });
    }
  }

  Future<void> _deleteRun(String runId) async {
    final colorScheme = Theme.of(context).colorScheme;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete Run'),
        content: const Text(
          'Are you sure you want to delete this run? This cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: colorScheme.error),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      await _api.deleteRun(runId);
      setState(() {
        if (_selectedRun?['id'] == runId) {
          _selectedRun = null;
        }
        _selectedIds.remove(runId);
      });
      await _loadData();
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to delete: $e')));
      }
    }
  }

  Future<void> _deleteSelected() async {
    if (_selectedIds.isEmpty) return;

    final colorScheme = Theme.of(context).colorScheme;
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete Selected Runs'),
        content: Text(
          'Are you sure you want to delete ${_selectedIds.length} runs? This cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: FilledButton.styleFrom(backgroundColor: colorScheme.error),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    for (final id in _selectedIds.toList()) {
      try {
        await _api.deleteRun(id);
      } catch (_) {}
    }

    setState(() {
      _selectedRun = null;
      _selectedIds.clear();
    });
    await _loadData();
  }

  Future<void> _exportSelected() async {
    if (_selectedIds.isEmpty) return;

    final selectedRuns = _allRuns
        .where((r) => _selectedIds.contains(r['id']))
        .toList();
    final jsonStr = const JsonEncoder.withIndent('  ').convert(selectedRuns);

    // Show export dialog with copyable content
    await showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Export Runs'),
        content: SizedBox(
          width: 500,
          height: 400,
          child: SingleChildScrollView(
            child: SelectableText(
              jsonStr,
              style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Center(child: CircularProgressIndicator());
    }

    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Error: $_error'),
            const SizedBox(height: 16),
            FilledButton(onPressed: _loadData, child: const Text('Retry')),
          ],
        ),
      );
    }

    final colorScheme = Theme.of(context).colorScheme;

    return Row(
      children: [
        // Main list
        Expanded(
          flex: 2,
          child: Column(
            children: [
              // Filters bar
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: colorScheme.surface,
                  border: Border(
                    bottom: BorderSide(color: colorScheme.outlineVariant),
                  ),
                ),
                child: Column(
                  children: [
                    Row(
                      children: [
                        // Status filter
                        SizedBox(
                          width: 150,
                          child: DropdownButtonFormField<String>(
                            initialValue: _statusFilter,
                            decoration: const InputDecoration(
                              labelText: 'Status',
                              contentPadding: EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 8,
                              ),
                            ),
                            items: const [
                              DropdownMenuItem(
                                value: 'all',
                                child: Text('All Statuses'),
                              ),
                              DropdownMenuItem(
                                value: 'completed',
                                child: Text('Completed'),
                              ),
                              DropdownMenuItem(
                                value: 'failed',
                                child: Text('Failed'),
                              ),
                              DropdownMenuItem(
                                value: 'stopped',
                                child: Text('Stopped'),
                              ),
                              DropdownMenuItem(
                                value: 'cancelled',
                                child: Text('Cancelled'),
                              ),
                            ],
                            onChanged: (v) {
                              setState(() {
                                _statusFilter = v ?? 'all';
                                _applyFilters();
                              });
                            },
                          ),
                        ),
                        const SizedBox(width: 12),

                        // Benchmark filter
                        SizedBox(
                          width: 180,
                          child: DropdownButtonFormField<String>(
                            initialValue: _benchmarkFilter,
                            decoration: const InputDecoration(
                              labelText: 'Benchmark',
                              contentPadding: EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 8,
                              ),
                            ),
                            items: [
                              const DropdownMenuItem(
                                value: 'all',
                                child: Text('All Benchmarks'),
                              ),
                              ..._benchmarks.map(
                                (b) => DropdownMenuItem(
                                  value: b['name'] as String,
                                  child: Text(b['label'] as String),
                                ),
                              ),
                            ],
                            onChanged: (v) {
                              setState(() {
                                _benchmarkFilter = v ?? 'all';
                                _applyFilters();
                              });
                            },
                          ),
                        ),
                        const SizedBox(width: 12),

                        // Search
                        Expanded(
                          child: TextFormField(
                            decoration: const InputDecoration(
                              labelText: 'Search',
                              prefixIcon: Icon(Icons.search, size: 20),
                              contentPadding: EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 8,
                              ),
                            ),
                            onChanged: (v) {
                              setState(() {
                                _searchQuery = v;
                                _applyFilters();
                              });
                            },
                          ),
                        ),
                        const SizedBox(width: 12),

                        // Sort
                        SizedBox(
                          width: 150,
                          child: DropdownButtonFormField<String>(
                            initialValue: _sortBy,
                            decoration: const InputDecoration(
                              labelText: 'Sort',
                              contentPadding: EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 8,
                              ),
                            ),
                            items: const [
                              DropdownMenuItem(
                                value: 'newest',
                                child: Text('Newest First'),
                              ),
                              DropdownMenuItem(
                                value: 'oldest',
                                child: Text('Oldest First'),
                              ),
                              DropdownMenuItem(
                                value: 'duration_desc',
                                child: Text('Longest'),
                              ),
                              DropdownMenuItem(
                                value: 'duration_asc',
                                child: Text('Shortest'),
                              ),
                            ],
                            onChanged: (v) {
                              setState(() {
                                _sortBy = v ?? 'newest';
                                _applyFilters();
                              });
                            },
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    // Bulk actions
                    Row(
                      children: [
                        Text(
                          '${_filteredRuns.length} runs',
                          style: Theme.of(context).textTheme.bodySmall,
                        ),
                        if (_selectedIds.isNotEmpty) ...[
                          const SizedBox(width: 8),
                          Text(
                            '(${_selectedIds.length} selected)',
                            style: Theme.of(context).textTheme.bodySmall
                                ?.copyWith(
                                  color: Theme.of(context).colorScheme.primary,
                                ),
                          ),
                        ],
                        const Spacer(),
                        if (_selectedIds.isNotEmpty) ...[
                          TextButton.icon(
                            onPressed: _exportSelected,
                            icon: const Icon(Icons.download, size: 18),
                            label: const Text('Export'),
                          ),
                          const SizedBox(width: 8),
                          TextButton.icon(
                            onPressed: _deleteSelected,
                            icon: Icon(
                              Icons.delete,
                              size: 18,
                              color: colorScheme.error,
                            ),
                            label: Text(
                              'Delete',
                              style: TextStyle(color: colorScheme.error),
                            ),
                          ),
                        ],
                        IconButton(
                          onPressed: _loadData,
                          icon: const Icon(Icons.refresh),
                          tooltip: 'Refresh',
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              // Table
              Expanded(
                child: _filteredRuns.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.history,
                              size: 64,
                              color: colorScheme.outlineVariant,
                            ),
                            const SizedBox(height: 16),
                            Text(
                              'No runs found',
                              style: TextStyle(
                                color: colorScheme.onSurfaceVariant,
                              ),
                            ),
                          ],
                        ),
                      )
                    : ListView.builder(
                        itemCount: _filteredRuns.length,
                        itemBuilder: (ctx, idx) {
                          final run = _filteredRuns[idx];
                          final id = run['id'] as String;
                          final benchmarks =
                              (run['benchmarks'] as List<dynamic>?)?.join(
                                ', ',
                              ) ??
                              '';
                          final isSelected = _selectedIds.contains(id);
                          final isViewing = _selectedRun?['id'] == id;

                          return Container(
                            decoration: BoxDecoration(
                              color: isViewing
                                  ? colorScheme.primary.withValues(alpha: 0.1)
                                  : (idx % 2 == 0
                                        ? colorScheme.surface
                                        : colorScheme.surfaceContainerLowest),
                              border: Border(
                                bottom: BorderSide(
                                  color: colorScheme.outlineVariant,
                                ),
                              ),
                            ),
                            child: InkWell(
                              onTap: () => _viewRun(run),
                              child: Padding(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 16,
                                  vertical: 12,
                                ),
                                child: Row(
                                  children: [
                                    // Checkbox
                                    Checkbox(
                                      value: isSelected,
                                      onChanged: (v) {
                                        setState(() {
                                          if (v == true) {
                                            _selectedIds.add(id);
                                          } else {
                                            _selectedIds.remove(id);
                                          }
                                        });
                                      },
                                    ),
                                    const SizedBox(width: 8),

                                    // Date/Time
                                    SizedBox(
                                      width: 120,
                                      child: Text(
                                        _formatDateTime(
                                          run['started_at'] as String?,
                                        ),
                                        style: const TextStyle(fontSize: 13),
                                      ),
                                    ),

                                    // Benchmarks
                                    Expanded(
                                      child: Text(
                                        benchmarks,
                                        overflow: TextOverflow.ellipsis,
                                        style: const TextStyle(
                                          fontWeight: FontWeight.w500,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 16),

                                    // Status
                                    SizedBox(
                                      width: 100,
                                      child: _buildStatusBadge(
                                        run['status'] as String? ?? '',
                                        colorScheme: colorScheme,
                                      ),
                                    ),

                                    // Duration
                                    SizedBox(
                                      width: 80,
                                      child: Text(
                                        _formatDuration(_getDuration(run)),
                                        style: TextStyle(
                                          color: colorScheme.onSurfaceVariant,
                                          fontSize: 13,
                                        ),
                                      ),
                                    ),

                                    // Actions
                                    IconButton(
                                      icon: const Icon(
                                        Icons.visibility,
                                        size: 20,
                                      ),
                                      onPressed: () => _viewRun(run),
                                      tooltip: 'View',
                                    ),
                                    IconButton(
                                      icon: Icon(
                                        Icons.delete,
                                        size: 20,
                                        color: colorScheme.error,
                                      ),
                                      onPressed: () => _deleteRun(id),
                                      tooltip: 'Delete',
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          );
                        },
                      ),
              ),
            ],
          ),
        ),

        // Detail panel
        if (_selectedRun != null) ...[
          const VerticalDivider(width: 1),
          Expanded(flex: 1, child: _buildDetailPanel()),
        ],
      ],
    );
  }

  Widget _buildDetailPanel() {
    final colorScheme = Theme.of(context).colorScheme;
    final run = _selectedRun!;
    final outputs = run['outputs'] as Map<String, dynamic>? ?? {};
    final images =
        (outputs['images'] as List<dynamic>?)?.cast<Map<String, dynamic>>() ??
        [];
    final data =
        (outputs['data'] as List<dynamic>?)?.cast<Map<String, dynamic>>() ?? [];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Card(
          elevation: 0,
          margin: EdgeInsets.zero,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.zero,
            side: BorderSide(
              color: colorScheme.outlineVariant,
              width: 0,
              strokeAlign: BorderSide.strokeAlignOutside,
            ),
          ),
          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              border: Border(
                bottom: BorderSide(color: colorScheme.outlineVariant),
              ),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        run['id'] as String,
                        style: const TextStyle(fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(height: 4),
                      _buildStatusBadge(
                        run['status'] as String? ?? '',
                        colorScheme: colorScheme,
                      ),
                    ],
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => setState(() => _selectedRun = null),
                ),
              ],
            ),
          ),
        ),

        // Log
        Expanded(
          child: Container(
            color: colorScheme.surfaceContainerHighest,
            padding: const EdgeInsets.all(12),
            child: _loadingDetail
                ? const Center(child: CircularProgressIndicator())
                : SingleChildScrollView(
                    child: SelectableText(
                      _selectedRunLog.isEmpty
                          ? 'No log available'
                          : _selectedRunLog,
                      style: TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 12,
                        color: colorScheme.onSurface,
                      ),
                    ),
                  ),
          ),
        ),

        // Outputs
        if (images.isNotEmpty || data.isNotEmpty)
          Card(
            elevation: 0,
            margin: EdgeInsets.zero,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.zero,
              side: BorderSide(color: colorScheme.outlineVariant),
            ),
            child: SizedBox(
              height: 200,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.all(12),
                    child: Text(
                      'Outputs',
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                  ),
                  Expanded(
                    child: ListView(
                      scrollDirection: Axis.horizontal,
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      children: [
                        ...images.map(
                          (img) => Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: InkWell(
                              onTap: () =>
                                  _showImagePreview(img['url'] as String),
                              child: ClipRRect(
                                borderRadius: BorderRadius.circular(8),
                                child: Image.network(
                                  _api.resolveUrl(img['url'] as String),
                                  height: 140,
                                  fit: BoxFit.cover,
                                  errorBuilder: (context, error, stackTrace) =>
                                      Container(
                                        width: 140,
                                        height: 140,
                                        color: colorScheme.surfaceContainerLow,
                                        child: const Icon(Icons.broken_image),
                                      ),
                                ),
                              ),
                            ),
                          ),
                        ),
                        ...data.map(
                          (file) => Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: InkWell(
                              onTap: () => _showFileContent(
                                file['name'] as String,
                                file['url'] as String,
                              ),
                              child: Container(
                                width: 120,
                                height: 140,
                                decoration: BoxDecoration(
                                  color: colorScheme.surfaceContainerLow,
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                child: Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Icon(
                                      Icons.description,
                                      size: 40,
                                      color: colorScheme.onSurfaceVariant,
                                    ),
                                    const SizedBox(height: 8),
                                    Padding(
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 8,
                                      ),
                                      child: Text(
                                        file['name'] as String,
                                        maxLines: 2,
                                        overflow: TextOverflow.ellipsis,
                                        textAlign: TextAlign.center,
                                        style: const TextStyle(fontSize: 11),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  void _showImagePreview(String url) {
    showDialog(
      context: context,
      builder: (ctx) => Dialog(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            AppBar(
              title: const Text('Preview'),
              automaticallyImplyLeading: false,
              actions: [
                IconButton(
                  icon: const Icon(Icons.close),
                  onPressed: () => Navigator.pop(ctx),
                ),
              ],
            ),
            Flexible(
              child: InteractiveViewer(
                child: Image.network(_api.resolveUrl(url)),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _showFileContent(String name, String url) async {
    try {
      final content = await _api.fetchText(url);
      if (!mounted) return;

      showDialog(
        context: context,
        builder: (ctx) => Dialog(
          child: Column(
            children: [
              AppBar(
                title: Text(name),
                automaticallyImplyLeading: false,
                actions: [
                  IconButton(
                    icon: const Icon(Icons.close),
                    onPressed: () => Navigator.pop(ctx),
                  ),
                ],
              ),
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: SelectableText(
                    content,
                    style: const TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 12,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Failed to load file: $e')));
      }
    }
  }
}
