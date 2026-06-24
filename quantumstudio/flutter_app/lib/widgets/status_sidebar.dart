import 'dart:async';
import 'package:flutter/material.dart';
import '../services/api_service.dart';

class StatusSidebar extends StatefulWidget {
  final Map<String, dynamic>? systemInfo;
  final List<Map<String, dynamic>> activeJobs;
  final VoidCallback? onRunLast;
  final VoidCallback? onStopAll;
  final VoidCallback? onViewAllJobs;
  final Function(String)? onCancelJob;

  const StatusSidebar({
    super.key,
    this.systemInfo,
    required this.activeJobs,
    this.onRunLast,
    this.onStopAll,
    this.onViewAllJobs,
    this.onCancelJob,
  });

  @override
  State<StatusSidebar> createState() => _StatusSidebarState();
}

class _StatusSidebarState extends State<StatusSidebar> {
  Map<String, dynamic>? _systemStats;
  bool _polling = false;

  @override
  void initState() {
    super.initState();
    _fetchStats();
    _startStatsPolling();
  }

  @override
  void dispose() {
    _polling = false;
    super.dispose();
  }

  void _startStatsPolling() {
    if (_polling) return;
    _polling = true;
    Future.doWhile(() async {
      if (!mounted || !_polling) {
        return false;
      }
      await Future.delayed(const Duration(seconds: 2));
      if (!mounted || !_polling) {
        return false;
      }
      await _fetchStats();
      return mounted && _polling;
    });
  }

  Future<void> _fetchStats() async {
    try {
      final stats = await ApiService().getSystemStats();
      if (mounted) {
        setState(() => _systemStats = stats);
      }
    } catch (_) {
      // Ignore errors, keep last known stats
    }
  }

  Color _getUsageColor(double percentage) {
    if (percentage < 50) return Colors.green;
    if (percentage < 80) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Container(
      width: 240,
      decoration: BoxDecoration(
        color: colorScheme.surface,
        border: Border(right: BorderSide(color: colorScheme.outlineVariant)),
      ),
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            _buildHardwareInfoCard(context),
            const SizedBox(height: 12),
            _buildSystemStatsCard(context),
            const SizedBox(height: 12),
            _buildJobQueueCard(context),
            const SizedBox(height: 12),
            _buildQuickActionsCard(context),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionCard({
    required BuildContext context,
    required IconData icon,
    required String title,
    required Widget child,
  }) {
    final colorScheme = Theme.of(context).colorScheme;
    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 16, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }

  Widget _buildHardwareInfoCard(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final info = widget.systemInfo;

    return _buildSectionCard(
      context: context,
      icon: Icons.memory,
      title: 'Hardware',
      child: info == null
          ? Text(
              'Loading...',
              style: TextStyle(
                fontSize: 11,
                color: colorScheme.onSurfaceVariant,
              ),
            )
          : Column(
              children: [
                _infoRow('Chip', info['chip'] ?? 'Unknown'),
                _infoRow('Memory', '${info['memory_gb'] ?? '?'} GB'),
                _infoRow('CPU', '${info['cpu_cores'] ?? '?'} cores'),
                _infoRow('GPU', '${info['gpu_cores'] ?? '?'} cores'),
              ],
            ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
          Text(
            value,
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w500),
          ),
        ],
      ),
    );
  }

  Widget _buildSystemStatsCard(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final stats = _systemStats;

    return _buildSectionCard(
      context: context,
      icon: Icons.monitor_heart,
      title: 'System',
      child: stats == null
          ? Text(
              'Connecting...',
              style: TextStyle(
                fontSize: 11,
                color: colorScheme.onSurfaceVariant,
              ),
            )
          : Column(
              children: [
                _buildStatBar(
                  context: context,
                  label: 'CPU',
                  value: '${(stats['cpu_percent'] ?? 0).toStringAsFixed(0)}%',
                  percentage: (stats['cpu_percent'] ?? 0) / 100,
                ),
                const SizedBox(height: 8),
                _buildStatBar(
                  context: context,
                  label: 'RAM',
                  value:
                      '${((stats['ram_used_gb'] ?? 0) as num).toStringAsFixed(1)} / ${((stats['ram_total_gb'] ?? 0) as num).toStringAsFixed(0)} GB',
                  percentage: (stats['ram_percent'] ?? 0) / 100,
                ),
                if (stats['gpu_memory_used_gb'] != null) ...[
                  const SizedBox(height: 8),
                  _buildStatBar(
                    context: context,
                    label: 'GPU',
                    value:
                        '${((stats['gpu_memory_used_gb'] ?? 0) as num).toStringAsFixed(1)} GB',
                    percentage: (stats['gpu_memory_percent'] ?? 0) / 100,
                  ),
                ],
              ],
            ),
    );
  }

  Widget _buildStatBar({
    required BuildContext context,
    required String label,
    required String value,
    required double percentage,
  }) {
    final color = _getUsageColor(percentage * 100);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: TextStyle(
                fontSize: 10,
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            Text(
              value,
              style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w500),
            ),
          ],
        ),
        const SizedBox(height: 4),
        ClipRRect(
          borderRadius: BorderRadius.circular(2),
          child: LinearProgressIndicator(
            value: percentage.clamp(0, 1),
            minHeight: 4,
            backgroundColor: Theme.of(
              context,
            ).colorScheme.surfaceContainerHighest,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
      ],
    );
  }

  Widget _buildJobQueueCard(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final jobs = widget.activeJobs;
    final runningCount = jobs.where((j) => j['status'] == 'running').length;
    final queuedCount = jobs.where((j) => j['status'] == 'queued').length;

    return _buildSectionCard(
      context: context,
      icon: Icons.queue,
      title: 'Jobs',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              if (runningCount > 0) ...[
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.blue.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '$runningCount running',
                    style: const TextStyle(fontSize: 10, color: Colors.blue),
                  ),
                ),
                const SizedBox(width: 6),
              ],
              if (queuedCount > 0)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.orange.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    '$queuedCount queued',
                    style: const TextStyle(fontSize: 10, color: Colors.orange),
                  ),
                ),
              if (jobs.isEmpty)
                Text(
                  'No active jobs',
                  style: TextStyle(
                    fontSize: 10,
                    color: colorScheme.onSurfaceVariant,
                  ),
                ),
            ],
          ),
          if (jobs.isNotEmpty) ...[
            const SizedBox(height: 8),
            ...jobs.take(5).map((job) => _buildJobItem(context, job)),
            if (jobs.length > 5)
              TextButton(
                onPressed: widget.onViewAllJobs,
                style: TextButton.styleFrom(
                  padding: EdgeInsets.zero,
                  minimumSize: const Size(0, 24),
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
                child: Text(
                  'View all (${jobs.length})',
                  style: const TextStyle(fontSize: 10),
                ),
              ),
          ],
        ],
      ),
    );
  }

  Widget _buildJobItem(BuildContext context, Map<String, dynamic> job) {
    final isRunning = job['status'] == 'running';
    final benchmarks = (job['benchmarks'] as List?)?.whereType<String>().toList() ?? const <String>[];
    final label = benchmarks.isNotEmpty ? benchmarks.join(', ') : (job['benchmark'] ?? job['id'] ?? 'Unknown');
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          if (isRunning)
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          else
            Icon(
              Icons.hourglass_empty,
              size: 12,
              color: Colors.orange.shade400,
            ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              label,
              style: const TextStyle(fontSize: 10),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          if (widget.onCancelJob != null)
            IconButton(
              icon: const Icon(Icons.close, size: 12),
              onPressed: () => widget.onCancelJob!(job['id']),
              padding: EdgeInsets.zero,
              constraints: const BoxConstraints(minWidth: 20, minHeight: 20),
              tooltip: 'Cancel',
            ),
        ],
      ),
    );
  }

  Widget _buildQuickActionsCard(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return _buildSectionCard(
      context: context,
      icon: Icons.flash_on,
      title: 'Quick Actions',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FilledButton.icon(
            onPressed: widget.onRunLast,
            icon: const Icon(Icons.replay, size: 14),
            label: const Text('Run Last', style: TextStyle(fontSize: 11)),
            style: FilledButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 8),
              minimumSize: const Size(0, 32),
            ),
          ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: widget.activeJobs.isEmpty ? null : widget.onStopAll,
            icon: Icon(
              Icons.stop,
              size: 14,
              color: widget.activeJobs.isEmpty ? null : colorScheme.error,
            ),
            label: Text(
              'Stop All',
              style: TextStyle(
                fontSize: 11,
                color: widget.activeJobs.isEmpty ? null : colorScheme.error,
              ),
            ),
            style: OutlinedButton.styleFrom(
              padding: const EdgeInsets.symmetric(vertical: 8),
              minimumSize: const Size(0, 32),
              side: widget.activeJobs.isEmpty
                  ? null
                  : BorderSide(color: colorScheme.error),
            ),
          ),
        ],
      ),
    );
  }
}
