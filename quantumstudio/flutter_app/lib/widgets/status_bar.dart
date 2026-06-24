import 'dart:async';
import 'package:flutter/material.dart';
import '../services/api_service.dart';

class StatusBar extends StatefulWidget {
  final Map<String, dynamic>? systemInfo;

  const StatusBar({super.key, this.systemInfo});

  @override
  State<StatusBar> createState() => _StatusBarState();
}

class _StatusBarState extends State<StatusBar> {
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
    final info = widget.systemInfo;
    final stats = _systemStats;

    final cpuPercent = (stats?['cpu_percent'] ?? 0).toDouble();
    final ramUsed = (stats?['ram_used_gb'] ?? 0).toDouble();
    final ramTotal = (stats?['ram_total_gb'] ?? 0).toDouble();
    final ramPercent = (stats?['ram_percent'] ?? 0).toDouble();
    final jobsRunning = stats?['jobs_running'] ?? 0;
    final jobsQueued = stats?['jobs_queued'] ?? 0;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      decoration: BoxDecoration(
        color: colorScheme.surfaceContainerHigh,
        border: Border(bottom: BorderSide(color: colorScheme.outlineVariant)),
      ),
      child: Row(
        children: [
          // Hardware info (from systemInfo)
          if (info != null) ...[
            _buildChip(context, Icons.memory, info['chip'] ?? 'Unknown'),
            _buildDivider(context),
            _buildChip(
              context,
              Icons.storage,
              '${info['memory_gb'] ?? '?'} GB Unified',
            ),
            _buildDivider(context),
            _buildChip(
              context,
              Icons.grid_view,
              'CPU ${info['cpu_cores'] ?? '?'} cores',
            ),
            _buildDivider(context),
            _buildChip(
              context,
              Icons.auto_awesome,
              'GPU ${info['gpu_cores'] ?? '?'} cores',
            ),
            _buildDivider(context),
            _buildChip(
              context,
              Icons.circle,
              'MLX ${info['mlx_version'] ?? '?'}',
            ),
          ],
          const Spacer(),
          // Live system stats (from polling)
          if (stats != null) ...[
            _buildStatChip(
              context,
              'CPU',
              '${cpuPercent.toStringAsFixed(0)}%',
              cpuPercent / 100,
              _getUsageColor(cpuPercent),
            ),
            const SizedBox(width: 12),
            _buildStatChip(
              context,
              'RAM',
              '${ramUsed.toStringAsFixed(1)} / ${ramTotal.toStringAsFixed(0)} GB',
              ramPercent / 100,
              _getUsageColor(ramPercent),
            ),
            const SizedBox(width: 16),
            _buildJobsChip(context, jobsRunning, jobsQueued),
          ],
        ],
      ),
    );
  }

  Widget _buildChip(BuildContext context, IconData icon, String label) {
    final colorScheme = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, size: 14, color: colorScheme.onSurfaceVariant),
        const SizedBox(width: 6),
        Text(
          label,
          style: TextStyle(fontSize: 12, color: colorScheme.onSurface),
        ),
      ],
    );
  }

  Widget _buildDivider(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      child: Container(
        width: 1,
        height: 16,
        color: Theme.of(context).colorScheme.outlineVariant,
      ),
    );
  }

  Widget _buildStatChip(
    BuildContext context,
    String label,
    String value,
    double percentage,
    Color color,
  ) {
    final colorScheme = Theme.of(context).colorScheme;
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label,
          style: TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w500,
            color: colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(width: 6),
        SizedBox(
          width: 40,
          height: 4,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(2),
            child: LinearProgressIndicator(
              value: percentage.clamp(0, 1),
              backgroundColor: colorScheme.outlineVariant,
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Text(
          value,
          style: TextStyle(fontSize: 11, color: colorScheme.onSurface),
        ),
      ],
    );
  }

  Widget _buildJobsChip(BuildContext context, int running, int queued) {
    final colorScheme = Theme.of(context).colorScheme;
    final hasJobs = running > 0 || queued > 0;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: hasJobs
            ? colorScheme.primaryContainer
            : colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: hasJobs ? colorScheme.primary : colorScheme.outlineVariant,
          width: hasJobs ? 1.5 : 1,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (running > 0) ...[
            SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: colorScheme.primary,
              ),
            ),
            const SizedBox(width: 6),
          ] else
            Icon(
              Icons.check_circle_outline,
              size: 14,
              color: hasJobs
                  ? colorScheme.primary
                  : colorScheme.onSurfaceVariant,
            ),
          const SizedBox(width: 4),
          Text(
            running > 0 || queued > 0
                ? '$running running${queued > 0 ? ' / $queued queued' : ''}'
                : 'No jobs',
            style: TextStyle(
              fontSize: 11,
              fontWeight: hasJobs ? FontWeight.w600 : FontWeight.normal,
              color: hasJobs
                  ? colorScheme.onPrimaryContainer
                  : colorScheme.onSurfaceVariant,
            ),
          ),
        ],
      ),
    );
  }
}
