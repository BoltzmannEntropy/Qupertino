import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../services/log_service.dart';

class LogsPanel extends StatefulWidget {
  final bool isCollapsed;
  final VoidCallback onToggleCollapse;
  final double height;
  final ValueChanged<double> onHeightChanged;

  const LogsPanel({
    super.key,
    required this.isCollapsed,
    required this.onToggleCollapse,
    required this.height,
    required this.onHeightChanged,
  });

  @override
  State<LogsPanel> createState() => _LogsPanelState();
}

class _LogsPanelState extends State<LogsPanel> {
  final ScrollController _scrollController = ScrollController();
  LogLevel? _filterLevel;
  final bool _autoScroll = true;

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    if (_autoScroll && _scrollController.hasClients) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 100),
          curve: Curves.easeOut,
        );
      });
    }
  }

  Color _getLevelColor(BuildContext context, LogLevel level) {
    final colorScheme = Theme.of(context).colorScheme;
    switch (level) {
      case LogLevel.debug:
        return colorScheme.outline;
      case LogLevel.info:
        return Colors.blue;
      case LogLevel.warning:
        return Colors.orange;
      case LogLevel.error:
        return colorScheme.error;
    }
  }

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    if (widget.isCollapsed) {
      return GestureDetector(
        onDoubleTap: widget.onToggleCollapse,
        child: Container(
          height: 32,
          decoration: BoxDecoration(
            color: colorScheme.surfaceContainerHighest,
            border: Border(top: BorderSide(color: colorScheme.outlineVariant)),
          ),
          child: Row(
            children: [
              const SizedBox(width: 12),
              Icon(Icons.terminal, size: 16, color: colorScheme.onSurfaceVariant),
              const SizedBox(width: 8),
              Text('Logs', style: TextStyle(fontSize: 12, color: colorScheme.onSurfaceVariant)),
              const Spacer(),
              IconButton(
                icon: const Icon(Icons.expand_less, size: 16),
                onPressed: widget.onToggleCollapse,
                tooltip: 'Expand',
                iconSize: 16,
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
              ),
              const SizedBox(width: 8),
            ],
          ),
        ),
      );
    }

    return ListenableBuilder(
      listenable: logger,
      builder: (context, _) {
        final logs = logger.filtered(level: _filterLevel);
        _scrollToBottom();

        return Column(
          children: [
            // Resize handle
            GestureDetector(
              onVerticalDragUpdate: (details) {
                final newHeight = (widget.height - details.delta.dy).clamp(80.0, 400.0);
                widget.onHeightChanged(newHeight);
              },
              onDoubleTap: widget.onToggleCollapse,
              child: MouseRegion(
                cursor: SystemMouseCursors.resizeRow,
                child: Container(
                  height: 6,
                  decoration: BoxDecoration(
                    color: colorScheme.surfaceContainerHighest,
                    border: Border(top: BorderSide(color: colorScheme.outlineVariant)),
                  ),
                  child: Center(
                    child: Container(
                      width: 40,
                      height: 3,
                      decoration: BoxDecoration(
                        color: colorScheme.outlineVariant,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                ),
              ),
            ),
            // Header
            Container(
              height: 32,
              color: colorScheme.surfaceContainerHighest,
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: Row(
                children: [
                  Icon(Icons.terminal, size: 16, color: colorScheme.primary),
                  const SizedBox(width: 8),
                  Text('Logs', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: colorScheme.onSurface)),
                  const SizedBox(width: 8),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                    decoration: BoxDecoration(
                      color: colorScheme.primaryContainer,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text('${logs.length}', style: TextStyle(fontSize: 10, color: colorScheme.onPrimaryContainer)),
                  ),
                  const SizedBox(width: 16),
                  // Filter dropdown
                  DropdownButton<LogLevel?>(
                    value: _filterLevel,
                    isDense: true,
                    underline: const SizedBox(),
                    style: TextStyle(fontSize: 11, color: colorScheme.onSurface),
                    items: [
                      const DropdownMenuItem(value: null, child: Text('All')),
                      ...LogLevel.values.map((l) => DropdownMenuItem(
                            value: l,
                            child: Text(l.name.toUpperCase()),
                          )),
                    ],
                    onChanged: (v) => setState(() => _filterLevel = v),
                  ),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.delete_outline, size: 16),
                    onPressed: logger.clear,
                    tooltip: 'Clear logs',
                    iconSize: 16,
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                  ),
                  IconButton(
                    icon: const Icon(Icons.copy_all_outlined, size: 16),
                    onPressed: () async {
                      final buffer = StringBuffer();
                      for (final entry in logs) {
                        final hh = entry.timestamp.hour.toString().padLeft(2, '0');
                        final mm = entry.timestamp.minute.toString().padLeft(2, '0');
                        final ss = entry.timestamp.second.toString().padLeft(2, '0');
                        buffer.writeln(
                          '[$hh:$mm:$ss] ${entry.levelName.toUpperCase()} '
                          '${entry.source}: ${entry.message}',
                        );
                      }
                      await Clipboard.setData(ClipboardData(text: buffer.toString()));
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(content: Text('Logs copied to clipboard')),
                        );
                      }
                    },
                    tooltip: 'Copy logs',
                    iconSize: 16,
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                  ),
                  IconButton(
                    icon: const Icon(Icons.download, size: 16),
                    onPressed: () async {
                      final path = await logger.export();
                      if (context.mounted && path != null) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text('Logs exported to $path')),
                        );
                      }
                    },
                    tooltip: 'Export logs',
                    iconSize: 16,
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                  ),
                  IconButton(
                    icon: const Icon(Icons.expand_more, size: 16),
                    onPressed: widget.onToggleCollapse,
                    tooltip: 'Collapse',
                    iconSize: 16,
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(minWidth: 32, minHeight: 32),
                  ),
                ],
              ),
            ),
            // Log content
            Container(
              height: widget.height - 38,
              color: colorScheme.surfaceContainerLowest,
              child: logs.isEmpty
                  ? Center(
                      child: Text(
                        'No logs yet',
                        style: TextStyle(color: colorScheme.onSurfaceVariant, fontFamily: 'monospace', fontSize: 12),
                      ),
                    )
                  : ListView.builder(
                      controller: _scrollController,
                      itemCount: logs.length,
                      padding: const EdgeInsets.all(8),
                      itemBuilder: (context, index) {
                        final entry = logs[index];
                        final time = '${entry.timestamp.hour.toString().padLeft(2, '0')}:'
                            '${entry.timestamp.minute.toString().padLeft(2, '0')}:'
                            '${entry.timestamp.second.toString().padLeft(2, '0')}';
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 1),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              SizedBox(
                                width: 60,
                                child: Text(
                                  time,
                                  style: TextStyle(color: colorScheme.onSurfaceVariant, fontFamily: 'monospace', fontSize: 11),
                                ),
                              ),
                              Container(
                                width: 16,
                                height: 16,
                                margin: const EdgeInsets.only(right: 8),
                                decoration: BoxDecoration(
                                  color: _getLevelColor(context, entry.level),
                                  borderRadius: BorderRadius.circular(2),
                                ),
                                child: Center(
                                  child: Text(
                                    entry.levelName[0],
                                    style: TextStyle(color: colorScheme.onPrimary, fontSize: 9, fontWeight: FontWeight.bold),
                                  ),
                                ),
                              ),
                              SizedBox(
                                width: 100,
                                child: Text(
                                  entry.source,
                                  style: TextStyle(color: colorScheme.primary, fontFamily: 'monospace', fontSize: 11, fontWeight: FontWeight.w600),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  entry.message,
                                  style: TextStyle(color: colorScheme.onSurface, fontFamily: 'monospace', fontSize: 11),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
            ),
          ],
        );
      },
    );
  }
}
