import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../services/api_service.dart';
import '../version.dart';
import 'privacy_policy_screen.dart';
import 'terms_of_service_screen.dart';
import 'license_screen.dart';

class AboutScreen extends StatefulWidget {
  const AboutScreen({super.key});

  @override
  State<AboutScreen> createState() => _AboutScreenState();
}

class _AboutScreenState extends State<AboutScreen> {
  Map<String, dynamic>? _systemInfo;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadSystemInfo();
  }

  Future<void> _loadSystemInfo() async {
    try {
      final info = await ApiService().getSystemInfo();
      if (mounted) {
        setState(() {
          _systemInfo = info;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildHeaderCard(context),
              const SizedBox(height: 24),
              _buildSection(
                context: context,
                icon: Icons.warning_amber_rounded,
                title: 'Important Notice',
                content:
                    'QuantumStudio is intended for research, benchmarking, and education. '
                    'Benchmark outputs and simulation results should be independently validated '
                    'before use in publications or production decisions.',
              ),
              const SizedBox(height: 16),
              _buildSection(
                context: context,
                icon: Icons.info_outline,
                title: 'What This App Does',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _bulletPoint(
                      'Run reproducible benchmark suites (QFT, Grover, VQE, QAOA, and more).',
                    ),
                    _bulletPoint(
                      'Compare simulation behavior across backends and qubit ranges.',
                    ),
                    _bulletPoint(
                      'Capture plots, logs, and outputs for research documentation.',
                    ),
                    _bulletPoint(
                      'Monitor local hardware and runtime metrics while jobs execute.',
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              _buildSystemInfoCard(context),
              const SizedBox(height: 16),
              _buildSection(
                context: context,
                icon: Icons.layers,
                title: 'Technology Stack',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _techItem(
                      'MLX',
                      'Apple\'s machine learning framework optimized for Apple Silicon',
                    ),
                    _techItem(
                      'mlx-quantum',
                      'Quantum circuit simulation library built on MLX',
                    ),
                    _techItem(
                      'Flutter',
                      'Cross-platform UI framework for the desktop application',
                    ),
                    _techItem(
                      'FastAPI',
                      'Python backend server for benchmark execution',
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              _buildSection(
                context: context,
                icon: Icons.science,
                title: 'Supported Benchmarks',
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _benchmarkItem('QFT', 'Quantum Fourier Transform'),
                    _benchmarkItem('Grover', 'Grover\'s search algorithm'),
                    _benchmarkItem('VQE', 'Variational Quantum Eigensolver'),
                    _benchmarkItem('QAOA', 'Quantum Approximate Optimization'),
                    _benchmarkItem('Random Circuit', 'Random circuit sampling'),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              _buildSection(
                context: context,
                icon: Icons.warning_amber,
                title: 'Limitations & Notes',
                content:
                    'Memory usage grows exponentially with qubit count. '
                    'State vector simulation requires 2^n complex amplitudes, limiting practical simulation to ~25-30 qubits on most systems.\n\n'
                    'Matrix Product State (MPS) backend can handle more qubits for certain circuit types but may be slower for highly entangled circuits.\n\n'
                    'Performance varies significantly based on circuit depth and gate types.',
              ),
              const SizedBox(height: 16),
              _buildSection(
                context: context,
                icon: Icons.code,
                title: 'Model Credits & Licenses',
                content:
                    'MLX (Apache-2.0 style ecosystem)\n'
                    'mlx-quantum (project license)\n'
                    'Flutter (BSD-3)\n'
                    'FastAPI (MIT)\n\n'
                    'See repository LICENSE files for exact legal terms.',
              ),
              const SizedBox(height: 16),
              _buildLegalSection(context),
              const SizedBox(height: 16),
              _buildSupportSection(context),
              const SizedBox(height: 24),
              Center(
                child: Column(
                  children: [
                    Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          'Copyright 2026 ',
                          style: TextStyle(
                            fontSize: 12,
                            color: Theme.of(
                              context,
                            ).colorScheme.onSurface.withValues(alpha: 0.5),
                          ),
                        ),
                        MouseRegion(
                          cursor: SystemMouseCursors.click,
                          child: GestureDetector(
                            onTap: () async {
                              final uri = Uri.parse(appWebsite);
                              if (await canLaunchUrl(uri)) {
                                await launchUrl(uri);
                              }
                            },
                            child: Text(
                              appAuthor,
                              style: TextStyle(
                                fontSize: 12,
                                color: Theme.of(context).colorScheme.primary,
                                decoration: TextDecoration.underline,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Source: BSL-1.1 | Binary: QuantumStudio Binary Distribution License',
                      style: TextStyle(
                        fontSize: 11,
                        color: Theme.of(
                          context,
                        ).colorScheme.onSurface.withValues(alpha: 0.6),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 48),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeaderCard(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Card(
      elevation: 0,
      color: colorScheme.primaryContainer,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Row(
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF6E40C9), Color(0xFF2188FF)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Center(
                child: Text(
                  'Q',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 48,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 24),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'QuantumStudio',
                    style: TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: colorScheme.onPrimaryContainer,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'MLX Quantum Benchmarking Suite',
                    style: TextStyle(
                      fontSize: 14,
                      color: colorScheme.onPrimaryContainer.withValues(
                        alpha: 0.8,
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Version $versionString',
                    style: TextStyle(
                      fontSize: 12,
                      color: colorScheme.onPrimaryContainer.withValues(
                        alpha: 0.6,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSystemInfoCard(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    final info = _systemInfo;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.computer, size: 20, color: colorScheme.primary),
                const SizedBox(width: 8),
                const Text(
                  'System Information',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 16),
            if (_isLoading)
              const Center(child: CircularProgressIndicator())
            else if (info == null)
              Text(
                'Unable to load system information',
                style: TextStyle(color: colorScheme.error),
              )
            else
              Table(
                columnWidths: const {
                  0: IntrinsicColumnWidth(),
                  1: FlexColumnWidth(),
                },
                children: [
                  _tableRow('Chip', info['chip'] ?? 'Unknown'),
                  _tableRow('Memory', '${info['memory_gb'] ?? '?'} GB'),
                  _tableRow('CPU Cores', '${info['cpu_cores'] ?? '?'}'),
                  _tableRow('GPU Cores', '${info['gpu_cores'] ?? '?'}'),
                  _tableRow('Architecture', info['arch'] ?? 'Unknown'),
                  _tableRow('macOS Version', info['os_version'] ?? 'Unknown'),
                  _tableRow(
                    'Python Version',
                    info['python_version'] ?? 'Unknown',
                  ),
                  _tableRow('MLX Version', info['mlx_version'] ?? 'Unknown'),
                ],
              ),
          ],
        ),
      ),
    );
  }

  TableRow _tableRow(String label, String value) {
    return TableRow(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 6),
          child: Text(
            label,
            style: TextStyle(
              color: Theme.of(context).colorScheme.onSurfaceVariant,
              fontSize: 13,
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 16),
          child: Text(
            value,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
          ),
        ),
      ],
    );
  }

  Widget _buildSection({
    required BuildContext context,
    required IconData icon,
    required String title,
    String? content,
    Widget? child,
  }) {
    final colorScheme = Theme.of(context).colorScheme;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 20, color: colorScheme.primary),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            if (content != null)
              Text(
                content,
                style: TextStyle(
                  fontSize: 13,
                  color: colorScheme.onSurface,
                  height: 1.5,
                ),
              ),
            if (child != null) child,
          ],
        ),
      ),
    );
  }

  Widget _bulletPoint(String text) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 6,
            height: 6,
            margin: const EdgeInsets.only(top: 6, right: 10),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.primary,
              shape: BoxShape.circle,
            ),
          ),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 13,
                color: Theme.of(context).colorScheme.onSurface,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _techItem(String name, String description) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 6,
            height: 6,
            margin: const EdgeInsets.only(top: 6, right: 12),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.primary,
              shape: BoxShape.circle,
            ),
          ),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: TextStyle(
                  fontSize: 13,
                  color: Theme.of(context).colorScheme.onSurface,
                ),
                children: [
                  TextSpan(
                    text: name,
                    style: const TextStyle(fontWeight: FontWeight.w600),
                  ),
                  TextSpan(text: ' - $description'),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _benchmarkItem(String name, String description) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.secondaryContainer,
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              name,
              style: const TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                fontFamily: 'monospace',
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              description,
              style: TextStyle(
                fontSize: 13,
                color: Theme.of(context).colorScheme.onSurface,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLegalSection(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.gavel, size: 20, color: colorScheme.primary),
                const SizedBox(width: 8),
                const Text(
                  'Legal',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton.tonalIcon(
                  onPressed: () => _navigateToPrivacyPolicy(context),
                  icon: const Icon(Icons.privacy_tip, size: 18),
                  label: const Text('Privacy Policy'),
                ),
                FilledButton.tonalIcon(
                  onPressed: () => _navigateToTermsOfService(context),
                  icon: const Icon(Icons.description, size: 18),
                  label: const Text('Terms of Service'),
                ),
                FilledButton.tonalIcon(
                  onPressed: () => _navigateToLicense(context),
                  icon: const Icon(Icons.article, size: 18),
                  label: const Text('License'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _navigateToPrivacyPolicy(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => const PrivacyPolicyScreen()),
    );
  }

  void _navigateToTermsOfService(BuildContext context) {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => const TermsOfServiceScreen()),
    );
  }

  void _navigateToLicense(BuildContext context) {
    Navigator.of(
      context,
    ).push(MaterialPageRoute(builder: (context) => const LicenseScreen()));
  }

  Widget _buildSupportSection(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;

    return Card(
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: colorScheme.outlineVariant),
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.help_outline, size: 20, color: colorScheme.primary),
                const SizedBox(width: 8),
                const Text(
                  'Support',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton.tonalIcon(
                  onPressed: () async {
                    final uri = Uri.parse(appWebsite);
                    if (await canLaunchUrl(uri)) {
                      await launchUrl(uri);
                    }
                  },
                  icon: const Icon(Icons.language, size: 18),
                  label: const Text('Website'),
                ),
                FilledButton.tonalIcon(
                  onPressed: () async {
                    final uri = Uri.parse(appGitHub);
                    if (await canLaunchUrl(uri)) {
                      await launchUrl(uri);
                    }
                  },
                  icon: const Icon(Icons.code, size: 18),
                  label: const Text('GitHub'),
                ),
                FilledButton.tonalIcon(
                  onPressed: () async {
                    final uri = Uri.parse(
                      'mailto:solomon@qneura.ai?subject=QuantumStudio%20Issue',
                    );
                    if (await canLaunchUrl(uri)) {
                      await launchUrl(uri);
                    }
                  },
                  icon: const Icon(Icons.bug_report, size: 18),
                  label: const Text('Report Issue'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
