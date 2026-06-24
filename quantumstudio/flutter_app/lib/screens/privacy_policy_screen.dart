import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class PrivacyPolicyScreen extends StatelessWidget {
  const PrivacyPolicyScreen({super.key});

  Future<void> _launchUrl(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri, mode: LaunchMode.externalApplication);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Privacy Policy'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 700),
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Header
                Center(
                  child: Column(
                    children: [
                      Container(
                        width: 80,
                        height: 80,
                        decoration: BoxDecoration(
                          color: theme.colorScheme.primaryContainer,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Icon(
                          Icons.privacy_tip,
                          size: 40,
                          color: theme.colorScheme.primary,
                        ),
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Privacy Policy',
                        style: theme.textTheme.headlineMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'QuantumStudio by QNeura.ai',
                        style: theme.textTheme.bodyLarge?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'Last updated: February 2026',
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 32),

                // Local-First Notice
                Card(
                  color: theme.colorScheme.primaryContainer.withValues(alpha: 0.3),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Icon(
                          Icons.shield_outlined,
                          color: theme.colorScheme.primary,
                          size: 32,
                        ),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Local-First by Default',
                                style: theme.textTheme.titleMedium?.copyWith(
                                  fontWeight: FontWeight.bold,
                                  color: theme.colorScheme.primary,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                'QuantumStudio processes simulations locally and does not send data externally by default.',
                                style: theme.textTheme.bodyMedium,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Section 1: Data Collection
                _buildSection(
                  context,
                  'Data Collection',
                  Icons.data_usage,
                  [
                    'QuantumStudio does not collect personal information by default.',
                    'We do not track usage behavior or sell data to third parties.',
                    'Your quantum circuit designs and simulation results remain on your device unless you choose to share them.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 2: Local Processing
                _buildSection(
                  context,
                  'Local Processing',
                  Icons.computer,
                  [
                    'All quantum circuit simulations occur locally on your Mac using Apple Silicon.',
                    'MLX-based computations are performed using your device\'s unified memory architecture.',
                    'Benchmark results and simulation outputs are stored only in locations you specify.',
                    'No simulation data is uploaded to cloud services by default.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 3: File Access
                _buildSection(
                  context,
                  'File Access Permissions',
                  Icons.folder_open,
                  [
                    'QuantumStudio may request access to save benchmark results and simulation outputs.',
                    'Write access is requested only to save results to your chosen output folder.',
                    'File access is limited to the specific folders and files you explicitly select.',
                    'We do not access any files outside of your explicit selections.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 4: Network Usage
                _buildSection(
                  context,
                  'Network Usage',
                  Icons.wifi_off,
                  [
                    'QuantumStudio operates entirely offline for quantum simulations.',
                    'The application communicates only with the local backend server running on your machine.',
                    'No user data or simulation results are transmitted externally by default.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 5: Third-Party Services
                _buildSection(
                  context,
                  'Third-Party Services',
                  Icons.extension,
                  [
                    'QuantumStudio uses open-source libraries including MLX, mlx-quantum, Flutter, and FastAPI.',
                    'These components run locally and do not transmit data externally.',
                    'No third-party analytics, advertising, or tracking services are integrated.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 6: Data Security
                _buildSection(
                  context,
                  'Data Security',
                  Icons.lock_outline,
                  [
                    'Your data is stored locally and protected by your device security settings.',
                    'You maintain control over your simulation data and benchmark results.',
                    'We recommend using your operating system\'s built-in security features to protect your files.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 7: Research & Scientific Use
                _buildSection(
                  context,
                  'Research & Scientific Use',
                  Icons.science,
                  [
                    'QuantumStudio is designed for research and educational purposes.',
                    'Simulation results are yours to use, publish, or share as you see fit.',
                    'We do not claim any rights to your research outputs or simulation data.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 8: Children\'s Privacy
                _buildSection(
                  context,
                  'Children\'s Privacy',
                  Icons.child_care,
                  [
                    'The Service does not address anyone under the age of 13.',
                    'We do not knowingly collect personal information from children under 13.',
                  ],
                ),
                const SizedBox(height: 20),

                // Section 9: Changes to Policy
                _buildSection(
                  context,
                  'Changes to This Policy',
                  Icons.update,
                  [
                    'We may update this Privacy Policy from time to time.',
                    'Changes will be reflected in the "Last updated" date at the top of this policy.',
                    'Continued use of QuantumStudio after changes constitutes acceptance of the updated policy.',
                  ],
                ),
                const SizedBox(height: 32),

                // Contact Section
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Contact Us',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'If you have any questions about this Privacy Policy, please contact '
                          'solomon@qneura.ai or visit:',
                          style: theme.textTheme.bodyMedium,
                        ),
                        const SizedBox(height: 12),
                        FilledButton.icon(
                          onPressed: () => _launchUrl('https://qneura.ai/apps.html'),
                          icon: const Icon(Icons.language),
                          label: const Text('qneura.ai/apps.html'),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 32),

                // Footer
                Center(
                  child: Text(
                    '2026 QNeura.ai - All rights reserved',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSection(
    BuildContext context,
    String title,
    IconData icon,
    List<String> points,
  ) {
    final theme = Theme.of(context);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(
              icon,
              size: 24,
              color: theme.colorScheme.primary,
            ),
            const SizedBox(width: 12),
            Text(
              title,
              style: theme.textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        ...points.map((point) => Padding(
              padding: const EdgeInsets.only(left: 8, bottom: 8),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(
                    Icons.check_circle_outline,
                    size: 18,
                    color: theme.colorScheme.secondary,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      point,
                      style: theme.textTheme.bodyMedium,
                    ),
                  ),
                ],
              ),
            )),
      ],
    );
  }
}
