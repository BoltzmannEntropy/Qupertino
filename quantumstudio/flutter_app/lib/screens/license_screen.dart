import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

class LicenseScreen extends StatelessWidget {
  const LicenseScreen({super.key});

  static const String _websiteUrl = 'https://qneura.ai/apps.html';

  static const Map<String, Map<String, String>> _thirdPartyLibraries = {
    'MLX': {
      'description': 'Apple\'s machine learning framework optimized for Apple Silicon',
      'license': 'MIT License',
      'url': 'https://github.com/ml-explore/mlx',
    },
    'Flutter': {
      'description': 'Google\'s UI toolkit for building cross-platform applications',
      'license': 'BSD 3-Clause License',
      'url': 'https://flutter.dev',
    },
    'FastAPI': {
      'description': 'Modern, fast web framework for building APIs with Python',
      'license': 'MIT License',
      'url': 'https://fastapi.tiangolo.com',
    },
  };

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
        title: const Text('License'),
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
                        child: Icon(Icons.description, size: 40, color: theme.colorScheme.primary),
                      ),
                      const SizedBox(height: 16),
                      Text('License', style: theme.textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold)),
                      const SizedBox(height: 8),
                      Text('Qupertino by QNeura.ai', style: theme.textTheme.bodyLarge?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
                    ],
                  ),
                ),
                const SizedBox(height: 32),

                // MIT Notice
                Card(
                  color: theme.colorScheme.primaryContainer.withValues(alpha: 0.3),
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      children: [
                        Icon(Icons.verified, color: theme.colorScheme.primary, size: 32),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('MIT License', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold, color: theme.colorScheme.primary)),
                              const SizedBox(height: 4),
                              Text('Free and open source. Use, modify, and redistribute freely, including commercially.', style: theme.textTheme.bodyMedium),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // MIT Text
                Text('Source Code & Binaries', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                const SizedBox(height: 12),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Copyright (c) 2026 Shlomo Kashani', style: theme.textTheme.bodyMedium?.copyWith(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 12),
                        Text('Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software.', style: theme.textTheme.bodyMedium),
                        const SizedBox(height: 12),
                        Text('Both the source code and the compiled binaries (DMG, app bundles) are distributed under the MIT License. No purchase or license key is required.', style: theme.textTheme.bodyMedium),
                        const SizedBox(height: 12),
                        Text('THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.', style: theme.textTheme.bodyMedium?.copyWith(fontStyle: FontStyle.italic)),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Third-Party
                Row(
                  children: [
                    Icon(Icons.extension, size: 24, color: theme.colorScheme.primary),
                    const SizedBox(width: 12),
                    Text('Third-Party Libraries', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                  ],
                ),
                const SizedBox(height: 12),
                ..._thirdPartyLibraries.entries.map((entry) => _buildLibraryCard(context, entry.key, entry.value)),

                const SizedBox(height: 32),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Questions?', style: theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
                        const SizedBox(height: 8),
                        FilledButton.icon(
                          onPressed: () => _launchUrl(_websiteUrl),
                          icon: const Icon(Icons.language),
                          label: const Text('QNeura.ai'),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 32),
                Center(child: Text('2026 QNeura.ai', style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant))),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildLibraryCard(BuildContext context, String name, Map<String, String> info) {
    final theme = Theme.of(context);
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: InkWell(
        onTap: () => _launchUrl(info['url']!),
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(name, style: theme.textTheme.titleSmall?.copyWith(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 4),
                    Text(info['description']!, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
                    const SizedBox(height: 4),
                    Text(info['license']!, style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.primary, fontWeight: FontWeight.w500)),
                  ],
                ),
              ),
              Icon(Icons.open_in_new, size: 20, color: theme.colorScheme.onSurfaceVariant),
            ],
          ),
        ),
      ),
    );
  }
}
