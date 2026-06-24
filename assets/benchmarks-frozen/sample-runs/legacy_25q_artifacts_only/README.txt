25q-only bundle created: 2026-05-06 18:41:27
Source logs: assets/benchmarks-frozen/sample-runs/legacy_25q_logs_2025-10
Source artifacts: assets/benchmarks-frozen/sample-runs/legacy_dmg_stage_bench_snapshot
Selection rule: circuits with completed 24q/25q lines in curated logs.

Circuits:

  --auto-hybrid-regex             (DEPRECATED) Use PCRE2 if appropriate.
  --binary                        Search binary files.
  --block-buffered                Force block buffering.
  --color=WHEN                    When to use color.
  --colors=COLOR_SPEC             Configure color settings and styles.
  --column                        Show column numbers.
  --context-separator=SEP         Set the separator for contextual chunks.
  --count-matches                 Show count of every match for each file.
  --crlf                          Use CRLF line terminators (nice for Windows).
  --debug                         Show debug messages.
  --dfa-size-limit=NUM            The upper size limit of the regex DFA.
  --engine=ENGINE                 Specify which regex engine to use.
  --field-context-separator=SEP   Set the field context separator.
  --field-match-separator=SEP     Set the field match separator.
  --files                         Print each file that would be searched.
  --files-without-match           Print the paths that contain zero matches.
  --generate=KIND                 Generate man pages and completion scripts.
  --glob-case-insensitive         Process all glob patterns case insensitively.
  --heading                       Print matches grouped by each file.
  --hostname-bin=COMMAND          Run a program to get this system's hostname.
  --hyperlink-format=FORMAT       Set the format of hyperlinks.
  --iglob=GLOB                    Include/exclude paths case insensitively.
  --ignore-file-case-insensitive  Process ignore files case insensitively.
  --ignore-file=PATH              Specify additional ignore files.
  --include-zero                  Include zero matches in summary output.
  --json                          Show search results in a JSON Lines format.
  --line-buffered                 Force line buffering.
  --max-columns-preview           Show preview for lines exceeding the limit.
  --max-filesize=NUM              Ignore files larger than NUM in size.
  --mmap                          Search with memory maps when possible.
  --multiline-dotall              Make '.' match line terminators.
  --no-config                     Never read configuration files.
  --no-ignore                     Don't use ignore files.
  --no-ignore-dot                 Don't use .ignore or .rgignore files.
  --no-ignore-exclude             Don't use local exclusion files.
  --no-ignore-files               Don't use --ignore-file arguments.
  --no-ignore-global              Don't use global ignore files.
  --no-ignore-messages            Suppress gitignore parse error messages.
  --no-ignore-parent              Don't use ignore files in parent directories.
  --no-ignore-vcs                 Don't use ignore files from source control.
  --no-messages                   Suppress some error messages.
  --no-pcre2-unicode              (DEPRECATED) Disable Unicode mode for PCRE2.
  --no-require-git                Use .gitignore outside of git repositories.
  --no-unicode                    Disable Unicode mode.
  --null-data                     Use NUL as a line terminator.
  --one-file-system               Skip directories on other file systems.
  --passthru                      Print both matching and non-matching lines.
  --path-separator=SEP            Set the path separator for printing paths.
  --pcre2-version                 Print the version of PCRE2 that ripgrep uses.
  --pre-glob=GLOB                 Include or exclude files from a preprocessor.
  --pre=COMMAND                   Search output of COMMAND for each PATH.
  --regex-size-limit=NUM          The size limit of the compiled regex.
  --sort-files                    (DEPRECATED) Sort results by file path.
  --sort=SORTBY                   Sort results in ascending order.
  --sortr=SORTBY                  Sort results in descending order.
  --stats                         Print statistics about the search.
  --stop-on-nonmatch              Stop searching after a non-match.
  --trace                         Show trace messages.
  --trim                          Trim prefix whitespace from matches.
  --type-add=TYPESPEC             Add a new glob for a file type.
  --type-clear=TYPE               Clear globs for a file type.
  --type-list                     Show all supported file types.
  --vimgrep                       Print results in a vim compatible format.
  -., --hidden                    Search hidden files and directories.
  -0, --null                      Print a NUL byte after file paths.
  -A, --after-context=NUM         Show NUM lines after each match.
  -B, --before-context=NUM        Show NUM lines before each match.
  -C, --context=NUM               Show NUM lines before and after each match.
  -E, --encoding=ENCODING         Specify the text encoding of files to search.
  -F, --fixed-strings             Treat all patterns as literals.
  -H, --with-filename             Print the file path with each matching line.
  -I, --no-filename               Never print the path with each matching line.
  -L, --follow                    Follow symbolic links.
  -M, --max-columns=NUM           Omit lines longer than this limit.
  -N, --no-line-number            Suppress line numbers.
  -P, --pcre2                     Enable PCRE2 matching.
  -S, --smart-case                Smart case search.
  -T, --type-not=TYPE             Do not search files matching TYPE.
  -U, --multiline                 Enable searching across multiple lines.
  -V, --version                   Print ripgrep's version.
  -a, --text                      Search binary files as if they were text.
  -b, --byte-offset               Print the byte offset for each matching line.
  -c, --count                     Show count of matching lines for each file.
  -d, --max-depth=NUM             Descend at most NUM directories.
  -e, --regexp=PATTERN            A pattern to search for.
  -f, --file=PATTERNFILE          Search for patterns from the given file.
  -g, --glob=GLOB                 Include or exclude file paths.
  -h, --help                      Show help output.
  -i, --ignore-case               Case insensitive search.
  -j, --threads=NUM               Set the approximate number of threads to use.
  -l, --files-with-matches        Print the paths with at least one match.
  -m, --max-count=NUM             Limit the number of matching lines.
  -n, --line-number               Show line numbers.
  -o, --only-matching             Print only matched parts of a line.
  -p, --pretty                    Alias for colors, headings and line numbers.
  -q, --quiet                     Do not print anything to stdout.
  -r, --replace=TEXT              Replace matches with the given text.
  -s, --case-sensitive            Search case sensitively (default).
  -t, --type=TYPE                 Only search files matching TYPE.
  -u, --unrestricted              Reduce the level of "smart" filtering.
  -v, --invert-match              Invert matching.
  -w, --word-regexp               Show matches surrounded by word boundaries.
  -x, --line-regexp               Show matches surrounded by line boundaries.
  -z, --search-zip                Search in compressed files.
  <PATH>...   A file or directory to search.
  <PATTERN>   A regular expression used for searching.
  rg [OPTIONS] PATTERN [PATH ...]
Andrew
FILTER
INPUT
LOGGING
OTHER
OUTPUT
POSITIONAL
Project
SEARCH
USAGE
Use
a
automatically
ripgrep
