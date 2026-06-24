from typing import Iterable

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False


if _HAS_RICH:
    # Force color for consistent output (like C++ fmt with colors)
    console = Console(highlight=False, force_terminal=True, color_system="truecolor")

    def info(msg: str):
        console.print(f"[bold cyan]{msg}[/bold cyan]")

    def success(msg: str):
        console.print(f"[bold green]{msg}[/bold green]")

    def warn(msg: str):
        console.print(f"[bold yellow]{msg}[/bold yellow]")

    def error(msg: str):
        console.print(f"[bold red]{msg}[/bold red]")

    def table(title: str, columns: Iterable[str], rows: Iterable[Iterable[str]]):
        t = Table(title=title, box=box.SIMPLE_HEAVY)
        for c in columns:
            t.add_column(str(c))
        for r in rows:
            t.add_row(*[str(x) for x in r])
        console.print(t)
else:
    class _PlainConsole:
        def print(self, *args, **kwargs):  # type: ignore
            # Strip any simple rich tags and print plainly
            s = " ".join(str(a) for a in args)
            for tag in ("[bold]", "[/bold]", "[bold cyan]", "[/bold cyan]",
                        "[bold green]", "[/bold green]", "[bold yellow]", "[/bold yellow]",
                        "[bold red]", "[/bold red]"):
                s = s.replace(tag, "")
            print(s, flush=True)

    console = _PlainConsole()

    def info(msg: str):
        print(f"{msg}", flush=True)

    def success(msg: str):
        print(f"{msg}", flush=True)

    def warn(msg: str):
        print(f"{msg}", flush=True)

    def error(msg: str):
        print(f"{msg}", flush=True)

    def table(title: str, columns: Iterable[str], rows: Iterable[Iterable[str]]):
        print(title, flush=True)
        cols = [str(c) for c in columns]
        print(" | ".join(cols), flush=True)
        print("-+-".join('-' * len(c) for c in cols), flush=True)
        for r in rows:
            print(" | ".join(str(x) for x in r), flush=True)
