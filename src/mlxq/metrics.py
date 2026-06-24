import sys
import time
import resource


def cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return float(ru.ru_utime + ru.ru_stime)


def peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rss = float(ru.ru_maxrss)
    # On macOS ru_maxrss is in bytes; on Linux it's kilobytes
    if sys.platform == 'darwin':
        return rss / (1024.0 * 1024.0)
    else:
        return rss / 1024.0


def now_ms() -> float:
    return time.perf_counter() * 1000.0

