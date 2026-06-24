import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mlxq.vendor import BENCH_KEYS, VENDOR_BENCHMARKS, ALGORITHM_BENCHMARKS
from mlxq.mlxQpretty import table, info, success


def test_bench_catalog_vendor_keys_subset():
    info("Catalog: vendor groups reference supported bench keys")
    bad = []
    for vendor, keys in VENDOR_BENCHMARKS.items():
        for k in keys:
            if k not in BENCH_KEYS:
                bad.append((vendor, k))
    table("Vendor → keys (sanity)", ("vendor","key"), bad or [("all","ok")])
    assert not bad


def test_bench_catalog_algorithm_keys_subset():
    info("Catalog: algorithm groups reference supported bench keys")
    bad = []
    for group, keys in ALGORITHM_BENCHMARKS.items():
        for k in keys:
            if k not in BENCH_KEYS:
                bad.append((group, k))
    table("Algorithm → keys (sanity)", ("group","key"), bad or [("all","ok")])
    assert not bad

