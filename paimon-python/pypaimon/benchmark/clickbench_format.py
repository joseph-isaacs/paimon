# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


"""
ClickBench benchmark for Paimon file formats.

Loads the ClickBench ``hits.parquet`` dataset and compares on-disk size, write time,
and read performance (wall time and peak RSS) across Paimon file formats.

Scans and filters run against a plain append table, so filters are pushed down to the
format readers. Row-ID lookups need row tracking, so they run against a second table
with data evolution enabled.

Every read runs ``--repeat`` times, each in a fresh process so its peak RSS covers only
that read. The summary reports the median time and the largest peak RSS.

Usage:
    python pypaimon/benchmark/clickbench_format.py [--rows N] [--data-path PATH]
        [--formats parquet,vortex] [--repeat N]

    --rows N          Rows to use (default: 3_000_000, 0 for the full dataset)
    --data-path PATH  Existing hits.parquet (skips the ~14GB download)
    --formats LIST    Comma-separated formats (default: every installed format)
    --repeat N        Runs per read benchmark (default: 3)
"""

import argparse
import importlib.util
import multiprocessing
import os
import random
import resource
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypaimon import CatalogFactory, Schema  # noqa: E402
from pypaimon.globalindex.indexed_split import IndexedSplit  # noqa: E402
from pypaimon.utils.range import Range  # noqa: E402

CLICKBENCH_URL = "https://datasets.clickhouse.com/hits_compatible/hits.parquet"

# Formats and the optional module each one needs.
ALL_FORMATS = {"parquet": None, "orc": None, "lance": "lance", "vortex": "vortex"}

# ClickBench stores these as Unix timestamps in seconds and days since the epoch.
TIMESTAMP_COLUMNS = ("EventTime", "ClientEventTime", "LocalEventTime")
DATE_COLUMNS = ("EventDate",)

NUM_LOOKUPS = 20
ROWS_PER_LOOKUP = 100


def download_clickbench(dest_path: str):
    """Download ClickBench hits.parquet if not already present."""
    if os.path.exists(dest_path):
        print(f"[INFO] Using cached dataset: {dest_path}")
        return
    print(f"[INFO] Downloading ClickBench dataset to {dest_path} ...")
    print(f"[INFO] URL: {CLICKBENCH_URL}")
    print("[INFO] This is ~14GB, may take a while.")

    import urllib.request
    urllib.request.urlretrieve(CLICKBENCH_URL, dest_path)
    print("[INFO] Download complete.")


def _target_type(field: pa.Field) -> pa.DataType:
    if field.name in TIMESTAMP_COLUMNS:
        return pa.timestamp("s")
    if field.name in DATE_COLUMNS:
        return pa.date32()
    if pa.types.is_binary(field.type):
        # ClickBench text columns are stored as binary but hold UTF-8 strings.
        return pa.string()
    # Paimon doesn't support unsigned integers.
    return {
        pa.uint8(): pa.int16(),
        pa.uint16(): pa.int32(),
        pa.uint32(): pa.int64(),
        pa.uint64(): pa.int64(),
    }.get(field.type, field.type)


def _cast_column(column: pa.ChunkedArray, target: pa.DataType) -> pa.ChunkedArray:
    if pa.types.is_date(target) and not pa.types.is_date(column.type):
        column = column.cast(pa.int32())
    return column.cast(target)


def load_data(data_path: str, max_rows: int) -> pa.Table:
    """Load ClickBench parquet data, optionally limiting rows, with SQL-friendly types."""
    print(f"[INFO] Loading data from {data_path} ...")
    pf = pq.ParquetFile(data_path)
    if max_rows > 0:
        # Read enough row groups to satisfy max_rows
        batches = []
        total = 0
        for rg_idx in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            batches.append(rg)
            total += rg.num_rows
            if total >= max_rows:
                break
        table = pa.concat_tables(batches)
        if table.num_rows > max_rows:
            table = table.slice(0, max_rows)
    else:
        table = pf.read()

    fields = []
    columns = []
    for field, column in zip(table.schema, table.columns):
        target = _target_type(field)
        fields.append(pa.field(field.name, target, nullable=field.nullable))
        columns.append(column if target == field.type else _cast_column(column, target))
    table = pa.Table.from_arrays(columns, schema=pa.schema(fields))

    print(f"[INFO] Loaded {table.num_rows:,} rows, {table.num_columns} columns")
    print(f"[INFO] In-memory size: {table.nbytes / 1024 / 1024:.1f} MB")
    return table


def get_dir_size(path: str) -> int:
    """Get total size of all files under a directory."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def write_paimon_table(catalog, table_name: str, data: pa.Table, options: dict) -> float:
    """Write data to a new Paimon table and return the write time."""
    schema = Schema.from_pyarrow_schema(data.schema, options=options)
    catalog.create_table(f'default.{table_name}', schema, False)
    table = catalog.get_table(f'default.{table_name}')

    write_builder = table.new_batch_write_builder()
    table_write = write_builder.new_write()
    table_commit = write_builder.new_commit()

    t0 = time.perf_counter()
    table_write.write_arrow(data)
    table_commit.commit(table_write.prepare_commit())
    write_time = time.perf_counter() - t0

    table_write.close()
    table_commit.close()
    return write_time


def _count_rows(read_builder) -> int:
    """Stream every batch of a read without materializing the result."""
    splits = read_builder.new_scan().plan().splits()
    reader = read_builder.new_read().to_arrow_batch_reader(splits)
    return sum(batch.num_rows for batch in reader)


def full_scan(table) -> int:
    return _count_rows(table.new_read_builder())


def projected_scan(table) -> int:
    return _count_rows(
        table.new_read_builder().with_projection(["URL", "UserID", "EventTime"]))


def numeric_filter(table) -> int:
    predicate_builder = table.new_read_builder().new_predicate_builder()
    return _count_rows(
        table.new_read_builder()
        .with_projection(["URL", "CounterID"])
        .with_filter(predicate_builder.equal("CounterID", 62)))


def string_filter(table) -> int:
    predicate_builder = table.new_read_builder().new_predicate_builder()
    return _count_rows(
        table.new_read_builder()
        .with_projection(["URL", "SearchPhrase"])
        .with_filter(predicate_builder.contains("URL", "google")))


def point_lookup(table) -> int:
    """Row-ID lookups through IndexedSplit on a data-evolution table."""
    read_builder = table.new_read_builder()
    splits = read_builder.new_scan().plan().splits()
    total_rows = sum(s.row_count for s in splits)

    random.seed(42)
    total_result_rows = 0
    for _ in range(NUM_LOOKUPS):
        start = random.randint(0, total_rows - ROWS_PER_LOOKUP)
        rng = Range(start, start + ROWS_PER_LOOKUP - 1)
        indexed_splits = []
        for s in splits:
            file_ranges = [
                Range(f.first_row_id, f.first_row_id + f.row_count - 1)
                for f in s.files if f.first_row_id is not None
            ]
            if Range.and_([rng], file_ranges):
                indexed_splits.append(IndexedSplit(s, [rng]))
        if indexed_splits:
            total_result_rows += read_builder.new_read().to_arrow(indexed_splits).num_rows
    return total_result_rows


def predicate_lookup(table) -> int:
    """Row-ID lookups through a _ROW_ID filter on a data-evolution table."""
    total_rows = sum(s.row_count for s in table.new_read_builder().new_scan().plan().splits())
    all_cols = [f.name for f in table.fields] + ['_ROW_ID']
    pb = table.new_read_builder().with_projection(all_cols).new_predicate_builder()

    random.seed(42)
    total_result_rows = 0
    for _ in range(NUM_LOOKUPS):
        start = random.randint(0, total_rows - ROWS_PER_LOOKUP)
        read_builder = table.new_read_builder().with_filter(
            pb.between('_ROW_ID', start, start + ROWS_PER_LOOKUP - 1))
        splits = read_builder.new_scan().plan().splits()
        total_result_rows += read_builder.new_read().to_arrow(splits).num_rows
    return total_result_rows


# (name, function, table suffix)
READS = [
    ("full_scan", full_scan, ""),
    ("projected_scan", projected_scan, ""),
    ("numeric_filter", numeric_filter, ""),
    ("string_filter", string_filter, ""),
    ("point_lookup", point_lookup, "_lookup"),
    ("predicate_lookup", predicate_lookup, "_lookup"),
]
READ_FUNCTIONS = {name: fn for name, fn, _ in READS}


def _peak_rss_mb() -> float:
    # Linux carries ru_maxrss across exec, so a spawned child would report the parent's
    # peak. VmHWM belongs to the child's own address space.
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    # ru_maxrss is in KiB on Linux and bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 / 1024 if sys.platform == "darwin" else peak / 1024


def _run_read(warehouse_dir: str, table_name: str, read_name: str, results):
    """Child process body: time one read and report it with this process's peak RSS."""
    catalog = CatalogFactory.create({'warehouse': warehouse_dir})
    table = catalog.get_table(f'default.{table_name}')
    t0 = time.perf_counter()
    rows = READ_FUNCTIONS[read_name](table)
    elapsed = time.perf_counter() - t0
    results.put((elapsed, rows, _peak_rss_mb()))


def measure_read(warehouse_dir: str, table_name: str, read_name: str, repeat: int) -> dict:
    ctx = multiprocessing.get_context("spawn")
    times, peaks, rows = [], [], None
    for _ in range(repeat):
        results = ctx.Queue()
        process = ctx.Process(target=_run_read, args=(warehouse_dir, table_name, read_name, results))
        process.start()
        elapsed, rows, peak_mb = results.get()
        process.join()
        times.append(elapsed)
        peaks.append(peak_mb)
    return {'time': statistics.median(times), 'peak_mb': max(peaks), 'rows': rows}


def run_benchmark(data: pa.Table, warehouse_dir: str, formats, repeat: int) -> dict:
    """Run the benchmark across all formats."""
    catalog = CatalogFactory.create({'warehouse': warehouse_dir})
    catalog.create_database('default', True)
    in_memory_mb = data.nbytes / 1024 / 1024
    results = {}

    for fmt in formats:
        table_name = f"clickbench_{fmt}"
        print(f"\n{'=' * 60}")
        print(f"  Format: {fmt.upper()}")
        print(f"{'=' * 60}")

        print(f"  Writing {data.num_rows:,} rows ...")
        write_time = write_paimon_table(catalog, table_name, data, {'file.format': fmt})
        write_paimon_table(catalog, f"{table_name}_lookup", data, {
            'file.format': fmt,
            'data-evolution.enabled': 'true',
            'row-tracking.enabled': 'true',
        })
        print(f"  Write time: {write_time:.2f}s")

        disk_mb = get_dir_size(catalog.get_table(f'default.{table_name}').table_path) / 1024 / 1024
        ratio = in_memory_mb / disk_mb if disk_mb > 0 else 0
        print(f"  On-disk size: {disk_mb:.1f} MB  (ratio: {ratio:.2f}x)")

        results[fmt] = {'write_time': write_time, 'disk_mb': disk_mb, 'ratio': ratio}
        for read_name, _, suffix in READS:
            metrics = measure_read(warehouse_dir, table_name + suffix, read_name, repeat)
            print(f"  {read_name:<17} {metrics['time']:>8.3f}s  {metrics['peak_mb']:>8.0f} MB peak"
                  f"  ({metrics['rows']:,} rows)")
            results[fmt][read_name] = metrics

    return results


def print_summary(results: dict, in_memory_mb: float, num_rows: int):
    """Print a summary comparison table."""
    formats = list(results)
    print(f"\n{'=' * 80}")
    print("  CLICKBENCH FORMAT BENCHMARK SUMMARY")
    print(f"  Rows: {num_rows:,}  |  In-memory: {in_memory_mb:.1f} MB")
    print(f"{'=' * 80}")
    print(f"  {'':<17}" + "".join(f"{fmt:>20}" for fmt in formats))
    print(f"  {'disk (MB)':<17}" + "".join(f"{results[f]['disk_mb']:>20.1f}" for f in formats))
    print(f"  {'write (s)':<17}" + "".join(f"{results[f]['write_time']:>20.2f}" for f in formats))
    for read_name, _, _ in READS:
        cells = "".join(
            f"{results[f][read_name]['time']:>9.3f}s / {results[f][read_name]['peak_mb']:>5.0f}MB"
            for f in formats)
        print(f"  {read_name:<17}" + cells)

    if 'vortex' in results and 'parquet' in results:
        v = results['vortex']
        p = results['parquet']
        print("\n  Vortex vs Parquet (time ratio, lower is better for Vortex):")
        print(f"    {'size':<17} {v['disk_mb'] / p['disk_mb']:.2f}x")
        print(f"    {'write':<17} {v['write_time'] / p['write_time']:.2f}x")
        for read_name, _, _ in READS:
            print(f"    {read_name:<17} {v[read_name]['time'] / p[read_name]['time']:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="ClickBench format benchmark for Paimon")
    parser.add_argument("--rows", type=int, default=3_000_000,
                        help="Number of rows to use (0 = full dataset, default: 3000000)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to existing hits.parquet (skips download)")
    parser.add_argument("--formats", type=str, default=None,
                        help="Comma-separated formats (default: every installed format)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Runs per read benchmark (default: 3)")
    args = parser.parse_args()

    if args.formats:
        formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    else:
        formats = [fmt for fmt, module in ALL_FORMATS.items()
                   if module is None or importlib.util.find_spec(module) is not None]
    print(f"[INFO] Formats: {', '.join(formats)}")

    if args.data_path:
        data_path = args.data_path
    else:
        cache_dir = os.path.join(Path.home(), ".cache", "paimon-bench")
        os.makedirs(cache_dir, exist_ok=True)
        data_path = os.path.join(cache_dir, "hits.parquet")
        download_clickbench(data_path)

    data = load_data(data_path, args.rows)
    in_memory_mb = data.nbytes / 1024 / 1024

    warehouse_dir = tempfile.mkdtemp(prefix="paimon_bench_")
    print(f"[INFO] Warehouse: {warehouse_dir}")

    try:
        results = run_benchmark(data, warehouse_dir, formats, args.repeat)
        print_summary(results, in_memory_mb, data.num_rows)
    finally:
        shutil.rmtree(warehouse_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
