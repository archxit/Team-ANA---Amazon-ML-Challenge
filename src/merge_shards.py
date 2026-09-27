from __future__ import annotations

import argparse
from pathlib import Path


EXPECTED_HEADERS = {
    "matching_results.tsv": b"source1_entity_id\tmatched_entity_ids\n",
    "candidate_pairs.tsv": b"source1_entity_id\tcandidate_entity_ids\n",
}


def merge_file(shard_dirs: list[Path], output_path: Path, expected_rows: int | None) -> int:
    expected_header = EXPECTED_HEADERS[output_path.name]
    rows = 0
    with output_path.open("wb") as destination:
        destination.write(expected_header)
        for shard_dir in shard_dirs:
            path = shard_dir / output_path.name
            with path.open("rb") as source:
                header = source.readline()
                if header != expected_header:
                    raise ValueError(f"Unexpected header in {path}: {header!r}")
                while chunk := source.read(8 * 1024 * 1024):
                    rows += chunk.count(b"\n")
                    destination.write(chunk)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"{output_path.name}: expected {expected_rows:,} rows, found {rows:,}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ordered prediction shards")
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    shard_dirs = [args.shard_root / f"shard_{number}" for number in range(args.shard_count)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in EXPECTED_HEADERS:
        rows = merge_file(shard_dirs, args.output_dir / filename, args.expected_rows)
        print(f"{filename}: {rows:,} rows")


if __name__ == "__main__":
    main()
