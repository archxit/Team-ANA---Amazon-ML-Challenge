from __future__ import annotations

import csv
import math
import sqlite3
import time
from collections import Counter
from pathlib import Path

from .common import record_keys

KEY_COLUMNS = [
    "k_name", "k_base", "k_signature", "k_ascii", "k_address", "k_digits", "k_prefix", "k_phonetic",
    *[f"name_mh{i}" for i in range(5)], *[f"address_mh{i}" for i in range(4)],
    *[f"name_token_mh{i}" for i in range(2)], *[f"address_token_mh{i}" for i in range(2)],
]


def connect(path: Path, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-262144")
        conn.execute("PRAGMA mmap_size=1073741824")
    else:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=FILE")
        conn.execute("PRAGMA cache_size=-524288")
    return conn


def build_index(source2: Path, source3: Path, database: Path, batch_size: int = 50_000) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {database}")
    conn = connect(database)
    columns_sql = ",\n".join(f"{column} INTEGER" for column in KEY_COLUMNS)
    conn.execute(f"""
        CREATE TABLE records (
            entity_id TEXT NOT NULL UNIQUE,
            source INTEGER NOT NULL,
            country TEXT NOT NULL,
            business_name TEXT NOT NULL,
            business_address TEXT NOT NULL,
            {columns_sql}
        )
    """)
    placeholders = ",".join("?" for _ in range(5 + len(KEY_COLUMNS)))
    insert_sql = f"INSERT INTO records VALUES ({placeholders})"
    total = 0
    for source, path in ((2, source2), (3, source3)):
        batch = []
        started = time.time()
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                keys = record_keys(row["business_name"], row["business_address"])
                batch.append((
                    row["entity_id"], source, row["country"], row["business_name"],
                    row["business_address"], *[keys[column] for column in KEY_COLUMNS],
                ))
                if len(batch) >= batch_size:
                    conn.executemany(insert_sql, batch)
                    conn.commit()
                    total += len(batch)
                    batch.clear()
                    if total % 500_000 == 0:
                        print(f"indexed {total:,} records", flush=True)
        if batch:
            conn.executemany(insert_sql, batch)
            conn.commit()
            total += len(batch)
        print(f"loaded source {source} in {time.time() - started:.1f}s", flush=True)
    for column in KEY_COLUMNS:
        print(f"building index on {column}", flush=True)
        conn.execute(f"CREATE INDEX idx_{column} ON records(country, {column}) WHERE {column} IS NOT NULL")
        conn.commit()
    conn.execute("ANALYZE")
    conn.close()


def prepare_oversized_blocks(database: Path, max_block: int) -> None:
    """Materialize only keys whose postings exceed the allowed block size."""
    conn = connect(database)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS oversized_blocks ("
        "column_name TEXT, country TEXT, value INTEGER, "
        "PRIMARY KEY (column_name,country,value)) WITHOUT ROWID"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS oversized_metadata ("
        "column_name TEXT PRIMARY KEY, max_block INTEGER) WITHOUT ROWID"
    )
    configured = dict(conn.execute("SELECT column_name,max_block FROM oversized_metadata"))
    if configured and any(value != max_block for value in configured.values()):
        conn.execute("DELETE FROM oversized_blocks")
        conn.execute("DELETE FROM oversized_metadata")
        configured = {}
        conn.commit()
    for column in KEY_COLUMNS:
        if configured.get(column) == max_block:
            continue
        print(f"profiling oversized blocks for {column}", flush=True)
        conn.execute(
            f"INSERT OR IGNORE INTO oversized_blocks "
            f"SELECT ?,country,{column} FROM records INDEXED BY idx_{column} "
            f"WHERE {column} IS NOT NULL GROUP BY country,{column} HAVING COUNT(*)>?",
            (column, max_block),
        )
        conn.execute("INSERT OR REPLACE INTO oversized_metadata VALUES (?,?)", (column, max_block))
        conn.commit()
    conn.close()


def retrieve_raw_candidates(conn: sqlite3.Connection, left: dict, max_block: int) -> tuple[dict[str, dict], Counter, Counter]:
    keys = record_keys(left["business_name"], left["business_address"])
    rows: dict[str, dict] = {}
    evidence: Counter = Counter()
    weighted: Counter = Counter()
    for column, value in keys.items():
        if value is None:
            continue
        found = conn.execute(
            f"SELECT entity_id,source,country,business_name,business_address FROM records "
            f"WHERE country=? AND {column}=? LIMIT ?",
            (left["country"], value, max_block + 1),
        ).fetchall()
        if len(found) > max_block:
            continue
        weight = 1.0 / math.log2(2.0 + len(found))
        for entity_id, source, country, name, address in found:
            rows[entity_id] = {
                "entity_id": entity_id, "source": source, "country": country,
                "business_name": name, "business_address": address,
            }
            evidence[entity_id] += 1
            weighted[entity_id] += weight
    return rows, evidence, weighted


def retrieve_batch_candidates(
    conn: sqlite3.Connection, left_rows: list[dict], max_block: int,
    prelimit_per_source: int = 100,
) -> list[tuple[dict[str, dict], Counter, Counter]]:
    """Retrieve blocks for many S1 rows with batched UNION ALL index probes.

    This is equivalent to ``retrieve_raw_candidates`` but avoids issuing one
    Python/SQLite round trip for every key of every entity. Each individual
    block remains capped independently, preserving the candidate definition.
    """
    outputs = [({}, Counter(), Counter()) for _ in left_rows]
    keys_by_row = [record_keys(row["business_name"], row["business_address"]) for row in left_rows]
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS query_keys (qid INTEGER PRIMARY KEY, country TEXT, value INTEGER)")
    for column in KEY_COLUMNS:
        block_to_rows: dict[tuple[str, int], list[int]] = {}
        for row_number, (left, keys) in enumerate(zip(left_rows, keys_by_row)):
            value = keys[column]
            if value is not None:
                block_to_rows.setdefault((left["country"], value), []).append(row_number)
        blocks = list(block_to_rows)
        conn.execute("DELETE FROM query_keys")
        conn.executemany(
            "INSERT INTO query_keys VALUES (?,?,?)",
            ((qid, country, value) for qid, (country, value) in enumerate(blocks)),
        )
        conn.execute(
            "DELETE FROM query_keys WHERE EXISTS ("
            "SELECT 1 FROM oversized_blocks o WHERE o.column_name=? "
            "AND o.country=query_keys.country AND o.value=query_keys.value)",
            (column,),
        )
        found_by_block: dict[int, list[tuple]] = {}
        for qid, rowid, entity_id, source in conn.execute(
            f"SELECT q.qid,r.rowid,r.entity_id,r.source "
            f"FROM query_keys q "
            f"CROSS JOIN records r INDEXED BY idx_{column} "
            f"ON r.country=q.country AND r.{column}=q.value"
        ):
            found_by_block.setdefault(qid, []).append((rowid, entity_id, source))
        for qid, found in found_by_block.items():
            weight = 1.0 / math.log2(2.0 + len(found))
            for row_number in block_to_rows[blocks[qid]]:
                rows, evidence, weighted = outputs[row_number]
                for rowid, entity_id, source in found:
                    rows[entity_id] = {"_rowid": rowid, "entity_id": entity_id, "source": source}
                    evidence[entity_id] += 1
                    weighted[entity_id] += weight
    needed_rowids = set()
    selected_ids = []
    for rows, evidence, weighted in outputs:
        prelim = {2: [], 3: []}
        for entity_id, row in rows.items():
            prelim[row["source"]].append((weighted[entity_id], entity_id))
        chosen = []
        for source in (2, 3):
            prelim[source].sort(reverse=True)
            chosen.extend(entity_id for _, entity_id in prelim[source][:prelimit_per_source])
        selected_ids.append(chosen)
        needed_rowids.update(rows[entity_id]["_rowid"] for entity_id in chosen)
    full_rows = {}
    rowids = list(needed_rowids)
    for start in range(0, len(rowids), 500):
        chunk = rowids[start:start + 500]
        placeholders = ",".join("?" for _ in chunk)
        for rowid, entity_id, source, country, name, address in conn.execute(
            f"SELECT rowid,entity_id,source,country,business_name,business_address "
            f"FROM records WHERE rowid IN ({placeholders})", chunk,
        ):
            full_rows[rowid] = {
                "entity_id": entity_id, "source": source, "country": country,
                "business_name": name, "business_address": address,
            }
    for output, chosen in zip(outputs, selected_ids):
        rows, evidence, weighted = output
        retained = {}
        for entity_id in chosen:
            retained[entity_id] = full_rows[rows[entity_id]["_rowid"]]
        rows.clear()
        rows.update(retained)
    return outputs
