from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np

from .common import pair_features, text_views
from .index_db import build_index, connect, prepare_oversized_blocks, retrieve_batch_candidates
from .train import select_candidates


def predict(test_dir: Path, work_dir: Path, model_path: Path, output_dir: Path, batch_entities=1000,
            max_entities: int | None = None, skip_entities: int = 0):
    payload = joblib.load(model_path)
    model, threshold = payload["model"], payload["threshold"]
    max_block = payload.get("max_block", 500)
    top_k = payload.get("top_k_per_source", 30)
    prelimit = payload.get("prelimit_per_source", 100)
    index_path = work_dir / "test_targets.sqlite"
    if not index_path.exists():
        build_index(test_dir / "test_source2.tsv", test_dir / "test_source3.tsv", index_path)
    prepare_oversized_blocks(index_path, max_block)
    conn = connect(index_path, readonly=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    with (
        (test_dir / "test_source1.tsv").open("r", encoding="utf-8", newline="") as source,
        matching_path.open("w", encoding="utf-8", newline="") as matching,
        candidate_path.open("w", encoding="utf-8", newline="") as candidate,
    ):
        reader = csv.DictReader(source, delimiter="\t")
        match_writer = csv.writer(matching, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate, delimiter="\t", lineterminator="\n")
        match_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        pending = []

        def flush():
            retrieved = retrieve_batch_candidates(conn, pending, max_block, prelimit)
            feature_matrix, offsets = [], []
            prepared = []
            for left, (rows, evidence, weighted) in zip(pending, retrieved):
                selected = select_candidates(
                    left, rows, evidence, top_k, weighted_evidence=weighted,
                    prelimit_per_source=prelimit,
                )
                start = len(feature_matrix)
                left_views = text_views(left)
                feature_matrix.extend(pair_features(left, rows[tid], evidence[tid], left_views) for tid in selected)
                offsets.append((start, len(feature_matrix)))
                prepared.append((left, selected))
            probabilities = model.predict_proba(np.asarray(feature_matrix, dtype=np.float32))[:, 1] if feature_matrix else np.empty(0)
            for (left, selected), (start, end) in zip(prepared, offsets):
                matches = [tid for tid, prob in zip(selected, probabilities[start:end]) if prob >= threshold]
                candidate_writer.writerow([left["entity_id"], ",".join(selected)])
                match_writer.writerow([left["entity_id"], ",".join(matches)])

        count = 0
        for input_number, left in enumerate(reader):
            if input_number < skip_entities:
                continue
            if max_entities is not None and count + len(pending) >= max_entities:
                break
            pending.append(left)
            if len(pending) >= batch_entities:
                flush()
                count += len(pending)
                pending.clear()
                if count <= 1_000 or count % 10_000 == 0:
                    print(f"predicted {count:,} Source-1 entities", flush=True)
        if pending:
            flush()
            count += len(pending)
    conn.close()
    return matching_path, candidate_path
