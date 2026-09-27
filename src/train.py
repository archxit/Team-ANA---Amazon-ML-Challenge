from __future__ import annotations

import csv
import json
import sqlite3
import zlib
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import FEATURE_NAMES, cheap_score, pair_features, text_views
from .index_db import build_index, connect, retrieve_raw_candidates
from .metrics import macro_metrics, optimize_threshold


def bucket(value: str, modulo: int) -> int:
    return zlib.crc32(value.encode("utf-8")) % modulo


def load_sample_truth(path: Path, sample_percent: int) -> dict[str, set[str]]:
    truth = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            sid = row["source1_entity_id"]
            if bucket(sid, 100) < sample_percent:
                raw = row["matched_entity_ids"].strip()
                truth[sid] = set(raw.split(",")) if raw else set()
    return truth


def select_candidates(left, rows, evidence, top_k_per_source, weighted_evidence=None, rank_model=None,
                      prelimit_per_source=100):
    ranked = {2: [], 3: []}
    entity_ids = list(rows)
    left_views = text_views(left)
    if weighted_evidence is not None:
        prelim = {2: [], 3: []}
        for entity_id in entity_ids:
            source = 3 if entity_id.startswith("S3-") else 2
            prelim[source].append((weighted_evidence[entity_id], entity_id))
        entity_ids = []
        for source in (2, 3):
            prelim[source].sort(reverse=True)
            entity_ids.extend(entity_id for _, entity_id in prelim[source][:prelimit_per_source])
        if prelimit_per_source <= top_k_per_source:
            return entity_ids
        scores = [cheap_score(left, rows[entity_id], evidence[entity_id], left_views) for entity_id in entity_ids]
    elif rank_model is None:
        scores = [cheap_score(left, rows[entity_id], evidence[entity_id], left_views) for entity_id in entity_ids]
    else:
        vectors = np.asarray(
            [pair_features(left, rows[entity_id], evidence[entity_id], left_views) for entity_id in entity_ids],
            dtype=np.float32,
        )
        scores = rank_model.predict_proba(vectors)[:, 1] if len(vectors) else []
    for entity_id, score in zip(entity_ids, scores):
        source = 3 if entity_id.startswith("S3-") else 2
        ranked[source].append((float(score), entity_id))
    selected = []
    for source in (2, 3):
        ranked[source].sort(reverse=True)
        selected.extend(entity_id for _, entity_id in ranked[source][:top_k_per_source])
    return selected


def prepare_training_data(train_dir: Path, index_path: Path, sample_percent: int, max_block: int,
                          top_k: int, prelimit_per_source: int):
    truth = load_sample_truth(train_dir / "train_ground_truth.tsv", sample_percent)
    conn = connect(index_path, readonly=True)
    features, labels, groups, group_ids, truth_count, countries = [], [], [], [], [], []
    raw_links = kept_links = raw_pairs = kept_pairs = 0
    with (train_dir / "train_source1.tsv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            sid = row["entity_id"]
            if sid not in truth:
                continue
            group = len(group_ids)
            group_ids.append(sid)
            truth_count.append(len(truth[sid]))
            countries.append(row["country"])
            candidates, evidence, weighted = retrieve_raw_candidates(conn, row, max_block)
            selected = select_candidates(
                row, candidates, evidence, top_k, weighted_evidence=weighted,
                prelimit_per_source=prelimit_per_source,
            )
            raw_links += len(truth[sid] & candidates.keys())
            kept_links += len(truth[sid] & set(selected))
            raw_pairs += len(candidates)
            kept_pairs += len(selected)
            left_views = text_views(row)
            for target_id in selected:
                features.append(pair_features(row, candidates[target_id], evidence[target_id], left_views))
                labels.append(int(target_id in truth[sid]))
                groups.append(group)
            if len(group_ids) % 1000 == 0:
                print(f"training candidates for {len(group_ids):,} entities", flush=True)
    conn.close()
    return (
        np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.uint8),
        np.asarray(groups, dtype=np.int32), group_ids, np.asarray(truth_count, dtype=np.int16),
        np.asarray(countries), {
            "truth_links": sum(map(len, truth.values())), "raw_links": raw_links,
            "kept_links": kept_links, "raw_pairs": raw_pairs, "kept_pairs": kept_pairs,
        },
    )


def train_pipeline(train_dir: Path, work_dir: Path, model_path: Path, report_path: Path,
                   sample_percent=1, max_block=500, top_k=30, prelimit_per_source=30):
    work_dir.mkdir(parents=True, exist_ok=True)
    index_path = work_dir / "train_targets.sqlite"
    if not index_path.exists():
        build_index(train_dir / "train_source2.tsv", train_dir / "train_source3.tsv", index_path)
    X, y, groups, group_ids, truth_count, countries, blocking = prepare_training_data(
        train_dir, index_path, sample_percent, max_block, top_k, prelimit_per_source
    )
    folds_group = np.asarray([bucket(sid, 3) for sid in group_ids], dtype=np.uint8)
    folds_row = folds_group[groups]
    specs = {
        "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(
            C=2.0, max_iter=500, class_weight="balanced", random_state=2026
        )),
        "lightgbm": lambda: lgb.LGBMClassifier(
            objective="binary", n_estimators=450, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, subsample=0.85, colsample_bytree=0.9,
            reg_lambda=2.0, class_weight="balanced", random_state=2026, n_jobs=4, verbosity=-1,
        ),
    }
    results, probabilities = {}, {}
    for name, factory in specs.items():
        oof = np.zeros(len(y), dtype=np.float32)
        fold_results = []
        for fold in range(3):
            model = factory()
            train_rows, valid_rows = folds_row != fold, folds_row == fold
            model.fit(X[train_rows], y[train_rows])
            oof[valid_rows] = model.predict_proba(X[valid_rows])[:, 1]
            best, _ = optimize_threshold(y, oof, groups, truth_count, folds_group == fold)
            best["fold"] = fold
            fold_results.append(best)
        best, grid = optimize_threshold(y, oof, groups, truth_count)
        results[name] = {"oof_best": best, "fold_best": fold_results}
        probabilities[name] = oof
    winner = max(results, key=lambda name: results[name]["oof_best"]["macro_f05"])
    threshold = results[winner]["oof_best"]["threshold"]
    probability = probabilities[winner]
    breakdown = {}
    for country in sorted(set(countries)):
        breakdown[f"country={country}"] = macro_metrics(y, probability, groups, truth_count, threshold, countries == country)
    breakdown["singletons"] = macro_metrics(y, probability, groups, truth_count, threshold, truth_count == 0)
    breakdown["one_match"] = macro_metrics(y, probability, groups, truth_count, threshold, truth_count == 1)
    breakdown["multi_match"] = macro_metrics(y, probability, groups, truth_count, threshold, truth_count > 1)
    final_model = specs[winner]()
    final_model.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model, "threshold": threshold, "feature_names": FEATURE_NAMES,
        "max_block": max_block, "top_k_per_source": top_k,
        "prelimit_per_source": prelimit_per_source,
    }, model_path)
    importance = None
    if winner == "lightgbm":
        importance = sorted(zip(FEATURE_NAMES, map(float, final_model.feature_importances_)), key=lambda x: -x[1])
    report = {
        "sample_percent": sample_percent, "entities": len(group_ids), "pairs": len(y),
        "prelimit_per_source": prelimit_per_source,
        "positives": int(y.sum()), "blocking": blocking, "models": results, "winner": winner,
        "threshold": threshold, "breakdown": breakdown, "feature_importance": importance,
        "validation": "Three-fold entity-level out-of-fold validation.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
