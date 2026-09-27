# Amazon ML Challenge 2026 - Business Entity Resolution

This package implements an out-of-core, multi-match entity-resolution pipeline. It uses only the supplied challenge data. No external entity lookup, external database, geocoder, API, or internet-derived augmentation is used.

## Environment

- Python 3.12 (64-bit)
- Recommended: at least 16 GB RAM and 25 GB free disk space
- The source TSVs are streamed; they are never loaded in full.
- SQLite indexes are written under the chosen work directory and may take several hours to build on a laptop.

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Expected data layout

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

## Train

Run from `code/business_entity_resolution/`:

```powershell
.venv\Scripts\python -m src.pipeline train `
  --train-dir C:\path\to\dataset\train `
  --work-dir C:\path\to\work `
  --model C:\path\to\work\model.joblib `
  --report C:\path\to\work\validation_report.json `
  --sample-percent 1 `
  --max-block 500 `
  --top-k 30 `
  --prelimit 30
```

The deterministic sample is selected by Source-1 entity ID. Validation is three-fold and entity-level, preventing pairs from one reference entity from leaking across train and validation. Increase `--sample-percent` if compute permits.

## Predict

```powershell
.venv\Scripts\python -m src.pipeline predict `
  --test-dir C:\path\to\dataset\test `
  --work-dir C:\path\to\work `
  --model C:\path\to\work\model.joblib `
  --output-dir C:\path\to\submission\output
```

This creates exactly:

- `matching_results.tsv`
- `candidate_pairs.tsv`

Each Source-1 test entity is streamed once and written exactly once. The candidates file contains the final top-K-per-source set actually scored by the model. Empty fields represent singletons.

The submission also includes `artifacts/model.joblib`, the exact fitted model used for the final outputs. For multi-process inference, `--skip-entities`, `--max-entities`, and `--batch-entities` create non-overlapping ordered shards; concatenate them in increasing skip order while retaining only the first header.

Eight equal shards for the official 1,732,544-row test set can be merged and row-count checked with:

```powershell
.venv\Scripts\python -m src.merge_shards `
  --shard-root C:\path\to\work\shards `
  --shard-count 8 `
  --output-dir C:\path\to\submission\output `
  --expected-rows 1732544
```

## Validate

Run the official validator from the supplied `student_resource` directory:

```powershell
python utils\validate_submission.py `
  --matching C:\path\to\submission\output\matching_results.tsv `
  --candidate C:\path\to\submission\output\candidate_pairs.tsv `
  --test-dir dataset\test
```

Use `--check-ids` only on a machine with sufficient memory, as noted by the official validator.

The bundled final outputs passed the official validator both normally and with `--check-ids`: 1,732,544 rows were accepted in each TSV and every referenced S2/S3 ID was valid.

## Reproducibility and fair play

- Random-state-bearing estimators use seed `2026`.
- Sampling and folds use deterministic CRC32 buckets of supplied entity IDs.
- Normalization is deterministic and preserves raw values in the SQLite record index.
- Country is treated as an open-set string and is never limited to the training labels.
- The trained model is LightGBM or logistic regression. LightGBM is MIT-licensed; no pretrained model is used and the 8B-parameter limit is therefore inapplicable in practice.
- This solution code is provided under the bundled MIT `LICENSE`.
