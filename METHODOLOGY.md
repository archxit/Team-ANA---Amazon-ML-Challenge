# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Fill before submission]  
**Team Members:** [Fill before submission]  
**Submission Date:** 26 September 2026

## 1. Executive Summary

We formulate the task as high-recall, multi-pass blocking followed by a precision-oriented pair classifier and explicit singleton decisions. The solution is fully out-of-core: supplied TSVs are streamed, target records are indexed on disk, candidates are ranked to a bounded final set, and a LightGBM classifier is thresholded using entity-level out-of-fold macro F0.5. No external entity data or lookup service is used.

## 2. Problem Analysis and EDA

The official task permits zero, one, or many Source-2/Source-3 matches for each deduplicated Source-1 entity. The metric is macro F0.5, including exact credit for correctly empty singleton predictions. Therefore no one-to-one assignment is imposed.

### 2.1 Scale and schema

All seven files are UTF-8 TSVs. Every source table contains `entity_id`, `business_name`, `business_address`, and `country`; ground truth contains `source1_entity_id` and `matched_entity_ids`.

| Split/file | Rows |
|---|---:|
| train_source1 | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| train_ground_truth | 2,206,821 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

Training Source 1 contains 1,323,633 US and 883,188 India rows. Test Source 1 contains 663,106 US, 809,986 India, and 259,452 France rows. Country is consequently treated as an open-set string rather than a training-only category.

Names, IDs, and countries have no missing values. Addresses are complete in Source 1, but missing in 3.356% of training Source 2 and 3.328% of training Source 3. Median name length is 24-25 characters (four words); median address length is 37-42 characters in noisy training sources. Non-ASCII names occur in 15.19% of training Source 2 and 11.48% of Source 3, motivating a parallel transliterated representation while retaining the native form.

All entity IDs are unique. Source 1 has no duplicate full rows. Exact full-row duplicates excluding ID are 0.514% in training Source 2 and 0.357% in Source 3; after conservative normalization they are 1.337% and 0.947%. These IDs are preserved independently because multiple records may legitimately match one reference entity.

### 2.2 Ground-truth structure

Training ground truth contains 7,638,365 links: 3,693,619 to Source 2 and 3,944,746 to Source 3.

- 123,247 Source-1 entities are singletons (5.585%).
- 1,964,417 entities have more than one match (89.016%).
- 1,129,968 have multiple Source-2 matches.
- 1,224,128 have multiple Source-3 matches.
- 1,776,047 match both Source 2 and Source 3.
- Observed total cardinality ranges from 0 through 11.

These measurements rule out one-to-one matching and justify independent pair decisions plus an entity-level empty-set threshold.

## 3. Preprocessing

Raw values are preserved in the disk index. Derived views use Unicode NFKC, case folding, ampersand expansion, punctuation-to-space conversion, and whitespace collapse. Business-name features additionally use a reversible parallel view with common legal suffix tokens removed, sorted token signatures, transliterated compact forms, phonetic signatures, and character/token MinHash keys. Address views retain native text, normalized text, number groups, and transliterated character/token keys. Missing addresses remain explicit and produce missingness features; no placeholder is allowed to create accidental equality.

Transformations intentionally do not discard the raw name or address. Native-script and transliterated similarities coexist, so transliteration cannot erase useful discriminative evidence.

## 4. Candidate Generation (Blocking)

Candidates are generated within country using multiple disk-indexed passes:

- exact normalized name, legal-suffix-reduced name, and sorted name signature;
- transliterated exact name and name prefix;
- exact normalized address and digit signature;
- fixed-seed character MinHash keys for names and addresses;
- fixed-seed token MinHash keys;
- a phonetic name signature.

Blocks larger than 500 records are ignored to prevent generic names, cities, or address fragments from exploding the pair space. Remaining evidence is rarity weighted, with weight inversely related to block size. Candidates are ranked independently for Source 2 and Source 3, and the top 30 per source becomes the final set sent to the matching classifier and written to `candidate_pairs.tsv`.

On a deterministic 1% entity sample (22,182 Source-1 entities and 76,730 true links), the enhanced blocker reaches 92.832% raw recall. Rarity-ranked top-30 candidates per source retain 82.236% of true links at a 99.99943% reduction ratio (84.228% for S1-S2 and 80.369% for S1-S3). Direct true-pair analysis found that 99.75% share at least one uncapped direct key and that enhanced phonetic/token keys cover 99.98%, demonstrating that block frequency control and ranking—not absence of a usable key—are the remaining bottlenecks.

## 5. Pairwise Features and Matching Model

The final feature vector contains:

- normalized-name and suffix-reduced exact equality;
- name edit ratio, weighted ratio, partial ratio, token-set ratio, token Jaccard, and length ratio;
- normalized-address exact equality and the analogous similarity family;
- numeric-token Jaccard and full digit-signature equality;
- missing-address indicator;
- source indicator;
- count of independent blocking passes supporting the pair.

Logistic regression is the linear baseline. LightGBM is the nonlinear model. Both use exactly the same entity-level folds and candidate set. LightGBM 4.7.0 is MIT-licensed; no pretrained model is used.

## 6. Validation and Threshold Selection

Sampling and folds are deterministic functions of supplied Source-1 IDs. Three-fold validation is split by Source-1 entity, so no pair from an entity can occur in both training and validation within a fold. Candidate misses remain false negatives in macro scoring, and singleton entities remain in the denominator even when they have no candidates.

On the final deployable candidate set, logistic regression reached macro F0.5 0.7641. LightGBM reached 0.8531 with precision 0.9676, recall 0.7332, and singleton accuracy 0.8976. The selected probability threshold is 0.960; per-fold optima were 0.960, 0.960, and 0.955, indicating stable threshold selection rather than one-split tuning. A slower fuzzy pre-ranking variant reached 0.8695, but its single-process test runtime was projected at approximately 40 hours; the rarity-ranked variant was selected for the final reproducible run after measuring this quality/runtime trade-off.

Breakdown at the selected threshold:

| Slice | Macro F0.5 |
|---|---:|
| US | 0.8817 |
| India | 0.8096 |
| Singletons | 0.8976 |
| One-match entities | 0.6720 |
| Multi-match entities | 0.8613 |

The model is notably harder on India and on one-match entities. Source-pair precision is similar: 0.9671 for S1-S2 and 0.9682 for S1-S3.

## 7. Error Analysis and Iteration

The initial heuristic matcher achieved macro F0.5 0.736 with precision 0.820 on the enhanced blocks. Replacing it with LightGBM increased macro F0.5 to 0.853 while raising precision to 0.968. Feature importance is distributed across name Jaccard/partial/edit similarities, address token and Jaccard features, length ratios, number overlap, and block evidence; exact equality alone is insufficient.

False negatives arise from two distinct stages: candidates lost through high-frequency block caps/ranking, and blocked positives below the conservative final threshold. False positives are concentrated among high-similarity businesses with shared generic name/address fragments. We therefore add independent rare locality keys and retain a high probability threshold instead of globally relaxing precision. Missing addresses and source-specific noise are exposed to the classifier rather than handled with forced rules.

## 8. Final Inference and Singleton Logic

For each test Source-1 row, the pipeline queries the disk index, skips oversized blocks, ranks candidates per noisy source, computes the final feature matrix, and applies the OOF-selected threshold. Any number of S2/S3 pairs may pass. If none pass—or no candidates exist—the output list is empty. This is the explicit singleton decision; no forced best match is emitted.

The writer streams both required TSVs and guarantees:

- exactly one output row per test Source-1 ID;
- only Source-2/Source-3 IDs retrieved from the supplied test tables;
- no duplicate IDs within a list;
- every final match is in the final candidate list;
- UTF-8, tab-separated output with exact headers.

The official `validate_submission.py` was run after inference in both modes. The standard validation passed, and the full `--check-ids` validation also passed against all 9,969,589 supplied test S2/S3 IDs. The final files contain exactly 1,732,544 rows each. `matching_results.tsv` contains 168,426 empty predictions and 1,564,118 non-empty predictions; `candidate_pairs.tsv` contains 141 empty candidate lists and 1,732,403 non-empty lists.

## 9. Computational Considerations

Source TSVs are streamed. High-cardinality lookup structures reside in SQLite, not Python dictionaries. Oversized blocks are profiled once, compact row IDs carry blocking evidence, and full text is fetched only for bounded top-K candidates. Final inference was divided into eight ordered, non-overlapping shards of 216,568 rows and concatenated without changing predictions. The definitive run started at 15:44 on 26 September and completed at 15:14 on 27 September (about 23.5 hours elapsed, including an overnight host suspension); observed active-compute progress corresponds to roughly 2.5-3 hours. The merged candidate file is approximately 1.34 GB and the matching file approximately 87.5 MB. The approach is CPU- and disk-intensive during one-time index construction but has bounded memory and is reproducible on a workstation.

## 10. Limitations

France has no labeled training entities, so its performance cannot be measured offline. The pipeline handles it structurally through open-set country blocking and language-agnostic character/token features, but threshold calibration transfers from US/India. Candidate recall is sensitive to block caps and top-K; systems with more disk/time may increase these values and revalidate. Transliteration is algorithmic and may not capture all pronunciation variants.

## Appendix: Code Artefacts

All executable code is under `code/business_entity_resolution/src/`. `src.pipeline train` creates the disk index, entity-level validation report, and model. `src.pipeline predict` creates both required TSVs. Exact commands and pinned dependencies are in `code/business_entity_resolution/README.md` and `requirements.txt`.
