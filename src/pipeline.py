from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predict import predict
from .train import train_pipeline


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge entity resolution pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--train-dir", type=Path, required=True)
    train.add_argument("--work-dir", type=Path, required=True)
    train.add_argument("--model", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--sample-percent", type=int, default=1)
    train.add_argument("--max-block", type=int, default=500)
    train.add_argument("--top-k", type=int, default=30)
    train.add_argument("--prelimit", type=int, default=30)
    infer = sub.add_parser("predict")
    infer.add_argument("--test-dir", type=Path, required=True)
    infer.add_argument("--work-dir", type=Path, required=True)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--max-entities", type=int)
    infer.add_argument("--skip-entities", type=int, default=0)
    infer.add_argument("--batch-entities", type=int, default=1000)
    args = parser.parse_args()
    if args.command == "train":
        report = train_pipeline(args.train_dir, args.work_dir, args.model, args.report,
                                args.sample_percent, args.max_block, args.top_k, args.prelimit)
        print(json.dumps(report, indent=2))
    else:
        matching, candidates = predict(
            args.test_dir, args.work_dir, args.model, args.output_dir,
            batch_entities=args.batch_entities, max_entities=args.max_entities,
            skip_entities=args.skip_entities,
        )
        print(matching)
        print(candidates)


if __name__ == "__main__":
    main()
