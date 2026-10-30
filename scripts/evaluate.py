"""Score a submission in-process with the official kit's scoring functions.

Same frame, cells, floor and skill as kit/score.py, without the per-run venv,
so a candidate can be scored on many datasets quickly. Exact (unnoised) skill.

    python scripts/evaluate.py --sub surveycast --data work/sbx/unicef \
        --schema kit/data/unicef.json [--phase 2] [--no-deadline] [--json out.json]

The kit is the official starter kit (github.com/SituatedEvals/public), cloned
into ./kit or pointed to with the KIT environment variable.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.environ.get("KIT", os.path.join(ROOT, "kit"))
sys.path.insert(0, KIT)

import score as kit  # noqa: E402


def load_module(sub):
    spec = importlib.util.spec_from_file_location(
        "sub_main_%d" % abs(hash(sub)), os.path.join(sub, "main.py"))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, sub)
    spec.loader.exec_module(module)
    return module


def evaluate(module, data, schema_path, phase=1):
    config = kit.load_config(os.path.join(KIT, "config.yml"))
    schema = kit.load_schema(schema_path, config)
    respondents = kit.load_frames(data, schema)
    masked, cells, truth = kit.sample_rows(schema, respondents, phase)
    floor = config["scoring"]["floor"]
    reference = kit.marginal_reference(schema, masked, cells, truth, floor)
    frame = masked.copy()
    started = time.time()
    vectors = module.predict(frame, json.loads(json.dumps(schema)))
    seconds = time.time() - started
    kit.check(schema, vectors, cells)
    vectors = kit.floored(vectors, schema, cells, floor)
    result = kit.score(schema, config, vectors, truth, cells, reference)
    return {"skill": result["skill"], "std_error": result["std_error"],
            "log_score": result["log_score"], "seconds": seconds,
            "n_cells": result["n_cells"], "by_item": result["by_item"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--json", default=None)
    parser.add_argument("--no-deadline", action="store_true",
                        help="lift the submission's time guard (CPU stand-in for the GPU run)")
    args = parser.parse_args()
    module = load_module(os.path.abspath(args.sub))
    if args.no_deadline and hasattr(module, "BUDGET"):
        module.BUDGET = {1: 1e9, 2: 1e9}
    result = evaluate(module, args.data, args.schema, args.phase)
    by_item = result.pop("by_item")
    print(json.dumps(result))
    if args.json:
        result["by_item"] = by_item["mean"].to_dict()
        with open(args.json, "w") as fh:
            json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
