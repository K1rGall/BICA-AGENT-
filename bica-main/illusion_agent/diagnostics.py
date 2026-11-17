"""Technical run diagnostics (not claims of model accuracy)."""
import json
from collections import defaultdict
from pathlib import Path


def analyze_run(run):
    root = Path(run)
    results = [json.loads(x) for x in (root / "results.jsonl").read_text().splitlines() if x.strip()]
    indexed = {r["case_id"]: r for r in results}
    if len(indexed) != len(results):
        raise ValueError("Duplicate case_id in results")
    by_mode = defaultdict(lambda: {"cases": 0, "errors": 0, "attempted_calls": 0, "completed_calls": 0,
                                   "elapsed_s": 0.0, "input_tokens": 0, "output_tokens": 0,
                                   "reflection_disagreement": 0, "flagged_cases": 0})
    for row in results:
        b = by_mode[row["mode"]]
        b["cases"] += 1
        b["errors"] += int(row["status"] != "ok")
    trace = root / "trace.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            if not line.strip(): continue
            event = json.loads(line)
            case_id = event.get("case_id")
            # New traces include a mode on each call, including interrupted
            # attempts not yet represented in results.jsonl. Old traces can be
            # interpreted by looking up completed case metadata.
            mode = event.get("mode") or indexed.get(case_id, {}).get("mode")
            if not mode:
                continue
            b = by_mode[mode]
            if event.get("event") == "request":
                b["attempted_calls"] += 1
            if event.get("event") == "response":
                b["completed_calls"] += 1
                b["elapsed_s"] += float(event.get("elapsed_s", 0))
                usage = event.get("usage") or {}
                b["input_tokens"] += int(usage.get("input_tokens") or usage.get("prompt_eval_count") or 0)
                b["output_tokens"] += int(usage.get("output_tokens") or usage.get("eval_count") or 0)
            elif event.get("event") == "reflection_flags":
                b["flagged_cases"] += 1
                b["reflection_disagreement"] += int(event["label_disagreement"])
    return {"note": "Technical diagnostics only: not visual correctness.",
            "synthetic": bool(json.loads((root / "config.json").read_text()).get("synthetic", False)),
            "by_mode": [{"mode": mode, **values,
                          "completion_rate": (values["cases"] - values["errors"]) / values["cases"] if values["cases"] else None}
                         for mode, values in sorted(by_mode.items())]}
