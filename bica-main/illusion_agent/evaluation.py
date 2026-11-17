"""Aggregate adjudicated human ratings; never use the agent as its own judge."""
import csv
import json
from collections import defaultdict
from pathlib import Path

METRICS = ("both_valid", "false_alternative", "cue_correct", "grounded")


def aggregate(mapping, ratings, annotations):
    items = {r["review_id"]: r for r in mapping}
    scores = {r["review_id"]: r for r in ratings}
    if len(items) != len(mapping) or len(scores) != len(ratings) or items.keys() != scores.keys():
        raise ValueError("Ratings must contain every review ID exactly once")
    groups = defaultdict(lambda: {"n": 0, "technical_errors": 0,
                                  **{m: [] for m in METRICS}})
    for rid, row in items.items():
        gold = annotations[row["image_id"]]
        if gold.get("review_status") != "adjudicated":
            raise ValueError("Annotations must be adjudicated before scoring")
        group = gold["group"]
        cue_kind = gold["cues"][row["cue_id"]]
        if group not in {"ambiguous", "ordinary", "abstract"} or cue_kind not in {"neutral", "supported", "misleading"}:
            raise ValueError("Unknown group or cue kind")
        bucket = groups[(row["mode"], group, cue_kind)]
        bucket["n"] += 1
        failed = row["status"] != "ok"
        bucket["technical_errors"] += int(failed)
        applicability = {"both_valid": group == "ambiguous", "false_alternative": group == "ordinary",
                         "cue_correct": cue_kind != "neutral", "grounded": True}
        for metric in METRICS:
            score = scores[rid].get(metric, "").strip()
            applicable = applicability[metric]
            if not applicable or failed:
                if score != "NA":
                    raise ValueError(f"{rid}/{metric}: use NA for inapplicable or technical errors")
                # Technical failures are failures for success metrics, not evidence of false alternatives.
                if applicable and failed and metric != "false_alternative":
                    bucket[metric].append(0)
            else:
                if score not in {"0", "1"}:
                    raise ValueError(f"{rid}/{metric}: completed applicable rating must be 0 or 1")
                bucket[metric].append(int(score))
    result = []
    for (mode, group, cue), bucket in sorted(groups.items()):
        record = {"mode": mode, "group": group, "cue_kind": cue,
                  "n": bucket["n"], "technical_errors": bucket["technical_errors"]}
        for metric in METRICS:
            values = bucket[metric]
            record[metric + "_n"] = len(values)
            record[metric + "_rate"] = sum(values) / len(values) if values else None
        result.append(record)
    return result


def score(args):
    run_dir = Path(args.run)
    if json.loads((run_dir / "config.json").read_text())["synthetic"]:
        raise ValueError("Synthetic runs cannot produce research metrics")
    mapping = json.loads(Path(args.mapping).read_text())
    with Path(args.ratings).open() as f:
        ratings = list(csv.DictReader(f))
    raw = [json.loads(s) for s in Path(args.annotations).read_text().splitlines() if s.strip()]
    annotations = {r["id"]: r for r in raw}
    if len(raw) != len(annotations):
        raise ValueError("Duplicate annotations")
    run_rows = [json.loads(s) for s in (run_dir / "results.jsonl").read_text().splitlines()]
    indexed = {r["case_id"]: r for r in run_rows}
    if {r["case_id"] for r in mapping} != indexed.keys() or len(mapping) != len(indexed):
        raise ValueError("Mapping must cover the entire run")
    for row in mapping:
        if any(row[k] != indexed[row["case_id"]][k] for k in row if k != "review_id"):
            raise ValueError("Mapping does not match saved run")
    output = aggregate(mapping, ratings, annotations)
    with Path(args.output).open("x") as f:
        json.dump({"note": "Descriptive rates only; repeat/image dependence not resolved by these counts",
                   "results": output}, f, ensure_ascii=False, indent=2)
    print(Path(args.output).resolve())


def paired_bootstrap(mapping, ratings, annotations, baseline, candidate, metric,
                     iterations=2000, seed=42):
    """Paired score difference with cluster resampling by original_id.

    Missing matched conditions or human labels are errors, never imputed. Repeated
    images and repeated model outputs share the same original sampling unit.
    """
    import random
    if baseline == candidate:
        raise ValueError("Choose distinct modes")
    if metric not in METRICS:
        raise ValueError("Unknown metric")
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap draws")
    ratings_index = {r["review_id"]: r for r in ratings}
    if len(ratings_index) != len(ratings) or {r["review_id"] for r in mapping} != ratings_index.keys():
        raise ValueError("Ratings must cover all review IDs")
    grouped = {baseline: {}, candidate: {}}
    for row in mapping:
        if row["mode"] not in grouped:
            continue
        key = (row["original_id"], row["image_id"], row["cue_id"], row["repeat"])
        if key in grouped[row["mode"]]:
            raise ValueError("Duplicate matched condition for " + row["mode"])
        grouped[row["mode"]][key] = (row, ratings_index[row["review_id"]])
    keys_a, keys_b = set(grouped[baseline]), set(grouped[candidate])
    if not keys_a or keys_a != keys_b:
        raise ValueError("Both modes require exactly the same matched conditions")
    clusters = defaultdict(list)
    failures = {baseline: 0, candidate: 0}
    excluded = 0
    for key in sorted(keys_a):
        _, image_id, cue_id, _ = key
        gold = annotations[image_id]
        if gold.get("review_status") != "adjudicated":
            raise ValueError("Image annotations must be adjudicated")
        group = gold["group"]
        cue_kind = gold["cues"][cue_id]
        applicable = (metric == "grounded" or
                      metric == "both_valid" and group == "ambiguous" or
                      metric == "false_alternative" and group == "ordinary" or
                      metric == "cue_correct" and cue_kind != "neutral")
        if not applicable:
            continue
        values = []
        for mode in [baseline, candidate]:
            row, rating = grouped[mode][key]
            failed = row["status"] != "ok"
            failures[mode] += int(failed)
            score = rating.get(metric, "").strip()
            if failed:
                if score != "NA":
                    raise ValueError("Failures require NA rating")
                values.append(None if metric == "false_alternative" else 0)
            else:
                if score not in {"0", "1"}:
                    raise ValueError("Missing human rating for " + metric)
                values.append(int(score))
        if None in values:
            excluded += 1
        else:
            clusters[key[0]].append(values[1] - values[0])
    clusters = {k: v for k, v in clusters.items() if v}
    if not clusters:
        raise ValueError("No applicable matched, scored cases")
    original_ids = sorted(clusters)
    deltas = [d for vals in clusters.values() for d in vals]
    estimate = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(iterations):
        sampled = [clusters[rng.choice(original_ids)] for _ in original_ids]
        draws.append(sum(sum(group) for group in sampled) / sum(len(group) for group in sampled))
    draws.sort()
    low = draws[int((iterations - 1) * 0.025)]
    high = draws[int((iterations - 1) * 0.975)]
    return {"baseline": baseline, "candidate": candidate, "metric": metric,
            "delta_candidate_minus_baseline": estimate,
            "cluster_bootstrap_95ci": [low, high], "bootstrap_iterations": iterations,
            "originals": len(original_ids), "matched_scored_pairs": len(deltas),
            "excluded_pair_technical_failure": excluded,
            "technical_failures_among_applicable": failures,
            "interpretation": "Exploratory cluster bootstrap CI, not proof of novelty or causal generalization."}


def paired_score(args):
    run_dir = Path(args.run)
    if json.loads((run_dir / "config.json").read_text())["synthetic"]:
        raise ValueError("Synthetic runs cannot produce research metrics")
    mapping = json.loads(Path(args.mapping).read_text())
    with Path(args.ratings).open(newline="") as f:
        ratings = list(csv.DictReader(f))
    raw = [json.loads(s) for s in Path(args.annotations).read_text().splitlines() if s.strip()]
    annotations = {r["id"]: r for r in raw}
    if len(raw) != len(annotations):
        raise ValueError("Duplicate annotations")
    run_rows = [json.loads(s) for s in (run_dir / "results.jsonl").read_text().splitlines()]
    indexed = {r["case_id"]: r for r in run_rows}
    if len(run_rows) != len(indexed) or len(mapping) != len(indexed) or {r["case_id"] for r in mapping} != indexed.keys():
        raise ValueError("Mapping does not cover the run exactly")
    for row in mapping:
        if any(row[k] != indexed[row["case_id"]][k] for k in row if k != "review_id"):
            raise ValueError("Mapping does not match saved run")
    output = paired_bootstrap(mapping, ratings, annotations, args.baseline,
                              args.candidate, args.metric, args.iterations, args.seed)
    with Path(args.output).open("x") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(Path(args.output).resolve())
