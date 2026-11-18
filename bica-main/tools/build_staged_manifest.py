"""Build cue levels A-D ONLY after human adjudication of image annotations.

Truth is kept in annotation file; the inference manifest exposes the actual
prompt text and its type, never whether a suggested interpretation is true.
"""
import argparse
import json
from pathlib import Path

from illusion_agent.core import load_manifest

LEVEL_B = ("Could this image support more than one genuinely different visual "
           "interpretation? If not, say so; describe the evidence you see.")
LEVEL_C = ("Examine contours, negative space and how local details might compose a "
           "different overall figure. Do not invent another image; explain visible support.")


def build(original_manifest, gold_file, output):
    base = load_manifest(original_manifest)
    raw = [json.loads(s) for s in Path(gold_file).read_text(encoding="utf-8").splitlines() if s.strip()]
    gold = {r["id"]: r for r in raw}
    if len(raw) != len(gold):
        raise ValueError("Duplicate gold IDs")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Never overwrite a prepared manifest: " + str(output))
    rows = []
    for row in base:
        annotation = gold.get(row["id"])
        if annotation is None or annotation.get("review_status") != "adjudicated":
            raise ValueError("Human-adjudicated annotation missing: " + row["id"])
        group = annotation.get("group")
        labels = annotation.get("approved_interpretations")
        target = annotation.get("target_label", "")
        if group not in {"ambiguous", "ordinary"}:
            raise ValueError("Unrecognized group for " + row["id"])
        if not isinstance(labels, list) or len(labels) != (2 if group == "ambiguous" else 1):
            raise ValueError("Approved interpretation count wrong: " + row["id"])
        if not all(isinstance(x, str) and x.strip() for x in labels):
            raise ValueError("Empty approved interpretation: " + row["id"])
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Target label is needed for D condition: " + row["id"])
        # D checks a known alternative on ambiguous images, but is a
        # PRE-REGISTERED false suggestion on ordinary controls. A real human
        # must document why the ordinary image does not support that object.
        normalized_labels = {x.strip().casefold() for x in labels}
        if group == "ambiguous" and target.strip().casefold() not in normalized_labels:
            raise ValueError("D target must match an approved alternative: " + row["id"])
        if group == "ordinary" and target.strip().casefold() in normalized_labels:
            raise ValueError("D target must be a misleading suggestion on ordinary control: " + row["id"])
        row = {k: row[k] for k in ("id", "original_id", "image", "sha256", "split")}
        row["cues"] = [
            {"id":"A", "text":"", "kind":"instruction"},
            {"id":"B", "text":LEVEL_B, "kind":"instruction"},
            {"id":"C", "text":LEVEL_C, "kind":"instruction"},
            {"id":"D", "text":target, "kind":"hypothesis"},
        ]
        # No class/group/truth labels are copied into the model-facing manifest.
        rows.append(row)
    output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    # Resolve images relative to the location of the NEW output manifest.
    # Require output beside the original manifest to keep safe relative paths.
    if output.resolve().parent != Path(original_manifest).resolve().parent:
        output.unlink()
        raise ValueError("Place output in same directory as original manifest to preserve image paths")
    load_manifest(output)
    return len(rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print('Prepared images:',build(args.manifest,args.annotations,args.output))
