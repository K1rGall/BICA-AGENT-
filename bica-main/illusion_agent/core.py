"""Experimental conditions, strict response validation, and isolated conversations."""
import base64
import hashlib
import json
import re

from .reflection import build_closeups, find_candidate_overlap, decide_visual_challenge
import time
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"
MODES = {
    "direct": ("describe",),
    "repeat": ("describe", "revise", "revise"),
    "hypotheses": ("describe", "alternatives", "retain"),
    "evidence": ("describe", "alternatives", "verify"),
    "reflect": ("describe", "independent", "adjudicate"),
    "reflect_zoom": ("describe", "independent", "adjudicate"),
    "reflect_adaptive": ("describe", "independent", "challenge", "adjudicate"),
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _json_payload(raw):
    """Accept one outer Markdown fence, never extract/repair JSON from prose."""
    stripped = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", stripped,
                          flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else stripped


def parse_response(raw):
    value = json.loads(_json_payload(raw))
    if not isinstance(value, dict) or set(value) != {"interpretations", "cue_verdict"}:
        raise ValueError("Response must contain interpretations and cue_verdict only")
    if value["cue_verdict"] not in {"supported", "rejected", "uncertain", "not_applicable"}:
        raise ValueError("Invalid cue_verdict")
    items = value["interpretations"]
    if not isinstance(items, list) or len(items) > 4:
        raise ValueError("Expected at most four interpretations")
    labels = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"label", "status", "evidence", "against"}:
            raise ValueError("Invalid interpretation fields")
        label = item["label"]
        if not isinstance(label, str) or not label.strip() or label.casefold().strip() in labels:
            raise ValueError("Labels must be nonempty and unique")
        labels.add(label.casefold().strip())
        if item["status"] not in {"supported", "rejected", "uncertain"}:
            raise ValueError("Invalid hypothesis status")
        if not isinstance(item["evidence"], list) or not isinstance(item["against"], list):
            raise ValueError("Evidence and against must be lists")
        if not all(isinstance(x, str) and x.strip() for x in item["against"]):
            raise ValueError("Counterevidence must be text")
        if item["status"] == "supported" and not item["evidence"]:
            raise ValueError("Supported hypotheses need evidence")
        for ev in item["evidence"]:
            if not isinstance(ev, dict) or set(ev) != {"region", "feature"}:
                raise ValueError("Evidence requires region and feature")
            if not all(isinstance(v, str) and v.strip() for v in ev.values()):
                raise ValueError("Empty evidence")
    # This validates structure, NOT whether the visual evidence is correct.
    return value


def load_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty manifest")
    ids, splits = set(), {}
    for row in rows:
        allowed = {"id", "original_id", "image", "sha256", "split", "cues"}
        if set(row) != allowed:
            raise ValueError("Manifest fields must be " + str(sorted(allowed)))
        if not all(isinstance(row[k], str) and row[k].strip() for k in allowed - {"cues"}):
            raise ValueError("Invalid manifest text field")
        if row["id"] in ids:
            raise ValueError("Duplicate image ID")
        ids.add(row["id"])
        if row["split"] not in {"debug", "dev", "test"}:
            raise ValueError("Unknown split")
        image = (path.parent / row["image"]).resolve()
        if not image.is_relative_to(path.parent):
            raise ValueError("Image path escapes manifest directory")
        data = image.read_bytes()
        if image.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Use PNG, JPEG or WebP")
        if sha256(data) != row["sha256"]:
            raise ValueError("Image hash mismatch: " + row["id"])
        for key in ("original:" + row["original_id"], "hash:" + row["sha256"]):
            if key in splits and splits[key] != row["split"]:
                raise ValueError("Original/image hash occurs across splits")
            splits[key] = row["split"]
        cues = row["cues"]
        if not isinstance(cues, list) or not cues:
            raise ValueError("At least one cue condition is required")
        cue_ids = set()
        for cue in cues:
            if not isinstance(cue, dict) or set(cue) not in ({"id", "text"}, {"id", "text", "kind"}):
                raise ValueError("Cue contains unexpected fields (do not include its truth label)")
            if cue.get("kind", "hypothesis") not in {"instruction", "hypothesis"}:
                raise ValueError("Cue.kind must be instruction or hypothesis (never its truth)")
            if not isinstance(cue["id"], str) or not cue["id"] or cue["id"] in cue_ids:
                raise ValueError("Cue IDs must be unique")
            if not isinstance(cue["text"], str):
                raise ValueError("Cue must be text")
            cue_ids.add(cue["id"])
        row["image_path"] = image
    return rows


def run_case(client, row, cue, mode, seed, emit, *, schema_repair=True, repair_allowed=None):
    """Run one condition. Reflection uses a blind independent second reading.

    Calls 1 and 2 cannot see one another; call 2 cannot see the text cue. The
    third call inspects the image and both fallible readings. This is a genuine
    orchestrated arbitration step, not merely a 'think again' message.
    """
    if mode not in MODES:
        raise ValueError("Unknown mode: " + str(mode))
    system = (PROMPTS / "system.txt").read_text(encoding="utf-8")
    messages = [{"role": "system", "content": system}]
    image_bytes = row["image_path"].read_bytes()
    image = base64.b64encode(image_bytes).decode("ascii")
    answer = None
    readings = []
    repair_attempted = False  # At most ONE extra generation per case, not per stage.
    reflective_modes = {"reflect", "reflect_zoom", "reflect_adaptive"}
    challenge_answer = None
    adaptive_policy = None
    for step, stage in enumerate(MODES[mode]):
        # The adaptive policy can only inspect fallible model outputs, not gold labels.
        # Skipping a challenge saves one VLM call while retaining final visual arbitration.
        if mode == "reflect_adaptive" and stage == "challenge":
            adaptive_policy = decide_visual_challenge(readings)
            emit({"event": "adaptive_decision", "mode": mode,
                  "need_challenge": adaptive_policy["need_challenge"],
                  "reasons": adaptive_policy["reasons"],
                  "note": "Deterministic heuristic; not a correctness estimate"})
            if not adaptive_policy["need_challenge"]:
                continue
        content = (PROMPTS / (stage + ".txt")).read_text(encoding="utf-8")
        extra_images, image_metadata = [], []
        if mode in reflective_modes and step:
            # New independent message context at each stage. Stage 1 sees neither
            # the first reading nor the external suggestion; stages 2/3 do.
            messages = [{"role": "system", "content": system}]
            if stage in {"challenge", "adjudicate"}:
                content += "\n\nTwo fallible prior readings (not ground truth):\n"
                for number, reading in enumerate(readings, 1):
                    content += f"\nReading {number}: " + json.dumps(reading, ensure_ascii=False)
                if stage == "challenge":
                    # Challenge uses one localized image, selected without gold labels.
                    extra_images, image_metadata = build_closeups(image_bytes, readings, maximum=1)
                    content += "\nCheck original before the close-up; crops may remove global evidence."
                    for number, meta in enumerate(image_metadata, 1):
                        content += (f"\nClose-up {number}: region {meta['region']!r}, "
                                    f"box [left, top, right, bottom]={meta['box_xyxy']}")
                if stage == "adjudicate":
                    if challenge_answer is not None:
                        content += ("\n\nAdditional fallible visual challenge (NOT truth): " +
                                    json.dumps(challenge_answer, ensure_ascii=False))
                    if cue["text"]:
                        if cue.get("kind", "hypothesis") == "instruction":
                            content += ("\nViewing instruction (not a factual suggestion): " +
                                        json.dumps(cue["text"], ensure_ascii=False))
                        else:
                            content += ("\nUntrusted suggestion to check: " +
                                        json.dumps(cue["text"], ensure_ascii=False))
                    if mode == "reflect_zoom":
                        extra_images, image_metadata = build_closeups(image_bytes, readings)
                        content += "\n" + (PROMPTS / "adjudicate_zoom.txt").read_text(encoding="utf-8")
                        for number, meta in enumerate(image_metadata, 1):
                            content += (f"\nClose-up {number}: region {meta['region']!r}, "
                                        f"box [left, top, right, bottom]={meta['box_xyxy']}")
        elif step == 0 and cue["text"]:
            if cue.get("kind", "hypothesis") == "instruction":
                content += "\nViewing instruction (not a factual suggestion): " + cue["text"]
            else:
                content += "\nSuggested interpretation to check (may be wrong): " + cue["text"]
        message = {"role": "user", "content": content}
        if step == 0 or mode in reflective_modes:
            message["images"] = [image] + extra_images
        messages.append(message)
        logged = [{k: v for k, v in m.items() if k != "images"} for m in messages]
        step_seed = seed + 1009 if mode in reflective_modes and stage == "independent" else seed
        emit({"event": "request", "mode": mode, "stage": stage, "step": step, "seed": step_seed,
              "image_sha256": row["sha256"], "closeups": image_metadata,
              "images_sha256": [row["sha256"]] + [m["sha256"] for m in image_metadata]
                               if "images" in message else [], "messages": logged})
        start = time.monotonic()
        try:
            raw, usage = client.complete(messages, step_seed)
        except Exception as exc:
            emit({"event": "transport_error", "mode": mode, "step": step, "error": type(exc).__name__,
                  "elapsed_s": time.monotonic() - start})
            raise
        emit({"event": "response", "mode": mode, "step": step, "raw": raw, "usage": usage,
              "elapsed_s": time.monotonic() - start})
        try:
            answer = parse_response(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            # Do not silently alter the candidate list. Ask the VLM once to
            # repair its own output, preserving the raw invalid answer in trace.
            error_text = f"{type(exc).__name__}: {exc}"
            emit({"event": "schema_failure", "mode": mode, "stage": stage,
                  "step": step, "reason": error_text})
            permitted = (schema_repair and not repair_attempted and
                         (repair_allowed is None or repair_allowed(step)))
            if not permitted:
                emit({"event": "repair_skipped", "mode": mode, "stage": stage,
                      "step": step, "reason": "disabled_or_budget"})
                if mode == "reflect_adaptive" and stage == "challenge":
                    # This stage is OPTIONAL. A malformed challenge cannot turn an
                    # otherwise answerable case into a failed experiment.
                    emit({"event": "optional_challenge_discarded", "mode": mode,
                          "step": step, "reason": error_text})
                    continue
                raise
            repair_attempted = True
            repair_prompt = (PROMPTS / "schema_repair.txt").read_text(encoding="utf-8")
            repair_prompt += ("\n\nValidation failure: " + error_text +
                              "\nThe prior invalid response is included immediately before "
                              "this instruction. Return ONE complete corrected JSON object.")
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_prompt},
            ]
            # A repair of the INDEPENDENT second pass sees only that pass's
            # context and its invalid output; never another reading or cue.
            repair_log = [{k: v for k, v in m.items() if k != "images"}
                          for m in repair_messages]
            repair_seed = step_seed + 42437
            emit({"event": "request", "mode": mode, "stage": "schema_repair",
                  "step": step, "seed": repair_seed,
                  "image_sha256": row["sha256"],
                  "repair_of": stage, "messages": repair_log})
            repair_start = time.monotonic()
            try:
                replacement_raw, repair_usage = client.complete(repair_messages, repair_seed)
            except Exception as transport_exc:
                emit({"event": "transport_error", "mode": mode, "stage": "schema_repair",
                      "step": step, "error": type(transport_exc).__name__,
                      "elapsed_s": time.monotonic() - repair_start})
                raise
            emit({"event": "response", "mode": mode, "stage": "schema_repair",
                  "step": step, "raw": replacement_raw, "usage": repair_usage,
                  "elapsed_s": time.monotonic() - repair_start})
            try:
                answer = parse_response(replacement_raw)
            except (ValueError, json.JSONDecodeError) as repair_exc:
                emit({"event": "repair_failure", "mode": mode, "stage": stage,
                      "step": step, "reason": str(repair_exc)})
                if mode == "reflect_adaptive" and stage == "challenge":
                    emit({"event": "optional_challenge_discarded", "mode": mode,
                          "step": step, "reason": str(repair_exc)})
                    continue
                raise
            emit({"event": "repair_success", "mode": mode, "stage": stage,
                  "step": step})
            raw = replacement_raw  # Subsequent stages see ONLY a valid response.
        if _json_payload(raw) != raw.strip():
            emit({"event": "normalization", "step": step, "kind": "outer_json_fence"})
        # The blind independent pass has no cue even when the overall case has one.
        cue_present = (bool(cue["text"]) and cue.get("kind", "hypothesis") == "hypothesis"
                       and not (mode in reflective_modes and stage in {"independent", "challenge"}))
        if not cue_present and answer["cue_verdict"] != "not_applicable":
            emit({"event": "normalization", "step": step, "kind": "absent_cue",
                  "field": "cue_verdict", "original": answer["cue_verdict"],
                  "value": "not_applicable"})
            answer["cue_verdict"] = "not_applicable"
        if cue_present and answer["cue_verdict"] == "not_applicable":
            raise ValueError("cue_verdict inconsistent with presence/absence of cue")
        if mode in reflective_modes:
            if stage in {"describe", "independent"}:
                readings.append(answer)
            elif stage == "challenge":
                challenge_answer = answer
        else:
            messages.append({"role": "assistant", "content": raw})
    overlap = find_candidate_overlap(answer)
    if overlap:
        emit({"event": "candidate_overlap_warning", "mode": mode,
              "flags": overlap,
              "note": "Lexical warning only; human/image-grounded assessment required."})
    if mode in reflective_modes:
        def labels(value):
            return {h["label"].strip().casefold() for h in value["interpretations"]
                    if h["status"] == "supported"}
        first, second = labels(readings[0]), labels(readings[1])
        last = labels(answer)
        emit({"event": "reflection_flags", "mode": mode, "label_disagreement": first != second,
              "first_supported_labels": len(first), "second_supported_labels": len(second),
              "final_supported_labels": len(last), "new_final_labels": len(last - first - second),
              "note": "Lexical diagnostics only; not correctness or semantic equivalence"})
        if mode == "reflect_adaptive":
            emit({"event": "adaptive_summary", "mode": mode,
                  "extra_visual_challenge": challenge_answer is not None,
                  "reasons": adaptive_policy["reasons"] if adaptive_policy else []})
    return answer
