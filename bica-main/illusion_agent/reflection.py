"""Evidence-targeted visual revisit for the experimental reflect_zoom condition.

Close-ups are chosen using *only* the model's candidate region descriptions. No
reference labels/annotations are consulted, and full image always remains visible.
"""
import base64
import hashlib
import io
import re


def _region_key(region):
    words = set(re.findall(r"[a-z]+", region.casefold()))
    y = "top" if words & {"top", "upper", "above"} else (
        "bottom" if words & {"bottom", "lower", "below"} else "center")
    x = "left" if "left" in words else ("right" if "right" in words else "center")
    return y, x


def select_regions(answers, maximum=2):
    """Stable, label-free region selection; fallback is a center inspection."""
    selected = []
    for answer in answers:
        for hypothesis in answer["interpretations"]:
            if hypothesis["status"] == "rejected":
                continue
            for ev in hypothesis["evidence"]:
                region = ev["region"]
                key = _region_key(region)
                if key not in [x[0] for x in selected]:
                    selected.append((key, region))
                    if len(selected) == maximum:
                        return selected
    if not selected:
        selected = [(('center', 'center'), 'center (fallback)')]
    return selected


def build_closeups(image_bytes, answers, maximum=2):
    """Return (base64 images, trace metadata); requires Pillow only for this mode."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("reflect_zoom requires Pillow: pip install Pillow") from exc
    with Image.open(io.BytesIO(image_bytes)) as im:
        source = ImageOps.exif_transpose(im).convert("RGB")
    width, height = source.size
    encoded, metadata = [], []
    for (y, x), label in select_regions(answers, maximum):
        # Overlapping 60%-wide windows preserve more surrounding Gestalt context.
        cw, ch = max(1, round(width * .6)), max(1, round(height * .6))
        left = {"left": 0, "center": (width-cw)//2, "right": width-cw}[x]
        top = {"top": 0, "center": (height-ch)//2, "bottom": height-ch}[y]
        crop = source.crop((left, top, left+cw, top+ch))
        if max(crop.size) < 640:
            scale = 640 / max(crop.size)
            crop = crop.resize((max(1, round(crop.width*scale)),
                                max(1, round(crop.height*scale))), Image.Resampling.BICUBIC)
        output = io.BytesIO()
        crop.save(output, format="JPEG", quality=90)
        raw = output.getvalue()
        encoded.append(base64.b64encode(raw).decode("ascii"))
        metadata.append({"region": label, "box_xyxy": [left, top, left+cw, top+ch],
                         "sha256": hashlib.sha256(raw).hexdigest()})
    return encoded, metadata


def find_candidate_overlap(answer):
    """Cheap, conservative warnings about near-duplicate labels.

    Does NOT decide visual equivalence, merge predictions, or alter scores.
    Matching the same head or a contextual suffix can yield false positives.
    """
    items = [x for x in answer.get("interpretations", [])
             if x.get("status") in {"supported", "uncertain"}]
    labels = [x.get("label", "").strip() for x in items]
    flags = []
    for i, left in enumerate(labels):
        lt = re.findall(r"[a-zа-яё]+", left.casefold())
        for j in range(i + 1, len(labels)):
            right = labels[j]
            rt = re.findall(r"[a-zа-яё]+", right.casefold())
            reason = None
            if lt and rt and len(lt) > 1 and len(rt) > 1 and lt[0] == rt[0]:
                reason = "shared_subject_word"
            elif len(lt) >= 3 and len(rt) >= 3 and lt[-2:] == rt[-2:]:
                reason = "shared_context_tail"
            elif (len(lt) == 1 and lt[0] in rt) or (len(rt) == 1 and rt[0] in lt):
                reason = "nested_label"
            if reason:
                flags.append({"left": left, "right": right, "reason": reason})
    return flags


def decide_visual_challenge(readings):
    """Choose whether to spend an extra VLM call, without gold or label truth.

    A lexical difference does NOT imply a visual difference. This conservative
    trigger only requests a closer inspection; neither policy nor overlaps
    count toward scientific correctness metrics.
    """
    if len(readings) != 2:
        raise ValueError("Adaptive policy requires two independent readings")
    reasons = []
    supported = []
    for answer in readings:
        labels = {item["label"].strip().casefold() for item in answer["interpretations"]
                  if item["status"] == "supported"}
        supported.append(labels)
        if any(item["status"] == "uncertain" for item in answer["interpretations"]):
            if "uncertain_interpretation" not in reasons:
                reasons.append("uncertain_interpretation")
        if find_candidate_overlap(answer) and "possible_label_overlap" not in reasons:
            reasons.append("possible_label_overlap")
    if supported[0] != supported[1]:
        reasons.append("lexical_disagreement")
    if not supported[0] or not supported[1]:
        reasons.append("missing_supported_interpretation")
    return {"need_challenge": bool(reasons), "reasons": reasons}
