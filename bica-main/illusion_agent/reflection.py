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
