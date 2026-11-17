"""Restore the original 6 debug images from the included Colab bundle.

The upstream ZIP omitted the data/ images although its nested Colab ZIP had
all six. Preserve the dataset bytes; refuse any hash mismatch.
"""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/ambibench_debug"
BUNDLE = ROOT / "deliverables/bica_colab_bundle.zip"


def main():
    manifest = [json.loads(x) for x in (DATASET / "manifest.jsonl").read_text().splitlines() if x.strip()]
    restored = 0
    with zipfile.ZipFile(BUNDLE) as z:
        for row in manifest:
            path = (DATASET / row["image"]).resolve()
            if not path.is_relative_to(DATASET.resolve()):
                raise ValueError("Invalid relative image path")
            if not path.exists():
                archive_path = (Path("data/ambibench_debug") / row["image"]).as_posix()
                raw = z.read(archive_path)
                if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError("Bundle image hash mismatch: " + row["id"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                restored += 1
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Local image hash mismatch: " + row["id"])
    print(f"Verified {len(manifest)} originals; restored {restored} missing images")


if __name__ == "__main__":
    main()
