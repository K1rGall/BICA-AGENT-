"""Restore debug imagery from revision-pinned upstream URLs, verifying SHA-256.

Individual image rights must be checked before redistribution. Only files
identified in the local manifest/provenance are fetched.
"""
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/ambibench_debug"
MAX_SIZE = 8_000_000


def main():
    manifest = [json.loads(line) for line in (DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    provenance = json.loads((DATASET / "provenance.json").read_text(encoding="utf-8"))
    sources = {x["id"]: x for x in provenance["items"]}
    root = DATASET.resolve()
    restored = 0
    for row in manifest:
        destination = (DATASET / row["image"]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("Image path escapes dataset folder")
        reference = sources[row["id"]]
        if reference["sha256"] != row["sha256"]:
            raise ValueError("Manifest/provenance hash mismatch: " + row["id"])
        if not destination.exists():
            url = reference["source_url"]
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or not parsed.path.startswith("/datasets/BLNL/AmbiBench/resolve/"):
                raise ValueError("Unexpected upstream URL: " + url)
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read(MAX_SIZE + 1)
            if len(content) > MAX_SIZE:
                raise ValueError("Image exceeds size limit")
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("Downloaded image hash mismatch: " + row["id"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            restored += 1
        if hashlib.sha256(destination.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Local image hash mismatch: " + row["id"])
    print(f"Verified {len(manifest)} originals; restored {restored} missing images")


if __name__ == "__main__":
    main()
