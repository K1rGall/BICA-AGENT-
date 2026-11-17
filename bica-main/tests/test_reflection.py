"""Behavioral tests of independent visual reflection and targeted evidence revisit."""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from illusion_agent.core import load_manifest, run_case, sha256
from illusion_agent.reflection import build_closeups, select_regions


def reading(label, region="top left"):
    return {"interpretations": [{"label": label, "status": "supported",
        "evidence": [{"region": region, "feature": "curved outline"}], "against": []}],
        "cue_verdict": "not_applicable"}


class ReflectionTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = self.root / "PRIVATE.png"
        img = Image.new("RGB", (60, 40), "white")
        for x in range(0, 24):
            for y in range(0, 20):
                img.putpixel((x, y), (220, 40, 60))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        self.image.write_bytes(buf.getvalue())
        row = {"id": "PRIVATE_CASE", "original_id": "PRIVATE_ORIGIN",
               "image": self.image.name, "sha256": sha256(self.image.read_bytes()),
               "split": "debug", "cues": [{"id": "c1", "text": "It is definitely a rabbit"}]}
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(json.dumps(row) + "\n")
        self.row = load_manifest(self.manifest)[0]

    def tearDown(self): self.tmp.cleanup()

    def test_reflect_stage_two_is_independent_and_blind_to_cue(self):
        histories = []
        class Client:
            synthetic = True
            def complete(self, messages, seed):
                histories.append(json.loads(json.dumps(messages)))
                response = reading("duck" if len(histories) != 2 else "rabbit")
                if len(histories) != 2:
                    response["cue_verdict"] = "rejected"
                return json.dumps(response), {"model": "unit-test"}
        events = []
        result = run_case(Client(), self.row, self.row["cues"][0], "reflect", 40, events.append)
        self.assertEqual(result["cue_verdict"], "rejected")
        self.assertEqual(len(histories), 3)
        self.assertTrue(all(len(history) == 2 for history in histories))
        self.assertNotIn("rabbit", histories[1][1]["content"].lower())
        self.assertNotIn("duck", histories[1][1]["content"].lower())
        self.assertIn('"duck"', histories[2][1]["content"])
        self.assertIn('"rabbit"', histories[2][1]["content"])
        self.assertEqual([r["seed"] for r in events if r["event"] == "request"], [40, 1049, 40])
        self.assertNotIn("PRIVATE", json.dumps(histories))
        self.assertEqual(len(histories[2][1]["images"]), 1)

    def test_zoom_pass_attaches_traceable_crops(self):
        histories = []
        class Client:
            def complete(self, messages, seed):
                histories.append(json.loads(json.dumps(messages)))
                value = reading("vase", "top left" if len(histories) == 1 else "bottom right")
                return json.dumps(value), {}
        events = []
        result = run_case(Client(), self.row, {"id": "c0", "text": ""},
                          "reflect_zoom", 7, events.append)
        self.assertEqual(result["cue_verdict"], "not_applicable")
        self.assertEqual(len(histories[2][1]["images"]), 3)  # original + two regions
        requests = [e for e in events if e["event"] == "request"]
        self.assertEqual(len(requests[2]["closeups"]), 2)
        self.assertEqual(len(requests[2]["images_sha256"]), 3)
        for encoded, meta in zip(histories[2][1]["images"][1:], requests[2]["closeups"]):
            self.assertEqual(meta["sha256"], sha256(base64.b64decode(encoded)))
        self.assertNotEqual(requests[2]["images_sha256"][1], requests[2]["images_sha256"][2])

    def test_regions_and_fallback(self):
        self.assertEqual(select_regions([reading("a", "top left")])[0][0], ("top", "left"))
        self.assertEqual(select_regions([{ "interpretations": []}]),
                         [(("center", "center"), "center (fallback)")])

    def test_reflect_zoom_no_pillow_fails_explicitly(self):
        with patch.dict("sys.modules", {"PIL": None}):
            with self.assertRaisesRegex(RuntimeError, "Pillow"):
                build_closeups(self.image.read_bytes(), [reading("duck")])

    def test_manifest_cannot_reference_outside_directory(self):
        row = json.loads(self.manifest.read_text())
        row["image"] = "../SECRET.png"
        self.manifest.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_manifest(self.manifest)


if __name__ == "__main__": unittest.main()
