import unittest

from illusion_agent.evaluation import paired_bootstrap


class PairedTests(unittest.TestCase):
    def test_pairs_clustered_by_original_not_condition(self):
        mapping, ratings, annotations = [], [], {}
        for i in range(5):
            original = "orig-a" if i < 3 else "orig-b"
            image = f"image-{i}"
            annotations[image] = {"review_status": "adjudicated", "group": "ambiguous", "cues": {"c0": "neutral"}}
            for mode in ("direct", "reflect"):
                rid = f"{i}-{mode}"
                mapping.append({"review_id": rid, "original_id": original, "image_id": image,
                                "cue_id": "c0", "repeat": 0, "mode": mode, "status": "ok"})
                ratings.append({"review_id": rid, "both_valid": str(int(mode == "reflect")),
                                "false_alternative": "NA", "cue_correct": "NA", "grounded": "0"})
        output = paired_bootstrap(mapping, ratings, annotations, "direct", "reflect", "both_valid", 200)
        self.assertEqual(output["originals"], 2)
        self.assertEqual(output["matched_scored_pairs"], 5)
        self.assertEqual(output["delta_candidate_minus_baseline"], 1)
        self.assertEqual(output["cluster_bootstrap_95ci"], [1.0, 1.0])

    def test_missing_pairs_rejected(self):
        mapping = [{"review_id": "r0", "original_id": "a", "image_id": "a",
                    "cue_id": "c0", "repeat": 0, "mode": "direct", "status": "ok"}]
        with self.assertRaises(ValueError):
            paired_bootstrap(mapping, [{"review_id": "r0", "grounded": "1"}], {},
                             "direct", "reflect", "grounded")

    def test_technical_error_zero_for_success_and_excluded_for_false_positive(self):
        annotations = {"i": {"review_status": "adjudicated", "group": "ordinary", "cues": {"c0": "neutral"}}}
        mapping = [{"review_id": mode, "original_id": "i", "image_id": "i", "cue_id": "c0",
                    "repeat": 0, "mode": mode, "status": status} for mode, status in
                   [("direct", "error"), ("reflect", "ok")]]
        ratings = [{"review_id": "direct", "grounded": "NA", "false_alternative": "NA"},
                   {"review_id": "reflect", "grounded": "1", "false_alternative": "1"}]
        result = paired_bootstrap(mapping, ratings, annotations, "direct", "reflect", "grounded", 100)
        self.assertEqual(result["delta_candidate_minus_baseline"], 1)
        with self.assertRaisesRegex(ValueError, "No applicable matched"):
            paired_bootstrap(mapping, ratings, annotations, "direct", "reflect", "false_alternative", 100)


if __name__ == "__main__": unittest.main()
