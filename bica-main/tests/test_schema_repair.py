"""Regression tests for bounded schema repair, blind context and retry costs."""
import json
import tempfile
import unittest
from pathlib import Path

from illusion_agent.core import load_manifest, run_case, sha256
from illusion_agent.colab import run_pilot
from illusion_agent.reflection import find_candidate_overlap


def response(*labels):
    return json.dumps({"interpretations": [
        {"label": label, "status": "supported",
         "evidence": [{"region": "center", "feature": "visible outline"}], "against": []}
        for label in labels], "cue_verdict": "not_applicable"})


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "image.jpg").write_bytes(b"mock-image")
        self.manifest = self.dir / "manifest.jsonl"
        self.manifest.write_text(json.dumps({"id": "case-a", "original_id": "case-a",
            "image": "image.jpg", "sha256": sha256(b"mock-image"),
            "split": "debug", "cues": [{"id": "c0", "text": ""}]}) + "\n")
        self.row = load_manifest(self.manifest)[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_repaired_without_silent_merging(self):
        class Client:
            def __init__(self): self.n = 0
            def complete(self, messages, seed):
                self.n += 1
                return (response("castle", "castle") if self.n == 1
                        else response("castle", "ladder")), {"output_tokens": 5}
        c = Client(); events=[]
        result = run_case(c,self.row,self.row["cues"][0],"direct",42,events.append)
        self.assertEqual([x["label"] for x in result["interpretations"]], ["castle","ladder"])
        self.assertEqual(c.n,2)
        self.assertEqual(len([e for e in events if e["event"] == "schema_failure"]),1)
        self.assertEqual(len([e for e in events if e["event"] == "repair_success"]),1)
        self.assertEqual([e["stage"] for e in events if e["event"]=="request"],
                         ["describe", "schema_repair"])
        self.assertIn('"castle"', [e for e in events if e["event"] == "response"][0]["raw"])

    def test_max_one_extra_call_even_if_invalid_twice(self):
        class Client:
            def __init__(self): self.n = 0
            def complete(self, messages, seed):
                self.n += 1
                return response("castle", "castle"), {}
        c = Client(); events=[]
        with self.assertRaisesRegex(ValueError, "unique"):
            run_case(c,self.row,self.row["cues"][0],"direct",1,events.append)
        self.assertEqual(c.n,2)
        self.assertEqual(len([e for e in events if e["event"]=="repair_failure"]),1)

    def test_more_than_four_repaired(self):
        class Client:
            def __init__(self): self.n = 0
            def complete(self, messages, seed):
                self.n += 1
                return response(*(list("abcde") if self.n == 1 else list("abcd"))), {}
        events=[]
        self.assertEqual(len(run_case(Client(),self.row,self.row["cues"][0],
                                 "direct",2,events.append)["interpretations"]),4)

    def test_blind_independent_pass_repair_does_not_see_first_reading(self):
        histories=[]
        class Client:
            def complete(self, messages, seed):
                histories.append(messages)
                if len(histories)==1:
                    obj=json.loads(response("first_unique_label")); obj["cue_verdict"]="uncertain"
                    return json.dumps(obj), {}
                if len(histories)==2: return response("second", "second"), {}
                if len(histories)==3: return response("second"), {}
                obj=json.loads(response("final")); obj["cue_verdict"]="rejected"
                return json.dumps(obj), {}
        run_case(Client(),self.row,{"id":"c9","text":"proposed mystery object"},
                 "reflect",7,lambda e:None)
        self.assertEqual(len(histories),4)
        blinded = json.dumps(histories[1:3])
        self.assertNotIn("first_unique_label",blinded)
        self.assertNotIn("proposed mystery object",blinded)
        self.assertIn("first_unique_label", json.dumps(histories[3]))

    def test_colab_budget_stops_hidden_repair(self):
        class Broken:
            synthetic=True
            metadata={"revision":"test"}
            def complete(self, messages, seed): return response("castle", "castle"), {}
        result = run_pilot(Broken(),self.manifest,self.dir / "run",modes=("direct",),max_calls=1)
        self.assertEqual(result["attempted_calls"],1)
        self.assertEqual(result["errors"],1)

    def test_colab_retry_accounted_as_extra_call(self):
        class Fixed:
            synthetic=True
            metadata={"revision":"test"}
            def __init__(self): self.n=0
            def complete(self,messages,seed):
                self.n+=1
                return response("castle", "castle") if self.n==1 else response("castle"), {}
        result=run_pilot(Fixed(),self.manifest,self.dir / "run",modes=("direct",),max_calls=2)
        self.assertEqual(result["attempted_calls"],2)
        self.assertEqual(result["errors"],0)

    def test_overlap_is_warning_only(self):
        val={"interpretations": [
            {"label":"man with hat", "status":"supported"},
            {"label":"man with pipe", "status":"supported"},
            {"label":"landscape", "status":"supported"}]}
        warnings=find_candidate_overlap(val)
        self.assertEqual(len(warnings),1)
        self.assertEqual(warnings[0]["reason"],"shared_subject_word")
        self.assertEqual(len(val["interpretations"]),3)


if __name__ == "__main__": unittest.main()
