"""Adaptive-reflection behavioral invariants; no accuracy claims from fake VLMs."""
import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from illusion_agent.core import MODES, load_manifest, run_case, sha256
from illusion_agent.reflection import decide_visual_challenge
from illusion_agent.colab import run_pilot
from illusion_agent.diagnostics import analyze_run


def answer(*labels, status='supported', verdict='not_applicable'):
    return json.dumps({'interpretations': [
        {'label': l, 'status': status,
         'evidence': [{'region': 'top left', 'feature': 'black curved outline'}],
         'against': []} for l in labels], 'cue_verdict': verdict})


class AdaptiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        img = Image.new('RGB', (80, 60), 'white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        self.im = self.root/'im.png'
        self.im.write_bytes(buf.getvalue())
        self.manifest = self.root/'manifest.jsonl'
        self.manifest.write_text(json.dumps({'id':'x', 'original_id':'x',
            'image':'im.png','sha256':sha256(self.im.read_bytes()),
            'split':'debug','cues':[{'id':'D','kind':'hypothesis','text':'RABBIT_CUE'}]})+'\n')
        self.row = load_manifest(self.manifest)[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_gate_trigger_has_no_truth_dependency(self):
        left = json.loads(answer('duck in grass', 'bird in grass'))
        right = json.loads(answer('duck in grass'))
        d = decide_visual_challenge([left, right])
        self.assertTrue(d['need_challenge'])
        self.assertIn('possible_label_overlap', d['reasons'])
        self.assertIn('lexical_disagreement', d['reasons'])
        self.assertEqual(decide_visual_challenge([right,right]),
                         {'need_challenge':False, 'reasons':[]})

    def test_adaptive_skips_optional_call_when_readings_agree(self):
        calls,events=[],[]
        class Client:
            def complete(self,messages,seed):
                calls.append((messages,seed))
                return answer('vase',verdict='uncertain' if len(calls)!=2 else 'not_applicable'), {}
        result=run_case(Client(),self.row,self.row['cues'][0],
                        'reflect_adaptive',42,events.append)
        self.assertEqual(len(calls),3)
        self.assertEqual([e['stage'] for e in events if e['event']=='request'],
                         ['describe','independent','adjudicate'])
        self.assertFalse(next(e for e in events if e['event']=='adaptive_decision')['need_challenge'])
        self.assertEqual(result['cue_verdict'],'uncertain')
        self.assertEqual(len(calls[2][0][-1]['images']),1)

    def test_challenge_reads_image_and_isolated_from_external_cue(self):
        calls,events=[],[]
        class Client:
            def complete(self,messages,seed):
                calls.append(json.loads(json.dumps(messages)))
                i=len(calls)
                label='first_visual_hypothesis' if i==1 else 'second_visual_hypothesis'
                return answer(label,verdict='rejected' if i in (1,4) else 'not_applicable'), {}
        result=run_case(Client(),self.row,self.row['cues'][0],
                        'reflect_adaptive',7,events.append)
        self.assertEqual(len(calls),4)
        self.assertEqual([e['stage'] for e in events if e['event']=='request'],
                         ['describe','independent','challenge','adjudicate'])
        self.assertNotIn('RABBIT_CUE',json.dumps(calls[1]))
        self.assertNotIn('RABBIT_CUE',json.dumps(calls[2]))
        self.assertNotIn('first_visual_hypothesis',json.dumps(calls[1]))
        self.assertIn('first_visual_hypothesis',json.dumps(calls[2]))
        self.assertIn('RABBIT_CUE',json.dumps(calls[3]))
        self.assertEqual(len(calls[2][-1]['images']),2)
        self.assertEqual(result['cue_verdict'],'rejected')
        self.assertTrue(next(e for e in events if e['event']=='adaptive_summary')['extra_visual_challenge'])

    def test_budget_and_diagnostics_track_extra_check(self):
        class Fake:
            synthetic=True
            metadata={'model':'fake','revision':'test'}
            def __init__(self):self.n=0
            def complete(self,messages,seed):
                self.n+=1
                return answer('vase' if self.n==1 else 'faces', verdict='uncertain' if self.n in (1,4) else 'not_applicable'), {'input_tokens':12,'output_tokens':4}
        result=run_pilot(Fake(),self.manifest,self.root/'out',modes=('reflect_adaptive',),
                         max_calls=4,seed=4)
        self.assertEqual(result['attempted_calls'],4)
        self.assertEqual(result['errors'],0)
        diagnostics=analyze_run(self.root/'out')['by_mode'][0]
        self.assertEqual(diagnostics['adaptive_challenges'],1)
        self.assertEqual(diagnostics['adaptive_skips'],0)
        self.assertEqual(diagnostics['completed_calls'],4)
        self.assertEqual(diagnostics['adaptive_trigger_reasons'].get('lexical_disagreement'),1)

    def test_budget_makes_room_for_one_repair_and_required_steps(self):
        class Fake:
            synthetic=True
            metadata={'model':'fake','revision':'test'}
            def __init__(self):self.n=0
            def complete(self,messages,seed):
                self.n+=1
                if self.n==1:return answer('a',verdict='uncertain'),{}
                if self.n==2:return answer('b'),{}
                if self.n==3:return '{not json}',{}
                return answer('c',verdict='uncertain'),{}
        x=run_pilot(Fake(),self.manifest,self.root/'out',modes=('reflect_adaptive',),max_calls=4)
        self.assertEqual(x['attempted_calls'],4)
        self.assertEqual(x['errors'],0)  # Invalid optional challenge skipped; final answer still produced.
        self.assertEqual(x['stopped_on_budget'],False)


if __name__=='__main__':unittest.main()
