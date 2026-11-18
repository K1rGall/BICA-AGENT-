import csv
import json
import tempfile
import unittest
from pathlib import Path

from illusion_agent.core import load_manifest, run_case, sha256
from tools.build_staged_manifest import build
from tools.compare_reviews import compare


class NewStudyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'i.jpg').write_bytes(b'fixture')
        self.manifest=self.root/'manifest.jsonl'
        self.manifest.write_text(json.dumps({'id':'x','original_id':'x','image':'i.jpg',
            'sha256':sha256(b'fixture'),'split':'debug',
            'cues':[{'id':'c0','text':''}]})+'\n')
        self.gold=self.root/'annotations.jsonl'
        self.gold.write_text(json.dumps({'id':'x','group':'ambiguous',
                                        'review_status':'pending_two_human_reviewers'})+'\n')
    def tearDown(self):self.tmp.cleanup()

    def test_no_staged_manifest_before_human_approval(self):
        with self.assertRaisesRegex(ValueError, 'adjudicated'):
            build(self.manifest,self.gold,self.root/'staged.jsonl')
        self.assertFalse((self.root/'staged.jsonl').exists())

    def test_independent_instruction_uses_no_cue_verdict(self):
        self.gold.write_text(json.dumps({'id':'x','group':'ambiguous',
            'review_status':'adjudicated','approved_interpretations':['vase','faces'],
            'target_label':'faces'})+'\n')
        new=self.root/'staged.jsonl'
        self.assertEqual(build(self.manifest,self.gold,new),1)
        row=load_manifest(new)[0]
        self.assertEqual([c['id'] for c in row['cues']],list('ABCD'))
        self.assertEqual([c['kind'] for c in row['cues']],
                         ['instruction','instruction','instruction','hypothesis'])
        class Client:
            def complete(self,messages,seed):
                return json.dumps({'interpretations':[],'cue_verdict':'not_applicable'}),{}
        events=[]
        run_case(Client(),row,row['cues'][1],'direct',3,events.append)
        self.assertIn('Viewing instruction',events[0]['messages'][-1]['content'])
        self.assertNotIn('Suggested interpretation',events[0]['messages'][-1]['content'])

    def test_compare_does_not_automatically_adjudicate(self):
        for which,rows in [('a',['1','','NA']),('b',['0','','NA'])]:
            with (self.root/f'{which}.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['review_id','both_valid','false_alternative',
                                              'cue_correct','grounded','notes'])
                w.writerow(['r1',rows[0],'NA','NA','1',''])
                w.writerow(['r2',rows[1],'NA','NA','',''])
                w.writerow(['r3',rows[2],rows[2],rows[2],rows[2],''])
        result=compare(self.root/'a.csv',self.root/'b.csv',self.root/'comparison')
        self.assertEqual(result,{'agreed':1,'disagreements':1,'incomplete':1,'total':3})


if __name__=='__main__':unittest.main()

class AdjudicationGateTests(unittest.TestCase):
    def test_ordinary_cue_cannot_be_the_only_correct_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'im.jpg').write_bytes(b'bytes')
            m=root/'manifest.jsonl'
            m.write_text(json.dumps({'id':'x','original_id':'x','image':'im.jpg',
                'sha256':sha256(b'bytes'),'split':'debug',
                'cues':[{'id':'c0','text':''}]})+'\n')
            gold=root/'gold.jsonl'
            gold.write_text(json.dumps({'id':'x','group':'ordinary',
                'review_status':'adjudicated','approved_interpretations':['duck'],
                'target_label':'duck'})+'\n')
            with self.assertRaisesRegex(ValueError, 'misleading suggestion'):
                build(m,gold,root/'ABCD.jsonl')
