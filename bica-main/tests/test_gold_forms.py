import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from tools.create_gold_forms import build
from illusion_agent.core import sha256

class GoldFormTests(unittest.TestCase):
    def test_gold_form_is_blinded_and_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/'im.png'; buffer=io.BytesIO(); Image.new('RGB',(32,32),'blue').save(buffer,format='PNG')
            p.write_bytes(buffer.getvalue())
            manifest=root/'manifest.jsonl'
            manifest.write_text(json.dumps({'id':'private-001','original_id':'private-origin',
                'image':'im.png','sha256':sha256(p.read_bytes()),'split':'debug',
                'cues':[{'id':'c0','text':''}]})+'\n')
            out=root/'review'
            self.assertEqual(build(manifest,out),1)
            sheet=(out/'gold_reviewer_A.csv').read_text(encoding='utf-8-sig')
            self.assertNotIn('private-001',sheet)
            self.assertNotIn('private-origin',sheet)
            self.assertEqual((out/'gold_reviewer_B.csv').read_text(encoding='utf-8-sig'),sheet)
            self.assertIn('private-001',(root/'review_OPERATOR_ONLY_mapping.json').read_text())
            with self.assertRaises(FileExistsError):build(manifest,out)
