"""Add a locally supplied image as an UNREVIEWED ordinary-control candidate.

No claim of ground truth is inferred from a filename or generated automatically.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def add(image, example_id, folder, source_url, license_note):
    image=Path(image).resolve(); folder=Path(folder).resolve()
    if not image.is_file() or image.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'}:
        raise ValueError('Provide an existing JPG/PNG/WebP image')
    if not example_id.startswith('ordinary-') or not example_id.replace('-','').isalnum():
        raise ValueError('Use ordinary-NNN alphanumeric ID')
    if not source_url.strip() or not license_note.strip():
        raise ValueError('Record original source and rights/permission')
    manifest=folder/'manifest.jsonl'; annotations=folder/'annotations.jsonl'
    for path in (manifest,annotations):
        if not path.exists(): raise FileNotFoundError(path)
    rows=[json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    gold=[json.loads(s) for s in annotations.read_text().splitlines() if s.strip()]
    if any(x['id']==example_id for x in rows+gold):
        raise ValueError('Duplicate ID')
    digest=hashlib.sha256(image.read_bytes()).hexdigest()
    if any(x['sha256']==digest for x in rows):
        raise ValueError('Duplicate image SHA256')
    newfile=folder/'images'/(example_id+image.suffix.lower())
    newfile.parent.mkdir(exist_ok=True)
    shutil.copy2(image,newfile)
    rows.append({'id':example_id,'original_id':example_id,
                 'image':'images/'+newfile.name, 'sha256':digest,'split':'debug',
                 'cues':[{'id':'c0','text':''}]})
    gold.append({'id':example_id,'group':'ordinary',
                 'reference_answer':'', 'source_question':'',
                 'review_status':'pending_two_human_reviewers',
                 'evidence':[], 'cues':{'c0':'neutral'},
                 'source':source_url,'rights':license_note})
    # Atomic append to verified copies; no hidden automatic adjudication.
    for path,data in [(manifest,rows),(annotations,gold)]:
        tmp=path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
        tmp.replace(path)
    return newfile

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True); p.add_argument('--id',required=True)
    p.add_argument('--dataset',default='data/ambibench_debug')
    p.add_argument('--source',required=True);p.add_argument('--rights',required=True)
    x=p.parse_args();print(add(x.image,x.id,x.dataset,x.source,x.rights))
