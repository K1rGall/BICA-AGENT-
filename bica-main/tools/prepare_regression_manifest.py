"""Prepare a 2-image regression subset of the ORIGINAL v0.2 failures."""
import argparse
import json
from pathlib import Path
from illusion_agent.core import load_manifest


def build(manifest, output, target_ids=('ambi-debug-004','ambi-debug-005')):
    rows=load_manifest(manifest)
    selected=[r for r in rows if r['id'] in target_ids]
    if len(selected)!=len(target_ids):
        raise ValueError('Not all selected cases were found')
    output=Path(output)
    if output.parent.resolve()!=Path(manifest).resolve().parent:
        raise ValueError('Output must share the source manifest directory')
    with output.open('x',encoding='utf-8') as f:
        for row in selected:
            f.write(json.dumps({k:row[k] for k in ('id','original_id','image','sha256','split','cues')},ensure_ascii=False)+'\n')
    load_manifest(output)
    return selected

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',default='data/ambibench_debug/manifest.jsonl')
    p.add_argument('--output',default='data/ambibench_debug/manifest_regression_v021.jsonl')
    x=p.parse_args();print('Subset images:', len(build(x.manifest,x.output)))
