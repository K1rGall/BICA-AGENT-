"""Compare two blind raters without guessing consensus for disagreements."""
import argparse
import csv
from pathlib import Path

METRICS=("both_valid", "false_alternative", "cue_correct", "grounded")


def read(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows or len({r['review_id'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate review_id in '+str(path))
    return {r['review_id']:r for r in rows}


def compare(a,b,out):
    a,b=read(a),read(b)
    if a.keys()!=b.keys():
        raise ValueError('Reviewers must score exactly the same review IDs')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    agreements=[]; unresolved=[]; incomplete=[]
    for rid in sorted(a):
        left,right=a[rid],b[rid]
        values_a={m:left.get(m,'').strip() for m in METRICS}
        values_b={m:right.get(m,'').strip() for m in METRICS}
        if any(x not in {'0','1','NA'} for x in (*values_a.values(),*values_b.values())):
            incomplete.append({'review_id':rid, 'reason':'missing_or_invalid_rating'})
            continue
        differing=[k for k in METRICS if values_a[k]!=values_b[k]]
        if differing:
            unresolved.append({'review_id':rid, 'fields':';'.join(differing),
                               **{m+'_A':values_a[m] for m in METRICS},
                               **{m+'_B':values_b[m] for m in METRICS},
                               **{m+'_final':'' for m in METRICS},
                               'adjudication_reason':''})
        else:
            agreements.append({'review_id':rid,**values_a,
                               'notes':'Two independent raters agreed'})
    def write(path,rows,fields):
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write(out/'agreements_ONLY.csv',agreements,['review_id',*METRICS,'notes'])
    write(out/'disagreements_TO_ADJUDICATE.csv',unresolved,
          ['review_id','fields',*[m+'_A' for m in METRICS],
           *[m+'_B' for m in METRICS],*[m+'_final' for m in METRICS],
           'adjudication_reason'])
    write(out/'incomplete_ROWS.csv',incomplete,['review_id','reason'])
    return {'agreed':len(agreements),'disagreements':len(unresolved),
            'incomplete':len(incomplete),'total':len(a)}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reviewer-a',required=True)
    p.add_argument('--reviewer-b',required=True)
    p.add_argument('--output',required=True)
    x=p.parse_args()
    print(compare(x.reviewer_a,x.reviewer_b,x.output))
