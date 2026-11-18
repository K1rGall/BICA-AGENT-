"""Prepare two separate image-only annotation sheets; never auto-create gold truth.

Example: python -m tools.create_gold_forms --manifest ... --output gold_pack
"""
import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from illusion_agent.core import load_manifest

FIELDS = ('blind_id', 'image', 'group', 'interpretation_1', 'interpretation_2',
          'evidence_1', 'evidence_2', 'composition_mechanism', 'source_url',
          'rights', 'confidence_0_to_2', 'notes')


def build(manifest, output, seed=109):
    rows = load_manifest(manifest)
    out = Path(output)
    if out.exists():
        raise FileExistsError('Review folder already exists: ' + str(out))
    out.mkdir(parents=True)
    (out/'images').mkdir()
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    table, mapping = [], []
    for count, idx in enumerate(indices):
        entry = rows[idx]
        blind_id = f'gold-{count:05d}'
        basename = blind_id + entry['image_path'].suffix.lower()
        shutil.copy2(entry['image_path'], out/'images'/basename)
        table.append(dict.fromkeys(FIELDS, ''))
        table[-1].update(blind_id=blind_id, image='images/'+basename)
        mapping.append({'blind_id':blind_id, 'image_id':entry['id'],
                        'original_id':entry['original_id'], 'sha256':entry['sha256']})
    for who in ('A', 'B'):
        with (out/f'gold_reviewer_{who}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader();writer.writerows(table)
    # The mapping is OUTSIDE the shared reviewer folder (one parent higher).
    (out.parent/(out.name+'_OPERATOR_ONLY_mapping.json')).write_text(
        json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'INSTRUCTIONS_RU.txt').write_text('''BICA: независимая разметка изображений

Передайте каждому участнику только папку images и ЕГО gold_reviewer_A/B.csv.
НЕ передавайте соседний файл *_OPERATOR_ONLY_mapping.json и ответы моделей.

Для каждой картинки:
  group = ambiguous | ordinary | exclude | unsure
  interpretation_1/2 = визуально РАЗНЫЕ формы, не синонимы
  evidence_1/2 = расположение, линии, общие контуры и их роль
  composition_mechanism = figure-ground | composite | orientation | reflection | other
  source_url и rights = по проверяемому источнику (не придумывайте лицензию)
  confidence_0_to_2 = 0 | 1 | 2
  notes = сомнения, спорные детали.

После получения двух CSV сравните результаты и оформите отдельную
ручную адъюдикацию. Нельзя автоматически признавать исходные AmbiBench
метки правильными или считать согласие двух участников доказательством.
Список утверждённых интерпретаций и источников оформляется в отдельном
gold_adjudicated.jsonl для генератора A–D. При споре исключите пример.
''',encoding='utf-8')
    return len(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=109)
    args=p.parse_args()
    print('Prepared:',build(args.manifest,args.output,args.seed),'images')
