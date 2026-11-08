"""Local diagnostics, NOT the official ranking formula. Defaults to original AI results."""
import argparse
from collections import Counter
from decimal import Decimal
from cqa import digest, load_records, read_json, validate_checklist, write_json


def make_template(checklist, records):
    validate_checklist(checklist)
    return dict(checklist_hash=digest(checklist), annotation_source='REPLACE_WITH_HUMAN_ANNOTATORS', records=[
        dict(conversation_id=r['conversation_id'], language=r['language'], channel=r['channel'], total_score=None,
             findings=[dict(criterion_id=c['id'], status=None, score=None, annotation_note='Independently annotate from official rubric; do not copy predictions') for c in checklist['criteria']]) for r in records])


def compare(gold, runs, stage='original',reference_kind='human'):
    if gold.get('schema_version')==2:
        from eval_metrics import compare_v2
        return compare_v2(gold,runs,stage,reference_kind)
    if reference_kind!='human':raise ValueError('AI reference comparison requires schema v2')
    if stage not in ('original', 'final') or not gold.get('checklist_hash') or not gold.get('records'):
        raise ValueError('Valid stage, checklist hash and nonempty gold required')
    by_run = {}
    for r in runs:
        if r['conversation_id'] in by_run: raise ValueError('Duplicate prediction ID')
        by_run[r['conversation_id']] = r
    seen, rows, totals, cs, ls = set(), [], [], {}, {}
    for r in gold['records']:
        cid = r['conversation_id']
        if cid in seen: raise ValueError('Duplicate gold ID')
        seen.add(cid)
        run = by_run.get(cid)
        if run and run['checklist_hash'] != gold['checklist_hash']: raise ValueError('Checklist versions differ')
        config = {c['id']: c for c in run['checklist_snapshot']['criteria']} if run else {}
        predictions = {}
        for f in run[f'{stage}_findings'] if run else []:
            if f['criterion_id'] in predictions: raise ValueError('Duplicate prediction criterion')
            predictions[f['criterion_id']] = f
        if not r.get('findings'): raise ValueError('No gold findings')
        cseen, items = set(), []
        for truth in r['findings']:
            crit, status = truth['criterion_id'], truth.get('status')
            if crit in cseen: raise ValueError('Duplicate gold criterion')
            cseen.add(crit)
            if status not in ('scored', 'na'): raise ValueError('Finish independent annotation first')
            if status == 'na' and truth.get('score') is not None: raise ValueError('N/A has no score')
            if status == 'scored' and (isinstance(truth.get('score'), bool) or not isinstance(truth.get('score'), (int, float))): raise ValueError('Numeric gold score required')
            if crit in config:
                c = config[crit]
                if status == 'na' and not c['allow_na']: raise ValueError('Gold N/A forbidden')
                if status == 'scored':
                    value = Decimal(str(truth['score']))
                    if not value.is_finite() or not Decimal(str(c['min_score'])) <= value <= Decimal(str(c['max_score'])): raise ValueError('Gold outside scale')
                    if c['type'] in ('binary','flag') and value not in (Decimal(str(c['min_score'])),Decimal(str(c['max_score']))): raise ValueError('Gold binary score not endpoint')
                    if 'allowed_scores' in c and value not in {Decimal(str(v)) for v in c['allowed_scores']}: raise ValueError('Gold score not a rubric step')
            p = predictions.get(crit)
            ps = p.get('status') if p else 'missing'
            valid = bool(p and ps in ('scored', 'na'))
            exact = valid and ps == status
            error = norm = None
            if status == 'scored' and ps == 'scored':
                pv = p.get('score')
                if isinstance(pv, bool) or not isinstance(pv, (int, float)): raise ValueError('Prediction must be numeric')
                error = abs(float(pv) - float(truth['score']))
                exact = exact and Decimal(str(pv)) == Decimal(str(truth['score']))
                if crit in config: norm = error / (float(config[crit]['max_score']) - float(config[crit]['min_score']))
            item = dict(conversation_id=cid, criterion_id=crit, gold_status=status, predicted_status=ps,
                        answered=valid, exact=bool(exact), absolute_error=error, normalized_absolute_error=norm)
            items.append(item); cs.setdefault(crit, []).append(item); ls.setdefault(r['language'], []).append(item)
        if run and cseen != set(config): raise ValueError('Gold must label every criterion')
        gt = r.get('total_score')
        if gt is not None:
            if isinstance(gt, bool) or not isinstance(gt,(int,float)) or not 0 <= gt <= 100: raise ValueError('Gold total must be 0..100')
            pt = run[f'{stage}_summary'].get('score') if run else None
            totals.append(dict(conversation_id=cid,gold_total=gt,predicted_total=pt,absolute_error=abs(pt-gt) if pt is not None else None))
        rows.append(dict(conversation_id=cid,all_criteria_exact=all(i['exact'] for i in items),findings=items))
    def summary(items):
        es = [i['absolute_error'] for i in items if i['absolute_error'] is not None]
        ns = [i['normalized_absolute_error'] for i in items if i['normalized_absolute_error'] is not None]
        return dict(criteria_count=len(items),answered_count=sum(i['answered'] for i in items),
            answer_coverage=sum(i['answered'] for i in items)/len(items),exact_accuracy_including_unanswered=sum(i['exact'] for i in items)/len(items),
            scored_pairs_for_mae=len(es),gold_scored_count=sum(i['gold_status']=='scored' for i in items),
            mae_on_scored_pairs_only=sum(es)/len(es) if es else None,normalized_mae_on_scored_pairs_only=sum(ns)/len(ns) if ns else None,
            status_confusion=dict(Counter(i['gold_status']+' -> '+str(i['predicted_status']) for i in items)))
    all_items = [i for row in rows for i in row['findings']]
    te = [t['absolute_error'] for t in totals if t['absolute_error'] is not None]
    return dict(warning='Local diagnostics, NOT official ranking formula. Coverage is not accuracy.',stage=stage,gold_records=len(rows),
        checklist_hash=gold['checklist_hash'],all_gold_criteria_answered=all(i['answered'] for i in all_items),
        exact_dialogue_accuracy=sum(r['all_criteria_exact'] for r in rows)/len(rows),overall=summary(all_items),
        by_criterion={k:summary(v) for k,v in cs.items()},by_language={k:summary(v) for k,v in ls.items()},
        total_score_comparison=dict(gold_totals=len(totals),available_predictions=len(te),mae_on_available_totals_only=sum(te)/len(te) if te else None),
        unused_prediction_records=sorted(set(by_run)-seen),rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('template'); a.add_argument('--data',required=True);a.add_argument('--checklist',required=True);a.add_argument('--out',required=True)
    a=sub.add_parser('compare');a.add_argument('--gold',required=True);a.add_argument('--predictions',required=True);a.add_argument('--out',required=True);a.add_argument('--stage',choices=['original','final','primary','review'],default='original')
    a.add_argument('--reference-kind',choices=['human','ai'],default='human',help='AI mode preserves AI provenance and reports agreement, not independent accuracy.')
    a=p.parse_args()
    result=make_template(read_json(a.checklist),load_records(read_json(a.data))) if a.command=='template' else compare(read_json(a.gold),read_json(a.predictions),a.stage,a.reference_kind)
    write_json(a.out,result)
    print('Annotation template written; no labels yet.' if a.command=='template' else __import__('json').dumps(result['overall'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
