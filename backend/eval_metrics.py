"""Human-reference metrics with explicit abstentions, denominators and source checks."""
from collections import Counter,defaultdict
from math import ceil,isfinite
from statistics import median

import cqa


def divide(a,b):return a/b if b else None


def summary(items):
    errors=[i['absolute_error'] for i in items if i['absolute_error'] is not None]
    norms=[i['normalized_absolute_error'] for i in items if i['normalized_absolute_error'] is not None]
    return dict(criteria_count=len(items),answered_count=sum(i['answered'] for i in items),
                answer_coverage=divide(sum(i['answered'] for i in items),len(items)),
                exact_accuracy_including_unanswered=divide(sum(i['exact'] for i in items),len(items)),
                scored_pairs_for_mae=len(errors),gold_scored_count=sum(i['gold_status']=='scored' for i in items),
                mae_on_scored_pairs_only=divide(sum(errors),len(errors)),
                normalized_mae_on_scored_pairs_only=divide(sum(norms),len(norms)),
                status_confusion=dict(Counter(i['gold_status']+' -> '+i['predicted_status'] for i in items)))


def compare_v2(gold,runs,stage='original',reference_kind='human'):
    if stage not in ('original','final','primary','review'):raise ValueError('Unknown evaluation stage')
    if reference_kind=='human':
        if gold.get('annotation_state')!='adjudicated' or gold.get('annotation_source')!='human':raise ValueError('An adjudicated human reference is required')
    elif reference_kind=='ai':
        if gold.get('annotation_source')!='ai_assistant' or gold.get('annotation_state')!='ai_reviewed' or gold.get('human_reviewed') is not False:
            raise ValueError('Explicitly identified AI annotation required; do not relabel human provenance')
    else:raise ValueError('Unknown reference kind')
    checklist=cqa.validate_checklist(gold['checklist_snapshot'])
    if cqa.digest(checklist)!=gold['checklist_hash']:raise ValueError('Reference checklist hash mismatch')
    config={c['id']:c for c in checklist['criteria']}
    predictions={}
    for wrapper in runs:
        identifier=wrapper['conversation_id']
        if identifier in predictions:raise ValueError('Duplicate prediction ID')
        predictions[identifier]=wrapper
    all_items=[];rows=[];unresolved=[];totals=[];flags=defaultdict(Counter)
    groups={name:defaultdict(list) for name in ('criterion','language','channel')}
    technical_errors=0;quoted=valid_quotes=0;durations=[];costs=[];seen=set()
    for reference in gold['records']:
        identifier=reference['conversation_id']
        if identifier in seen:raise ValueError('Duplicate reference ID')
        seen.add(identifier)
        turns=reference['messages']
        if cqa.digest([{k:m[k] for k in ('timestamp','speaker','text')} for m in turns])!=reference['transcript_hash']:
            raise ValueError('Reference transcript changed')
        refs=reference['findings']
        if len(refs)!=len(config) or {f['criterion_id'] for f in refs}!=set(config):raise ValueError('Incomplete reference criteria')
        reference_aggregate=[]
        for truth in refs:
            expected_state='adjudicated' if reference_kind=='human' else 'submitted'
            if truth.get('annotation_state')!=expected_state or not truth.get('annotator_id'):raise ValueError('Unapproved reference item')
            if reference_kind=='ai' and truth.get('ai_assisted') is not True:raise ValueError('AI item provenance must be explicit')
            if gold.get('split')=='test' and truth.get('ai_assisted'):raise ValueError('Holdout reference is not blind')
            c=config[truth['criterion_id']]
            if truth['status']=='unresolved':
                if truth.get('score') is not None or not truth.get('reason'):raise ValueError('Invalid unresolved reference')
                reference_aggregate.append(cqa.unknown(c,truth['reason']))
            else:
                checked=cqa.validate_finding(c,truth,turns,gold.get('knowledge_snapshot'))
                if checked['status'] not in ('scored','na'):raise ValueError('Invalid reference score, N/A or evidence')
                reference_aggregate.append(checked)
        expected_total=cqa.aggregate(checklist,reference_aggregate)['score']
        if reference.get('total_score')!=expected_total:raise ValueError('Reference total must be calculated from findings')
        wrapper=predictions.get(identifier)
        run=wrapper.get('result') if wrapper and 'result' in wrapper else wrapper
        error=bool(wrapper and wrapper.get('status')=='error');technical_errors+=int(error)
        if run:
            if run['checklist_hash']!=gold['checklist_hash'] or run['transcript_hash']!=reference['transcript_hash']:raise ValueError('Prediction source/checklist versions differ')
            if run.get('messages')!=turns:raise ValueError('Prediction messages differ from reference')
        metadata=(wrapper or {}).get('inference',{})
        duration=metadata.get('duration_seconds')
        if type(duration) in (int,float) and isfinite(duration) and duration>=0:durations.append(duration)
        cost=metadata.get('estimated_usd')
        if type(cost) in (int,float) and isfinite(cost) and cost>=0:costs.append(cost)
        if run and stage in ('primary','review'):
            selected=next((s for s in (wrapper.get('stages') or run.get('stages',[])) if s.get('name')==stage and s.get('status')=='completed'),None)
            raw=selected.get('findings',[]) if selected else []
        else:raw=run.get(stage+'_findings',[]) if run else []
        if len({f['criterion_id'] for f in raw})!=len(raw):raise ValueError('Duplicate predicted criterion')
        by_id={f['criterion_id']:f for f in raw}
        if set(by_id)-set(config):raise ValueError('Unknown predicted criterion')
        validated={}
        original={m['message_id']:m for m in turns}
        for key,f in by_id.items():
            for e in f.get('evidence',[]):
                quoted+=1
                m=original.get(e.get('message_id'))
                valid_quotes+=int(bool(m and isinstance(e.get('quote'),str) and e['quote'] and e['quote'] in m['text'] and m['speaker'] in config[key].get('evidence_roles',['operator'])))
            validated[key]=cqa.validate_finding(config[key],f,turns,run.get('knowledge_snapshot'))
        items=[];reference_complete=True
        for truth in refs:
            key=truth['criterion_id'];c=config[key];prediction=validated.get(key)
            ps=prediction['status'] if prediction else ('error' if error else 'missing')
            if truth['status']=='unresolved':
                reference_complete=False
                unresolved.append(dict(conversation_id=identifier,criterion_id=key,predicted_status=ps));continue
            answered=ps in ('scored','na');exact=answered and ps==truth['status']
            absolute=normalized=None
            if truth['status']=='scored' and ps=='scored':
                absolute=abs(float(prediction['score'])-float(truth['score']))
                normalized=absolute/(float(c['max_score'])-float(c['min_score']))
                exact=exact and absolute==0
            item=dict(conversation_id=identifier,criterion_id=key,gold_status=truth['status'],predicted_status=ps,
                      answered=answered,exact=bool(exact),absolute_error=absolute,normalized_absolute_error=normalized)
            items.append(item);all_items.append(item)
            for name,value in [('criterion',key),('language',reference['language']),('channel',reference['channel'])]:groups[name][value].append(item)
            if c['type']=='flag' and truth['status']=='scored':
                positive=truth['score']==c.get('flag_score',c['max_score'])
                if ps!='scored':flags[key]['unanswered_positive' if positive else 'unanswered_negative']+=1
                else:
                    predicted=prediction['score']==c.get('flag_score',c['max_score'])
                    flags[key]['true_positives' if positive and predicted else 'false_negatives' if positive else 'false_positives' if predicted else 'true_negatives']+=1
        predicted_total=cqa.aggregate(checklist,list(validated.values()))['score'] if run else None
        if expected_total is not None:totals.append(dict(conversation_id=identifier,gold_total=expected_total,predicted_total=predicted_total,
                                                        absolute_error=abs(expected_total-predicted_total) if predicted_total is not None else None))
        rows.append(dict(conversation_id=identifier,reference_complete=reference_complete,all_criteria_exact=bool(items) and all(i['exact'] for i in items),findings=items))
    if not rows:raise ValueError('Empty reference')
    flag_report={}
    for key,counts in flags.items():
        result={k:counts[k] for k in ('true_positives','false_positives','false_negatives','true_negatives','unanswered_positive','unanswered_negative')}
        result.update(precision=divide(counts['true_positives'],counts['true_positives']+counts['false_positives']),
                      recall_including_unanswered=divide(counts['true_positives'],counts['true_positives']+counts['false_negatives']+counts['unanswered_positive']))
        flag_report[key]=result
    complete=[r for r in rows if r['reference_complete']];available=[r['absolute_error'] for r in totals if r['absolute_error'] is not None]
    warning=('Agreement with AI-authored reference, NOT independent human accuracy. Reference was not blind to prior model outputs.' if reference_kind=='ai'
             else 'Local diagnostics, NOT official ranking formula. Small internal holdout is not a production accuracy guarantee.')
    return dict(schema_version=2,warning=warning,reference_kind=reference_kind,human_reference=reference_kind=='human',
                stage=stage,gold_records=len(rows),checklist_hash=gold['checklist_hash'],overall=summary(all_items),
                all_gold_criteria_answered=bool(all_items) and all(i['answered'] for i in all_items),
                unresolved_reference_count=len(unresolved),unresolved_references=unresolved,
                reference_complete_dialogues=len(complete),exact_dialogue_accuracy=divide(sum(r['all_criteria_exact'] for r in complete),len(complete)),
                by_criterion={k:summary(v) for k,v in groups['criterion'].items()},
                by_language={k:summary(v) for k,v in groups['language'].items()},by_channel={k:summary(v) for k,v in groups['channel'].items()},
                flags=flag_report,technical_error_count=technical_errors,
                evidence=dict(quotes=quoted,valid_role_quotes=valid_quotes,structural_validity=divide(valid_quotes,quoted),semantic_validity='Requires human review'),
                total_score_comparison=dict(gold_totals=len(totals),available_predictions=len(available),coverage=divide(len(available),len(totals)),
                                            mae_on_available_totals_only=divide(sum(available),len(available))),
                performance=dict(known_durations=len(durations),p50_seconds=median(durations) if durations else None,
                                 p95_seconds=sorted(durations)[max(0,ceil(.95*len(durations))-1)] if durations else None,
                                 known_costs=len(costs),known_cost_usd=sum(costs) if costs else None,unknown_costs=len(rows)-len(costs)),
                unused_prediction_records=sorted(set(predictions)-seen),rows=rows)
