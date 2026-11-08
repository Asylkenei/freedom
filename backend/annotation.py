"""Offline blind annotation packages. Never creates human labels or calls a model."""
import argparse
import copy
import csv
import json
from collections import Counter,defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import cqa


def dataset_hash(records):
    return cqa.digest(sorted(records,key=lambda r:r['conversation_id']))


def make_template(checklist,records):
    cqa.validate_checklist(checklist)
    cqa.load_records({'records':records})
    return dict(schema_version=2,annotation_state='draft',annotation_source='human_pending',annotation_version='1',
                source_dataset_hash=dataset_hash(records),checklist_hash=cqa.digest(checklist),
                checklist_snapshot=copy.deepcopy(checklist),knowledge_snapshot={'sources':[]},records=[dict(
                    conversation_id=r['conversation_id'],language=r['language'],channel=r['channel'],
                    transcript_hash=cqa.digest(r['transcript']),record_hash=cqa.digest(r),messages=cqa.messages(r),total_score=None,
                    findings=[dict(criterion_id=c['id'],status=None,score=None,reason='',evidence=[],checked_span=None,
                                   na_reason=None,knowledge_evidence=[],factual_status='not_assessed',verification_limits=[],
                                   annotator_id='',reviewer_id='',annotation_state='draft',ai_assisted=False)
                              for c in checklist['criteria']]) for r in records])


def prepare(records,checklist,seed='cqa-split-v1',development_ids=()):
    records=sorted(cqa.load_records({'records':copy.deepcopy(records)}),key=lambda r:r['conversation_id'])
    counts=Counter((r['language'],r['channel']) for r in records)
    official={('RU','agent chat'):30,('KZ','agent chat'):15,('EN','agent chat'):15,('RU','voice call'):5}
    quotas={('RU','agent chat'):8,('KZ','agent chat'):5,('EN','agent chat'):5,('RU','voice call'):2} if counts==official else {
        k:min(n-1,max(1,round(n*.3))) if n>=3 else 0 for k,n in counts.items()}
    forced=set(development_ids)
    for k in counts:
        forced.add(next(r['conversation_id'] for r in records if (r['language'],r['channel'])==k))
    if not forced<={r['conversation_id'] for r in records}:raise ValueError('Unknown development-only ID')
    groups=defaultdict(list)
    for r in records:groups[cqa.digest(r['transcript'])].append(r)
    assignments={};remaining=dict(quotas)
    for group_id,members in sorted(groups.items(),key=lambda item:cqa.digest([seed,item[0]])):
        needed=Counter((r['language'],r['channel']) for r in members)
        selected=not any(r['conversation_id'] in forced for r in members) and all(remaining[k]>=n for k,n in needed.items())
        for r in members:assignments[r['conversation_id']]='test' if selected else 'development'
        if selected:
            for k,n in needed.items():remaining[k]-=n
    if any(remaining.values()):raise ValueError('Duplicate groups/development exclusions prevent requested split; adjust the manifest explicitly')
    entries=[dict(conversation_id=r['conversation_id'],language=r['language'],channel=r['channel'],
                  transcript_hash=cqa.digest(r['transcript']),record_hash=cqa.digest(r),
                  group_id=cqa.digest(r['transcript']),split=assignments[r['conversation_id']]) for r in records]
    near=[]
    for index,a in enumerate(records):
        for b in records[index+1:]:
            if (a['language'],a['channel'])!=(b['language'],b['channel']) or assignments[a['conversation_id']]==assignments[b['conversation_id']]:continue
            text=lambda r:'\n'.join(m['speaker']+':'+m['text'].casefold() for m in r['transcript'] if m['speaker'] in ('client','operator'))
            ratio=SequenceMatcher(None,text(a),text(b),autojunk=False).ratio()
            if ratio>=.92:near.append(dict(left=a['conversation_id'],right=b['conversation_id'],similarity=round(ratio,4)))
    manifest=dict(schema_version=1,seed=seed,source_dataset_hash=dataset_hash(records),checklist_hash=cqa.digest(checklist),
                  development_only_ids=sorted(forced),records=entries,near_duplicate_candidates=near,
                  near_duplicate_review='pending' if near else 'no_candidates',
                  warning='Internal holdout; near-duplicate candidates require human review before independence claims.')
    manifest['manifest_hash']=cqa.digest(manifest)
    package={'manifest':manifest}
    for split in ('development','test'):
        doc=make_template(checklist,[r for r in records if assignments[r['conversation_id']]==split])
        doc.update(source_dataset_hash=manifest['source_dataset_hash'],manifest_hash=manifest['manifest_hash'],split=split)
        package[split]=doc
    return package


def validate_annotations(document,records,checklist,require_adjudicated=True):
    if document.get('schema_version')!=2 or document.get('checklist_hash')!=cqa.digest(checklist):raise ValueError('Wrong annotation/checklist version')
    if document.get('source_dataset_hash')!=dataset_hash(records):raise ValueError('Source dataset changed')
    source={r['conversation_id']:r for r in cqa.load_records({'records':records})}
    if not document.get('records'):raise ValueError('No annotations')
    output=copy.deepcopy(document);seen=set();config={c['id']:c for c in checklist['criteria']}
    for row in output['records']:
        identifier=row['conversation_id']
        if identifier in seen or identifier not in source:raise ValueError('Duplicate or unknown conversation')
        seen.add(identifier);r=source[identifier]
        if row.get('record_hash')!=cqa.digest(r) or row.get('transcript_hash')!=cqa.digest(r['transcript']):raise ValueError('Annotated source changed')
        turns=cqa.messages(r)
        if row.get('messages')!=turns:raise ValueError('Blind transcript was modified')
        findings=row.get('findings',[])
        if len(findings)!=len(config) or {f.get('criterion_id') for f in findings}!=set(config):raise ValueError('Label every current criterion exactly once')
        validated=[];aggregate=[]
        for f in findings:
            c=config[f['criterion_id']]
            states=('adjudicated',) if require_adjudicated else ('submitted','adjudicated')
            if f.get('annotation_state') not in states or not isinstance(f.get('annotator_id'),str) or not f['annotator_id'].strip():
                raise ValueError('Finish independent human annotation and identify annotator')
            if type(f.get('ai_assisted')) is not bool:raise ValueError('Declare AI assistance')
            if document.get('split')=='test' and f['ai_assisted']:raise ValueError('Holdout labels must be blind to model predictions')
            if f.get('status')=='unresolved':
                if f.get('score') is not None or not f.get('reason','').strip():raise ValueError('Unresolved reference needs reason and null score')
                aggregate.append(cqa.unknown(c,f['reason']));validated.append(f);continue
            if f.get('status') not in ('scored','na'):raise ValueError('Unfinished reference label')
            checked=cqa.validate_finding(c,f,turns,output.get('knowledge_snapshot'))
            if checked['status']=='needs_review':raise ValueError('Invalid human finding '+c['id']+': '+checked['reason'])
            aggregate.append(checked)
            checked.update({k:f[k] for k in ('annotator_id','annotation_state','ai_assisted')})
            checked['reviewer_id']=f.get('reviewer_id','');validated.append(checked)
        row['findings']=validated;row['total_score']=cqa.aggregate(checklist,aggregate)['score']
        row['language']=r['language'];row['channel']=r['channel']
    output.update(annotation_state='adjudicated' if require_adjudicated else 'submitted',annotation_source='human',
                  checklist_snapshot=copy.deepcopy(checklist))
    return output


def agreement(left,right,records,checklist):
    a=validate_annotations(left,records,checklist,False);b=validate_annotations(right,records,checklist,False)
    flatten=lambda d:{(r['conversation_id'],f['criterion_id']):f for r in d['records'] for f in r['findings']}
    aa,bb=flatten(a),flatten(b)
    if set(aa)!=set(bb):raise ValueError('Annotators must label the same items')
    differences=[dict(conversation_id=k[0],criterion_id=k[1],left_status=aa[k]['status'],left_score=aa[k]['score'],
                      right_status=bb[k]['status'],right_score=bb[k]['score']) for k in aa if
                 (aa[k]['status'],aa[k]['score'])!=(bb[k]['status'],bb[k]['score'])]
    same_author=sum(aa[k]['annotator_id']==bb[k]['annotator_id'] for k in aa)
    return dict(items=len(aa),exact_agreement=(len(aa)-len(differences))/len(aa),disagreements=differences,
                same_annotator_items=same_author,warning='Agreement is not accuracy. Same-author items are not independent double annotation.')


CSV_FIELDS=['conversation_id','criterion_id','status','score','reason','evidence','checked_span','na_reason',
            'knowledge_evidence','factual_status','verification_limits','annotator_id','reviewer_id','annotation_state','ai_assisted']
JSON_FIELDS={'evidence','checked_span','knowledge_evidence','verification_limits'}


def to_csv(document,path):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS);writer.writeheader()
        for row in document['records']:
            for f in row['findings']:
                item={k:f.get(k) for k in CSV_FIELDS};item['conversation_id']=row['conversation_id']
                for k in JSON_FIELDS:item[k]=json.dumps(item[k],ensure_ascii=False)
                item['ai_assisted']='true' if f.get('ai_assisted') else 'false'
                writer.writerow(item)


def from_csv(template,path):
    result=copy.deepcopy(template)
    findings={(r['conversation_id'],f['criterion_id']):f for r in result['records'] for f in r['findings']};seen=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames!=CSV_FIELDS:raise ValueError('Unexpected annotation CSV columns')
        for row in reader:
            key=row.pop('conversation_id'),row['criterion_id']
            if key in seen or key not in findings:raise ValueError('Duplicate/unknown CSV item')
            seen.add(key)
            row['score']=json.loads(row['score']) if row['score'] else None
            row['status']=row['status'] or None;row['na_reason']=row['na_reason'] or None
            if row['ai_assisted'] not in ('true','false'):raise ValueError('ai_assisted must be true/false')
            row['ai_assisted']=row['ai_assisted']=='true'
            for k in JSON_FIELDS:row[k]=json.loads(row[k])
            findings[key].update(row)
    if seen!=set(findings):raise ValueError('Missing CSV items')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--data',required=True);p.add_argument('--checklist',default='checklist.official.json')
    p.add_argument('--out',required=True);p.add_argument('--seed',default='cqa-split-v1');p.add_argument('--development-id',action='append',default=[])
    for name in ('validate','export'):
        p=sub.add_parser(name);p.add_argument('--annotations',required=True);p.add_argument('--data',required=True)
        p.add_argument('--checklist',default='checklist.official.json');p.add_argument('--out',required=name=='export')
    p=sub.add_parser('agreement');p.add_argument('--left',required=True);p.add_argument('--right',required=True)
    p.add_argument('--data',required=True);p.add_argument('--checklist',default='checklist.official.json');p.add_argument('--out',required=True)
    p=sub.add_parser('from-csv');p.add_argument('--template',required=True);p.add_argument('--csv',required=True);p.add_argument('--out',required=True)
    args=parser.parse_args()
    if args.command=='from-csv':result=from_csv(cqa.read_json(args.template),args.csv)
    else:
        records=cqa.load_records(cqa.read_json(args.data));checklist=cqa.read_json(args.checklist)
        if args.command=='prepare':
            package=prepare(records,checklist,args.seed,args.development_id);directory=Path(args.out)
            if directory.exists() and any(directory.iterdir()):raise ValueError('Output directory is not empty; human work will not be overwritten')
            directory.mkdir(parents=True,exist_ok=True)
            cqa.write_json(directory/'manifest.json',package['manifest'])
            for split in ('development','test'):
                cqa.write_json(directory/(split+'.blind.json'),package[split]);to_csv(package[split],directory/(split+'.labels.csv'))
            print('Blind packages written. Human labels are empty. No API calls.');return
        if args.command=='agreement':result=agreement(cqa.read_json(args.left),cqa.read_json(args.right),records,checklist)
        else:result=validate_annotations(cqa.read_json(args.annotations),records,checklist)
    if args.out:cqa.write_json(args.out,result)
    print(json.dumps({'records':len(result.get('records',[])),'state':result.get('annotation_state'),'exact_agreement':result.get('exact_agreement')},ensure_ascii=False))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,TypeError,OSError) as exc:raise SystemExit(str(exc)) from None
