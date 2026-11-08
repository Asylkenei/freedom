"""Explicit bounded experiments; default is a dry plan with zero model calls."""
import argparse
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import annotation
import cqa


def make_plan(manifest,records,pipeline_hash,mode,split,limit,max_calls,frozen=None):
    if manifest.get('manifest_hash')!=cqa.digest({k:v for k,v in manifest.items() if k!='manifest_hash'}):raise ValueError('Manifest changed without a new hash')
    if manifest['source_dataset_hash']!=annotation.dataset_hash(records):raise ValueError('Dataset changed')
    if mode not in ('single','cascade') or split not in ('development','test'):raise ValueError('Unknown pipeline/split')
    if type(limit) is not int or limit<=0 or type(max_calls) is not int or max_calls<=0:raise ValueError('Positive explicit limits required')
    source={r['conversation_id']:r for r in records};entries=manifest['records']
    if len(source)!=len(records) or len({e['conversation_id'] for e in entries})!=len(entries) or set(source)!={e['conversation_id'] for e in entries}:raise ValueError('Manifest record IDs differ')
    groups={}
    for entry in entries:
        r=source[entry['conversation_id']]
        if entry['record_hash']!=cqa.digest(r) or entry['transcript_hash']!=cqa.digest(r['transcript']):raise ValueError('Record hash differs')
        if entry['split'] not in ('development','test'):raise ValueError('Unknown split')
        group=entry['group_id']
        if group in groups and groups[group]!=entry['split']:raise ValueError('Duplicate group crosses the split')
        groups[group]=entry['split']
    if split=='test':
        expected=dict(pipeline_hash=pipeline_hash,manifest_hash=manifest['manifest_hash'],checklist_hash=manifest['checklist_hash'])
        if not frozen or any(frozen.get(k)!=v for k,v in expected.items()):raise ValueError('Holdout requires matching frozen pipeline, manifest and checklist')
        if manifest.get('near_duplicate_review') not in ('reviewed','no_candidates'):raise ValueError('Review near duplicates before using the holdout')
    selected=sorted((e['conversation_id'] for e in entries if e['split']==split),key=lambda cid:cqa.digest([manifest['seed'],cid]))
    if limit>len(selected):raise ValueError('Requested limit exceeds split size')
    calls_per_case=2 if mode=='cascade' else 1
    if limit*calls_per_case>max_calls:raise ValueError('Paid-call cap is smaller than the worst-case plan')
    return dict(schema_version=1,execute=False,split=split,conversation_ids=selected[:limit],pipeline=mode,pipeline_hash=pipeline_hash,
                manifest_hash=manifest['manifest_hash'],checklist_hash=manifest['checklist_hash'],source_dataset_hash=manifest['source_dataset_hash'],
                max_calls=max_calls,estimated_max_calls=limit*calls_per_case,
                warning='Human labels are never sent. Call limits are hard bounds; dollar cost requires per-model prices and actual usage.')


def api(server,path,payload=None):
    request=urllib.request.Request(server.rstrip('/')+'/api'+path,
        data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'} if payload is not None else {})
    try:
        with urllib.request.urlopen(request,timeout=30) as response:return json.load(response),response.status
    except urllib.error.HTTPError as exc:
        code=exc.code;exc.close();raise ValueError('Backend request failed: HTTP '+str(code)) from None


def execute_plan(plan,records,server,directory,force=False,client=api):
    if annotation.dataset_hash(records)!=plan['source_dataset_hash']:raise ValueError('Dataset changed after planning')
    ids=plan['conversation_ids']
    if len(set(ids))!=len(ids) or not set(ids)<={r['conversation_id'] for r in records}:raise ValueError('Invalid planned IDs')
    if len(ids)*(2 if plan['pipeline']=='cascade' else 1)>plan['max_calls']:raise ValueError('Plan exceeds call cap')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'report.json').exists() or (directory/'predictions.json').exists():raise ValueError('Use a new output directory; previous experiment is immutable')
    current,_=client(server,'/health')
    if current.get('pipeline_hash')!=plan['pipeline_hash']:raise ValueError('Server pipeline differs from plan')
    source={r['conversation_id']:r for r in records}
    client(server,'/conversations/import',{'records':[source[cid] for cid in plan['conversation_ids']]})
    report=dict(plan=plan,cases=[],observed_calls=0,complete=False,stopped_on_error=False)
    predictions=[]
    def save():
        cqa.write_json(directory/'report.json',report);cqa.write_json(directory/'predictions.json',predictions)
    save()
    for cid in plan['conversation_ids']:
        reserve=2 if plan['pipeline']=='cascade' else 1
        if report['observed_calls']+reserve>plan['max_calls']:raise ValueError('Call budget exhausted; no next request submitted')
        value,status=client(server,'/evaluations',dict(conversation_id=cid,checklist_hash=plan['checklist_hash'],pipeline_hash=plan['pipeline_hash'],force=force))
        cached=status==200
        deadline=time.monotonic()+660
        while value['status'] in ('queued','running'):
            if time.monotonic()>deadline:raise ValueError('Job may still run; inspect it before explicitly retrying')
            time.sleep(1);value,_=client(server,'/evaluations/'+value['evaluation_id'])
        calls=0 if cached else value.get('inference',{}).get('calls',0)
        report['observed_calls']+=calls
        if report['observed_calls']>plan['max_calls']:raise ValueError('Provider exceeded planned stage count')
        value['experiment_cache_hit']=cached;predictions.append(value)
        report['cases'].append(dict(conversation_id=cid,evaluation_id=value['evaluation_id'],status=value['status'],cache_hit=cached,new_calls=calls,inference=value.get('inference',{})))
        cqa.write_json(directory/(value['evaluation_id']+'.json'),value);save()
        print(json.dumps(report['cases'][-1],ensure_ascii=False),flush=True)
        if value['status']=='error':report['stopped_on_error']=True;save();return report
    report['complete']=True;save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--split',choices=['development','test'],default='development')
    parser.add_argument('--limit',type=int,default=4);parser.add_argument('--max-calls',type=int,required=True)
    parser.add_argument('--server',default='http://127.0.0.1:8000');parser.add_argument('--out',required=True)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--force',action='store_true')
    parser.add_argument('--frozen');parser.add_argument('--freeze-out')
    args=parser.parse_args()
    manifest=cqa.read_json(args.manifest);records=cqa.load_records(cqa.read_json(args.data))
    health,_=api(args.server,'/health')
    plan=make_plan(manifest,records,health['pipeline_hash'],health['pipeline'],args.split,args.limit,args.max_calls,cqa.read_json(args.frozen) if args.frozen else None)
    directory=Path(args.out);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'report.json').exists():raise ValueError('Use a new experiment directory')
    if args.freeze_out:cqa.write_json(args.freeze_out,{k:plan[k] for k in ('pipeline_hash','manifest_hash','checklist_hash')})
    cqa.write_json(directory/'plan.json',plan)
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return
    plan['execute']=True
    report=execute_plan(plan,records,args.server,directory,args.force)
    if not report['complete']:raise SystemExit(1)


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,TypeError) as exc:raise SystemExit(str(exc)) from None
