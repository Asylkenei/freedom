"""Explicit opt-in live smoke: at most four selected conversations, never the whole dataset.

Requires the backend already running with a real server-side key. No key is sent by this client.
Raw results/reports are private local data and must not be published.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from datetime import datetime,timezone

from cqa import load_records


def select_cases(records):
    selected=[]
    for label,language,channel in [('RU_CHAT','RU','agent chat'),('KZ_CHAT','KZ','agent chat'),('EN_CHAT','EN','agent chat'),('RU_CALL','RU','voice call')]:
        item=next((r for r in records if r['language']==language and r['channel']==channel),None)
        if item is None:raise ValueError(f'Missing case: {label}')
        selected.append((label,item))
    return selected


def api(server,path,payload=None):
    data=json.dumps(payload,ensure_ascii=False).encode() if payload is not None else None
    request=urllib.request.Request(server.rstrip('/')+'/api'+path,data=data,
        headers={'Content-Type':'application/json'} if data else {})
    try:
        with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
    except urllib.error.HTTPError as exc:
        try:value=json.load(exc)
        finally:exc.close()
        raise ValueError(value.get('error',{}).get('message','Backend request failed')) from None


def validate_saved_evidence(run):
    original={m['message_id']:m['text'] for m in run['messages']}
    rows=run['final_findings']
    expected={c['id'] for c in run['checklist_snapshot']['criteria']}
    if len(rows)!=len(expected) or {f['criterion_id'] for f in rows}!=expected:raise ValueError('Incomplete saved criteria')
    for f in rows:
        for e in f['evidence']:
            if e['message_id'] not in original or not e['quote'] or e['quote'] not in original[e['message_id']]:
                raise ValueError('Saved evidence does not match source')
    return sum(len(f['evidence']) for f in rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,help='Local organizer JSON, not published')
    p.add_argument('--server',default='http://127.0.0.1:8000')
    p.add_argument('--out',default='data/live-smoke-summary.json')
    p.add_argument('--retry-errors',action='store_true',help='Manually retry a previously saved error once. Does not retry a newly failed request.')
    args=p.parse_args()
    cases=select_cases(load_records(json.loads(Path(args.data).read_text(encoding='utf-8-sig'))))
    health=api(args.server,'/health')
    if not health['llm_configured']:raise SystemExit('Server CQA_API_KEY is not configured. No paid calls made.')
    checklist=api(args.server,'/checklists')['items'][0]
    imported=api(args.server,'/conversations/import',{'records':[r for label,r in cases]})
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    raw_dir=output.parent/'live-smoke';raw_dir.mkdir(exist_ok=True)
    report=dict(started_at=datetime.now(timezone.utc).isoformat(),requested_model=health['model'],
                checklist_hash=checklist['hash'],imported=imported,cases=[],complete=False,
                warning='Live connectivity/evidence smoke only; not independent gold accuracy. Raw transcripts/results are private.')
    def save():output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    for label,record in cases:
        print('Starting '+label,flush=True)
        value=api(args.server,'/evaluations',{'conversation_id':record['conversation_id'],'checklist_hash':checklist['hash']})
        if value['status']=='error' and args.retry_errors:
            value=api(args.server,'/evaluations',{'conversation_id':record['conversation_id'],'checklist_hash':checklist['hash'],'force':True})
        deadline=time.monotonic()+660
        while value['status'] in ('queued','running'):
            if time.monotonic()>deadline:raise ValueError('Backend job deadline exceeded; job may still be running, do not blindly retry')
            time.sleep(1)
            value=api(args.server,'/evaluations/'+value['evaluation_id'])
        (raw_dir/(label+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        case=dict(case=label,evaluation_id=value['evaluation_id'],status=value['status'],inference=value['inference'],error=value['error'])
        if value['result']:
            case.update(score=value['result']['final_summary']['score'],evidence_quotes_checked=validate_saved_evidence(value['result']),
                        needs_review=[f['criterion_id'] for f in value['result']['final_findings'] if f['status']=='needs_review'])
        report['cases'].append(case);save()
        print(json.dumps(case,ensure_ascii=False),flush=True)
        if value['status']=='error':
            print('Stopping after provider/backend error; remaining cases will not incur new requests.',flush=True)
            return 1
    report['complete']=len(report['cases'])==4 and all(c['status'] in ('completed','needs_review') for c in report['cases'])
    report['finished_at']=datetime.now(timezone.utc).isoformat();save()
    print('Four-case smoke finished; private report: '+str(output),flush=True)
    return 0 if report['complete'] else 1


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as exc:raise SystemExit(str(exc)) from None
