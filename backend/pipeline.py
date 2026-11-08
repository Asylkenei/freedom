"""Bounded, independently judged model stages. Human decisions never enter model inputs."""
import copy
import cqa
from provider import ProviderError
from costs import estimate

RULES_VERSION='cascade-v2-factual-guard'
REVIEW_PROMPT='\nIndependently evaluate the supplied transcript against every current criterion. Prior model scores are intentionally unavailable. Do not invent missing product facts.'


def profile(settings):
    return dict(mode=settings.pipeline,primary_model=settings.small_model if settings.pipeline=='cascade' else settings.model,
                review_model=settings.review_model or settings.model,review_all=settings.review_all,
                audit_rate=settings.audit_rate,audit_seed=settings.audit_seed,
                sensitive_criteria=list(settings.sensitive_criteria),routing_version=RULES_VERSION,
                primary_prompt=cqa.SYSTEM_PROMPT,review_prompt=cqa.SYSTEM_PROMPT+REVIEW_PROMPT,
                adapter='responses-v1-medium-16000',base_url=settings.base_url,
                reasoning='medium',store=False,max_output_tokens=16000)


def validate_stage(checklist,record,turns,knowledge,raw):
    expected={c['id'] for c in checklist['criteria']}
    if (not isinstance(raw,list) or len(raw)!=len(expected) or
        any(not isinstance(f,dict) or not isinstance(f.get('criterion_id'),str) for f in raw) or
        {f['criterion_id'] for f in raw}!=expected):
        raise ProviderError('invalid_model_response','Неверный набор критериев в ответе модели.')
    by_id={f['criterion_id']:f for f in raw};findings=[]
    for c in checklist['criteria']:
        source=cqa.unknown(c,'Audio has not been evaluated') if c.get('requires_audio') else by_id[c['id']]
        f=cqa.validate_finding(c,source,turns,knowledge)
        if f['status']=='scored' and f.get('factual_status')=='not_verified':
            f.update(status='needs_review',score=None,answer=None,critical_error=None,
                     reason='Product fact explicitly not verified; human review required. '+f['reason'])
        if record['channel']=='voice call' and c.get('optional_audio_features'):
            f.setdefault('verification_limits',[]).append('Аудио не анализировалось. Не проверены: '+', '.join(c['optional_audio_features']))
        findings.append(f)
    return findings


def inference(stages,p):
    completed=[s for s in stages if s['status'] in ('completed','error')]
    for s in completed:s['cost_estimate']=estimate(s.get('metadata',{}).get('model') or s['requested_model'],s.get('metadata',{}).get('usage'))
    known=[s['cost_estimate']['estimated_usd'] for s in completed if s['cost_estimate']['estimated_usd'] is not None]
    fields={k for s in completed for k in s.get('metadata',{}).get('usage',{})}
    usage={k:sum(s.get('metadata',{}).get('usage',{}).get(k,0) for s in completed) for k in fields}
    return dict(requested_model=p['primary_model'],model=completed[-1].get('metadata',{}).get('model') if completed else None,
                calls=sum(s.get('metadata',{}).get('calls',0) for s in completed),usage=usage,
                estimated_usd=round(sum(known),8) if completed and len(known)==len(completed) else None,
                known_estimated_usd=round(sum(known),8) if known else None,
                cost_estimate_complete=bool(completed) and len(known)==len(completed),
                cost_estimate_note='Dated token estimate, not invoice; see each stage assumptions.',
                usage_complete=bool(completed) and all(s['status']=='completed' and
                    all(k in s.get('metadata',{}).get('usage',{}) for k in ('prompt_tokens','completion_tokens')) for s in completed))


def routing(checklist,record,knowledge,raw,findings,p):
    by_raw={f['criterion_id']:f for f in raw};blocked=[];reasons=[]
    for c,f in zip(checklist['criteria'],findings):
        if (f['status']=='needs_review' and by_raw[c['id']].get('factual_status')=='not_verified'
            and not (knowledge or {}).get('sources')):
            blocked.append(c['id']);continue
        if f['status']=='needs_review':reasons.append(dict(code='unresolved_criterion',criterion_id=c['id']))
        if c['id'] in p['sensitive_criteria'] and f['status']=='scored':
            reasons.append(dict(code='sensitive_criterion',criterion_id=c['id']))
        if c['type']=='flag' and f['status']=='scored' and f['score']==c.get('flag_score',c['max_score']):
            reasons.append(dict(code='flag_detected',criterion_id=c['id']))
    sample=int(cqa.digest(dict(record=record,checklist=cqa.digest(checklist),seed=p['audit_seed']))[:16],16)/2**64
    if p['review_all']:reasons.append(dict(code='review_all'))
    elif sample<p['audit_rate']:reasons.append(dict(code='deterministic_audit'))
    return dict(required=bool(reasons),reasons=reasons,blocked_by_missing_sources=blocked,version=p['routing_version'])


def execute(checklist,record,turns,knowledge,p,primary,reviewer,update):
    stages=[]
    def call(name,provider,prompt,model):
        stage=dict(name=name,status='running',requested_model=model,metadata={'calls':0,'usage':{}})
        stages.append(stage);update(name,stages,inference(stages,p))
        try:
            raw,metadata=provider.evaluate(checklist,record,turns,knowledge,prompt)
            stage['metadata']=metadata
            findings=validate_stage(checklist,record,turns,knowledge,raw)
            stage.update(status='completed',raw_findings=copy.deepcopy(raw),findings=findings)
            update(name,stages,inference(stages,p))
            return raw,findings
        except ProviderError as exc:
            stage.update(status='error',error={'code':exc.code,'message':str(exc)})
            stage['metadata'].update({k:v for k,v in exc.metadata.items() if k!='usage' or v or not stage['metadata'].get('usage')})
            update(name,stages,inference(stages,p))
            raise ProviderError(exc.code,str(exc),inference(stages,p)) from None
        except Exception:
            stage.update(status='error',error={'code':'internal_error','message':'Не удалось завершить этап оценки.'})
            stage['metadata']['calls']=1
            update(name,stages,inference(stages,p))
            raise ProviderError('internal_error','Не удалось завершить этап оценки.',inference(stages,p)) from None
    raw,findings=call('primary',primary,p['primary_prompt'],p['primary_model'])
    route=dict(required=False,reasons=[],blocked_by_missing_sources=[],version=p['routing_version'])
    disagreements=[]
    if p['mode']=='cascade':
        route=routing(checklist,record,knowledge,raw,findings,p)
        update('routing',stages,inference(stages,p),route)
        if route['required']:
            _,reviewed=call('review',reviewer,p['review_prompt'],p['review_model'])
            final=[]
            for c,a,b in zip(checklist['criteria'],findings,reviewed):
                different=(a['status'],a.get('score'))!=(b['status'],b.get('score'))
                if different:disagreements.append(c['id'])
                if c['id'] in route['blocked_by_missing_sources']:
                    final.append(a)
                elif different and (c['type']=='flag' or c['id'] in p['sensitive_criteria']) and a['status'] in ('scored','na') and b['status'] in ('scored','na'):
                    final.append(cqa.unknown(c,'Models disagree on a sensitive criterion; human review required.'))
                else:final.append(b)
            findings=final
    return findings,inference(stages,p),stages,route,disagreements
