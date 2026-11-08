"""Server-only OpenAI Responses adapter. One request, no implicit retries."""
import json
import socket
import time
import urllib.error
import urllib.request


class ProviderError(ValueError):
    def __init__(self, code, message, metadata=None):
        super().__init__(message)
        self.code=code
        self.metadata=metadata or {'calls':1,'usage':{}}


def object_schema(properties):
    return dict(type='object',additionalProperties=False,properties=properties,required=list(properties))


def make_payload(checklist, record, turns, knowledge, model, system_prompt):
    ids=[m['message_id'] for m in turns]
    message_id={'type':'string','enum':ids} if ids else {'type':'string'}
    evidence=object_schema({'message_id':message_id,'quote':{'type':'string'}})
    span=object_schema({'from':message_id,'to':message_id})
    source_evidence=object_schema({'source_id':{'type':'string'},'quote':{'type':'string'}})
    variants=[]
    for c in checklist['criteria']:
        score={'type':['number','null']}
        steps=c.get('allowed_scores')
        if steps is None and c['type'] in ('binary','flag'): steps=[c['min_score'],c['max_score']]
        if steps is not None: score['enum']=[*steps,None]
        variants.append(object_schema({
            'criterion_id':{'type':'string','enum':[c['id']]},
            'status':{'type':'string','enum':['scored','needs_review']+(['na'] if c['allow_na'] else [])},
            'score':score,'reason':{'type':'string'},
            'evidence':{'type':'array','items':evidence},
            'checked_span':{'anyOf':[{'type':'null'},span]},
            'na_reason':{'type':['string','null']},'critical_error':{'type':['boolean','null']},
            'factual_status':{'type':'string','enum':['not_assessed','not_needed','not_verified','supported','contradicted']},
            'knowledge_evidence':{'type':'array','items':source_evidence},
            'verification_limits':{'type':'array','items':{'type':'string'}}}))
    schema=object_schema({'findings':{'type':'array','items':{'anyOf':variants}}})
    content=dict(language=record['language'],channel=record['channel'],criteria=checklist['criteria'],
                 checklist_policy=checklist.get('evaluation_policy',{}),transcript=turns,knowledge=knowledge or {'sources':[]})
    return dict(model=model,instructions=system_prompt,input=[{'role':'user','content':json.dumps(content,ensure_ascii=False)}],
                reasoning={'effort':'medium'},max_output_tokens=16000,store=False,
                text={'format':dict(type='json_schema',name='cqa_findings',strict=True,schema=schema)})


def normalized_metadata(value):
    usage=value.get('usage') or {}
    if not isinstance(usage,dict):usage={}
    clean={}
    for source,target in [('input_tokens','prompt_tokens'),('output_tokens','completion_tokens'),('total_tokens','total_tokens')]:
        n=usage.get(source)
        if type(n) is int and n>=0: clean[target]=n
    for source,key,target in [('input_tokens_details','cached_tokens','cached_input_tokens'),('output_tokens_details','reasoning_tokens','reasoning_tokens')]:
        detail=usage.get(source) or {}
        n=detail.get(key) if isinstance(detail,dict) else None
        if type(n) is int and n>=0: clean[target]=n
    details=usage.get('input_tokens_details')
    if isinstance(details,dict):
        written=details.get('cache_write_tokens')
        if type(written) is int and written>=0:clean['cache_write_input_tokens']=written
        breakdown=details.get('cache_write_tokens_details')
        if isinstance(breakdown,dict):
            for source,target in [('ttl_30m_tokens','cache_write_30m_input_tokens'),('ttl_12h_tokens','cache_write_12h_input_tokens')]:
                n=breakdown.get(source)
                if type(n) is int and n>=0:clean[target]=n
    return dict(model=value.get('model'),usage=clean,calls=1,response_id=value.get('id'))


def parse_response(value):
    if not isinstance(value,dict): raise ProviderError('invalid_model_response','Модель вернула неверную структуру ответа.')
    meta=normalized_metadata(value)
    if value.get('status')=='incomplete':
        raise ProviderError('incomplete_response','Модель не завершила ответ. Запустите оценку повторно вручную.',meta)
    if value.get('status')!='completed':
        raise ProviderError('invalid_model_response','Модель не вернула завершённую оценку.',meta)
    texts=[]
    try:
        for item in value.get('output',[]):
            if item.get('type')!='message': continue
            for block in item.get('content',[]):
                if block.get('type')=='refusal':
                    raise ProviderError('model_refusal','Модель отказалась оценивать диалог. Нужна ручная проверка.',meta)
                if block.get('type')=='output_text': texts.append(block['text'])
        parsed=json.loads(''.join(texts))
        if not isinstance(parsed,dict) or not isinstance(parsed.get('findings'),list): raise ValueError()
        if any(not isinstance(f,dict) for f in parsed['findings']): raise ValueError()
    except (KeyError,ValueError,TypeError,AttributeError) as exc:
        if isinstance(exc,ProviderError): raise
        raise ProviderError('invalid_model_response','Модель вернула некорректный JSON оценки.',meta) from None
    return parsed['findings'],meta


class ResponsesProvider:
    def __init__(self,api_key,model='gpt-6.1-sol',base_url='https://api.openai.com/v1',timeout=120):
        self.api_key=api_key;self.model=model;self.base_url=base_url;self.timeout=timeout

    def evaluate(self,checklist,record,turns,knowledge,system_prompt):
        if not self.api_key:
            raise ProviderError('not_configured','Не задан CQA_API_KEY на сервере.',{'calls':0,'usage':{}})
        payload=make_payload(checklist,record,turns,knowledge,self.model,system_prompt)
        request=urllib.request.Request(self.base_url.rstrip('/')+'/responses',data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),
            headers={'Authorization':'Bearer '+self.api_key,'Content-Type':'application/json','User-Agent':'Freedom-CQA-Backend/1.0'},method='POST')
        start=time.monotonic()
        def meta():return dict(model=self.model,calls=1,usage={},latency_seconds=round(time.monotonic()-start,3))
        try:
            with urllib.request.urlopen(request,timeout=self.timeout) as response:
                data=response.read(8_000_001)
            if len(data)>8_000_000: raise ProviderError('invalid_model_response','Ответ модели превышает допустимый размер.',meta())
            try: value=json.loads(data)
            except (ValueError,UnicodeError): raise ProviderError('invalid_model_response','API модели вернул некорректный JSON.',meta()) from None
            findings,metadata=parse_response(value)
            expected={c['id'] for c in checklist['criteria']}
            returned=[f.get('criterion_id') for f in findings]
            if len(returned)!=len(expected) or any(not isinstance(i,str) for i in returned) or set(returned)!=expected:
                raise ProviderError('invalid_model_response','Модель вернула пропущенные, повторные или неизвестные критерии.',metadata)
            by_id={f['criterion_id']:f for f in findings}
            metadata['latency_seconds']=round(time.monotonic()-start,3)
            return [by_id[c['id']] for c in checklist['criteria']],metadata
        except urllib.error.HTTPError as exc:
            status=exc.code;exc.close()
            code={401:'provider_auth',403:'provider_auth',429:'provider_rate_limit'}.get(status,'provider_unavailable' if status>=500 else 'provider_request')
            message={401:'API-ключ не принят OpenAI.',403:'У ключа нет доступа к модели или проекту.',429:'Достигнут лимит API. Проверьте баланс и ограничения проекта.'}.get(status,f'API модели вернул HTTP {status}. Проверьте серверные настройки.')
            raise ProviderError(code,message,meta()) from None
        except (TimeoutError,socket.timeout):
            raise ProviderError('provider_timeout','Истёк тайм-аут модели. Автоматический повтор не выполнялся; стоимость запроса может быть неизвестна.',meta()) from None
        except urllib.error.URLError as exc:
            code='provider_timeout' if isinstance(exc.reason,(TimeoutError,socket.timeout)) else 'provider_unavailable'
            raise ProviderError(code,'API модели недоступен: сеть, TLS или тайм-аут. Автоматического повтора нет.',meta()) from None
        except OSError:
            raise ProviderError('provider_unavailable','Не удалось получить ответ API модели. Автоматического повтора нет.',meta()) from None
        except ProviderError as exc:
            exc.metadata.setdefault('latency_seconds',round(time.monotonic()-start,3))
            exc.metadata.setdefault('model',self.model)
            raise
