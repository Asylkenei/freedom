"""Persistent single-worker evaluation service; the baseline remains the scoring authority."""
import copy
import json
import queue
import threading
import time
import uuid
from datetime import datetime,timezone
from urllib.parse import urlsplit

import cqa
import pipeline
from database import Database,encode
from provider import ProviderError,ResponsesProvider


def now():return datetime.now(timezone.utc).isoformat()


class ServiceError(ValueError):
    def __init__(self,status_code,code,message):
        super().__init__(message);self.status_code=status_code;self.code=code


class Service:
    def __init__(self,settings,provider=None):
        self.settings=settings;self.settings.data_dir.mkdir(parents=True,exist_ok=True)
        self.db=Database(settings.data_dir/'cqa.sqlite3');self.lock=threading.RLock()
        checklist=cqa.validate_checklist(cqa.read_json(settings.checklist_file))
        with self.db.connect() as db:
            db.execute('INSERT OR IGNORE INTO checklists(hash,created_at,payload) VALUES (?,?,?)',
                       (cqa.digest(checklist),now(),encode(checklist)))
        primary_model=settings.small_model if settings.pipeline=='cascade' else settings.model
        self.provider=provider or ResponsesProvider(settings.api_key,primary_model,settings.base_url,settings.timeout_seconds)
        self.reviewer=ResponsesProvider(settings.api_key,settings.review_model or settings.model,settings.base_url,settings.timeout_seconds)
        self.queue=queue.Queue();self.stopping=threading.Event();self._recover()
        self.worker=threading.Thread(target=self._work,name='cqa-single-worker',daemon=True);self.worker.start()

    def close(self):
        self.stopping.set();self.queue.put(None)
        if self.worker.is_alive():self.worker.join(timeout=2)

    def health(self):
        return dict(status='ok',llm_configured=bool(self.settings.api_key),model=self.settings.model,
                    storage='sqlite',worker='running' if self.worker.is_alive() else 'stopped',api_version='2.0.0',
                    pipeline=self.settings.pipeline,small_model=self.settings.small_model or None,
                    review_model=self.settings.review_model or self.settings.model,pipeline_hash=cqa.digest(pipeline.profile(self.settings)))

    def _required(self,table,column,value):
        found=self.db.get(table,column,value)
        if found is None:raise ServiceError(404,'not_found','Запись не найдена.')
        return found

    def list_checklists(self):
        with self.db.connect() as db:
            items=[dict(hash=row['hash'],**json.loads(row['payload'])) for row in db.execute('SELECT hash,payload FROM checklists ORDER BY rowid DESC')]
        return {'items':items}

    def get_checklist(self,identifier):
        return dict(hash=identifier,**self._required('checklists','hash',identifier))

    def create_checklist(self,checklist,parent_hash,author):
        if parent_hash:self._required('checklists','hash',parent_hash)
        try:
            if not isinstance(checklist,dict) or not 1<=len(checklist['criteria'])<=200:
                raise ValueError()
            for c in checklist['criteria']:
                if not isinstance(c['id'],str) or not c['id'].strip() or len(c['id'])>200:raise ValueError()
                if any(type(c[k]) not in (int,float) for k in ('weight','min_score','max_score')):raise ValueError()
                if c.get('section') is not None and (not isinstance(c['section'],str) or not c['section'].strip()):raise ValueError()
                if any(len(c[k])>100000 for k in ('title','instruction')):raise ValueError()
                if any(not isinstance(k,str) or not isinstance(v,str) for k,v in c.get('channel_instructions',{}).items()):raise ValueError()
                if c['type'] in ('binary','flag') and 'allowed_scores' in c and set(c['allowed_scores'])!={c['min_score'],c['max_score']}:raise ValueError()
            if checklist.get('rounding_mode','HALF_UP')!='HALF_UP':raise ValueError()
            cqa.validate_checklist(checklist)
            if len(encode(checklist))>1_000_000:raise ValueError()
        except (KeyError,ValueError,TypeError,ArithmeticError,AttributeError):
            raise ServiceError(422,'invalid_checklist','Проверьте ID, формулировки, веса, шкалы и правила N/A.') from None
        identifier=cqa.digest(checklist)
        with self.lock,self.db.connect() as db:
            exists=db.execute('SELECT 1 FROM checklists WHERE hash=?',(identifier,)).fetchone()
            if not exists:
                db.execute('INSERT INTO checklists VALUES (?,?,?)',(identifier,now(),encode(checklist)))
                db.execute('INSERT INTO checklist_versions VALUES (?,?,?)',(identifier,parent_hash,author))
        return dict(hash=identifier,**checklist),200 if exists else 201

    def list_appeals(self,evaluation_id):
        value=self.get_evaluation(evaluation_id)
        return dict(items=value['result']['appeals'] if value['result'] else [],revision=value.get('revision',0))

    def create_appeal(self,evaluation_id,criterion_id,comment,author,client_request_id):
        fingerprint=cqa.digest(dict(criterion_id=criterion_id,comment=comment,author=author))
        with self.lock,self.db.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT * FROM appeals WHERE evaluation_id=? AND request_key=?',(evaluation_id,client_request_id)).fetchone()
            if old:
                if old['request_hash']!=fingerprint:raise ServiceError(409,'idempotency_conflict','ID запроса уже использован с другим содержимым.')
                return json.loads(old['payload']),200
            row=db.execute('SELECT payload FROM evaluations WHERE id=?',(evaluation_id,)).fetchone()
            if row is None:raise ServiceError(404,'not_found','Оценка не найдена.')
            value=json.loads(row['payload'])
            if value['status'] not in ('completed','needs_review') or not value['result']:
                raise ServiceError(409,'evaluation_not_ready','Для апелляции нужна завершённая содержательная оценка.')
            run=value['result']
            try:appeal_id=cqa.add_appeal(run,criterion_id,comment)
            except ValueError:raise ServiceError(409,'invalid_appeal','Проверьте критерий и отсутствие открытой апелляции.') from None
            appeal=next(a for a in run['appeals'] if a['appeal_id']==appeal_id)
            value.update(revision=value.get('revision',0)+1,review_status='open',updated_at=now())
            appeal.update(evaluation_id=evaluation_id,author=author,client_request_id=client_request_id,revision=value['revision'])
            db.execute('INSERT INTO appeals VALUES (?,?,?,?,?)',(appeal_id,evaluation_id,client_request_id,fingerprint,encode(appeal)))
            db.execute('UPDATE evaluations SET payload=? WHERE id=?',(encode(value),evaluation_id))
            return copy.deepcopy(appeal),201

    def resolve_appeal(self,appeal_id,decision,reason,reviewer,replacement,expected_revision,client_request_id):
        payload=dict(decision=decision,reason=reason,reviewer=reviewer,replacement=replacement,expected_revision=expected_revision)
        fingerprint=cqa.digest(payload)
        with self.lock,self.db.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM appeals WHERE id=?',(appeal_id,)).fetchone()
            if row is None:raise ServiceError(404,'not_found','Апелляция не найдена.')
            old=db.execute('SELECT * FROM appeal_decisions WHERE appeal_id=?',(appeal_id,)).fetchone()
            if old:
                if old['request_key']==client_request_id and old['request_hash']==fingerprint:return json.loads(old['payload'])
                raise ServiceError(409,'already_resolved','Решение уже сохранено; повтор с другим содержимым отклонён.')
            value=json.loads(db.execute('SELECT payload FROM evaluations WHERE id=?',(row['evaluation_id'],)).fetchone()['payload'])
            if value.get('revision',0)!=expected_revision:raise ServiceError(409,'stale_revision','Оценка изменена. Получите текущую ревизию.')
            if (decision=='change')!=(replacement is not None):raise ServiceError(422,'invalid_decision','Для change нужен replacement; для uphold он должен быть null.')
            try:cqa.resolve_appeal(value['result'],appeal_id,reason,reviewer,replacement)
            except (ValueError,KeyError,TypeError,StopIteration):
                raise ServiceError(422,'invalid_decision','Решение требует допустимого ответа и подтверждённых доказательств.') from None
            value.update(revision=expected_revision+1,updated_at=now(),
                         status='completed' if value['result']['final_summary']['assessment_complete'] else 'needs_review',
                         review_status='open' if any(a['status']=='open' for a in value['result']['appeals']) else 'resolved')
            appeal=next(a for a in value['result']['appeals'] if a['appeal_id']==appeal_id)
            appeal['resolved_revision']=value['revision']
            db.execute('UPDATE appeals SET payload=? WHERE id=?',(encode(appeal),appeal_id))
            db.execute('UPDATE evaluations SET status=?,payload=? WHERE id=?',(value['status'],encode(value),value['evaluation_id']))
            db.execute('INSERT INTO appeal_decisions VALUES (?,?,?,?)',(appeal_id,client_request_id,fingerprint,encode(value)))
            return value

    def import_records(self,payload):
        try:
            rows=cqa.load_records(copy.deepcopy(payload))
            if not 1<=len(rows)<=2000:raise ValueError('Нужно от 1 до 2000 диалогов.')
            normalized=[]
            for r in rows:
                if len(r['conversation_id'])>200 or not r['conversation_id'].strip() or len(r['transcript'])>10000:
                    raise ValueError('Некорректный ID или слишком много реплик.')
                audio=r.get('audio_url') or None
                if audio:
                    url=urlsplit(audio)
                    if url.scheme not in ('http','https') or not url.hostname or url.username or url.password:raise ValueError('Некорректная ссылка на аудио.')
                turns=[]
                for m in r['transcript']:
                    if not m['text'].strip() or len(m['text'])>100000:raise ValueError('Пустая или слишком длинная реплика.')
                    turns.append({k:m[k] for k in ('speaker','timestamp','text')})
                normalized.append(dict(conversation_id=r['conversation_id'],language=r['language'],channel=r['channel'],audio_url=audio,transcript=turns))
        except (KeyError,TypeError,ValueError,OverflowError):
            raise ServiceError(422,'invalid_import','Некорректный пакет: проверьте records, уникальные ID, RU/KZ/EN, канал, роли и исходные временные метки.') from None
        added=skipped=0
        with self.lock,self.db.connect() as db:
            for r in normalized:
                content_hash=cqa.digest(r)
                old=db.execute('SELECT content_hash FROM conversations WHERE id=?',(r['conversation_id'],)).fetchone()
                if old:
                    if old['content_hash']!=content_hash:
                        raise ServiceError(409,'conversation_conflict','Диалог с этим ID уже существует с другим содержимым. Используйте новый ID; прежние доказательства сохраняются.')
                    skipped+=1
                else:
                    db.execute('INSERT INTO conversations VALUES (?,?,?,?)',(r['conversation_id'],content_hash,now(),encode(r)));added+=1
        warnings=[]
        data=payload.get('data',payload) if isinstance(payload,dict) else {}
        declared=data.get('total_records') if isinstance(data,dict) else None
        if declared is not None and declared!=len(rows):warnings.append(f'Заявлено {declared}, получено {len(rows)} диалогов.')
        return dict(imported=added,skipped=skipped,total=len(rows),warnings=warnings)

    def list_conversations(self):
        with self.db.connect() as db:
            latest={}
            for row in db.execute('SELECT conversation_id,payload FROM evaluations ORDER BY rowid DESC'):
                if row['conversation_id'] not in latest:latest[row['conversation_id']]=json.loads(row['payload'])
            out=[]
            for row in db.execute('SELECT payload,created_at FROM conversations ORDER BY rowid DESC'):
                r=json.loads(row['payload']);e=latest.get(r['conversation_id']);summary=e['result']['final_summary'] if e and e['result'] else None
                out.append(dict(conversation_id=r['conversation_id'],language=r['language'],channel=r['channel'],message_count=len(r['transcript']),
                                imported_at=row['created_at'],latest_evaluation_id=e['evaluation_id'] if e else None,
                                status=e['status'] if e else 'not_evaluated',score=summary['score'] if summary else None,
                                critical_error=summary['critical_error'] if summary else None))
        return {'items':out}

    def get_conversation(self,identifier):
        r=self._required('conversations','id',identifier)
        with self.db.connect() as db:
            history=[json.loads(row['payload']) for row in db.execute('SELECT payload FROM evaluations WHERE conversation_id=? ORDER BY rowid DESC',(identifier,))]
        return dict(conversation_id=r['conversation_id'],language=r['language'],channel=r['channel'],audio_url=r.get('audio_url'),
                    messages=cqa.messages(r),evaluations=[dict(evaluation_id=e['evaluation_id'],status=e['status'],created_at=e['created_at'],
                    checklist_hash=e['checklist_hash'],score=e['result']['final_summary']['score'] if e['result'] else None) for e in history])

    def get_evaluation(self,identifier):
        value=self._required('evaluations','id',identifier)
        value.setdefault('revision',0);value.setdefault('review_status','none')
        return value

    def _knowledge(self):
        if not self.settings.knowledge_file:return None
        try:return copy.deepcopy(cqa.validate_knowledge(cqa.read_json(self.settings.knowledge_file)))
        except (OSError,ValueError,TypeError,KeyError):
            raise ServiceError(503,'invalid_knowledge','Серверный файл продуктовых источников недоступен или некорректен.') from None

    def start_evaluation(self,conversation_id,checklist_hash,force=False,pipeline_hash=None):
        if self.stopping.is_set():raise ServiceError(503,'shutting_down','Сервер завершает работу.')
        record=self._required('conversations','id',conversation_id)
        checklist=self._required('checklists','hash',checklist_hash);knowledge=self._knowledge()
        pipeline_snapshot=pipeline.profile(self.settings)
        if pipeline_hash is not None and pipeline_hash!=cqa.digest(pipeline_snapshot):
            raise ServiceError(409,'pipeline_changed','Конфигурация модели изменилась; обновите план эксперимента.')
        if not pipeline_snapshot['primary_model'].strip():raise ServiceError(503,'pipeline_not_configured','Задайте CQA_SMALL_MODEL для каскада.')
        cache_key=cqa.digest(dict(record=record,checklist=checklist,knowledge=knowledge,pipeline=pipeline_snapshot))
        with self.lock,self.db.connect() as db:
            existing=db.execute('SELECT payload FROM evaluations WHERE cache_key=? AND status IN (\'queued\',\'running\') ORDER BY rowid DESC LIMIT 1',(cache_key,)).fetchone()
            if existing:return json.loads(existing['payload']),202
            if not force:
                existing=db.execute('SELECT payload FROM evaluations WHERE cache_key=? ORDER BY rowid DESC LIMIT 1',(cache_key,)).fetchone()
                if existing:return json.loads(existing['payload']),200
            if not self.settings.api_key:raise ServiceError(503,'not_configured','Не задан CQA_API_KEY на сервере. Заполните .env и перезапустите backend.')
            value=dict(evaluation_id=uuid.uuid4().hex,conversation_id=conversation_id,checklist_hash=checklist_hash,status='queued',
                       result=None,error=None,created_at=now(),updated_at=now(),
                       inference=dict(requested_model=pipeline_snapshot['primary_model'],calls=0,usage={}),
                       pipeline=self.settings.pipeline,pipeline_snapshot=pipeline_snapshot,stages=[],stage='queued',
                       routing=None,disagreements=[],revision=0,review_status='none')
            work=dict(record=record,checklist=checklist,knowledge=knowledge,pipeline_snapshot=pipeline_snapshot)
            db.execute('INSERT INTO evaluations(id,conversation_id,checklist_hash,cache_key,status,created_at,payload,input_json) VALUES (?,?,?,?,?,?,?,?)',
                       (value['evaluation_id'],conversation_id,checklist_hash,cache_key,'queued',value['created_at'],encode(value),encode(work)))
        self.queue.put(value['evaluation_id'])
        return value,202

    def _recover(self):
        with self.db.connect() as db:
            interrupted=[json.loads(row['payload']) for row in db.execute("SELECT payload FROM evaluations WHERE status IN ('queued','running')")]
        for value in interrupted:
            value.update(status='error',result=None,updated_at=now(),error=dict(code='interrupted',message='Обработка прервана перезапуском сервера. Повторите вручную с force=true.'))
            self.db.save_evaluation(value)

    def _work(self):
        while not self.stopping.is_set():
            identifier=self.queue.get()
            try:
                if identifier is None:return
                self._evaluate(identifier)
            finally:self.queue.task_done()

    def _evaluate(self,identifier):
        value=self.get_evaluation(identifier)
        with self.db.connect() as db:
            row=db.execute('SELECT input_json FROM evaluations WHERE id=?',(identifier,)).fetchone()
        work=json.loads(row['input_json']);record=work['record'];checklist=work['checklist'];knowledge=work['knowledge']
        value.update(status='running',updated_at=now());self.db.save_evaluation(value)
        turns=cqa.messages(record);start=time.monotonic()
        p=work.get('pipeline_snapshot') or pipeline.profile(self.settings)
        def stage_update(name,stages,metadata,route=None):
            value.update(stage=name,stages=copy.deepcopy(stages),updated_at=now())
            value['inference'].update(metadata)
            if route is not None:value['routing']=copy.deepcopy(route)
            self.db.save_evaluation(value)
        try:
            findings,metadata,stages,route,disagreements=pipeline.execute(checklist,record,turns,knowledge,p,self.provider,self.reviewer,stage_update)
            value['inference'].update(metadata)
            value.update(stages=stages,routing=route,disagreements=disagreements,stage='aggregation')
            summary=cqa.aggregate(checklist,findings)
            run=dict(run_id=identifier,created_at=value['created_at'],conversation_id=record['conversation_id'],language=record['language'],channel=record['channel'],
                     checklist_hash=cqa.digest(checklist),transcript_hash=cqa.digest(record['transcript']),knowledge_hash=cqa.digest(knowledge) if knowledge else None,
                     checklist_snapshot=copy.deepcopy(checklist),knowledge_snapshot=copy.deepcopy(knowledge) if knowledge else {'sources':[]},
                     messages=turns,original_findings=copy.deepcopy(findings),original_summary=copy.deepcopy(summary),
                     final_findings=findings,final_summary=summary,appeals=[],inference=copy.deepcopy(value['inference']),
                     pipeline_snapshot=p,stages=copy.deepcopy(stages))
            value.update(status='completed' if summary['assessment_complete'] else 'needs_review',result=run,error=None)
            value['stage']='finished'
        except ProviderError as exc:
            value['inference'].update(exc.metadata)
            value.update(status='error',result=None,error=dict(code=exc.code,message=str(exc)))
        except Exception:
            value.update(status='error',result=None,error=dict(code='internal_error',message='Не удалось завершить оценку. Готовые результаты сохранены.'))
        value['inference']['duration_seconds']=round(time.monotonic()-start,3)
        if value['result']:value['result']['inference']=copy.deepcopy(value['inference'])
        value['updated_at']=now();self.db.save_evaluation(value)
