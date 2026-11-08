import copy
import json
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from test_support import FixtureProvider, record
from provider import ProviderError

try:
    from config import Settings
    from service import Service, ServiceError
except ImportError:
    Settings=Service=ServiceError=None


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(Service),'Persistent service is not implemented')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.settings=Settings(data_dir=Path(self.temp.name),api_key='fixture-key')
        self.provider=FixtureProvider();self.service=Service(self.settings,self.provider)
        self.addCleanup(self.service.close)
        self.checklist=self.service.list_checklists()['items'][0]
        self.hash=self.checklist['hash']

    def wait(self,evaluation_id):
        for _ in range(200):
            value=self.service.get_evaluation(evaluation_id)
            if value['status'] not in ('queued','running'):return value
            time.sleep(.01)
        self.fail('Evaluation did not finish')

    def submit(self,cid='fixture-en',force=False):
        value,_=self.service.start_evaluation(cid,self.hash,force)
        return self.wait(value['evaluation_id'])

    def test_import_idempotence_conflict_and_atomic_rollback(self):
        result=self.service.import_records({'records':[record()]})
        self.assertEqual(result['imported'],1)
        self.assertEqual(self.service.import_records({'records':[record()]})['skipped'],1)
        changed=record();changed['transcript'][0]['text']='Different'
        with self.assertRaises(ServiceError) as e:self.service.import_records({'records':[record('new'),changed]})
        self.assertEqual(e.exception.status_code,409)
        self.assertEqual(len(self.service.list_conversations()['items']),1)

    def test_invalid_language_rejects_entire_batch(self):
        with self.assertRaises(ServiceError):self.service.import_records({'records':[record(),record('invalid','FR')]})
        self.assertEqual(self.service.list_conversations()['items'],[])

    def test_completed_result_has_baseline_shape_and_usage(self):
        self.service.import_records({'records':[record()]})
        value=self.submit()
        self.assertEqual(value['status'],'completed')
        self.assertEqual(value['result']['final_summary']['score'],100)
        self.assertEqual(value['result']['original_summary'],value['result']['final_summary'])
        self.assertEqual(value['inference']['usage']['completion_tokens'],200)
        self.assertEqual(value['result']['messages'][0]['message_id'],'m0001')

    def test_completed_cache_and_force_keep_history(self):
        self.service.import_records({'records':[record()]})
        first=self.submit();cached,status=self.service.start_evaluation('fixture-en',self.hash)
        self.assertEqual(status,200);self.assertEqual(cached['evaluation_id'],first['evaluation_id'])
        second=self.submit(force=True)
        self.assertNotEqual(first['evaluation_id'],second['evaluation_id'])
        self.assertEqual(self.provider.calls,2)
        self.assertEqual(len(self.service.get_conversation('fixture-en')['evaluations']),2)

    def test_active_duplicate_including_force_does_not_double_bill(self):
        self.service.import_records({'records':[record()]});self.provider.release.clear()
        first,status=self.service.start_evaluation('fixture-en',self.hash)
        self.assertEqual(status,202);self.assertTrue(self.provider.entered.wait(2))
        for force in [False,True]:
            again,status=self.service.start_evaluation('fixture-en',self.hash,force)
            self.assertEqual(again['evaluation_id'],first['evaluation_id']);self.assertEqual(status,202)
        self.provider.release.set();self.wait(first['evaluation_id'])
        self.assertEqual(self.provider.calls,1)

    def test_one_worker_runs_different_conversations_serially(self):
        self.service.import_records({'records':[record('a'),record('b')]})
        first,_=self.service.start_evaluation('a',self.hash);second,_=self.service.start_evaluation('b',self.hash)
        self.wait(first['evaluation_id']);self.wait(second['evaluation_id'])
        self.assertEqual(self.provider.max_active,1)

    def test_simultaneous_submissions_share_one_paid_attempt(self):
        self.service.import_records({'records':[record()]});self.provider.release.clear()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(lambda _:self.service.start_evaluation('fixture-en',self.hash),range(16)))
            self.assertEqual(len({value['evaluation_id'] for value,status in results}),1)
        finally:self.provider.release.set()
        self.wait(results[0][0]['evaluation_id'])
        self.assertEqual(self.provider.calls,1)

    def test_semantic_uncertainty_is_needs_review_not_zero(self):
        def uncertain(rows):
            rows[3].update(status='needs_review',score=None,reason='Нет решающего продуктового источника',evidence=[],checked_span=None)
            return rows
        self.provider.transform=uncertain;self.service.import_records({'records':[record()]})
        result=self.submit()
        self.assertEqual(result['status'],'needs_review');self.assertIsNone(result['error'])
        self.assertIsNone(result['result']['final_summary']['score'])
        self.assertEqual(result['result']['final_findings'][3]['status'],'needs_review')

    def test_invalid_quote_cannot_publish_a_successful_score(self):
        def fabricate(rows):rows[0]['evidence'][0]['quote']='Invented';return rows
        self.provider.transform=fabricate;self.service.import_records({'records':[record()]})
        result=self.submit();self.assertEqual(result['status'],'needs_review')
        self.assertIsNone(result['result']['final_summary']['score'])

    def test_transport_error_is_error_and_cache_requires_force_to_retry(self):
        self.provider.error=ProviderError('provider_auth','API-ключ не принят',{'calls':1,'usage':{},'latency_seconds':.01})
        self.service.import_records({'records':[record()]});value=self.submit()
        self.assertEqual(value['status'],'error');self.assertIsNone(value['result'])
        self.assertEqual(value['error']['code'],'provider_auth');self.assertEqual(self.provider.calls,1)
        cached,status=self.service.start_evaluation('fixture-en',self.hash)
        self.assertEqual(cached['evaluation_id'],value['evaluation_id']);self.assertEqual(status,200)
        self.provider.error=None;self.assertEqual(self.submit(force=True)['status'],'completed')

    def test_missing_key_blocks_creation_without_billing(self):
        self.service.import_records({'records':[record()]});self.settings.api_key=''
        with self.assertRaises(ServiceError) as e:self.service.start_evaluation('fixture-en',self.hash)
        self.assertEqual(e.exception.status_code,503);self.assertEqual(self.provider.calls,0)

    def test_restart_keeps_history_and_marks_unfinished_as_interrupted(self):
        self.service.import_records({'records':[record()]});done=self.submit();self.service.close()
        with self.service.db.connect() as db:
            payload=copy.deepcopy(done);payload.update(evaluation_id='interrupted-test',status='running',result=None)
            db.execute('INSERT INTO evaluations(id,conversation_id,checklist_hash,cache_key,status,created_at,payload) VALUES (?,?,?,?,?,?,?)',
                       ('interrupted-test','fixture-en',self.hash,'unfinished','running',done['created_at'],json.dumps(payload)))
        reopened=Service(self.settings,FixtureProvider());self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_evaluation(done['evaluation_id'])['result']['final_summary']['score'],100)
        bad=reopened.get_evaluation('interrupted-test')
        self.assertEqual(bad['status'],'error');self.assertEqual(bad['error']['code'],'interrupted')

    def test_knowledge_snapshot_is_preserved_and_changes_cache_key(self):
        self.settings.knowledge_file=Path(self.temp.name)/'knowledge.json'
        self.settings.knowledge_file.write_text(json.dumps({'version':'v1','sources':[{'source_id':'policy','text':'One policy'}]}),encoding='utf-8')
        self.service.import_records({'records':[record()]});first=self.submit()
        self.assertEqual(first['result']['knowledge_snapshot']['version'],'v1')
        self.settings.knowledge_file.write_text(json.dumps({'version':'v2','sources':[{'source_id':'policy','text':'New policy'}]}),encoding='utf-8')
        second=self.submit();self.assertNotEqual(first['evaluation_id'],second['evaluation_id'])
        self.assertEqual(first['result']['knowledge_snapshot']['sources'][0]['text'],'One policy')


if __name__=='__main__':unittest.main()
