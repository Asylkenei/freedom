import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from test_support import FixtureProvider,record
from config import Settings

try:
    from app import create_app
except ImportError:create_app=None


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(create_app),'FastAPI application not implemented')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.settings=Settings(data_dir=Path(self.temp.name),api_key='server-only-secret')
        self.provider=FixtureProvider()
        self.client=TestClient(create_app(self.settings,self.provider));self.client.__enter__()
        self.addCleanup(self.client.__exit__,None,None,None)
        self.hash=self.client.get('/api/checklists').json()['items'][0]['hash']

    def wait(self,eid):
        for _ in range(200):
            result=self.client.get('/api/evaluations/'+eid).json()
            if result['status'] not in ['running','queued']:return result
            time.sleep(.01)
        self.fail('Timed out')

    def test_health_docs_and_contract_never_expose_server_key(self):
        for path in ['/api/health','/api/checklists','/openapi.json','/docs']:
            response=self.client.get(path);self.assertEqual(response.status_code,200)
            self.assertNotIn('server-only-secret',response.text)
        schema=self.client.get('/openapi.json').json()
        for path in ['/api/conversations/import','/api/evaluations','/api/evaluations/{evaluation_id}']:
            self.assertIn(path,schema['paths'])
        self.assertNotIn('api_key',json.dumps(schema))

    def test_complete_frontend_round_trip_and_cached_result(self):
        imported=self.client.post('/api/conversations/import',json={'isSuccess':True,'data':{'total_records':1,'records':[record()]}})
        self.assertEqual(imported.status_code,200)
        detail=self.client.get('/api/conversations/fixture-en').json()
        self.assertEqual(detail['messages'][0]['message_id'],'m0001')
        request={'conversation_id':'fixture-en','checklist_hash':self.hash}
        started=self.client.post('/api/evaluations',json=request)
        self.assertEqual(started.status_code,202)
        value=self.wait(started.json()['evaluation_id'])
        self.assertEqual(value['status'],'completed')
        self.assertEqual(value['result']['final_summary']['score'],100)
        self.assertEqual(self.client.get('/api/conversations').json()['items'][0]['score'],100)
        cached=self.client.post('/api/evaluations',json=request)
        self.assertEqual(cached.status_code,200)
        self.assertEqual(cached.json()['evaluation_id'],value['evaluation_id'])
        self.assertNotIn('server-only-secret',json.dumps(value))

    def test_cors_accepts_frontend_and_rejects_other_origin(self):
        headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'}
        response=self.client.options('/api/evaluations',headers=headers)
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.headers['access-control-allow-origin'],'http://localhost:5173')
        self.assertNotIn('access-control-allow-credentials',response.headers)
        bad=self.client.options('/api/evaluations',headers={**headers,'Origin':'https://untrusted.example'})
        self.assertEqual(bad.status_code,400)
        self.assertNotIn('access-control-allow-origin',bad.headers)

    def test_missing_key_is_503_and_invalid_body_is_422(self):
        self.client.post('/api/conversations/import',json={'records':[record()]});self.settings.api_key=''
        response=self.client.post('/api/evaluations',json={'conversation_id':'fixture-en','checklist_hash':self.hash})
        self.assertEqual(response.status_code,503);self.assertEqual(response.json()['error']['code'],'not_configured')
        self.assertEqual(self.client.post('/api/evaluations',json={'conversation_id':'fixture-en','checklist_hash':self.hash,'force':'true'}).status_code,422)
        self.assertEqual(self.client.post('/api/conversations/import',content='{broken',headers={'Content-Type':'application/json'}).status_code,422)

    def test_conflict_not_found_and_private_files(self):
        self.client.post('/api/conversations/import',json={'records':[record()]})
        changed=record();changed['transcript'][0]['text']='Changed'
        self.assertEqual(self.client.post('/api/conversations/import',json={'records':[changed]}).status_code,409)
        self.assertEqual(self.client.get('/api/evaluations/unknown').status_code,404)
        for path in ['/.env','/data/cqa.sqlite3','/cqa.py']:
            self.assertEqual(self.client.get(path).status_code,404)

    def test_language_timestamp_and_ids_survive_http_for_voice(self):
        self.client.post('/api/conversations/import',json={'records':[record('call','KZ','voice call')]})
        data=self.client.get('/api/conversations/call').json()
        self.assertEqual(data['language'],'KZ')
        self.assertEqual(data['messages'][0]['timestamp'],'2026-11-07T10:00:01+05:00')


class ConfigurationTests(unittest.TestCase):
    def test_environment_overrides_file_without_leaking_key_in_repr(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'.env').write_text('CQA_API_KEY=from-file\nCQA_MODEL=from-file\n',encoding='utf-8')
            with patch.dict('os.environ',{'CQA_API_KEY':'from-environment','CQA_MODEL':'from-environment'},clear=True):
                settings=Settings.from_env(p)
            self.assertEqual(settings.api_key,'from-environment');self.assertEqual(settings.model,'from-environment')
            self.assertNotIn('from-environment',repr(settings).split('model=')[0])

    def test_empty_environment_value_disables_key_in_file(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);(p/'.env').write_text('CQA_API_KEY=from-file\n',encoding='utf-8')
            with patch.dict('os.environ',{'CQA_API_KEY':''},clear=True):settings=Settings.from_env(p)
            self.assertEqual(settings.api_key,'')


if __name__=='__main__':unittest.main()
