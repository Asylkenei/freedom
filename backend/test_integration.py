"""Full FastAPI → queue → actual local HTTP Responses fixture → baseline → SQLite.

These are transport/logic tests, not measurements of real model quality.
"""
import copy
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient
from app import create_app
from config import Settings
from cqa import read_json
from test_support import fixture_findings,record


class FullStackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.requests=[];self.variant='ok'
        case=self
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                case.requests.append(payload);data=json.loads(payload['input'][0]['content'])
                rows=fixture_findings({'criteria':data['criteria']},data['transcript'])
                if case.variant=='quote':rows[0]['evidence'][0]['quote']='NOT IN SOURCE'
                if case.variant=='missing':rows=rows[:-1]
                output=[dict(type='message',content=[dict(type='output_text',text=json.dumps({'findings':rows}))])]
                if case.variant=='refusal':output=[dict(type='message',content=[dict(type='refusal',refusal='Upstream private message')])]
                if case.variant=='json':output=[dict(type='message',content=[dict(type='output_text',text='{broken')])]
                body=dict(id='fixture-response',status='incomplete' if case.variant=='incomplete' else 'completed',
                    model='gpt-6.1-sol',usage=dict(input_tokens=1200,output_tokens=300,total_tokens=1500),output=output)
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(body).encode())
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=server.serve_forever,daemon=True).start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        self.settings=Settings(data_dir=Path(self.temp.name)/'db',api_key='fixture-not-real',
                               base_url=f'http://127.0.0.1:{server.server_port}/v1')
        self.client=TestClient(create_app(self.settings));self.client.__enter__()
        self.addCleanup(self.client.__exit__,None,None,None)
        self.hash=self.client.get('/api/checklists').json()['items'][0]['hash']

    def evaluate(self,r):
        self.assertEqual(self.client.post('/api/conversations/import',json={'records':[r]}).status_code,200)
        response=self.client.post('/api/evaluations',json={'conversation_id':r['conversation_id'],'checklist_hash':self.hash})
        self.assertEqual(response.status_code,202)
        eid=response.json()['evaluation_id']
        for _ in range(200):
            value=self.client.get('/api/evaluations/'+eid).json()
            if value['status'] not in ('queued','running'):return value
            time.sleep(.01)
        self.fail('Timed out')

    def test_four_language_channel_cases_through_real_http_transport(self):
        for cid,lang,channel in [('ru','RU','agent chat'),('kz','KZ','agent chat'),('en','EN','agent chat'),('call','RU','voice call')]:
            with self.subTest(cid=cid):
                value=self.evaluate(record(cid,lang,channel))
                self.assertEqual(value['status'],'completed')
                self.assertEqual(value['result']['final_summary']['score'],100)
                self.assertEqual(value['result']['language'],lang)
                self.assertEqual(value['inference']['usage']['completion_tokens'],300)
                self.assertEqual(value['inference']['calls'],1)
                self.assertTrue(value['result']['final_findings'][5]['verification_limits'] if channel=='voice call' else True)
        self.assertEqual(len(self.requests),4)
        self.assertTrue(all(r['model']=='gpt-6.1-sol' and r['reasoning']=={'effort':'medium'} for r in self.requests))

    def test_refusal_incomplete_json_and_missing_ids_are_saved_errors(self):
        for variant,code in [('refusal','model_refusal'),('incomplete','incomplete_response'),('json','invalid_model_response'),('missing','invalid_model_response')]:
            self.variant=variant
            with self.subTest(variant=variant):
                value=self.evaluate(record('error-'+variant))
                self.assertEqual(value['status'],'error');self.assertIsNone(value['result'])
                self.assertEqual(value['error']['code'],code)
                self.assertEqual(value['inference']['usage']['prompt_tokens'],1200)
                self.assertNotIn('Upstream private message',json.dumps(value))

    def test_fabricated_evidence_is_review_and_never_publishes_full_score(self):
        self.variant='quote';value=self.evaluate(record())
        self.assertEqual(value['status'],'needs_review')
        self.assertIsNone(value['result']['final_summary']['score'])
        self.assertEqual(value['result']['final_findings'][0]['status'],'needs_review')

    def test_new_checklist_is_passed_to_schema_without_hardcoding_eight(self):
        checklist=read_json('checklist.official.json')
        checklist['criteria'].append(dict(checklist['criteria'][0],id='custom',title='Custom check'))
        target=Path(self.temp.name)/'custom.json';target.write_text(json.dumps(checklist),encoding='utf-8')
        self.client.__exit__(None,None,None)
        self.settings.checklist_file=target
        self.client=TestClient(create_app(self.settings));self.client.__enter__()
        self.addCleanup(self.client.__exit__,None,None,None)
        self.hash=self.client.get('/api/checklists').json()['items'][0]['hash']
        value=self.evaluate(record())
        self.assertEqual(len(value['result']['final_findings']),9)
        self.assertEqual(len(self.requests[0]['text']['format']['schema']['properties']['findings']['items']['anyOf']),9)
        self.assertEqual(len(self.client.get('/api/checklists').json()['items']),2)


if __name__=='__main__':unittest.main()
