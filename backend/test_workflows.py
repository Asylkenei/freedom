"""HTTP business workflows use real SQLite and synthetic external model outputs."""
import copy
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from app import create_app
from config import Settings
from test_support import FixtureProvider, record


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = Settings(data_dir=Path(self.temp.name), api_key='private-fixture')
        self.client = TestClient(create_app(self.settings, FixtureProvider()))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        item = self.client.get('/api/checklists').json()['items'][0]
        self.hash = item.pop('hash')
        self.checklist = item

    def evaluate(self, checklist_hash=None):
        self.client.post('/api/conversations/import', json={'records': [record()]})
        started = self.client.post('/api/evaluations', json={
            'conversation_id': 'fixture-en', 'checklist_hash': checklist_hash or self.hash})
        self.assertEqual(started.status_code, 202)
        for _ in range(200):
            value = self.client.get('/api/evaluations/' + started.json()['evaluation_id']).json()
            if value['status'] not in ('queued', 'running'):
                return value
            time.sleep(.01)
        self.fail('Worker timeout')

    def appeal(self, value):
        return self.client.post('/api/evaluations/' + value['evaluation_id'] + '/appeals', json={
            'criterion_id': 'c01', 'comment': 'Please review the first reply.',
            'author': 'operator', 'client_request_id': 'appeal-request-1'})

    def decision(self, value, revision=1):
        finding = copy.deepcopy(value['result']['final_findings'][0])
        finding.update(score=0, evidence=[], reason='Synthetic human decision for arithmetic test.',
                       checked_span={'from': 'm0001', 'to': 'm0003'})
        return dict(decision='change', reason='Reviewed source messages.', reviewer='QA',
                    replacement=finding, expected_revision=revision, client_request_id='decision-request-1')

    def test_checklist_version_is_immutable_and_custom_criterion_is_evaluated(self):
        changed = copy.deepcopy(self.checklist)
        extra = copy.deepcopy(changed['criteria'][4])
        extra.update(id='custom-care', title='Custom care criterion', weight=5)
        changed['criteria'].append(extra)
        payload = dict(checklist=changed, parent_hash=self.hash, author='teammate')
        response = self.client.post('/api/checklists', json=payload)
        self.assertEqual(response.status_code, 201)
        new_hash = response.json()['hash']
        self.assertNotEqual(new_hash, self.hash)
        self.assertEqual(len(self.client.get('/api/checklists/' + self.hash).json()['criteria']), 8)
        self.assertEqual(self.client.post('/api/checklists', json=payload).json()['hash'], new_hash)
        value = self.evaluate(new_hash)
        self.assertEqual(value['status'], 'completed')
        self.assertEqual(len(value['result']['original_findings']), 9)

    def test_invalid_checklist_does_not_create_a_version(self):
        changed = copy.deepcopy(self.checklist)
        changed['criteria'][3]['allowed_scores'] = [0, 21]
        response = self.client.post('/api/checklists', json={'checklist': changed, 'parent_hash': self.hash, 'author': 'QA'})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(self.client.get('/api/checklists').json()['items']), 1)

    def test_nonstring_criterion_id_and_string_scale_are_rejected_before_model_call(self):
        for field,value in [('id',123),('min_score','0'),('section',{'bad':'section'})]:
            with self.subTest(field=field):
                changed=copy.deepcopy(self.checklist);changed['criteria'][0][field]=value
                response=self.client.post('/api/checklists',json={'checklist':changed,'parent_hash':self.hash,'author':'QA'})
                self.assertEqual(response.status_code,422)

    def test_stale_pipeline_guard_rejects_job_without_creating_history(self):
        self.client.post('/api/conversations/import',json={'records':[record()]})
        response=self.client.post('/api/evaluations',json={'conversation_id':'fixture-en','checklist_hash':self.hash,'pipeline_hash':'0'*64})
        self.assertEqual(response.status_code,409)
        self.assertEqual(self.client.get('/api/conversations/fixture-en').json()['evaluations'],[])

    def test_appeal_resolution_recalculates_and_preserves_original_after_restart(self):
        value = self.evaluate()
        response = self.appeal(value)
        self.assertEqual(response.status_code, 201)
        appeal_id = response.json()['appeal_id']
        resolved = self.client.post('/api/appeals/' + appeal_id + '/resolve', json=self.decision(value))
        self.assertEqual(resolved.status_code, 200)
        self.assertEqual(resolved.json()['revision'], 2)
        self.assertEqual(resolved.json()['result']['original_summary']['score'], 100)
        self.assertEqual(resolved.json()['result']['final_summary']['score'], 90)
        self.assertEqual(resolved.json()['result']['appeals'][0]['status'], 'changed')
        with TestClient(create_app(self.settings, FixtureProvider())) as reopened:
            saved = reopened.get('/api/evaluations/' + value['evaluation_id']).json()
            self.assertEqual(saved['result']['final_summary']['score'], 90)
            self.assertEqual(saved['result']['original_findings'][0]['score'], 1)

    def test_duplicate_appeal_and_decision_are_idempotent(self):
        value = self.evaluate()
        first = self.appeal(value)
        self.assertEqual(first.status_code, 201)
        repeated = self.appeal(value)
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(first.json()['appeal_id'], repeated.json()['appeal_id'])
        endpoint = '/api/appeals/' + first.json()['appeal_id'] + '/resolve'
        payload = self.decision(value)
        self.assertEqual(self.client.post(endpoint, json=payload).status_code, 200)
        repeated = self.client.post(endpoint, json=payload)
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.json()['revision'], 2)
        payload['reason'] = 'Changed request under the same key'
        self.assertEqual(self.client.post(endpoint, json=payload).status_code, 409)
        self.assertEqual(len(self.client.get('/api/evaluations/' + value['evaluation_id'] + '/appeals').json()['items']), 1)

    def test_stale_revision_and_invalid_evidence_do_not_modify_score(self):
        value = self.evaluate()
        response = self.appeal(value)
        self.assertEqual(response.status_code, 201)
        endpoint = '/api/appeals/' + response.json()['appeal_id'] + '/resolve'
        self.assertEqual(self.client.post(endpoint, json=self.decision(value, 0)).status_code, 409)
        payload = self.decision(value)
        payload['replacement']['evidence'] = [{'message_id': 'm0001', 'quote': 'Invented quote'}]
        self.assertEqual(self.client.post(endpoint, json=payload).status_code, 422)
        saved = self.client.get('/api/evaluations/' + value['evaluation_id']).json()
        self.assertEqual(saved['revision'], 1)
        self.assertEqual(saved['result']['final_summary']['score'], 100)

    def test_uphold_requires_reason_and_keeps_score(self):
        value = self.evaluate()
        response = self.appeal(value)
        self.assertEqual(response.status_code, 201)
        endpoint = '/api/appeals/' + response.json()['appeal_id'] + '/resolve'
        payload = dict(decision='uphold', reason=' ', reviewer='QA', replacement=None,
                       expected_revision=1, client_request_id='uphold-1')
        self.assertEqual(self.client.post(endpoint, json=payload).status_code, 422)
        payload['reason'] = 'Original assessment confirmed.'
        resolved = self.client.post(endpoint, json=payload)
        self.assertEqual(resolved.status_code, 200)
        self.assertEqual(resolved.json()['result']['final_summary']['score'], 100)
        self.assertEqual(resolved.json()['result']['appeals'][0]['status'], 'upheld')


if __name__ == '__main__':
    unittest.main()
