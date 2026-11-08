"""Cascade acceptance rules, paid-call boundaries and provenance."""
import tempfile
import time
import unittest
from pathlib import Path

from config import Settings
from provider import ProviderError
from service import Service
from test_support import FixtureProvider, record


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.settings=Settings(data_dir=Path(self.temp.name),api_key='fixture')
        self.settings.pipeline='cascade';self.settings.small_model='fixture-small'
        self.settings.review_model='fixture-review';self.settings.review_all=True
        self.settings.audit_rate=0;self.settings.sensitive_criteria=()
        self.primary=FixtureProvider();self.review=FixtureProvider()
        self.service=Service(self.settings,self.primary);self.service.reviewer=self.review
        self.addCleanup(self.service.close)
        self.hash=self.service.list_checklists()['items'][0]['hash']
        self.service.import_records({'records':[record()]})

    def run_job(self):
        value,_=self.service.start_evaluation('fixture-en',self.hash)
        for _ in range(200):
            value=self.service.get_evaluation(value['evaluation_id'])
            if value['status'] not in ('queued','running'):return value
            time.sleep(.01)
        self.fail('Worker timeout')

    def test_reviewed_automatic_result_is_distinct_from_small_stage(self):
        def partial(rows):rows[3]['score']=5;return rows
        self.primary.transform=partial
        value=self.run_job()
        self.assertEqual(value['inference']['calls'],2)
        self.assertEqual(value['result']['original_summary']['score'],100)
        self.assertEqual(value['stages'][0]['findings'][3]['score'],5)
        self.assertEqual(value['stages'][1]['findings'][3]['score'],20)
        self.assertEqual(value['result']['original_findings'],value['result']['final_findings'])
        self.assertEqual(value['inference']['usage']['completion_tokens'],400)

    def test_selective_route_can_accept_one_valid_call(self):
        self.settings.review_all=False
        value=self.run_job()
        self.assertEqual(value.get('pipeline'),'cascade')
        self.assertEqual(value['status'],'completed')
        self.assertEqual(len(value['stages']),1)
        self.assertEqual(value['inference']['calls'],1)

    def test_sensitive_disagreement_requires_human_and_has_no_full_score(self):
        self.settings.sensitive_criteria=('c03',)
        def unsafe(rows):rows[2]['score']=0;return rows
        self.primary.transform=unsafe
        value=self.run_job()
        self.assertEqual(value['status'],'needs_review')
        self.assertIsNone(value['result']['final_summary']['score'])
        self.assertEqual(value['result']['final_findings'][2]['status'],'needs_review')
        self.assertIn('c03',value['disagreements'])

    def test_missing_fact_does_not_get_fabricated_by_second_model(self):
        def missing(rows):
            rows[3].update(status='needs_review',score=None,factual_status='not_verified',
                           reason='Decisive product fact is missing.',evidence=[])
            return rows
        self.primary.transform=missing
        value=self.run_job()
        self.assertEqual(value['status'],'needs_review')
        self.assertIsNone(value['result']['final_findings'][3]['score'])

    def test_missing_fact_alone_does_not_trigger_another_paid_call(self):
        self.settings.review_all=False
        def missing(rows):
            rows[3].update(status='needs_review',score=None,factual_status='not_verified',
                           reason='Missing source.',evidence=[])
            return rows
        self.primary.transform=missing
        value=self.run_job()
        self.assertEqual(value['status'],'needs_review')
        self.assertEqual(value['inference']['calls'],1)
        self.assertEqual(value.get('routing',{}).get('blocked_by_missing_sources'),['c04'])

    def test_unverified_scored_fact_is_withheld_even_without_second_model(self):
        self.settings.review_all=False
        def unverified(rows):
            rows[3].update(score=20,factual_status='not_verified',reason='Client thanks alone is not product verification.')
            return rows
        self.primary.transform=unverified
        value=self.run_job()
        self.assertEqual(value['status'],'needs_review')
        self.assertIsNone(value['result']['final_summary']['score'])
        self.assertEqual(value['result']['final_findings'][3]['status'],'needs_review')
        self.assertEqual(value['inference']['calls'],1)

    def test_review_transport_error_keeps_stage_history_without_success(self):
        self.review.error=ProviderError('provider_timeout','Timed out',{'calls':1,'usage':{}})
        value=self.run_job()
        self.assertEqual(value['status'],'error')
        self.assertIsNone(value['result'])
        self.assertEqual(value['inference']['calls'],2)
        self.assertFalse(value['inference']['usage_complete'])
        self.assertEqual(value['stages'][0]['status'],'completed')
        self.assertEqual(value['stages'][1]['status'],'error')

    def test_changed_routing_creates_new_run_instead_of_reusing_cache(self):
        self.settings.review_all=False
        first=self.run_job()
        self.settings.audit_rate=1
        second=self.run_job()
        self.assertNotEqual(first['evaluation_id'],second['evaluation_id'])
        self.assertEqual(second['inference']['calls'],2)

    def test_invalid_primary_json_does_not_trigger_review(self):
        self.primary.error=ProviderError('invalid_model_response','Invalid JSON')
        value=self.run_job()
        self.assertEqual(value['status'],'error')
        self.assertEqual(value.get('stage'),'primary')
        self.assertEqual(value['inference']['calls'],1)

    def test_unexpected_provider_failure_is_sanitized_and_stage_is_not_left_running(self):
        self.primary.error=RuntimeError('private-provider-internal-content')
        value=self.run_job()
        self.assertEqual(value['status'],'error')
        self.assertEqual(value['stages'][0]['status'],'error')
        self.assertEqual(value['inference']['calls'],1)
        self.assertNotIn('private-provider-internal-content',str(value))


if __name__=='__main__':unittest.main()
