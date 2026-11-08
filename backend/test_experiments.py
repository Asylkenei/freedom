import copy
import unittest
import tempfile

import annotation
import cqa
from config import ROOT
from test_support import record
try:
    import experiments
except ImportError:experiments=None


class ExperimentPlanTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(experiments,'Bounded experiment runner is not implemented')
        self.checklist=cqa.read_json(ROOT/'checklist.official.json')
        self.records=[]
        for i in range(10):
            r=record('case-'+str(i));r['transcript'][1]['text']='Unique topic '+str(i)
            self.records.append(r)
        self.manifest=annotation.prepare(self.records,self.checklist)['manifest']

    def test_dry_plan_reserves_two_calls_per_cascade_case(self):
        plan=experiments.make_plan(self.manifest,self.records,'a'*64,'cascade','development',2,4)
        self.assertEqual(plan['estimated_max_calls'],4)
        self.assertEqual(plan['execute'],False)
        self.assertEqual(len(plan['conversation_ids']),2)
        with self.assertRaises(ValueError):experiments.make_plan(self.manifest,self.records,'a'*64,'cascade','development',2,3)

    def test_changed_dataset_and_manifest_are_rejected(self):
        changed=copy.deepcopy(self.records);changed[0]['transcript'][1]['text']='Changed'
        with self.assertRaises(ValueError):experiments.make_plan(self.manifest,changed,'a'*64,'single','development',2,2)
        self.manifest['records'][0]['split']='test'
        with self.assertRaises(ValueError):experiments.make_plan(self.manifest,self.records,'a'*64,'single','development',2,2)

    def test_holdout_requires_frozen_profile_and_duplicate_review(self):
        with self.assertRaises(ValueError):experiments.make_plan(self.manifest,self.records,'a'*64,'single','test',1,1)
        self.manifest['near_duplicate_review']='reviewed'
        self.manifest['manifest_hash']=cqa.digest({k:v for k,v in self.manifest.items() if k!='manifest_hash'})
        frozen={'pipeline_hash':'a'*64,'manifest_hash':self.manifest['manifest_hash'],'checklist_hash':self.manifest['checklist_hash']}
        plan=experiments.make_plan(self.manifest,self.records,'a'*64,'single','test',1,1,frozen)
        self.assertEqual(plan['split'],'test')
        with self.assertRaises(ValueError):experiments.make_plan(self.manifest,self.records,'b'*64,'single','test',1,1,frozen)

    def test_execution_rechecks_source_before_any_backend_request(self):
        plan=experiments.make_plan(self.manifest,self.records,'a'*64,'single','development',2,2)
        changed=copy.deepcopy(self.records);changed[0]['transcript'][1]['text']='Changed after planning'
        def no_request(*args):self.fail('Changed source reached the backend')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):experiments.execute_plan(plan,changed,'http://localhost',directory,client=no_request)


if __name__=='__main__':unittest.main()
