import copy
import unittest
import inspect

import annotation
import cqa
from benchmark import compare
from config import ROOT
from test_support import record,fixture_findings


class MetricsV2Tests(unittest.TestCase):
    def setUp(self):
        self.checklist=cqa.read_json(ROOT/'checklist.official.json');self.record=record()
        draft=annotation.make_template(self.checklist,[self.record])
        draft['records'][0]['findings']=fixture_findings(self.checklist,cqa.messages(self.record))
        for f in draft['records'][0]['findings']:f.update(annotator_id='human-fixture',annotation_state='adjudicated',ai_assisted=False)
        self.gold=annotation.validate_annotations(draft,[self.record],self.checklist)
        self.run=cqa.evaluate(self.checklist,self.record)
        self.run['original_findings']=copy.deepcopy(self.gold['records'][0]['findings'])
        self.run['original_summary']=dict(score=100)
        self.run['final_findings']=copy.deepcopy(self.run['original_findings'])
        self.run['final_summary']=dict(score=100)

    def test_unresolved_reference_is_reported_without_discarding_known_criteria(self):
        f=self.gold['records'][0]['findings'][3];f.update(status='unresolved',score=None,reason='Source unavailable')
        self.gold['records'][0]['total_score']=None
        report=compare(self.gold,[self.run])
        self.assertEqual(report['overall']['criteria_count'],7)
        self.assertEqual(report.get('unresolved_reference_count'),1)
        self.assertEqual(report['overall']['exact_accuracy_including_unanswered'],1)
        self.assertIsNone(report['exact_dialogue_accuracy'])

    def test_technical_error_is_unanswered_and_not_a_zero_score(self):
        report=compare(self.gold,[dict(conversation_id='fixture-en',status='error',result=None,error={'code':'provider_timeout'},inference={})])
        self.assertEqual(report['overall']['answer_coverage'],0)
        self.assertEqual(report['overall']['exact_accuracy_including_unanswered'],0)
        self.assertIsNone(report['total_score_comparison']['mae_on_available_totals_only'])
        self.assertEqual(report['technical_error_count'],1)

    def test_na_mismatch_counts_as_error_instead_of_disappearing_from_mae(self):
        self.run['original_findings'][1].update(status='na',score=None,na_reason='Enough context')
        report=compare(self.gold,[self.run])
        self.assertEqual(report['overall']['exact_accuracy_including_unanswered'],.875)
        self.assertEqual(report['overall']['answer_coverage'],1)
        self.assertEqual(report['overall']['status_confusion']['scored -> na'],1)

    def test_flag_false_negative_is_visible_even_with_identical_total(self):
        self.gold['records'][0]['findings'][7]['score']=1
        report=compare(self.gold,[self.run])
        self.assertEqual(report.get('flags',{}).get('c08',{}).get('false_negatives'),1)
        self.assertEqual(report['flags']['c08']['recall_including_unanswered'],0)
        self.assertEqual(report['total_score_comparison']['mae_on_available_totals_only'],0)

    def test_reference_scale_is_validated_without_any_prediction(self):
        self.gold['records'][0]['findings'][0]['score']=7
        with self.assertRaises(ValueError):compare(self.gold,[])

    def test_changed_transcript_and_unapproved_labels_are_rejected(self):
        self.run['transcript_hash']='changed'
        with self.assertRaises(ValueError):compare(self.gold,[self.run])
        self.gold['annotation_state']='draft'
        with self.assertRaises(ValueError):compare(self.gold,[])

    def test_default_evaluates_automatic_pipeline_and_groups_by_channel(self):
        self.run['final_findings'][0]['score']=0
        report=compare(self.gold,[self.run])
        self.assertEqual(report['overall']['exact_accuracy_including_unanswered'],1)
        self.assertIn('agent chat',report.get('by_channel',{}))
        self.assertEqual(compare(self.gold,[self.run],'final')['overall']['exact_accuracy_including_unanswered'],.875)

    def ai_reference(self):
        value=copy.deepcopy(self.gold)
        value.update(annotation_source='ai_assistant',annotation_state='ai_reviewed',human_reviewed=False)
        for f in value['records'][0]['findings']:
            f.update(annotator_id='Codex AI',annotation_state='submitted',ai_assisted=True)
        return value

    def test_ai_reference_requires_explicit_mode_and_retains_ai_provenance(self):
        value=self.ai_reference()
        with self.assertRaises(ValueError):compare(value,[self.run])
        self.assertIn('reference_kind',inspect.signature(compare).parameters,'Explicit AI-reference comparison is missing')
        report=compare(value,[self.run],reference_kind='ai')
        self.assertEqual(report['reference_kind'],'ai')
        self.assertFalse(report['human_reference'])
        self.assertEqual(report['overall']['exact_accuracy_including_unanswered'],1)
        self.assertEqual(value['annotation_source'],'ai_assistant')
        self.assertEqual(value['records'][0]['findings'][0]['annotation_state'],'submitted')

    def test_ai_uncertainty_is_not_scored_as_correct_or_as_wrong(self):
        self.assertIn('reference_kind',inspect.signature(compare).parameters,'Explicit AI-reference comparison is missing')
        value=self.ai_reference()
        value['records'][0]['findings'][3].update(status='unresolved',score=None,reason='Missing product source')
        value['records'][0]['total_score']=None
        report=compare(value,[self.run],reference_kind='ai')
        self.assertEqual(report['overall']['criteria_count'],7)
        self.assertEqual(report['unresolved_reference_count'],1)
        self.assertEqual(report['reference_complete_dialogues'],0)


if __name__=='__main__':unittest.main()
