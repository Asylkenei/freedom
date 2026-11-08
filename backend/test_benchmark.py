import unittest
from benchmark import compare,make_template
from cqa import digest,evaluate
from test_cqa import criterion,finding

class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.checklist={'criteria':[criterion()]}
        self.record=dict(conversation_id='fixture',channel='agent chat',language='EN',audio_url=None,transcript=[dict(speaker='operator',timestamp='00:01',text='Hello')])
        self.run=evaluate(self.checklist,self.record)
        self.gold=dict(checklist_hash=digest(self.checklist),records=[dict(conversation_id='fixture',language='EN',findings=[dict(criterion_id='one',status='scored',score=1)],total_score=100)])
    def test_unanswered_is_counted_and_does_not_get_fake_mae(self):
        r=compare(self.gold,[self.run]);self.assertEqual(r['overall']['answer_coverage'],0);self.assertEqual(r['overall']['exact_accuracy_including_unanswered'],0);self.assertIsNone(r['overall']['mae_on_scored_pairs_only']);self.assertFalse(r['all_gold_criteria_answered'])
    def test_missing_conversation_is_unanswered(self):
        self.assertEqual(compare(self.gold,[])['overall']['criteria_count'],1);self.assertEqual(compare(self.gold,[])['overall']['answer_coverage'],0)
    def test_na_mismatch_is_error_despite_absent_numeric_pair(self):
        self.run['original_findings'][0].update(status='na',score=None);r=compare(self.gold,[self.run]);self.assertEqual(r['overall']['exact_accuracy_including_unanswered'],0);self.assertEqual(r['overall']['status_confusion'],{'scored -> na':1})
    def test_default_does_not_compare_human_overrides(self):
        self.run['final_findings']=[finding()];self.assertEqual(compare(self.gold,[self.run])['overall']['exact_accuracy_including_unanswered'],0);self.assertEqual(compare(self.gold,[self.run],'final')['overall']['exact_accuracy_including_unanswered'],1)
    def test_wrong_checklist_rejected(self):
        self.gold['checklist_hash']='different'
        with self.assertRaises(ValueError):compare(self.gold,[self.run])
    def test_unannotated_template_cannot_be_used_as_gold(self):
        with self.assertRaises(ValueError):compare(make_template(self.checklist,[self.record]),[self.run])

if __name__=='__main__':unittest.main()
