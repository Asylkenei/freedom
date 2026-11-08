import copy
import unittest
from unittest.mock import patch
from cqa import aggregate,validate_finding,validate_checklist,messages,lexical,evaluate,add_appeal,resolve_appeal,load_records


def criterion(cid='one',weight=1,max_score=1,allow_na=True):
    return dict(id=cid,title=cid,instruction='Check operator',section='Test',weight=weight,min_score=0,max_score=max_score,type='binary' if max_score==1 else 'scale',allow_na=allow_na)


def finding(cid='one',score=1):
    return dict(criterion_id=cid,status='scored',score=score,reason='Observed action',evidence=[dict(message_id='m0002',quote='Hello')],checked_span=None,critical_error=None,na_reason=None)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.record=dict(conversation_id='test',channel='agent chat',language='EN',audio_url=None,transcript=[
            dict(speaker='bot',text='Hello bot',timestamp='00:01'),dict(speaker='operator',text='Hello operator',timestamp='00:05')])
        self.turns=messages(self.record)
    def test_weight_normalization_and_na(self):
        cs=[criterion('a',30,5),criterion('b',20),criterion('c',50)]
        fs=[dict(criterion_id='a',status='scored',score=4),dict(criterion_id='b',status='scored',score=0),dict(criterion_id='c',status='na',score=None)]
        self.assertEqual(aggregate({'criteria':cs},fs)['score'],48)
    def test_unknown_does_not_inflate_final(self):
        result=aggregate({'criteria':[criterion('a'),criterion('b')]},[dict(criterion_id='a',status='scored',score=1),dict(criterion_id='b',status='needs_review',score=None)])
        self.assertIsNone(result['score']);self.assertEqual(result['partial_score_on_assessed'],100);self.assertEqual(result['coverage_of_applicable_weight'],.5)
    def test_all_na_has_no_score(self):
        self.assertIsNone(aggregate({'criteria':[criterion()]},[dict(criterion_id='one',status='na')])['score'])
    def test_rounding_half_up(self):
        self.assertEqual(aggregate({'criteria':[criterion(max_score=10000)]},[dict(criterion_id='one',status='scored',score=8765.5)])['score'],87.66)
    def test_valid_operator_quote_enriched_with_original_time(self):
        f=validate_finding(criterion(),finding(),self.turns);self.assertEqual(f['status'],'scored');self.assertEqual(f['evidence'][0]['timestamp'],'00:05')
    def test_fabricated_quote_requires_review(self):
        f=finding();f['evidence'][0]['quote']='Invented!';self.assertEqual(validate_finding(criterion(),f,self.turns)['status'],'needs_review')
    def test_bot_cannot_earn_operator_points(self):
        f=finding();f['evidence'][0]['message_id']='m0001';self.assertEqual(validate_finding(criterion(),f,self.turns)['status'],'needs_review')
    def test_missing_action_uses_operator_span(self):
        f=finding(score=0);f.update(evidence=[],checked_span={'from':'m0002','to':'m0002'})
        self.assertEqual(validate_finding(criterion(),f,self.turns)['status'],'scored')
        f['checked_span']={'from':'m0001','to':'m0001'};self.assertEqual(validate_finding(criterion(),f,self.turns)['status'],'needs_review')
    def test_na_permissions_and_reason(self):
        f=finding();f.update(status='na',score=None,na_reason='Not applicable')
        self.assertEqual(validate_finding(criterion(allow_na=True),f,self.turns)['status'],'na')
        self.assertEqual(validate_finding(criterion(allow_na=False),f,self.turns)['status'],'needs_review')
    def test_binary_midpoint_rejected(self):
        self.assertEqual(validate_finding(criterion(),finding(score=.5),self.turns)['status'],'needs_review')
    def test_lexical_missing_match_is_unknown_not_zero(self):
        c=criterion();c['rule']=dict(mode='regex',scope='all_operator',patterns=['goodbye'])
        self.assertEqual(lexical(c,self.turns)['status'],'needs_review')
    def test_review_preserves_original_and_recalculates(self):
        c=criterion();c['rule']=dict(mode='regex',scope='all_operator',patterns=['Hello']);run=evaluate({'criteria':[c]},self.record)
        before=copy.deepcopy(run['original_findings']);aid=add_appeal(run,'one','Disagree')
        resolve_appeal(run,aid,'Human review','reviewer',finding(score=0))
        self.assertEqual(run['original_findings'],before);self.assertEqual(run['original_summary']['score'],100);self.assertEqual(run['final_summary']['score'],0)
        with self.assertRaises(ValueError):resolve_appeal(run,aid,'Again','reviewer')
    def test_invalid_replacement_does_not_resolve_appeal(self):
        run=evaluate({'criteria':[criterion()]},self.record);aid=add_appeal(run,'one','Review');f=finding();f['evidence'][0]['quote']='not in transcript'
        with self.assertRaises(ValueError):resolve_appeal(run,aid,'Human','reviewer',f)
        self.assertEqual(run['appeals'][0]['status'],'open')
    def test_llm_failure_is_review_not_pass(self):
        with patch('cqa.llm_evaluate',side_effect=RuntimeError('private server info')):run=evaluate({'criteria':[criterion()]},self.record,'llm')
        self.assertIsNone(run['final_summary']['score']);self.assertNotIn('private server info',str(run))
    def test_audio_not_judged_from_text(self):
        c=criterion();c['requires_audio']=True;self.assertEqual(evaluate({'criteria':[c]},self.record)['final_findings'][0]['status'],'needs_review')
    def test_iso_fraction_preserved(self):
        r=copy.deepcopy(self.record);r['channel']='voice call'
        for m in r['transcript']:m['timestamp']='2026-11-03T15:49:13.1900670+00:00'
        self.assertEqual(messages(load_records([r])[0])[0]['timestamp'],r['transcript'][0]['timestamp'])
    def test_no_operator_blocks_scoring(self):
        r=copy.deepcopy(self.record);r['transcript']=r['transcript'][:1];self.assertIsNone(evaluate({'criteria':[criterion()]},r)['final_summary']['score'])
    def test_empty_scale_invalid(self):
        c=criterion();c['max_score']=0
        with self.assertRaises(ValueError):validate_checklist({'criteria':[c]})

if __name__=='__main__':unittest.main()
