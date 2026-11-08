import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock,patch
from cqa import aggregate,evaluate,messages,read_json,validate_checklist,validate_finding,validate_knowledge

class OfficialChecklistTests(unittest.TestCase):
    def setUp(self):
        self.checklist=read_json(Path(__file__).with_name('checklist.official.json'));self.cs={c['id']:c for c in self.checklist['criteria']}
        self.record=dict(conversation_id='fixture',channel='agent chat',language='EN',audio_url=None,transcript=[
            dict(speaker='operator',text='Hello, my name is Jane.',timestamp='00:01'),dict(speaker='system',text='Identity verification completed.',timestamp='00:02'),dict(speaker='operator',text='Account checked. Goodbye.',timestamp='00:03')])
        self.turns=messages(self.record)
    def finding(self,cid,score,quote='Hello',mid='m0001'):
        return dict(criterion_id=cid,status='scored',score=score,reason='Test fixture',evidence=[dict(message_id=mid,quote=quote)],critical_error=None,checked_span={'from':'m0001','to':'m0003'},na_reason=None)
    def all_pass(self):
        return [validate_finding(c,self.finding(c['id'],0 if c['type']=='flag' else c['max_score']),self.turns) for c in self.checklist['criteria']]
    def test_official_has_seven_weighted_and_one_zero_weight_flag(self):
        validate_checklist(self.checklist);self.assertEqual(sum(c['weight'] for c in self.cs.values()),100);self.assertEqual(self.cs['c08']['type'],'flag')
    def test_flag_true_does_not_zero_or_raise_total(self):
        fs=self.all_pass();fs[-1]=validate_finding(self.cs['c08'],self.finding('c08',1),self.turns);s=aggregate(self.checklist,fs)
        self.assertEqual(s['score'],100);self.assertTrue(s['critical_error']);self.assertEqual(fs[-1]['answer'],'Да')
    def test_flag_false_is_explicit_not_unknown(self):
        s=aggregate(self.checklist,self.all_pass());self.assertFalse(s['critical_error']);self.assertTrue(s['assessment_complete'])
    def test_unresolved_flag_retains_score_but_not_complete_assessment(self):
        fs=self.all_pass();fs[-1].update(status='needs_review',score=None,critical_error=None);s=aggregate(self.checklist,fs)
        self.assertEqual(s['score'],100);self.assertFalse(s['assessment_complete']);self.assertEqual(s['unresolved_flags'],['c08'])
    def test_discrete_scale_steps_enforced(self):
        for cid,x in [('c04',17),('c06',8),('c06',5)]:
            with self.subTest(cid=cid,score=x):self.assertEqual(validate_finding(self.cs[cid],self.finding(cid,x),self.turns)['status'],'needs_review')
        for x in [0,4,7,10]:self.assertEqual(validate_finding(self.cs['c06'],self.finding('c06',x),self.turns)['status'],'scored')
    def test_official_na_and_scale_example(self):
        fs=self.all_pass()
        for i in [1,2]:fs[i].update(status='na',score=None,na_reason='Not applicable')
        fs[3]['score']=15;fs[5]['score']=7;fs[6]['score']=0;s=aggregate(self.checklist,fs)
        self.assertEqual(s['score'],77.69);self.assertEqual(next(i for i in s['criteria'] if i['criterion_id']=='c06')['weighted_points'],10.5)
    def test_system_can_confirm_identity_but_not_operator_greeting(self):
        f=self.finding('c03',1,'Identity verification completed.','m0002');self.assertEqual(validate_finding(self.cs['c03'],f,self.turns)['status'],'scored')
        f['criterion_id']='c01';self.assertEqual(validate_finding(self.cs['c01'],f,self.turns)['status'],'needs_review')
    def test_late_greeting_evidence_not_accepted(self):
        self.assertEqual(validate_finding(self.cs['c01'],self.finding('c01',1,'Goodbye','m0003'),self.turns)['status'],'needs_review')
    def test_no_gross_errors_requires_full_operator_span(self):
        f=self.finding('c08',0);f['checked_span']={'from':'m0001','to':'m0001'};self.assertEqual(validate_finding(self.cs['c08'],f,self.turns)['status'],'needs_review')
    def test_voice_c06_not_marked_as_audio_only(self):
        r=copy.deepcopy(self.record);r['channel']='voice call';fake=[self.finding(c['id'],0 if c['type']=='flag' else c['max_score']) for c in self.cs.values()]
        with patch('cqa.llm_evaluate',return_value=(fake,{'calls':1})):run=evaluate(self.checklist,r,'llm')
        f=next(f for f in run['original_findings'] if f['criterion_id']=='c06');self.assertEqual(f['status'],'scored');self.assertTrue(any('Аудио не анализировалось' in s for s in f['verification_limits']))
    def test_product_verification_needs_exact_source_quote(self):
        kb={'sources':[{'source_id':'policy','text':'Test policy text.'}]};f=self.finding('c04',20);f['factual_status']='supported'
        self.assertEqual(validate_finding(self.cs['c04'],f,self.turns,kb)['status'],'needs_review')
        f['knowledge_evidence']=[{'source_id':'policy','quote':'Test policy'}];self.assertEqual(validate_finding(self.cs['c04'],f,self.turns,kb)['status'],'scored')
        f['knowledge_evidence'][0]['quote']='Invented policy';self.assertEqual(validate_finding(self.cs['c04'],f,self.turns,kb)['status'],'needs_review')
    def test_duplicate_knowledge_sources_rejected(self):
        with self.assertRaises(ValueError):validate_knowledge({'sources':[{'source_id':'x','text':'a'},{'source_id':'x','text':'b'}]})
    def test_forbidden_na_does_not_erase_weight(self):
        f=self.finding('c06',10);f.update(status='na',score=None,na_reason='No audio');self.assertEqual(validate_finding(self.cs['c06'],f,self.turns)['status'],'needs_review')
    def test_actual_adapter_serializes_official_policy_without_audio_url(self):
        raw=[self.finding(c['id'],0 if c['type']=='flag' else c['max_score']) for c in self.cs.values()]
        response=dict(status='completed',model='fixture-model',usage=dict(input_tokens=10,output_tokens=20),output=[{'type':'message','content':[{'type':'output_text','text':json.dumps({'findings':raw})}]}])
        mock_response=MagicMock();mock_response.__enter__.return_value.read.return_value=json.dumps(response).encode()
        record=copy.deepcopy(self.record);record['audio_url']='https://private.invalid/recording?signature=secret'
        with patch.dict(os.environ,{'CQA_API_KEY':'fixture-key'}),patch('cqa.urllib.request.urlopen',return_value=mock_response) as call:run=evaluate(self.checklist,record,'llm','fixture-model','https://provider.invalid/v1')
        payload=json.loads(call.call_args.args[0].data);body=json.loads(payload['input'][0]['content']);self.assertEqual(len(body['criteria']),8)
        self.assertTrue(call.call_args.args[0].full_url.endswith('/responses'));self.assertTrue(payload['text']['format']['strict']);self.assertNotIn('temperature',payload)
        self.assertIn('порог первой реакции в 3 минуты',body['checklist_policy']['excluded']);self.assertNotIn('private.invalid',json.dumps(payload))
        self.assertEqual(run['original_summary']['score'],100);self.assertFalse(run['original_summary']['critical_error']);self.assertEqual(run['inference']['usage']['completion_tokens'],20)

if __name__=='__main__':unittest.main()
