"""Synthetic provider fixtures, not gold labels or real model predictions."""
import copy
import threading


def record(cid='fixture-en',language='EN',channel='agent chat'):
    stamps=['00:01','00:02','00:03'] if channel=='agent chat' else [f'2026-11-07T10:00:0{i}+05:00' for i in [1,2,3]]
    return dict(conversation_id=cid,language=language,channel=channel,audio_url=None,transcript=[
        dict(timestamp=stamps[0],speaker='operator',text='Hello, my name is Jane.'),
        dict(timestamp=stamps[1],speaker='client',text='Thank you.'),
        dict(timestamp=stamps[2],speaker='operator',text='Anything else? Goodbye.')])


def fixture_findings(checklist,turns):
    return [dict(criterion_id=c['id'],status='scored',score=c['min_score'] if c['type']=='flag' else c['max_score'],
                 reason='Synthetic transport fixture; not a semantic quality assessment.',
                 evidence=[dict(message_id=turns[0]['message_id'],quote=turns[0]['text'])],
                 checked_span={'from':turns[0]['message_id'],'to':turns[-1]['message_id']},
                 critical_error=False if c['type']=='flag' else None,na_reason=None,
                 factual_status='not_needed',knowledge_evidence=[],verification_limits=[])
            for c in checklist['criteria']]


class FixtureProvider:
    def __init__(self):
        self.calls=0;self.active=0;self.max_active=0
        self.release=threading.Event();self.release.set();self.entered=threading.Event()
        self.error=None;self.transform=None;self.lock=threading.Lock()

    def evaluate(self,checklist,conversation,turns,knowledge,prompt):
        with self.lock:
            self.calls+=1;self.active+=1;self.max_active=max(self.max_active,self.active)
        self.entered.set()
        try:
            if not self.release.wait(5):raise RuntimeError('Fixture timeout')
            if self.error:raise self.error
            rows=fixture_findings(checklist,turns)
            if self.transform:rows=self.transform(copy.deepcopy(rows))
            return rows,dict(model='fixture-model',usage={'prompt_tokens':1000,'completion_tokens':200},calls=1,latency_seconds=.01)
        finally:
            with self.lock:self.active-=1
