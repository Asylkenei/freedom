import copy
import csv
import tempfile
import unittest
from pathlib import Path

import cqa
from config import ROOT
from test_support import record, fixture_findings
try:
    import annotation
except ImportError:
    annotation=None


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(annotation,'Annotation tooling is not implemented')
        self.checklist=cqa.read_json(ROOT/'checklist.official.json')
        self.records=[]
        for language,channel,count in [('RU','agent chat',30),('KZ','agent chat',15),('EN','agent chat',15),('RU','voice call',5)]:
            for i in range(count):
                r=record(f'{language}-{channel}-{i}',language,channel)
                r['transcript'][1]['text']=f'Topic {language} {channel} number {i}.'
                self.records.append(r)

    def labelled(self,records=None):
        records=records or self.records[:1]
        value=annotation.make_template(self.checklist,records)
        for row,r in zip(value['records'],records):
            row['findings']=fixture_findings(self.checklist,cqa.messages(r))
            for f in row['findings']:
                f.update(annotator_id='human-a',annotation_state='adjudicated',ai_assisted=False)
        return value

    def test_split_is_repeatable_stratified_and_blind(self):
        first=annotation.prepare(self.records,self.checklist,seed='fixture-seed')
        second=annotation.prepare(list(reversed(self.records)),self.checklist,seed='fixture-seed')
        split=lambda p:{r['conversation_id']:r['split'] for r in p['manifest']['records']}
        self.assertEqual(split(first),split(second))
        self.assertEqual(len(first['development']['records']),45)
        self.assertEqual(len(first['test']['records']),20)
        test_counts={}
        for r in first['test']['records']:
            k=(r['language'],r['channel']);test_counts[k]=test_counts.get(k,0)+1
            self.assertTrue(all(f['status'] is None and f['score'] is None for f in r['findings']))
            self.assertNotIn('original_findings',r)
        self.assertEqual(test_counts,{('RU','agent chat'):8,('KZ','agent chat'):5,('EN','agent chat'):5,('RU','voice call'):2})

    def test_draft_is_rejected_and_source_changes_invalidate_annotation(self):
        draft=annotation.make_template(self.checklist,self.records[:1])
        with self.assertRaises(ValueError):annotation.validate_annotations(draft,self.records[:1],self.checklist)
        labelled=self.labelled();changed=copy.deepcopy(self.records[:1])
        changed[0]['transcript'][0]['text']='Edited source'
        with self.assertRaises(ValueError):annotation.validate_annotations(labelled,changed,self.checklist)

    def test_unresolved_fact_is_preserved_without_numeric_total(self):
        value=self.labelled()
        value['records'][0]['findings'][3].update(status='unresolved',score=None,reason='No decisive source.',evidence=[])
        gold=annotation.validate_annotations(value,self.records[:1],self.checklist)
        self.assertIsNone(gold['records'][0]['total_score'])
        self.assertEqual(gold['records'][0]['findings'][3]['status'],'unresolved')
        self.assertEqual(gold['annotation_state'],'adjudicated')

    def test_invalid_quote_and_forbidden_na_are_rejected(self):
        value=self.labelled();value['records'][0]['findings'][0]['evidence'][0]['quote']='Not present'
        with self.assertRaises(ValueError):annotation.validate_annotations(value,self.records[:1],self.checklist)
        value=self.labelled();value['records'][0]['findings'][0].update(status='na',score=None,na_reason='Not applicable')
        with self.assertRaises(ValueError):annotation.validate_annotations(value,self.records[:1],self.checklist)

    def test_independent_agreement_lists_actual_disagreement(self):
        a=self.labelled();b=copy.deepcopy(a)
        b['records'][0]['findings'][6]['score']=0
        b['records'][0]['findings'][6]['annotator_id']='human-b'
        report=annotation.agreement(a,b,self.records[:1],self.checklist)
        self.assertEqual(report['exact_agreement'],.875)
        self.assertEqual(report['disagreements'][0]['criterion_id'],'c07')

    def test_csv_edit_is_imported_without_losing_unicode_or_evidence(self):
        value=self.labelled()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.csv'
            annotation.to_csv(value,path)
            with path.open(encoding='utf-8-sig',newline='') as stream:
                rows=list(csv.DictReader(stream));fields=list(rows[0])
            rows[0]['score']='0';rows[0]['reason']='Проверено, вручную\nsecond line'
            with path.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            result=annotation.from_csv(value,path)
            gold=annotation.validate_annotations(result,self.records[:1],self.checklist)
            self.assertEqual(gold['records'][0]['findings'][0]['score'],0)
            self.assertEqual(gold['records'][0]['findings'][0]['reason'],'Проверено, вручную\nsecond line')


if __name__=='__main__':unittest.main()
