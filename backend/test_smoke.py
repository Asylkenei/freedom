import unittest
from test_support import record

try:from smoke_live import select_cases
except ImportError:select_cases=None


class SelectionTests(unittest.TestCase):
    def test_selects_four_not_full_dataset_and_voice_is_separate(self):
        self.assertTrue(callable(select_cases),'Live smoke selector not implemented')
        records=[record('voice','RU','voice call'),record('ru','RU'),record('kz','KZ'),record('en','EN')]+[record('extra-'+str(i),'RU') for i in range(61)]
        cases=select_cases(records)
        self.assertEqual([label for label,r in cases],['RU_CHAT','KZ_CHAT','EN_CHAT','RU_CALL'])
        self.assertEqual([r['conversation_id'] for label,r in cases],['ru','kz','en','voice'])

    def test_missing_language_is_explicit_error_not_a_fake_case(self):
        self.assertTrue(callable(select_cases),'Live smoke selector not implemented')
        with self.assertRaises(ValueError):select_cases([record('only-ru','RU')])


if __name__=='__main__':unittest.main()
