import unittest
try:
    from costs import estimate
except ImportError:estimate=None


class CostTests(unittest.TestCase):
    def test_cached_input_and_reasoning_are_not_double_counted(self):
        self.assertTrue(callable(estimate),'Cost estimation is missing')
        result=estimate('gpt-6.1-sol',dict(prompt_tokens=100000,cached_input_tokens=20000,completion_tokens=10000,reasoning_tokens=2000))
        self.assertEqual(result['estimated_usd'],.262)
        self.assertFalse(result['invoice_total'])

    def test_unknown_usage_or_model_has_no_fabricated_zero_cost(self):
        self.assertTrue(callable(estimate),'Cost estimation is missing')
        for model,usage in [('unknown',{'prompt_tokens':1,'completion_tokens':2}),('gpt-6.1-sol',{}),
                            ('gpt-6.1-sol',{'prompt_tokens':5,'cached_input_tokens':6,'completion_tokens':2})]:
            self.assertIsNone(estimate(model,usage)['estimated_usd'])

    def test_cache_writes_replace_ordinary_input_rate_and_include_both_lifetimes(self):
        usage=dict(prompt_tokens=12000,cached_input_tokens=4000,completion_tokens=1000,
                   cache_write_input_tokens=7000,cache_write_30m_input_tokens=3000,cache_write_12h_input_tokens=4000)
        self.assertEqual(estimate('gpt-6.1-sol',usage)['estimated_usd'],.0399)

    def test_inconsistent_write_breakdown_is_not_priced_as_zero(self):
        usage=dict(prompt_tokens=12000,cached_input_tokens=4000,completion_tokens=1000,
                   cache_write_input_tokens=7000,cache_write_30m_input_tokens=3000,cache_write_12h_input_tokens=5000)
        self.assertIsNone(estimate('gpt-6.1-sol',usage)['estimated_usd'])


if __name__=='__main__':unittest.main()
