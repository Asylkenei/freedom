"""Dated standard-tier token estimates, not invoices or live pricing lookups."""
from decimal import Decimal

AS_OF='2026-11-08'
SOURCE='https://developers.openai.com/api/docs/pricing'
RATES={'gpt-6.1-sol':('2','0.10','10'),'gpt-6-luna':('0.10','0.01','0.50')}
WRITE_RATES={'gpt-6.1-sol':('2.50','5'),'gpt-6-luna':('0.125','0.25')}


def estimate(model,usage):
    result=dict(estimated_usd=None,invoice_total=False,pricing_as_of=AS_OF,pricing_source=SOURCE,
                cache_write_usage_available=False,
                assumptions='Standard tier, <=272K input tokens. Includes cache writes when reported; otherwise their uplift is unknown. Excludes regional and nonstandard-tier charges. Not an invoice.')
    if model not in RATES or not isinstance(usage,dict):return result
    values=[usage.get('prompt_tokens'),usage.get('cached_input_tokens',0),usage.get('completion_tokens')]
    if any(type(v) is not int or v<0 for v in values):return result
    inp,cached,out=values
    if cached>inp or inp>272000:return result
    written=usage.get('cache_write_input_tokens',0)
    if type(written) is not int or written<0 or cached+written>inp:return result
    if 'cache_write_input_tokens' in usage:
        w30=usage.get('cache_write_30m_input_tokens',0 if written==0 else None)
        w12=usage.get('cache_write_12h_input_tokens',0 if written==0 else None)
        if any(type(v) is not int or v<0 for v in (w30,w12)) or w30+w12!=written:return result
        result['cache_write_usage_available']=True
    else:w30=w12=0
    rates=[Decimal(v) for v in RATES[model]]
    write_rates=[Decimal(v) for v in WRITE_RATES[model]]
    result['estimated_usd']=float(((inp-cached-written)*rates[0]+cached*rates[1]+w30*write_rates[0]+w12*write_rates[1]+out*rates[2])/1_000_000)
    return result
