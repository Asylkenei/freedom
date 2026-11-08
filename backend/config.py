"""Explicit server configuration. Environment variables override .env, even when empty."""
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

ROOT=Path(__file__).resolve().parent
DEFAULT_ORIGINS=('http://localhost:3000','http://127.0.0.1:3000','http://localhost:5173','http://127.0.0.1:5173')


@dataclass
class Settings:
    api_key:str=field(default='',repr=False)
    model:str='gpt-6.1-sol'
    base_url:str='https://api.openai.com/v1'
    data_dir:Path=ROOT/'data'
    checklist_file:Path=ROOT/'checklist.official.json'
    knowledge_file:Path|None=None
    timeout_seconds:float=120
    allowed_origins:tuple[str,...]=DEFAULT_ORIGINS
    pipeline:str='single'
    small_model:str=''
    review_model:str=''
    review_all:bool=False
    audit_rate:float=0.1
    audit_seed:str='cqa-audit-v1'
    sensitive_criteria:tuple[str,...]=('c03',)

    def __post_init__(self):
        self.data_dir=Path(self.data_dir).resolve();self.checklist_file=Path(self.checklist_file).resolve()
        if self.knowledge_file:self.knowledge_file=Path(self.knowledge_file).resolve()
        url=urlsplit(self.base_url)
        if url.scheme not in ('http','https') or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError('CQA_BASE_URL must be an HTTP(S) API base URL without credentials or query parameters')
        if not self.model.strip() or not 0<self.timeout_seconds<=600:raise ValueError('Invalid CQA model or timeout')
        if self.pipeline not in ('single','cascade') or not 0<=self.audit_rate<=1:
            raise ValueError('Invalid CQA pipeline or audit rate')
        for origin in self.allowed_origins:
            parsed=urlsplit(origin)
            if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
                raise ValueError('CQA_ALLOWED_ORIGINS must contain exact HTTP(S) origins')

    @classmethod
    def from_env(cls,root=ROOT):
        root=Path(root)
        values={**dotenv_values(root/'.env'),**os.environ}
        def path(name,default):
            value=Path(values.get(name) or default)
            return value if value.is_absolute() else root/value
        def boolean(name):
            value=(values.get(name) or 'false').lower()
            if value not in ('1','0','true','false'):raise ValueError('Invalid boolean setting: '+name)
            return value in ('1','true')
        return cls(api_key=(values.get('CQA_API_KEY') or '').strip(),model=values.get('CQA_MODEL') or 'gpt-6.1-sol',
                   base_url=values.get('CQA_BASE_URL') or 'https://api.openai.com/v1',data_dir=path('CQA_DATA_DIR','data'),
                   checklist_file=path('CQA_CHECKLIST_FILE','checklist.official.json'),
                   knowledge_file=path('CQA_KNOWLEDGE_FILE','') if values.get('CQA_KNOWLEDGE_FILE') else None,
                   timeout_seconds=float(values.get('CQA_TIMEOUT_SECONDS') or '120'),
                   allowed_origins=tuple(s.strip() for s in (values.get('CQA_ALLOWED_ORIGINS') or ','.join(DEFAULT_ORIGINS)).split(',') if s.strip()),
                   pipeline=values.get('CQA_PIPELINE') or 'single',small_model=values.get('CQA_SMALL_MODEL') or '',
                   review_model=values.get('CQA_REVIEW_MODEL') or '',review_all=boolean('CQA_REVIEW_ALL'),
                   audit_rate=float(values.get('CQA_AUDIT_RATE') or '.1'),audit_seed=values.get('CQA_AUDIT_SEED') or 'cqa-audit-v1',
                   sensitive_criteria=tuple(s.strip() for s in (values.get('CQA_SENSITIVE_CRITERIA') if 'CQA_SENSITIVE_CRITERIA' in values else 'c03').split(',') if s.strip()))
