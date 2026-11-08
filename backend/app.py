"""Local JSON API. Run: python -m uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1"""
from contextlib import asynccontextmanager
from typing import Any,Literal

from fastapi import FastAPI,Request,Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict,Field,StrictBool,field_validator

from config import Settings
from service import Service,ServiceError


class MessageInput(BaseModel):
    timestamp:str=Field(min_length=1,description='Original MM:SS for chats or timezone-aware ISO 8601 for calls. Never converted.')
    speaker:Literal['client','operator','bot','system']
    text:str=Field(min_length=1,max_length=100000)


class ConversationInput(BaseModel):
    conversation_id:str=Field(min_length=1,max_length=200)
    language:Literal['RU','KZ','EN']
    channel:Literal['agent chat','voice call']
    transcript:list[MessageInput]=Field(min_length=1,max_length=10000)
    audio_url:str|None=None


class ImportData(BaseModel):
    records:list[ConversationInput]=Field(min_length=1,max_length=2000)
    total_records:int|None=Field(default=None,ge=0)
    dataset_group_id:str|None=None


class OrganizerResponse(BaseModel):
    isSuccess:StrictBool
    data:ImportData
    error:str=''


class EvaluationRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    conversation_id:str=Field(min_length=1,max_length=200)
    checklist_hash:str=Field(pattern=r'^[0-9a-f]{64}$')
    force:StrictBool=Field(default=False,description='Creates a new paid attempt only after previous work is finished. An active duplicate always returns the same ID.')
    pipeline_hash:str|None=Field(default=None,pattern=r'^[0-9a-f]{64}$',description='Optional experiment guard; changed server configuration returns 409 before any model call.')


class ErrorDetail(BaseModel):
    code:str
    message:str


class WorkflowRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')

    @field_validator('author','comment','reason','reviewer','client_request_id',check_fields=False)
    @classmethod
    def nonblank(cls,value):
        if not value.strip():raise ValueError('Nonblank value required')
        return value.strip()


class ChecklistRequest(WorkflowRequest):
    checklist:dict[str,Any]
    parent_hash:str|None=Field(default=None,pattern=r'^[0-9a-f]{64}$')
    author:str=Field(min_length=1,max_length=200)


class AppealRequest(WorkflowRequest):
    criterion_id:str=Field(min_length=1,max_length=200)
    comment:str=Field(min_length=1,max_length=10000)
    author:str=Field(min_length=1,max_length=200)
    client_request_id:str=Field(min_length=1,max_length=200)


class DecisionRequest(WorkflowRequest):
    decision:Literal['uphold','change']
    reason:str=Field(min_length=1,max_length=10000)
    reviewer:str=Field(min_length=1,max_length=200)
    replacement:dict[str,Any]|None=None
    expected_revision:int=Field(ge=0,strict=True)
    client_request_id:str=Field(min_length=1,max_length=200)


class EvaluationResponse(BaseModel):
    evaluation_id:str
    conversation_id:str
    checklist_hash:str
    status:Literal['queued','running','completed','needs_review','error']
    result:dict[str,Any]|None=Field(description='Baseline run. Use result.final_summary.score for the main score; null is valid. Exact fields and synthetic samples are supplied in FRONTEND_INTEGRATION.md and examples/.')
    error:ErrorDetail|None
    inference:dict[str,Any]=Field(description='Requested/actual model, normalized usage, calls and duration; no credentials.')
    created_at:str
    updated_at:str
    revision:int=0
    review_status:Literal['none','open','resolved']='none'
    pipeline:str='single'
    stage:str|None=None
    stages:list[dict[str,Any]]=Field(default_factory=list)
    pipeline_snapshot:dict[str,Any]|None=None
    routing:dict[str,Any]|None=None
    disagreements:list[str]=Field(default_factory=list)


def create_app(settings=None,provider=None):
    settings=settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application):
        service=Service(settings,provider)
        application.state.service=service
        try:yield
        finally:service.close()

    application=FastAPI(title='Freedom CQA Backend',version='2.0.0',lifespan=lifespan,
        description='Local transcript evaluation, bounded LLM cascade, immutable checklists, evidence validation, SQLite and human appeal decisions. No frontend or ASR.')
    application.add_middleware(CORSMiddleware,allow_origins=list(settings.allowed_origins),allow_credentials=False,
                               allow_methods=['GET','POST','OPTIONS'],allow_headers=['Content-Type'])

    @application.exception_handler(ServiceError)
    async def service_error(request,error):
        return JSONResponse(status_code=error.status_code,content={'error':{'code':error.code,'message':str(error)}})

    @application.exception_handler(RequestValidationError)
    async def invalid_body(request,error):
        # Do not echo arbitrary transcript text or submitted values in validation errors.
        return JSONResponse(status_code=422,content={'error':{'code':'invalid_request','message':'Некорректные поля JSON-запроса. Проверьте схему в /docs.',
            'fields':[{'location':list(item['loc']),'type':item['type']} for item in error.errors()]}})

    def service():return application.state.service

    @application.get('/api/health',tags=['System'])
    def health():
        """Configuration presence is not a live provider access/balance check."""
        return service().health()

    @application.post('/api/conversations/import',tags=['Conversations'])
    def import_conversations(payload:ImportData|OrganizerResponse):
        """Atomically import organizer JSON. Same ID/content is skipped; changed content returns 409."""
        return service().import_records(payload.model_dump())

    @application.get('/api/conversations',tags=['Conversations'])
    def list_conversations():return service().list_conversations()

    @application.get('/api/conversations/{conversation_id:path}',tags=['Conversations'])
    def get_conversation(conversation_id:str):
        """Immutable source messages with stable message_id and evaluation history."""
        return service().get_conversation(conversation_id)

    @application.get('/api/checklists',tags=['Checklists'])
    def checklists():return service().list_checklists()

    @application.get('/api/checklists/{checklist_hash}',tags=['Checklists'])
    def get_checklist(checklist_hash:str):return service().get_checklist(checklist_hash)

    @application.post('/api/checklists',tags=['Checklists'],status_code=201)
    def save_checklist(payload:ChecklistRequest,response:Response):
        value,status=service().create_checklist(**payload.model_dump());response.status_code=status
        return value

    @application.get('/api/evaluations/{evaluation_id}/appeals',tags=['Appeals'])
    def list_appeals(evaluation_id:str):return service().list_appeals(evaluation_id)

    @application.post('/api/evaluations/{evaluation_id}/appeals',tags=['Appeals'],status_code=201)
    def create_appeal(evaluation_id:str,payload:AppealRequest,response:Response):
        value,status=service().create_appeal(evaluation_id,**payload.model_dump());response.status_code=status
        return value

    @application.post('/api/appeals/{appeal_id}/resolve',response_model=EvaluationResponse,tags=['Appeals'])
    def resolve_appeal(appeal_id:str,payload:DecisionRequest):
        return service().resolve_appeal(appeal_id,**payload.model_dump())

    @application.post('/api/evaluations',response_model=EvaluationResponse,status_code=202,
                      responses={200:{'description':'Cached terminal evaluation; no new model call.'},503:{'description':'Model not configured or service is shutting down.'}},tags=['Evaluations'])
    def start_evaluation(payload:EvaluationRequest,response:Response):
        value,status=service().start_evaluation(payload.conversation_id,payload.checklist_hash,payload.force,payload.pipeline_hash)
        response.status_code=status
        response.headers['Location']='/api/evaluations/'+value['evaluation_id']
        return value

    @application.get('/api/evaluations/{evaluation_id}',response_model=EvaluationResponse,tags=['Evaluations'])
    def get_evaluation(evaluation_id:str):return service().get_evaluation(evaluation_id)

    return application


app=create_app()
