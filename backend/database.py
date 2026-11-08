import json
import sqlite3
from contextlib import contextmanager


def encode(value):return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))


class Database:
    def __init__(self,path):
        self.path=path
        with self.connect() as db:
            if db.execute('PRAGMA user_version').fetchone()[0] not in (0,1,2):raise ValueError('Unsupported database version')
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, content_hash TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS checklists (hash TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id),
                    checklist_hash TEXT NOT NULL REFERENCES checklists(hash),cache_key TEXT NOT NULL,status TEXT NOT NULL,
                    created_at TEXT NOT NULL,payload TEXT NOT NULL,input_json TEXT NOT NULL DEFAULT '{}');
                CREATE INDEX IF NOT EXISTS evaluation_cache ON evaluations(cache_key,status);
                CREATE INDEX IF NOT EXISTS evaluation_history ON evaluations(conversation_id,created_at);
                CREATE TABLE IF NOT EXISTS checklist_versions (
                    hash TEXT PRIMARY KEY REFERENCES checklists(hash),parent_hash TEXT REFERENCES checklists(hash),author TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS appeals (
                    id TEXT PRIMARY KEY,evaluation_id TEXT NOT NULL REFERENCES evaluations(id),
                    request_key TEXT NOT NULL,request_hash TEXT NOT NULL,payload TEXT NOT NULL,
                    UNIQUE(evaluation_id,request_key));
                CREATE TABLE IF NOT EXISTS appeal_decisions (
                    appeal_id TEXT PRIMARY KEY REFERENCES appeals(id),request_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,payload TEXT NOT NULL);
                PRAGMA user_version=2;
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30);db.row_factory=sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:yield db
        finally:db.close()

    def get(self,table,column,value):
        with self.connect() as db:
            row=db.execute(f'SELECT payload FROM {table} WHERE {column}=?',(value,)).fetchone()
        return json.loads(row['payload']) if row else None

    def save_evaluation(self,value):
        with self.connect() as db:
            db.execute('UPDATE evaluations SET status=?,payload=? WHERE id=?',
                       (value['status'],encode(value),value['evaluation_id']))
