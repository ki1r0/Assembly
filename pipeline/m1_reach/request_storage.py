"""Deduplicate recorded request messages across runs; never change HTTP payloads."""
import json
import os
import sqlite3
import zlib
from pathlib import Path

FORMAT = 'rpent-request-ref-v1'

def save_request(folder, body):
    folder = Path(folder)
    store = Path(os.environ['ASSEMBLY_REQUEST_STORE']).expanduser().resolve() if os.environ.get('ASSEMBLY_REQUEST_STORE') else Path(__file__).resolve().parents[2] / '_shared' / 'requests.sqlite3'
    store.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(store, timeout=60) as db:
        db.execute('CREATE TABLE IF NOT EXISTS fragments (id INTEGER PRIMARY KEY, value BLOB NOT NULL UNIQUE)')
        def intern(value):
            data = zlib.compress(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode(), 1)
            db.execute('INSERT OR IGNORE INTO fragments(value) VALUES (?)', (data,))
            return db.execute('SELECT id FROM fragments WHERE value=?', (data,)).fetchone()[0]
        fields = dict(body)
        messages = [intern(m) for m in fields.pop('messages', [])]
        tools = intern(fields.pop('tools')) if 'tools' in fields else None
        ref = {'format': FORMAT, 'store': __import__('os').path.relpath(store, folder),
               'fields': fields, 'messages': messages, 'tools': tools,
               'has_messages': 'messages' in body}
    target = folder / 'request.ref.json'
    temporary = folder / 'request.ref.json.partial'
    temporary.write_text(json.dumps(ref, ensure_ascii=False, indent=2))
    temporary.replace(target)
    return target

def load_request(folder):
    folder = Path(folder)
    raw = folder / 'request.json'
    if raw.exists():
        return json.loads(raw.read_text())
    return load_reference(folder / 'request.ref.json')

def load_reference(path):
    path = Path(path)
    ref = json.loads(path.read_text())
    if ref.get('format') != FORMAT:
        raise ValueError('Unsupported request reference format')
    store = (path.parent / ref['store']).resolve()
    with sqlite3.connect(store.as_uri() + '?mode=ro', uri=True) as db:
        def expand(index):
            row = db.execute('SELECT value FROM fragments WHERE id=?', (index,)).fetchone()
            if row is None:
                raise ValueError(f'Missing request fragment {index} in {store}')
            return json.loads(zlib.decompress(row[0]))
        body = dict(ref['fields'])
        if ref['has_messages']:
            body['messages'] = [expand(i) for i in ref['messages']]
        if ref['tools'] is not None:
            body['tools'] = expand(ref['tools'])
    return body
