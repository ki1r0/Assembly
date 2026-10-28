"""Check exact depth bytes and full request reconstruction before compacting evidence."""
import gzip
import importlib.util
import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest
import settings
from runtime.assembly_harness.depth_io import save_depth,load_depth
from m1_reach import request_storage

def test_depth_exact_bytes_and_legacy_fallback(tmp_path):
    data=np.array([0x7fc00001,0x7f800000,0xff800000,0x80000000,0x3f812345],dtype=np.uint32).view(np.float32)
    legacy=tmp_path/'depth.npy';np.save(legacy,data,allow_pickle=False)
    compressed=save_depth(tmp_path/'depth-new.npy',data)
    assert load_depth(compressed).tobytes()==data.tobytes()
    with gzip.open(compressed,'rb') as stream:assert stream.read()==legacy.read_bytes()
    legacy.unlink();compressed.rename(str(legacy)+'.gz')
    assert load_depth(legacy).tobytes()==data.tobytes()

def test_request_reconstruction_and_reuse(tmp_path,monkeypatch):
    monkeypatch.setattr(request_storage,'__file__',str(tmp_path/'project/m1_reach/request_storage.py'))
    folder=tmp_path/'project/runs/run/request_001';folder.mkdir(parents=True)
    body={'model':'example','messages':[{'role':'system','content':'指南'}, {'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,ABC'}}]}],'tools':[],'stream':True}
    ref=request_storage.save_request(folder,body)
    assert request_storage.load_reference(ref)==body
    count=lambda:sqlite3.connect(tmp_path/'_shared/requests.sqlite3').execute('select count(*) from fragments').fetchone()[0]
    first=count();request_storage.save_request(folder,body);assert count()==first
    second=folder.with_name('request_002');second.mkdir()
    changed=dict(body,messages=body['messages']+[{'role':'assistant','content':'下一步'}]);request_storage.save_request(second,changed)
    assert count()==first+1
    assert request_storage.load_request(second)==changed
    (folder/'request.json').write_text(json.dumps({'legacy':True}))
    assert request_storage.load_request(folder)=={'legacy':True}
    (tmp_path/'_shared/requests.sqlite3').unlink()
    with pytest.raises(sqlite3.OperationalError):request_storage.load_request(second)

def test_migration_retains_original_on_existing_mismatch(tmp_path):
    spec=importlib.util.spec_from_file_location('manage',Path(settings.ROOT).parent/'storage-tools/manage.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=tmp_path/'depth.npy';original.write_bytes(b'original')
    with gzip.open(str(original)+'.gz','wb') as stream:stream.write(b'different')
    with pytest.raises(ValueError):module.compress_depth(original)
    assert original.read_bytes()==b'original'
    Path(str(original)+'.gz').unlink()
    assert module.compress_depth(original)[0]==8
    module.restore_depth(str(original)+'.gz');assert original.read_bytes()==b'original'

def test_http_recorder_sends_original_payload_and_excludes_auth(tmp_path,monkeypatch):
    import asyncio
    import httpx2 as httpx
    from m1_reach import recording
    monkeypatch.setattr(request_storage,'__file__',str(tmp_path/'project/m1_reach/request_storage.py'))
    monkeypatch.setattr(recording,'_installed',False)
    monkeypatch.setenv('OPENAI_BASE_URL','https://example.test/v1')
    monkeypatch.setenv('M1_RECORD_DIR',str(tmp_path/'requests'))
    body={'model':'example','messages':[{'role':'user','content':'test'}]}
    received=[]
    async def send(client,request,*args,**kwargs):
        received.append(json.loads(request.content))
        return httpx.Response(200,content=b'{"ok":true}',request=request)
    monkeypatch.setattr(httpx.AsyncClient,'send',send)
    recording.install()
    async def exercise():
        async with httpx.AsyncClient() as client:
            await client.post('https://example.test/v1/chat/completions',json=body,headers={'Authorization':'Bearer TEST_SECRET_NOT_TO_RECORD'})
    asyncio.run(exercise())
    folder=tmp_path/'requests/request_001'
    assert received==[body]
    assert request_storage.load_request(folder)==body
    assert json.loads((folder/'response.json').read_text())=={'ok':True}
    assert b'TEST_SECRET_NOT_TO_RECORD' not in (tmp_path/'_shared/requests.sqlite3').read_bytes()
    assert 'TEST_SECRET_NOT_TO_RECORD' not in (folder/'request.ref.json').read_text()


def test_request_store_can_be_shared_across_implementation_checkouts(tmp_path,monkeypatch):
    from m1_reach.request_storage import save_request,load_request
    store=tmp_path/'shared'/'requests.sqlite3';monkeypatch.setenv('ASSEMBLY_REQUEST_STORE',str(store))
    first=tmp_path/'run-a';second=tmp_path/'run-b';first.mkdir();second.mkdir()
    body={'messages':[{'role':'user','content':'Public observation'}],'model':'test'}
    save_request(first,body);save_request(second,body)
    assert load_request(first)==body and load_request(second)==body
    assert store.is_file()
