"""Save actual GLM request bodies and response streams; never authentication headers."""
from m1_reach.request_storage import save_request
import os,json,time,itertools
from urllib.parse import urlparse
from pathlib import Path
_installed=False

def install():
    global _installed
    if _installed:return
    _installed=True
    import httpx2 as httpx
    original=httpx.AsyncClient.send;sequence=itertools.count(1)
    class Tee(httpx.AsyncByteStream):
        def __init__(self,stream,folder):self.stream=stream;self.folder=folder;self.parts=[];self.saved=False
        def save(self):
            if self.saved:return
            self.saved=True;body=b''.join(self.parts);(self.folder/'response.body').write_bytes(body)
            try:(self.folder/'response.json').write_text(json.dumps(json.loads(body),ensure_ascii=False,indent=2))
            except (ValueError,UnicodeError):pass
        async def __aiter__(self):
            try:
                async for part in self.stream:self.parts.append(part);yield part
            finally:self.save()
        async def aclose(self):
            try:await self.stream.aclose()
            finally:self.save()
    async def send(self,request,*args,**kwargs):
        if request.url.host!=urlparse(os.environ.get('OPENAI_BASE_URL', '')).hostname:return await original(self,request,*args,**kwargs)
        folder=Path(os.environ['M1_RECORD_DIR'])/f'request_{next(sequence):03d}';folder.mkdir(parents=True)
        # Body only: OpenAI authentication lives in HTTP headers.
        save_request(folder,json.loads(request.content));start=time.monotonic()
        try:
            response=await original(self,request,*args,**kwargs)
            (folder/'metadata.json').write_text(json.dumps({'status_code':response.status_code,'headers_elapsed_s':time.monotonic()-start}))
            tee=Tee(response.stream,folder);response.stream=tee
            if response.is_stream_consumed:tee.parts=[response.content];tee.save()
            return response
        except Exception as exc:
            (folder/'error.json').write_text(json.dumps({'error':repr(exc)}));raise
    httpx.AsyncClient.send=send
