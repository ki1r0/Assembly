"""Lossless experiment storage migration and explicit artifact restoration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def request_module(project):
    spec = importlib.util.spec_from_file_location('request_storage', project/'m1_reach/request_storage.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def compress_depth(path):
    path = Path(path)
    target = Path(str(path)+'.gz')
    temporary = Path(str(target)+'.partial')
    before = path.stat()
    if not target.exists():
        try:
            with path.open('rb') as source, gzip.open(temporary, 'wb', compresslevel=1) as out:
                shutil.copyfileobj(source, out, 1024*1024)
            with path.open('rb') as source, gzip.open(temporary, 'rb') as restored:
                while True:
                    a, b = source.read(1024*1024), restored.read(1024*1024)
                    if a != b:
                        raise ValueError(f'Compression verification failed: {path}')
                    if not a:break
            after = path.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
                raise RuntimeError(f'File changed while compressing: {path}')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        with path.open('rb') as source, gzip.open(target, 'rb') as restored:
            while True:
                a, b = source.read(1024*1024), restored.read(1024*1024)
                if a != b:raise ValueError(f'Existing compressed file differs: {path}')
                if not a:break
    path.unlink()
    return before.st_size, target.stat().st_size

def migrate(project):
    module = request_module(project)
    depths = sorted((project/'runs').rglob('*.npy'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes=list(pool.map(compress_depth, depths))
    print(json.dumps({'experiment':project.name,'depth_files':len(sizes),
        'depth_before':sum(x[0] for x in sizes),'depth_after':sum(x[1] for x in sizes)}),flush=True)
    count=before=0
    for path in sorted((project/'runs').rglob('request.json')):
        if not path.parent.name.startswith('request_'):continue
        body=json.loads(path.read_text())
        ref=module.save_request(path.parent,body)
        if module.load_reference(ref)!=body:
            raise ValueError(f'Request reconstruction mismatch: {path}')
        before+=path.stat().st_size;count+=1
        path.unlink()
    print(json.dumps({'experiment':project.name,'requests':count,'request_bytes_before':before}),flush=True)

def restore_depth(path):
    path=Path(path)
    if path.suffix!='.gz':path=Path(str(path)+'.gz')
    target=path.with_suffix('')
    if target.exists():raise FileExistsError(target)
    temporary=Path(str(target)+'.partial')
    try:
        with gzip.open(path,'rb') as source, temporary.open('xb') as out:
            shutil.copyfileobj(source,out,1024*1024)
        temporary.replace(target)
    finally:temporary.unlink(missing_ok=True)
    print(target)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('compact',help='Migrate completed experiment runs; stop writers first')
    request=sub.add_parser('request',help='Restore a full actual request.json from a request folder');request.add_argument('folder')
    depth=sub.add_parser('depth',help='Restore one NPY file alongside its gzip copy');depth.add_argument('path')
    args=p.parse_args()
    if args.command=='compact':
        for project in ROOT.iterdir():
            if (project/'m1_reach/request_storage.py').exists():migrate(project)
    elif args.command=='depth':restore_depth(args.path)
    else:
        folder=Path(args.folder).resolve()
        project=next(parent for parent in folder.parents if (parent/'m1_reach/request_storage.py').exists())
        body=request_module(project).load_request(folder)
        path=folder/'request.json'
        if path.exists():raise FileExistsError(path)
        with path.open('x') as stream:json.dump(body,stream,ensure_ascii=False,indent=2)
        print(path)

if __name__=='__main__':main()
