"""Readable activity index over untouched wire bodies and RPent artifacts."""
from m1_reach.request_storage import load_request
import json,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();run=Path(a.run).resolve();activity=run/'episode/rpent-activity'
text=['# GLM / RPent 活动记录\n','完整请求可通过 ../storage-tools/manage.py request <request目录> 还原。请求引用依赖 experiments/_shared/requests.sqlite3；迁移时须一起携带。实际请求JSON包含图像data URL，完整响应保留服务返回内容；不记录认证header。不补造服务未返回的内部推理。\n']
previous_messages=[]
for folder in sorted(activity.glob('request_*')):
 text += [f'## {folder.name}\n',f'[完整请求记录]({folder.name}/'+('request.json' if (folder/'request.json').exists() else 'request.ref.json')+')\n']
 if (folder/'request.json').exists() or (folder/'request.ref.json').exists():
  body=load_request(folder)
  messages=body.get('messages',[])
  shared=0
  for earlier,current in zip(previous_messages,messages):
   if earlier!=current:break
   shared+=1
  if shared:text += [f'本轮请求复用前 {shared} 条历史消息，以下仅展开新增／变化消息；完整请求可还原。\n']
  previous_messages=messages
  for message in messages[shared:]:
   content=message.get('content')
   if isinstance(content,str):text+=['```text\n'+content+'\n```\n']
   elif isinstance(content,list):
    for part in content:
     if part.get('type')=='text':text+=['```text\n'+part['text']+'\n```\n']
   if message.get('tool_calls'):text+=['```json\n'+json.dumps(message['tool_calls'],ensure_ascii=False,indent=2)+'\n```\n']
 for name in ['response.json','response.body','error.json','metadata.json']:
  f=folder/name
  if f.exists():
   text += [f'[{name}]({folder.name}/{name})\n']
   if name in {'response.json','error.json'}:text+=['```json\n'+f.read_text()+'\n```\n']
   if name=='response.body':
    if '[DONE]' not in f.read_text(errors='replace'):text += ['**此流未记录到 [DONE]；可能被中断，不能按完整模型回答解读。**\n']
    events=[]
    for line in f.read_text(errors='replace').splitlines():
     if line.startswith('data: ') and line[6:]!='[DONE]':
      try:events.append(json.loads(line[6:]))
      except ValueError:pass
    if events:
     (folder/'stream-events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2))
     reasoning=[];reasoning_keys=set();content=[];calls={}
     for event in events:
      for choice in event.get('choices',[]):
       delta=choice.get('delta',{})
       for key in ('reasoning_content','reasoning'):
        if isinstance(delta.get(key),str):reasoning.append(delta[key]);reasoning_keys.add(key)
       if isinstance(delta.get('content'),str):content.append(delta['content'])
       for call in delta.get('tool_calls',[]):
        item=calls.setdefault(call.get('index',0),{'name':'','arguments':''});fn=call.get('function',{})
        item['name']+=fn.get('name','');item['arguments']+=fn.get('arguments','')
     text += ['[实际流式事件全文]('+folder.name+'/stream-events.json)\n','服务实际返回的推理字段 '+', '.join(sorted(reasoning_keys))+'：\n```text\n'+''.join(reasoning)+'\n```\n','输出文本：\n```text\n'+''.join(content)+'\n```\n','工具调用：\n```json\n'+json.dumps(calls,ensure_ascii=False,indent=2)+'\n```\n']
(activity/'GLM_ACTIVITY_LOG_CN.md').write_text('\n'.join(text))
print(activity/'GLM_ACTIVITY_LOG_CN.md')
