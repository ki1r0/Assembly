"""GLM -> actual RPent ApiAgentLoop/Toolkit/MemoryManager -> Isaac direct policy."""
import argparse,os,json,time,threading
import settings
from settings import ROOT,configure

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--integration-smoke',action='store_true');p.add_argument('--scene-smoke',action='store_true');p.add_argument('--method',choices=['ours','vlm-dewm'],default='ours');p.add_argument('--budget-s',type=float,default=600)
    p.add_argument('--resume-existing',action='store_true',help='Separate planner recovery segment; no simulator reset or budget extension')
    args=p.parse_args();run=configure(args.run_dir)
    segment=None;cursor=None;history=[]
    if args.resume_existing:
        if args.method!='ours' or args.integration_smoke or args.scene_smoke:p.error('Recovery supports the generic ours rollout only')
        config=json.loads((run/'configuration.json').read_text())
        os.environ.update(ASSEMBLY_TASK=config['task'],ASSEMBLY_ROBOT=config['robot'],ASSEMBLY_CASE=config['scenario'])
        segment='resume-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())
        os.environ['M1_RECORD_DIR']=str(run/'episode'/('rpent-activity-'+segment))
    from common import write
    from m1_reach.control import snapshot,terminate,remaining
    from task_profiles import generic,instruction,GENERIC_SYSTEM
    if args.method=='vlm-dewm':
        if not generic():raise ValueError('VLM-DEWM adaptation requires the shared generic action backend')
        from dewm_toolkit import DEWMToolkit as IsaacToolkit
    elif generic():
        from generic_toolkit import RobotToolkit as IsaacToolkit
    else:
        from toolkit import IsaacToolkit
    from rpent.utils.logging import init_output_dir
    from rpent.planner.api_loop import ApiAgentLoop
    from rpent.dashboard.events import NullDashboardEventSink
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
    started=time.monotonic()
    while not (run/'ipc/reply_0000.json').exists():
        if (run/'private/backend-failed.json').exists():raise RuntimeError('Isaac backend startup failed')
        if time.monotonic()-started>600:raise TimeoutError('Isaac backend startup timeout')
        time.sleep(.2)
    if args.scene_smoke:
        observation=json.loads((run/'ipc/reply_0000.json').read_text())
        rgb=observation.get('rgb_path')
        depth_files=sorted((run/'episode/public').glob('depth_*.npy.gz'))
        rgb_exists=bool(rgb and os.path.isfile(rgb) and os.path.getsize(rgb)>0)
        depth_exists=bool(depth_files and all(path.stat().st_size>0 for path in depth_files))
        passed=rgb_exists and depth_exists
        write(run/'episode/scene-smoke.json',{
            'status':'passed' if passed else 'failed',
            'task':os.environ.get('ASSEMBLY_TASK'),
            'cad_meters_per_source_unit':settings.CAD_METERS_PER_SOURCE_UNIT,
            'robot':observation.get('robot'),
            'observation_id':observation.get('observation_id'),
            'initial_rgb_exists':rgb_exists,
            'initial_depth_file_count':len(depth_files),
            'initial_depth_exists':depth_exists,
            'initial_feature_count':len(observation.get('features',[])),
            'action_count':0,
            'model_called':False,
            'assembly_success_claimed':False,
        })
        terminal='scene_smoke_complete' if passed else 'scene_smoke_incomplete'
        error=None if passed else 'Initial public RGB-D files were missing or empty.'
        terminate(terminal,evidence='Isaac scene initialized and published its first public observation; no model call or robot action was performed.')
        write(run/'episode/planner-summary.json',{'terminal':snapshot()['terminal'],'error':error,'scene_smoke':True,'integration_smoke':False})
        if error:
            raise RuntimeError(error)
        return
    if segment:
        from planner_recovery import continuation_cursor,public_history
        cursor,reply=continuation_cursor(run)
        while not reply.exists():
            if snapshot()['terminal']:raise RuntimeError('Episode ended before pending action completed')
            if remaining()<=.001:terminate('wallclock_budget');raise RuntimeError('Original episode budget expired')
            if (run/'private/backend-failed.json').exists():raise RuntimeError('Backend failed during recovery wait')
            time.sleep(.2)
        history=public_history(run)
    trace_dir=run/'episode'/('rpent-'+segment if segment else 'rpent')
    init_output_dir(trace_dir);snapshot();toolkit=IsaacToolkit(run,trace_dir=trace_dir) if segment else IsaacToolkit(run)
    if segment:
        toolkit.index=cursor;toolkit.latest=json.loads(reply.read_text());toolkit.initial_delivered=True
        write(run/'episode'/('recovery-'+segment+'.json'),{'scope':'New planner conversation continuing the same physical episode; not an uninterrupted rollout','next_command_index':cursor,'public_reply':str(reply),'original_deadline_monotonic':snapshot()['deadline_monotonic'],'simulator_reset':False})
    profile=json.loads((run/'private/robot-profile.json').read_text())
    system=(GENERIC_SYSTEM if generic() else (ROOT/'SYSTEM.md').read_text())+'\nPublic embodiment profile:\n'+json.dumps(profile)+'\nRead-only RPent memory index: '+str(ROOT/'memory/MEMORY.md')
    if args.method=='vlm-dewm':
        from dewm_toolkit import DEWM_SYSTEM
        system+='\n'+DEWM_SYSTEM+'\nShared action capability schemas:\n'+toolkit.capability_context()
    user=instruction()
    if segment:user+='\nPlanner recovery segment after process interruption. The same physical episode continues, with no reset. First observe the current scene. Prior public commands and measured outcomes (not guaranteed current object state):\n'+json.dumps(history)
    if args.integration_smoke:
        user='Integration wiring test only, not a peg-insertion rollout. In order: call observe; call read_text_file on '+str(ROOT/'memory/MEMORY.md')+'; call move_relative with translation_m=[-0.005,0.0,0.0] and no rotation; call set_gripper(width_m=0.08); then stop with reason integration_test_complete. Do not perform the assembly task or any additional movement.'
        if args.method=='vlm-dewm':
            user='Integration wiring only, not assembly. Call observe. Submit one execute_ert for move_relative with translation_m=[-0.005,0,0], empty world_belief and causal_assump lists, confidence 0.5 and a recovery fallback to observe and stop. Then observe and verify_action (empty expectations must not commit semantic progress). Finally stop with reason integration_test_complete. Do not perform other movements or claim task success.'
    write(run/'private'/('rendered-context-'+segment+'.json' if segment else 'rendered-context.json'),{'system':system,'user':user,'profile':profile})
    model=OpenAIChatModel(os.environ['OPENAI_MODEL'],profile=({'context_window':int(os.environ['ASSEMBLY_CONTEXT_WINDOW'])} if os.environ.get('ASSEMBLY_CONTEXT_WINDOW') else None),settings={'openai_reasoning_effort':'low'},provider=OpenAIProvider(base_url=os.environ['OPENAI_BASE_URL'],api_key=os.environ['OPENAI_API_KEY']))
    done=threading.Event()
    def watchdog():
        while not done.wait(.2):
            state=snapshot()
            if state['deadline_monotonic'] is not None and remaining()<=.001:terminate('wallclock_budget');return
    threading.Thread(target=watchdog,daemon=True).start();result=None
    try:
        result=ApiAgentLoop(model=model,dashboard_events=NullDashboardEventSink(),max_tokens=int(os.environ.get("ASSEMBLY_MAX_TOKENS","16384")),reasoning_effort='none',timeout_s=args.budget_s).solve(system_prompt=system,user_message=user,toolkit=toolkit,max_turns=100)
        write(run/'episode'/('planner-result-'+segment+'.json' if segment else 'planner-result.json'),{'finish_result':result.finish_result,'stats':result.stats,'error':result.error,'messages':result.messages})
    finally:
        done.set()
        if snapshot()['terminal'] is None:terminate('planner_returned',error=result.error if result else 'exception')
        toolkit.close();write(run/'episode'/('planner-summary-'+segment+'.json' if segment else 'planner-summary.json'),{'terminal':snapshot()['terminal'],'error':result.error if result else 'exception','integration_smoke':args.integration_smoke})
    if result and result.error:raise RuntimeError(result.error)
if __name__=='__main__':main()
