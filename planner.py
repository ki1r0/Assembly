"""GLM -> actual RPent ApiAgentLoop/Toolkit/MemoryManager -> Isaac direct policy."""
import argparse,os,json,time,threading
import settings
from settings import ROOT,configure

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--integration-smoke',action='store_true');p.add_argument('--budget-s',type=float,default=600);args=p.parse_args();run=configure(args.run_dir)
    from common import write
    from m1_reach.control import snapshot,terminate,remaining
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
    init_output_dir(run/'episode/rpent');snapshot();toolkit=IsaacToolkit(run)
    profile=json.loads((run/'private/robot-profile.json').read_text())
    system=(ROOT/'SYSTEM.md').read_text()+'\nPublic embodiment profile:\n'+json.dumps(profile)+'\nRead-only RPent memory index: '+str(ROOT/'memory/MEMORY.md')
    user='Pick up the loose bolt, insert its shaft through the upper-right bolt hole in the raised hub-cover annulus (about two o-clock from its large center bore in the fixed task-camera view) into the corresponding casing opening, seat the head on the cover, release, and withdraw the gripper. Complete the assembly task. Call observe first.'
    if args.integration_smoke:
        user='Integration wiring test only, not a peg-insertion rollout. In order: call observe; call read_text_file on '+str(ROOT/'memory/MEMORY.md')+'; call move_relative with translation_m=[-0.005,0.0,0.0] and no rotation; call set_gripper(width_m=0.08); then stop with reason integration_test_complete. Do not perform the assembly task or any additional movement.'
    write(run/'private/rendered-context.json',{'system':system,'user':user,'profile':profile})
    model=OpenAIChatModel(os.environ['OPENAI_MODEL'],settings={'openai_reasoning_effort':'low'},provider=OpenAIProvider(base_url=os.environ['OPENAI_BASE_URL'],api_key=os.environ['OPENAI_API_KEY']))
    done=threading.Event()
    def watchdog():
        while not done.wait(.2):
            state=snapshot()
            if state['deadline_monotonic'] is not None and remaining()<=.001:terminate('wallclock_budget');return
    threading.Thread(target=watchdog,daemon=True).start();result=None
    try:
        result=ApiAgentLoop(model=model,dashboard_events=NullDashboardEventSink(),max_tokens=16384,reasoning_effort='none',timeout_s=args.budget_s).solve(system_prompt=system,user_message=user,toolkit=toolkit,max_turns=100)
        write(run/'episode/planner-result.json',{'finish_result':result.finish_result,'stats':result.stats,'error':result.error,'messages':result.messages})
    finally:
        done.set()
        if snapshot()['terminal'] is None:terminate('planner_returned',error=result.error if result else 'exception')
        toolkit.close();write(run/'episode/planner-summary.json',{'terminal':snapshot()['terminal'],'error':result.error if result else 'exception','integration_smoke':args.integration_smoke})
    if result and result.error:raise RuntimeError(result.error)
if __name__=='__main__':main()
