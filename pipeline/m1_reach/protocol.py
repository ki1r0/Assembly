"""Existing continue/help/stop/native-success semantics for the Isaac RPent adapter."""
from typing import Literal
from rpent.tools import tool, ToolResult

class RolloutTools:
    def __init__(self,toolkit):self.toolkit=toolkit

    @tool(readonly=True)
    def decision(self,choice:Literal['continue'],evidence:str,uncertainty:str,next_step_or_help:str)->ToolResult:
        """Record a continuing decision and return normally to observation/action planning. Does not end the episode or reserve action permission. Explain evidence, uncertainty and a concrete next step."""
        from .control import locked,event
        with locked() as data:
            terminal=data['terminal']
            event(data,'decision',choice=choice,evidence=evidence,uncertainty=uncertainty,next_step_or_help=next_step_or_help,terminal=terminal is not None)
        if terminal is not None:
            return ToolResult(data={'recorded':True,'choice':choice,'episode_ended':True,'_finish':True,'status':'success' if terminal['source']=='native_success' else 'failure','stop_reason':terminal['source']})
        return ToolResult(data={'recorded':True,'choice':choice,'episode_ended':False})

    @tool(readonly=True)
    def ask_human_help(self,evidence:str,uncertainty:str,request:str)->ToolResult:
        """Request operator assistance and end this rollout immediately. Assistance is assumed effective for this test but is not executed. Describe evidence, uncertainty and the help requested; no audit file is required first."""
        from .control import terminate
        end=terminate('model_ask_human_help',evidence=evidence,uncertainty=uncertainty,request=request)
        return ToolResult(data={'_finish':True,'status':'stuck','stop_reason':end['source'],'evidence':evidence,'uncertainty':uncertainty,'request':request,'help_requested':end['source']=='model_ask_human_help','help_executed':False})

    @tool(readonly=True)
    def stop(self,evidence:str,uncertainty:str,reason:str)->ToolResult:
        """End the current rollout without assistance or a success claim. Give evidence, uncertainty and stopping reason. Does not require a prior audit."""
        from .control import terminate
        end=terminate('model_stop',evidence=evidence,uncertainty=uncertainty,reason=reason)
        return ToolResult(data={'_finish':True,'status':'failure','stop_reason':end['source'],'evidence':evidence,'uncertainty':uncertainty,'reason':reason})

    @tool(readonly=True)
    def finish(self,status:Literal['success'],summary:str)->ToolResult:
        """Report already observed native task success only. Failure/stuck are not finish outcomes: use stop or ask_human_help. A textual declaration cannot establish environment success."""
        from .control import snapshot,terminate
        native=snapshot()['terminal']
        solved=getattr(self.toolkit,'solved',lambda:False)
        if not (native and native['source']=='native_success') and not solved():
            return ToolResult(data={'error':'Native success has not occurred. Continue with real tools, ask_human_help, or stop.','_finish':False})
        end=terminate('native_success',summary=summary)
        return ToolResult(data={'_finish':True,'status':'success','summary':summary,'stop_reason':end['source']})
