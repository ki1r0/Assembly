"""Repeatable actor-input noise; never use on safety/evaluator measurements."""
import numpy as np


class InputPerturbation:
    def __init__(self,condition):
        self.rng=np.random.default_rng(condition.get('seed',0))
        self.depth_std=float(condition.get('depth_std_m',0.))
        self.dropout=float(condition.get('depth_dropout_probability',0.))
        self.wrench_std=np.array([float(condition.get('force_std_n',0.))]*3+
                                 [float(condition.get('torque_std_nm',0.))]*3)
        if not np.isfinite([self.depth_std,self.dropout,*self.wrench_std]).all() or self.depth_std<0 or not 0<=self.dropout<=1 or (self.wrench_std<0).any():
            raise ValueError('Noise parameters must be finite, nonnegative, with dropout in [0,1]')
        self.frame_id=None
        self.frame=None

    def depth(self,depth,frame_id):
        if not isinstance(frame_id,str) or not frame_id:raise ValueError('Original frame ID required')
        if frame_id==self.frame_id:return self.frame.copy()
        output=np.asarray(depth,dtype=float).copy()
        valid=np.isfinite(output)&(output>0)
        if self.depth_std:
            output[valid]+=self.rng.normal(0,self.depth_std,int(valid.sum()))
            output[valid&(output<=0)]=np.nan
        if self.dropout:output[valid&(self.rng.random(output.shape)<self.dropout)]=np.nan
        self.frame_id=frame_id;self.frame=output.copy()
        return output

    def wrench(self,wrench):
        result=np.asarray(wrench,dtype=float).copy()
        if result.shape!=(6,) or not np.isfinite(result).all():raise ValueError('Finite six-axis wrench required')
        return result+self.rng.normal(0,self.wrench_std)
