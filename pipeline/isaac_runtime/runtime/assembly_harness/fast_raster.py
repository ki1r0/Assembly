"""Compiled equivalent of the original pinhole depth rasterizer (no fast-math)."""
import math
import numpy as np
from numba import njit

@njit(cache=True)
def _triangle(depth,tri,k):
    p=np.empty((3,2),np.float64)
    for i in range(3):
        z=tri[i,2]
        if z<=0:return
        p[i,0]=(k[0,0]*tri[i,0]+k[0,1]*tri[i,1])/z+k[0,2]
        p[i,1]=(k[1,0]*tri[i,0]+k[1,1]*tri[i,1])/z+k[1,2]
    area=(p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[1,1]-p[0,1])*(p[2,0]-p[0,0])
    if abs(area)<1e-12:return
    x0=max(0,int(math.ceil(min(p[0,0],p[1,0],p[2,0]))));x1=min(depth.shape[1]-1,int(math.floor(max(p[0,0],p[1,0],p[2,0]))))
    y0=max(0,int(math.ceil(min(p[0,1],p[1,1],p[2,1]))));y1=min(depth.shape[0]-1,int(math.floor(max(p[0,1],p[1,1],p[2,1]))))
    for y in range(y0,y1+1):
        for x in range(x0,x1+1):
            a=((p[2,0]-p[1,0])*(y-p[1,1])-(p[2,1]-p[1,1])*(x-p[1,0]))/area
            b=((p[0,0]-p[2,0])*(y-p[2,1])-(p[0,1]-p[2,1])*(x-p[2,0]))/area
            c=1.-a-b
            inv=a/tri[0,2]+b/tri[1,2]+c/tri[2,2]
            if a>=-1e-9 and b>=-1e-9 and c>=-1e-9 and inv>0:
                d=1./inv
                old=depth[y,x]
                if not np.isfinite(old) or d<old:depth[y,x]=np.float32(d)

@njit(cache=True)
def draw_clipped_triangles(depth,triangles,k,near,far):
    for tri in triangles:
        poly=np.empty((8,3),np.float64);poly[:3]=tri;n=3
        for j in range(2):
            bound=near if j==0 else far
            clipped=np.empty((8,3),np.float64);count=0
            if n==0:break
            prev=poly[n-1].copy();pin=prev[2]>=bound if j==0 else prev[2]<=bound
            for i in range(n):
                cur=poly[i].copy();cin=cur[2]>=bound if j==0 else cur[2]<=bound
                if cin!=pin:
                    denominator=cur[2]-prev[2]
                    if abs(denominator)>1e-15:
                        fraction=(bound-prev[2])/denominator
                        clipped[count]=prev+fraction*(cur-prev);count+=1
                if cin:clipped[count]=cur;count+=1
                prev=cur;pin=cin
            poly=clipped;n=count
        for i in range(1,n-1):
            t=np.empty((3,3),np.float64);t[0]=poly[0];t[1]=poly[i];t[2]=poly[i+1]
            _triangle(depth,t,k)
