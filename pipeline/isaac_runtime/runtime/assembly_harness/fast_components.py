"""Same ordered depth connectivity search compiled without fast-math."""
import numpy as np
from numba import njit

@njit(cache=True)
def connected_cells(points,valid,max_neighbor_distance_m):
    h,w=valid.shape;visited=np.zeros((h,w),np.bool_);components=[]
    for row in range(h):
        for col in range(w):
            if not valid[row,col] or visited[row,col]:continue
            visited[row,col]=True;stack=[(row,col)];cells=[]
            while stack:
                r,c=stack.pop();cells.append((r,c));current=points[r,c]
                for nr,nc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                    if nr<0 or nr>=h or nc<0 or nc>=w or visited[nr,nc] or not valid[nr,nc]:continue
                    threshold=max(max_neighbor_distance_m,.01*float(current[2]))
                    if np.linalg.norm(points[nr,nc]-current)<=threshold:
                        visited[nr,nc]=True;stack.append((nr,nc))
            rows=np.empty(len(cells),np.int32);cols=np.empty(len(cells),np.int32)
            for i in range(len(cells)):rows[i]=cells[i][0];cols[i]=cells[i][1]
            components.append((rows,cols))
    return components
