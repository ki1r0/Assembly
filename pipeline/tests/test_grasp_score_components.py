import ast
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize('robot,expected',[('xarm7',1.25),('panda',5.25)])
def test_g1_separates_insertion_access_from_pickup_motion_cost(robot,expected):
    tree=ast.parse((Path(__file__).parents[1]/'generic_backend.py').read_text())
    statements=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='candidate' and isinstance(t.slice,ast.Constant) and t.slice.value in {'score_components','score'} for t in node.targets):
            statements.append(node)
    statements.sort(key=lambda n:n.lineno)
    candidate={'assembly_compatibility':{'cost':4.}}
    namespace=dict(np=np,candidate=candidate,robot_name=robot,ik_pre={'joints':[1.,0.]},joints=np.zeros(2),pre=np.array([.5,0,0]),pose=lambda arm:np.zeros(7),arm='left')
    exec(compile(ast.Module(body=statements,type_ignores=[]),'score','exec'),namespace)
    assert candidate['score']==pytest.approx(expected)
    assert candidate['assembly_compatibility']['cost']==4.
