import time
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

device = torch.device('cpu')
# variables and corresponding dimensions.
var_in = {"x": [3, 1]}


def comb_fn(X_struct):
    x = X_struct.x

    # objective function
    f = -torch.norm(x - 0.7)

    # inequality constraint, matrix form
    ci = pygransoStruct()
    ci.c1 = -x
    ci.c2 = x - 1

    # equality constraint
    # ce = pygransoStruct()
    ce = None
    # ce.c1 = x * (x - 1)

    # solving binary can be done by enforcing
    # 0 <= x <= 1
    # <x, (x - 1)> == 0
    # since 0 <= x <= 1 implies x * (x - 1) >= 0 with equality iff x = 0, x = 1

    return [f, ci, ce]


opts = pygransoStruct()
# option for switching QP solver. We only have osqp as the only qp solver in current version. Default is osqp
# opts.QPsolver = 'osqp'

# set an intial point
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.

# Converges to wrong point. Should not initialize to boundary point?
opts.x0 = torch.ones((3, 1), device=device, dtype=torch.double)
# opts.x0 = torch.ones((3, 1), device=device, dtype=torch.double) / 2
opts.torch_device = device

start = time.time()
soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print(soln.final.x)

print(comb_fn(soln.final)[0])
