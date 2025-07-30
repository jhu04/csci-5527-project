import time
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
import math

device = torch.device('cpu')
# variables and corresponding dimensions.
n = 10
var_in = {"x": [n, 1]}


def comb_fn(X_struct):
    x = X_struct.x

    # objective function
    f = x.T @ x

    # inequality constraint, matrix form
    ci = pygransoStruct()
    # ci = None
    ci.c1 = x + 1

    # equality constraint
    # ce = pygransoStruct()
    ce = None
    # ce.c1 = x * (x - 1)

    return [f, ci, ce]


opts = pygransoStruct()
# option for switching QP solver. We only have osqp as the only qp solver in current version. Default is osqp
# opts.QPsolver = 'osqp'

# set an intial point
# All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
# As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
# Also, please make sure the device of provided torch tensor is the same as opts.torch_device.

# opts.x0 = torch.randn((n, 1), device=device, dtype=torch.double) / math.sqrt(n)
opts.x0 = torch.ones((n, 1), device=device, dtype=torch.double) * 3
opts.mu0 = 1e-2
opts.torch_device = device

start = time.time()
soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print("Most feasible:", soln.most_feasible.x)
print("Final:", soln.final.x)

print("Final Objective:", comb_fn(soln.final)[0])
