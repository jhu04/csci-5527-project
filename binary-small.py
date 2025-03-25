import time
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

device = torch.device('cpu')
# variables and corresponding dimensions.
n = 4
var_in = {"x": [n, 1]}

# G = I causes x.T @ x = 2 everywhere, which is weird?
G = torch.eye(n, device=device, dtype=torch.double)
# G = torch.zeros((n, n), device=device, dtype=torch.double) / 4
# G[0][0] = 2
# eps = 1e-5


def comb_fn(X_struct):
    x = X_struct.x

    # objective function
    f = x.T @ x

    # inequality constraint, matrix form
    ci = pygransoStruct()
    ci.c1 = -x
    ci.c2 = x - 1
    # ci.c3 = x * (1 - x) - eps

    # equality constraint
    ce = pygransoStruct()
    ce.c1 = x * (x - 1)
    ce.c2 = x.T @ G @ x - 2

    # print(x.T @ G @ x - 2)

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
# opts.x0 = torch.ones((3, 1), device=device, dtype=torch.double)

opts.x0 = torch.ones((n, 1), device=device, dtype=torch.double) / 3
# opts.x0[0][0] = 1
# opts.x0[1][0] = 1
opts.torch_device = device

start = time.time()
soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print("Most feasible:", soln.most_feasible.x)
print("Final:", soln.final.x)

print(comb_fn(soln.final)[0])
