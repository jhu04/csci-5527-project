import time
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct

# fix the random seed; don't modify this
torch.manual_seed(55272025)

# torch default precision is float32. Uncomment below if you want to use float64
# torch.set_default_dtype(torch.float64)

n = 50
m = 200
r = 10

dtype = torch.double

# generate a random dataset
A = torch.randn(n, r)
B = torch.randn(m, r)
X = B @ A.T + 0.01*torch.randn(m,n, dtype=dtype)

device = torch.device('cpu')
# variables and corresponding dimensions.
var_in = {"A3": [n, r]}


def comb_fn(X_struct):
    A3 = X_struct.A3
    A3.requires_grad = True

    # objective function
    f = 1 / m * torch.norm(X - X @ A3 @ A3.T / (n * r)) ** 2
    # print("Grad:", torch.norm(2 * (X - X @ A3 @ A3.T) @ A3 @ A3.T))

    # inequality constraint, matrix form
    # ci = pygransoStruct()
    ci = None
    # ci.c1 = -x
    # ci.c2 = x - 1

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

import math
opts.x0 = torch.randn(n * r, 1, device=device, dtype=torch.double)
opts.torch_device = device

start = time.time()
soln = pygranso(var_spec=var_in, combined_fn=comb_fn, user_opts=opts)
end = time.time()
print("Total Wall Time: {}s".format(end - start))
print(soln.final.x)

A3 = torch.reshape(soln.final.x, (n, r))
A3.requires_grad = True
f = 1 / m * torch.norm(X - X @ A3 @ A3.T) ** 2
print(f)
f.backward()
print("grad norm:", torch.norm(A3.grad))

_, A1_full = torch.linalg.eig(X.T @ X)
A1 = A1_full[:, :r]
print(torch.norm(A1 @ torch.linalg.pinv(A1) - A3 @ torch.linalg.pinv(A3)) / torch.norm(A3 @ torch.linalg.pinv(A3)))
