import time
import torch
import sys
import numpy as np
import scipy
import scipy.io

import matplotlib.pyplot as plt

from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
from pygranso.private.getObjGrad import getObjGradDL

# Physics-informed neural network - a straightforward MLP with tanh activations
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PINN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 1)])
        self.linear_out = nn.Linear(hidden_size, 1)
        self.activ = nn.Tanh()
    
    def forward(self, x):
        x = self.linear_in(x)
        x = self.activ(x)
        for l in self.linear_hidden:
            x = l(x)
            x = self.activ(x)
        out = self.linear_out(x)
        return out

# Helper function to extract gradients of the NN outputs
def get_grads(u, x, t):
    u_t = torch.autograd.grad(
        u, t, 
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        retain_graph=True,
        create_graph=True
    )[0]
    return u_t, u_x, u_xx

# User function specifying objective and constraints - required by PyGRANSO
def user_fn(model, sample_points, boundary_points, boundary_usol):
    # Forward pass for sample points
    x, t = sample_points
    xt = torch.cat((x, t), 1)
    u = model(xt)

    # Calculate gradients of network
    u_t, u_x, u_xx = get_grads(u, x, t)

    # Minimize residual
    res = u_t + u * u_x - 0.01 / np.pi * u_xx
    f = torch.sum(res ** 2)

    # No inequality constraints
    ci = None

    # Equality constraint on boundary points
    ce = pygransoStruct()
    xb, tb = boundary_points
    xtb = torch.cat((xb, tb), 1)
    ub = model(xtb)

    ce.c1 = ub - boundary_usol

    return [f,ci,ce]

# Evaluates the relative L2 error over all grid points
# Notably, this is NOT what the PINN is minimizing--it only has access to boundary points
def evaluate(iteration, model, xv, tv, test_usol, error):
    test_points = torch.stack((xv, tv)).transpose(0,1)
    pred_usol = model(test_points)
    L2_error = torch.sqrt(torch.sum((pred_usol - test_usol) ** 2) / torch.sum(test_usol ** 2))
    error[iteration-1] = L2_error.cpu().detach().item()

    # Save intermediate results (NN outputs + PDE residuals) as images
    if iteration % 25 == 0:
        outimg = pred_usol.cpu().detach().numpy()
        outimg = np.reshape(outimg, (xgridsize, tgridsize))
        plt.imsave("output_imgs/predicted_"+str(iteration)+".png", outimg, origin='upper')
        plt.close()
        evalu_t, evalu_x, evalu_xx = get_grads(pred_usol, xv, tv)
        evalres = evalu_t + torch.flatten(pred_usol) * evalu_x - 0.01 / np.pi * evalu_xx
        outimg = evalres.cpu().detach().numpy()
        outimg = np.reshape(outimg, (xgridsize, tgridsize))
        plt.imsave("output_imgs/pderesidual_"+str(iteration)+".png", outimg, vmin=-3, vmax=3, origin='upper')
        plt.close()


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    double_precision = torch.double
    torch.manual_seed(0)

    # NN hyperparams - width + depth are somewhat arbitrary and vary between papers
    input_size = 2
    hidden_size = 20
    num_layers = 7

    data = scipy.io.loadmat('./data/burgers_shock.mat')

    # Create PINN
    model = PINN(input_size, hidden_size, num_layers).to(device=device, dtype=double_precision)
    model.train()

    # Get boundary points along three sides (x = -1, x = 1, t = 0)
    tb_x = data['t']
    xb_t = data['x']
    xb_xlow = np.full_like(tb_x, data['x'][0])
    xb_xhigh = np.full_like(tb_x, data['x'][-1])
    tb_tlow = np.full_like(xb_t, data['t'][0])

    usol_xlow = data['usol'][0,:,None]
    usol_xhigh = data['usol'][-1,:,None]
    usol_tlow = data['usol'][:,0,None]

    xb = np.vstack((xb_xlow, xb_xhigh, xb_t))
    tb = np.vstack((tb_x, tb_x, tb_tlow))
    usolb = np.vstack((usol_xlow, usol_xhigh, usol_tlow))

    xb = torch.Tensor(xb).to(device=device, dtype=double_precision).requires_grad_()
    tb = torch.Tensor(tb).to(device=device, dtype=double_precision).requires_grad_()
    usolb = torch.Tensor(usolb).to(device=device, dtype=double_precision).requires_grad_()
    boundary_points = (xb, tb)

    # Ground-truth data - used for testing/evaluation
    usol_full = data['usol']
    usol_tensor = usol_full.flatten()
    usol_tensor = torch.Tensor(usol_tensor).to(device=device, dtype=double_precision)

    # Sample points. Following Dual-Cone Gradient Descent, 10x as many sample points as boundary points
    n_samples = 4560
    xs = -1 + 2 * np.random.rand(n_samples, 1)
    ts = np.random.rand(n_samples, 1)

    xs = torch.Tensor(xs).to(device=device, dtype=double_precision).requires_grad_()
    ts = torch.Tensor(ts).to(device=device, dtype=double_precision).requires_grad_()
    sample_points = (xs, ts)

    # Create grid inputs for visualization, comparison to GT
    xgridsize = 256
    tgridsize = 100
    tv, xv = np.meshgrid(data['t'], data['x'])
    tv = torch.Tensor(tv.flatten()).to(device=device, dtype=double_precision).requires_grad_()
    xv = torch.Tensor(xv.flatten()).to(device=device, dtype=double_precision).requires_grad_()
    grid_points = torch.stack((xv, tv)).transpose(0,1)

    # Tensors have fixed size and we need to modify in-place, so initialize with maximum possible size
    max_iters = 200
    error = torch.empty(max_iters, device=device, dtype=double_precision)

    # Functions for optimizer
    comb_fn = lambda model: user_fn(model, sample_points, boundary_points, usolb)
    halt_log_fn = lambda iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level: \
        evaluate(iteration, model, xv, tv, usol_tensor, error)

    # Pygranso Options
    opts = pygransoStruct()
    nvar = getNvarTorch(model.parameters())
    opts.x0 = nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
    opts.torch_device = device
    opts.print_level = 1
    opts.print_frequency = 10
    opts.disable_terminationcode_6 = True # Important for training NNs
    opts.maxit = max_iters
    opts.halt_log_fn = halt_log_fn

    # Hyperparameters
    # opts.mu0 = 1

    # Main algorithm
    start = time.time()
    soln = pygranso(var_spec= model, combined_fn = comb_fn, user_opts = opts)
    end = time.time()
    print("Total Wall Time: {}s".format(end - start))

    model.eval()

    test_output = model(grid_points)

    # Plot predictions, GT, and error over the full range
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(18, 12))
    outimg = test_output.cpu().detach().numpy()
    outimg = np.reshape(outimg, (xgridsize, tgridsize))

    global_min = np.min([np.min(outimg), np.min(usol_full), np.min(np.abs(outimg - usol_full))])
    global_max = np.max([np.max(outimg), np.max(usol_full), np.max(np.abs(outimg - usol_full))])

    ax1.set_title("Predicted outputs from PINN")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")
    ax1.set_box_aspect(1)
    ax1.imshow(outimg, vmin=global_min, vmax=global_max, extent=[0, 1, 1, -1], aspect='auto')

    ax2.set_title("Ground truth solution from burgers_shock.mat")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")
    ax2.set_box_aspect(1)
    ax2.imshow(usol_full, vmin=global_min, vmax=global_max, extent=[0, 1, 1, -1], aspect='auto')

    ax3.set_title("Difference")
    ax3.set_xlabel("t")
    ax3.set_ylabel("x")
    ax3.set_box_aspect(1)
    ax3.imshow(usol_full - outimg, vmin=global_min, vmax=global_max, extent=[0, 1, 1, -1], aspect='auto')

    # Calculate gradients of network
    testu_t, testu_x, testu_xx = get_grads(test_output, xv, tv)

    testres = testu_t + torch.flatten(test_output) * testu_x - 0.01 / np.pi * testu_xx

    test_ut_img = testu_t.cpu().detach().numpy()
    test_ut_img = np.reshape(test_ut_img, (xgridsize, tgridsize))
    test_ux_img = testu_x.cpu().detach().numpy()
    test_ux_img = np.reshape(test_ux_img, (xgridsize, tgridsize))
    test_res_img = testres.cpu().detach().numpy()
    test_res_img = np.reshape(test_res_img, (xgridsize, tgridsize))

    ax4.set_title("Predicted derivative w.r.t. t")
    ax4.set_xlabel("t")
    ax4.set_ylabel("x")
    ax4.set_box_aspect(1)
    ax4.imshow(test_ut_img, extent=[0, 1, 1, -1], aspect='auto')

    ax5.set_title("Predicted derivative w.r.t. x")
    ax5.set_xlabel("t")
    ax5.set_ylabel("x")
    ax5.set_box_aspect(1)
    ax5.imshow(test_ux_img, extent=[0, 1, 1, -1], aspect='auto')

    ax6.set_title("Predicted PDE residual")
    ax6.set_xlabel("t")
    ax6.set_ylabel("x")
    ax6.set_box_aspect(1)
    ax6.imshow(test_res_img, extent=[0, 1, 1, -1], aspect='auto')
    plt.show()

    # Plot L2 loss over full grid
    iter_range = np.arange(1, soln.iters+1)
    error = error.detach().cpu().numpy()
    plt.plot(iter_range, error[:soln.iters])
    plt.xlabel("Iteration")
    plt.ylabel("Relative L2 loss")
    plt.show()
