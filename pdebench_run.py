#!/usr/bin/env python
# coding: utf-8

# ## Modules Importing
# Import all necessary modules and add PyGRANSO src folder to system path. 

import time
import torch
import scipy.io
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('.')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch.nn as nn
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# from pygranso.private.getObjGrad import getObjGradDL

import scipy
import numpy as np
import matplotlib.pyplot as plt

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# fix the random seed
seed = 55272025
torch.manual_seed(seed)
np.random.seed(seed)

# w = 8

# ## Model architecture

# Physics-informed neural network - a straightforward MLP with tanh activations
class PINN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(PINN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 1)])
        self.linear_out = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.linear_in.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)
        for hidden in self.linear_hidden:
            nn.init.xavier_uniform_(hidden.weight)
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

    u_tt = torch.autograd.grad(
        u_t, t,
        grad_outputs=torch.ones_like(u_t),
        retain_graph=True,
        create_graph=True
    )[0]

    return u_t, u_x, u_xx, u_tt

# ## Data setup

###
### START data setup
double_precision = torch.double
data = {}
all_data = np.load('./data/advection/1D_Advection_Sols_beta1.0.npy') # You will need to follow PDEBench repo instructions to get this data
data['usol'] = all_data[0]
data['t'] = np.load("./data/advection/t_coordinate.npy")
data['usol_init'] = np.load("./data/advection/x_coordinate.npy")

# Get boundary points along three sides (x = -1, x = 1, t = 0)
tb_init = np.zeros_like(data['usol_init'])
xb_init = np.linspace(0, 1, num=len(data['usol_init']))

xb = xb_init[:,None]
tb = tb_init[:,None]
usolb = data['usol'][0]

plt.plot(usolb)
plt.title("Initial function")
plt.show()

print(data['t'].shape)
print(data['usol_init'].shape)
print(data['usol'].shape)

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
xs = np.random.rand(n_samples, 1)
ts = 2 * np.random.rand(n_samples, 1)

xs = torch.Tensor(xs).to(device=device, dtype=double_precision).requires_grad_()
ts = torch.Tensor(ts).to(device=device, dtype=double_precision).requires_grad_()
sample_points = (xs, ts)

# Create grid inputs for visualization, comparison to GT
xgridsize = 1024
tgridsize = 201
tv, xv = np.meshgrid(data['t'][:-1], xb_init) # PDEBench appends a surplus number for whatever reason
tv = torch.Tensor(tv.flatten()).to(device=device, dtype=double_precision).requires_grad_()
xv = torch.Tensor(xv.flatten()).to(device=device, dtype=double_precision).requires_grad_()
grid_points = torch.stack((xv, tv)).transpose(0,1)
### END data setup

# Evaluates the relative L2 error over all grid points
# Notably, this is NOT what the PINN is minimizing--it only has access to boundary points
# NOTE: This is only used in the PyGRANSO implementation
def evaluate(iteration, model, xv, tv, test_usol, metric_dict, mu):
    test_points = torch.stack((xv, tv)).transpose(0,1)
    pred_usol = model(test_points)

    # Get test loss (normalized to same scale as train loss)
    u = pred_usol.flatten()
    u_t, u_x, u_xx, u_tt = get_grads(u, xv, tv)
    test_res = u_t + u_x # advection
    curr_test_err = torch.norm(test_res) / test_res.numel() * np.sqrt(201 * 1024 / 4560)

    # Get u MSE
    u_mse = torch.norm(pred_usol.flatten() - test_usol) ** 2 / pred_usol.numel()

    # Track metrics
    metric_dict["train_err"][iteration-1] = metric_dict["curr_train_err"][0].cpu().detach().item()
    metric_dict["test_err"][iteration-1] = curr_test_err.cpu().detach().item()
    metric_dict["feas"][iteration-1] = metric_dict["curr_feas"][0].cpu().detach().item()
    metric_dict["u_mse"][iteration-1] = u_mse.cpu().detach().item()
    metric_dict["mu"][iteration-1] = mu

    # Save intermediate results (NN outputs + PDE residuals) as images
    if (iteration < 500 and iteration % 10 == 0
        ) or (iteration < 2000 and iteration % 50 == 0) or iteration % 200 == 0:
        outimg = pred_usol.cpu().detach().numpy()
        outimg = np.reshape(outimg, (xgridsize, tgridsize))
        plt.imsave("output_imgs/predicted_"+str(iteration)+".png", outimg, origin='upper')
        plt.close()
        evalu_t, evalu_x, evalu_xx, evalu_tt = get_grads(pred_usol, xv, tv)
        evalres = evalu_t + evalu_x # advection
        outimg = evalres.cpu().detach().numpy()
        outimg = np.reshape(outimg, (xgridsize, tgridsize))
        plt.imsave("output_imgs/pderesidual_"+str(iteration)+".png", outimg, vmin=-3, vmax=3, origin='upper')
        plt.close()

def f(model, sample_points): # objective
    x, t = sample_points
    xt = torch.cat((x, t), 1)
    u = model(xt)
    
    # Calculate gradients of network
    u_t, u_x, u_xx, u_tt = get_grads(u, x, t)
    
    # Minimize residual
    res = u_t + u_x
    objective = torch.norm(res) / res.numel() # * 0 + 3
    return objective

def penalty(model, boundary_points, boundary_usol):
    xb, tb = boundary_points
    xtb = torch.cat((xb, tb), 1)
    ub = model(xtb)
    
    boundary_errors = ub.flatten() - boundary_usol
    return torch.norm(boundary_errors, p=1) / boundary_errors.numel()

def l2_penalty(model, boundary_points, boundary_usol):
    xb, tb = boundary_points
    xtb = torch.cat((xb, tb), 1)
    ub = model(xtb)
    
    boundary_errors = ub.flatten() - boundary_usol
    penalty = torch.norm(boundary_errors, p=2) / boundary_errors.numel() * 5000
    return penalty

# User function specifying objective and constraints - required by PyGRANSO
# explicitly takes following arguments:
# sample_points: Tensor(2, n_sample_points)
# boundary_points: Tensor(2, n_boundary_points)
# boundary_usol: Tensor(n_boundary_points)
def user_fn(model, sample_points, boundary_points, boundary_usol, metric_dict):
    # Minimize residual
    objective = f(model, sample_points)

    xb, tb = boundary_points
    xtb = torch.cat((xb, tb), 1)
    ub = model(xtb)
    
    # No inequality constraints
    ci = pygransoStruct()
    # Constraint folding
    ci.c1 = penalty(model, boundary_points, boundary_usol) # CHANGE THIS BETWEEN penalty() AND l2_penalty()

    # Track error
    # This func can be called >1 times per PyGRANSO iter, so we can't add to loss array yet
    metric_dict["curr_train_err"][0] = objective.detach()
    metric_dict["curr_feas"][0] = ci.c1.detach()

    # Equality constraint on boundary points folded away into an inequality constraint
    # ce = pygransoStruct()
    # ce.c1 = ub - boundary_usol
    ce = None

    return [objective,ci,ce]

# Adam stuff

def train_loop(model, mu, optimizer, f_lambda, penalty_lambda):
    model.train()
    
    train_err = f_lambda(model)
    feas = penalty_lambda(model)
    loss = train_err + mu * feas

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return train_err, feas

# def val_loop(dataloader, model):
#     model.eval()
#     # size = len(dataloader.dataset)
#     size = batch_size
#     correct = 0

#     with torch.no_grad():
#         logits = model(inputs)
#         _, predicted = torch.max(logits.data, 1)
#         train_acc.append((predicted == labels).sum().item() / size)
        
#         logits = model(test_inputs)
#         _, predicted = torch.max(logits.data, 1)
#         c = (predicted == test_labels).sum().item()
#         test_acc.append(c / size)
#         correct += c
    
#     correct /= size
#     print(f"Error: \n Accuracy: {(100*correct):>0.1f}% \n")
#     return 100*correct

# ### Main training function

# `f_lambda` takes form `lambda model: loss_of_model_on_training_set`
# `penalty_lambda` takes form `lambda model: penalty_of_model`
# these two lambdas should have other required info (e.g. training points) already baked into them
def exact_penalty_with_adam(model, f_lambda, penalty_lambda, metric_dict, eval_fn, mu_0=1., mu_rho=1.1, mu_eps=1e-5, n_inner_iters=100, max_iters=200):
    mu = torch.tensor([mu_0], dtype=double_precision).to(device)
    h_prev = float('inf')
    
    optimizer = torch.optim.Adam(model.parameters())

    resets = 0
    for iteration in range(max_iters * n_inner_iters): # TODO: try smaller number of iterations (e.g. 2), and/or try Wenjie stopping strategy
        update = iteration % 500 == 0
        if update:
            print("Iter", iteration)
        
        train_err, feas = train_loop(model, mu, optimizer, f_lambda, penalty_lambda)
        
        # Exact penalty update
        h = penalty_lambda(model)
        if update:
            print("Objective:", f_lambda(model))
            print("Penalty parameter:", mu)
            print("Penalty:", h)
        if h < 1e-5:  # if h(xk ) ≤ τ
            break

        # Choose new penalty parameter µk+1 > µk ;
        # 100 inner iterations
        if update and h > h_prev:
            mu *= mu_rho
            optimizer = torch.optim.Adam(model.parameters())
            
            resets += 1
            print("Reset", resets, "times")
        if update:
            h_prev = h

        # Choose new starting point (stay as optimal x1, x2)
        if update:
            print()
        
        # Track loss, etc.
        if iteration % n_inner_iters == n_inner_iters - 1:
            metric_dict["curr_train_err"][0] = train_err
            metric_dict["curr_feas"][0] = feas
            eval_fn(iteration // n_inner_iters + 1, model, metric_dict, mu)


# `f_lambda` takes form `lambda model: loss_of_model_on_training_set`
# `penalty_lambda` takes form `lambda model: penalty_of_model`
# these two lambdas should have other required info (e.g. training points) already baked into them
def exact_penalty_with_pygranso(model, f_lambda, penalty_lambda, metric_dict, eval_fn, mu_0=1., mu_rho=1.1, mu_eps=1e-5, n_inner_iters=1000, max_iters=200):
    mu = torch.tensor([mu_0], dtype=double_precision).to(device)
    h_prev = float('inf')

    # for iteration in range(1000):
    for iteration in range(max_iters): # TODO: try smaller number of iterations (e.g. 2), and/or try Wenjie stopping strategy
        print("Iter", iteration)
        
        # PyGRANSO
        def comb_fn(model, metric_dict):
            # objective function
            train_err = f_lambda(model)
            feas = penalty_lambda(model)
            phi1_x_mu = train_err + mu * feas

            # Track metrics
            metric_dict["curr_train_err"][0] = train_err.detach()
            metric_dict["curr_feas"][0] = feas.detach()
        
            # inequality constraint, matrix form
            ci = None
        
            # equality constraint 
            ce = None
        
            return [phi1_x_mu,ci,ce]
        
        opts = pygransoStruct()
        # option for switching QP solver. We only have osqp as the only qp solver in current version. Default is osqp
        # opts.QPsolver = 'osqp'
        
        # set an intial point
        # All the user-provided data (vector/matrix/tensor) must be in torch tensor format. 
        # As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
        # Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
        nvar = getNvarTorch(model.parameters())
        opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
        opts.torch_device = device
        opts.opt_tol = 1e-11
        opts.viol_eq_tol = 1e-8
        opts.double_precision = True
        opts.print_level = 1
        opts.print_frequency = 50
        opts.maxit = n_inner_iters  # Inner epochs # TODO: perhaps tune
        opts.disable_terminationcode_6 = True # Important for training NNs
        
        start = time.time()
        soln = pygranso(var_spec = model, combined_fn = lambda model: comb_fn(model, metric_dict), user_opts = opts)
        end = time.time()
        print("Inner Loop Wall Time: {}s".format(end - start))
        torch.nn.utils.vector_to_parameters(soln.final.x, model.parameters())
        
        # Exact penalty update
        
        h = penalty_lambda(model)
        print("Objective:", f_lambda(model))
        print("Penalty parameter:", mu)
        print("Penalty:", h)
        if h < 1e-5:  # if h(xk ) ≤ τ
            break

        # Choose new penalty parameter µk+1 > µk ;
        if h > mu_eps:
            mu *= mu_rho

        eval_fn(iteration + 1, model, metric_dict, mu)


# `user_fn_lambda` takes form `lambda model: [objective, ci, ce]`
# this lambda should have other required info (e.g. training points) already baked into them
# TODO: these args are incorrect (e.g. inner iterations)
def directly_use_pygranso(model, user_fn_lambda, eval_fn, mu_0=1., max_iters=1000):
    # Functions for optimizer


    # make a copy of the user fn that does not need to see the data points?
    # def user_fn_2(model, f_lambda(model), ci_lambda, ce_lambda):
    #     objective = f_lambda(model)
    #     ce = ce_lambda(model)
    #     ci = ci_lambda(model)


    #     # # Minimize residual
    #     # objective = f(model, sample_points)
    #     # xb, tb = boundary_points
    #     # xtb = torch.cat((xb, tb), 1)
    #     # ub = model(xtb)
    #     # 
    #     # # No inequality constraints
    #     # ci = pygransoStruct()
    #     # # Constraint folding
    #     # ci.c1 = l2_penalty(model, boundary_points, boundary_usol)

    #     # # Equality constraint on boundary points
    #     # # folded away into an inequality constraint
    #     # # ce = pygransoStruct()
    #     # # ce.c1 = ub - boundary_usol
    #     # ce = None

    #     # return [objective,ci,ce]

    # PyGRANSO
    comb_fn = user_fn_lambda
    halt_log_fn = eval_fn

    # Pygranso Options
    opts = pygransoStruct()
    nvar = getNvarTorch(model.parameters())
    opts.x0 = nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
    opts.torch_device = device
    opts.opt_tol = 1e-11
    opts.viol_eq_tol = 1e-8
    opts.double_precision = True
    opts.print_level = 1
    opts.print_frequency = 50
    opts.disable_terminationcode_6 = True # Important for training NNs
    opts.maxit = max_iters
    opts.halt_log_fn = halt_log_fn

    # Hyperparameters
#     opts.mu0 = 0.1
    opts.mu0 = mu_0

    # Main algorithm
    start = time.time()
    soln = pygranso(
        var_spec= model, 
        combined_fn = comb_fn,
        user_opts = opts,
    )
    end = time.time()
    print("Total Wall Time: {}s".format(end - start))
    return soln

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device {device}")
    torch.manual_seed(seed)
    
    # TODO: Take optimizer in as command-line argument
    # hyperparams
    optimizer_options = ['ep_adam', 'ep_pygranso', 'pygranso']
    # optim = 'ep_adam'
    # optim = 'ep_pygranso'
    optim = 'pygranso'
    print(f"Using optimizer {optim}")
    print(f"Using seed {seed}")

    # NN hyperparams - width + depth are somewhat arbitrary and vary between papers
    input_size = 2
    hidden_size = 20
    num_layers = 7
    double_precision = torch.double

    # Create PINN
    torch.manual_seed(seed)
    model = PINN(input_size, hidden_size, num_layers).to(device=device, dtype=double_precision)
    model.train()

    # Tensors have fixed size and we need to modify in-place, so initialize with maximum possible size
    max_iters = 20000
    train_err = torch.empty(max_iters, device=device, dtype=double_precision)
    test_err = torch.empty(max_iters, device=device, dtype=double_precision)
    u_mse = torch.empty(max_iters, device=device, dtype=double_precision)
    feas = torch.empty(max_iters, device=device, dtype=double_precision)
    mu = torch.empty(max_iters, device=device, dtype=double_precision)
    curr_train_err = torch.empty(1, device=device, dtype=double_precision)
    curr_feas = torch.empty(1, device=device, dtype=double_precision)

    metric_dict = {"train_err": train_err,
                   "test_err": test_err,
                   "u_mse": u_mse,
                   "feas": feas,
                   "mu": mu,
                   "curr_train_err": curr_train_err,
                   "curr_feas": curr_feas}
    
    # for ep methods
    f_lambda = lambda model: f(model, sample_points)
    penalty_lambda = lambda model: penalty(model, boundary_points, boundary_usol=usolb)

    # for PyGRANSO
    user_fn_lambda = lambda model: user_fn(
        model,
        sample_points,
        boundary_points,
        boundary_usol=usolb,
        metric_dict=metric_dict
    )
    halt_log_fn = lambda iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level: \
        evaluate(iteration, model, xv, tv, usol_tensor, metric_dict, penaltyfn_parts.mu)
    ep_eval_fn = lambda iteration, model, metric_dict, mu: \
        evaluate(iteration, model, xv, tv, usol_tensor, metric_dict, mu)

    soln = None
    if optim == 'ep_adam':
        exact_penalty_with_adam(
            model,
            mu_0=0.1,
            mu_rho=1.1,
            mu_eps=1e-5,
            f_lambda=f_lambda,
            penalty_lambda=penalty_lambda,
            metric_dict=metric_dict,
            eval_fn=ep_eval_fn,
            n_inner_iters=1000,
            max_iters=max_iters,
        )
    elif optim == 'ep_pygranso':
        exact_penalty_with_pygranso(
            model,
            mu_0=0.1,
            mu_rho=1.1,
            mu_eps=1e-5,
            f_lambda=f_lambda,
            penalty_lambda=penalty_lambda,
            metric_dict=metric_dict,
            eval_fn=ep_eval_fn,
            n_inner_iters=200,
            max_iters=max_iters,
        )
    elif optim == 'pygranso':
        soln = directly_use_pygranso(
            model,
            mu_0=10,
            user_fn_lambda=user_fn_lambda,
            eval_fn=halt_log_fn,
            max_iters=max_iters,
        )

def plot_pinn(model):
    model.eval()

    test_output = model(grid_points)

    # Plot predictions, GT, and error over the full range
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(18, 12))
    outimg = test_output.cpu().detach().numpy()
    outimg = np.reshape(outimg, (xgridsize, tgridsize))

    usol_full = data['usol']
    usol_full = np.transpose(usol_full)

    global_min = np.min([np.min(outimg), np.min(usol_full), np.min(np.abs(outimg - usol_full))])
    global_max = np.max([np.max(outimg), np.max(usol_full), np.max(np.abs(outimg - usol_full))])
    
    global_max = max(abs(global_min), abs(global_max))
    global_min = -global_max

    ax1.set_title("Predicted outputs from PINN")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")
    ax1.set_box_aspect(1)
    ax1.imshow(outimg, vmin=global_min, vmax=global_max, extent=[0, 2, 1, 0], aspect='auto')

    ax2.set_title("Ground truth solution")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")
    ax2.set_box_aspect(1)
    ax2.imshow(usol_full, vmin=global_min, vmax=global_max, extent=[0, 2, 1, 0], aspect='auto')

    ax3.set_title("Difference")
    ax3.set_xlabel("t")
    ax3.set_ylabel("x")
    ax3.set_box_aspect(1)
    ax3.imshow(usol_full - outimg, vmin=global_min, vmax=global_max, extent=[0, 2, 1, 0], aspect='auto')

    # Calculate gradients of network
    testu_t, testu_x, testu_xx, testu_tt = get_grads(test_output, xv, tv)

    testres = testu_t + testu_x

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
    ax4.imshow(test_ut_img, extent=[0, 2, 1, 0], aspect='auto')

    ax5.set_title("Predicted derivative w.r.t. x")
    ax5.set_xlabel("t")
    ax5.set_ylabel("x")
    ax5.set_box_aspect(1)
    ax5.imshow(test_ux_img, extent=[0, 2, 1, 0], aspect='auto')

    ax6.set_title("Predicted PDE residual")
    ax6.set_xlabel("t")
    ax6.set_ylabel("x")
    ax6.set_box_aspect(1)
    ax6.imshow(test_res_img, extent=[0, 2, 1, 0], aspect='auto')
    plt.show()

    # Plot L2 loss (u MSE) over full grid
    fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4, figsize=(24, 6))
    iter_range = np.arange(1, max_iters+1)
    
    u_mse = metric_dict["u_mse"].detach().cpu().numpy()
    ax1.semilogy(iter_range, u_mse[:max_iters])
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("u MSE")
    
    train_err = metric_dict["train_err"].detach().cpu().numpy()
    test_err = metric_dict["test_err"].detach().cpu().numpy()
    ax2.semilogy(iter_range, train_err[:max_iters], color='blue', label='Train error')
    ax2.semilogy(iter_range, test_err[:max_iters], color='green', label='Test error')
    # ax2.plot(iter_range, train_err[:soln.iters] / test_err[:soln.iters])
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error (normalized for comparison)")
    ax2.legend()
    
    feas = metric_dict["feas"].detach().cpu().numpy()
    ax3.semilogy(iter_range, feas[:max_iters])
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Feasibility")

    mu = metric_dict["mu"].detach().cpu().numpy()
    ax4.semilogy(iter_range, mu[:max_iters])
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Mu")

    plt.show()


plot_pinn(model)
