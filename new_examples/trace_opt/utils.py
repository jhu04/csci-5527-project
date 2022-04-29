import time
import torch
import sys
## Adding PyGRANSO directories. Should be modified by user
sys.path.append('/home/buyun/Documents/GitHub/PyGRANSO')
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from torch.linalg import norm
from scipy.stats import ortho_group
import numpy as np
from matplotlib import pyplot as plt 
import os
from datetime import datetime

def get_name(square_flag,folding_list,n,d,total,maxclocktime,N,K):
    if square_flag:
        square_str = "square_"
    else:
        square_str = ""
    maxfolding = ''
    for str in folding_list:
        maxfolding = maxfolding + str + '_'
    # save file
    now = datetime.now() # current date and time
    date_time = now.strftime("%m%d%Y_%H:%M:%S")
    my_path = os.path.dirname(os.path.abspath(__file__))
    name_str = "N{}K{}_n{}_d{}_{}{}_total{}_maxtime{}".format(N,K,n,d,square_str,maxfolding,total,maxclocktime)
    log_name = "log/" + date_time + name_str + '.txt'

    print( name_str + "start\n\n")
    return [my_path, log_name, date_time, name_str]


def data_init(rng_seed, n, d, device):
    # fix random seeds
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)
    # data initialization
    A = torch.randn(n,n)
    A = (A + A.T)/2
    # All the user-provided data (vector/matrix/tensor) must be in torch tensor format.
    # As PyTorch tensor is single precision by default, one must explicitly set `dtype=torch.double`.
    # Also, please make sure the device of provided torch tensor is the same as opts.torch_device.
    A = A.to(device=device, dtype=torch.double)
    L, U = torch.linalg.eig(A)
    L = L.to(dtype=torch.double)
    U = U.to(dtype=torch.double)
    index = torch.argsort(L,descending=True)
    U = U[:,index[0:d]]
    ana_sol = -torch.trace(U.T@A@U).item()
    return [A, U, ana_sol]

def user_fn(X_struct,A,d,device,maxfolding,square_flag):
    V = X_struct.V
    # objective function
    f = -torch.trace(V.T@A@V)
    # inequality constraint, matrix form
    ci = None
    # equality constraint
    ce = pygransoStruct()
    constr_vec = (V.T@V - torch.eye(d).to(device=device, dtype=torch.double)).reshape(d**2,1)
    if maxfolding == 'l1':
        ce.c1 = torch.sum(torch.abs(constr_vec))
    elif maxfolding == 'l2':
        ce.c1 = torch.sum(constr_vec**2)**0.5
    elif maxfolding == 'linf':
        ce.c1 = torch.amax(torch.abs(constr_vec))
    elif maxfolding == 'unfolding':
        ce.c1 = V.T@V - torch.eye(d).to(device=device, dtype=torch.double)
    else:
        print("Please specficy you maxfolding type!")
        exit()
    if square_flag:
        ce.c1 = ce.c1**2
    return [f,ci,ce]

def opts_init(device,maxit,opt_tol,maxclocktime,QPsolver,mu0,ana_sol,threshold,n,d):
    opts = pygransoStruct()
    opts.torch_device = device
    opts.print_frequency = 10
    opts.maxit = maxit
    opts.print_use_orange = False
    opts.print_ascii = True
    opts.quadprog_info_msg  = False
    opts.opt_tol = opt_tol
    opts.maxclocktime = maxclocktime
    opts.QPsolver = QPsolver
    opts.mu0 = mu0
    opts.fvalquit = ana_sol*threshold
    opts.x0 =  torch.randn((n*d,1)).to(device=device, dtype=torch.double)
    opts.x0 = opts.x0/norm(opts.x0)
    return opts

def result_dict_init(N,folding_list):

    result_dict = {}
    for rng_seed in range(N):
        for maxfolding in folding_list:
            dict_key = str(rng_seed) + maxfolding

            tmp_dict = {
                'time':np.array([]),
                'iter': np.array([]),
                'F': np.array([]),
                'MF': np.array([]),
                'term_code_pass': np.array([]),
                'tv': np.array([]),
                'MF_tv': np.array([]),
                'term_code_fail': [],
                'E': np.array([]), # error
                'index_sort': np.array([])
                }
            
            result_dict[dict_key] = tmp_dict

    return result_dict

def store_result(soln,end,start,n,d,i,result_dict,U,rng_seed,maxfolding):
    dict_key = str(rng_seed) + maxfolding
    cur_result_dict = result_dict[dict_key]
    print("seed {} folding type {} Wall Time: {}s".format(rng_seed, maxfolding,end - start))
    if soln.termination_code != 12 and soln.termination_code != 8:
        # mean error
        V = torch.reshape(soln.final.x,(n,d))
        E = norm(V-U)/norm(U)
        cur_result_dict['E'] = np.append(cur_result_dict['E'],E.item())
        cur_result_dict['time'] = np.append(cur_result_dict['time'],end-start)
        cur_result_dict['F'] = np.append(cur_result_dict['F'],soln.final.f)
        cur_result_dict['MF'] = np.append(cur_result_dict['MF'],soln.most_feasible.f)
        cur_result_dict['term_code_pass'] = np.append(cur_result_dict['term_code_pass'],soln.termination_code)
        cur_result_dict['tv'] = np.append(cur_result_dict['tv'],soln.final.tv) #total violation at x (vi + ve)
        cur_result_dict['MF_tv'] = np.append(cur_result_dict['MF_tv'],soln.most_feasible.tv)
        cur_result_dict['iter'] = np.append(cur_result_dict['iter'],soln.iters)
    else:
        cur_result_dict['term_code_fail'].append("i = {}, code = {}\n ".format(i,soln.termination_code) )

    return result_dict

def sort_result(result_dict,rng_seed,maxfolding):

    dict_key = str(rng_seed) + maxfolding
    cur_result_dict = result_dict[dict_key]

    index_sort = np.argsort(cur_result_dict['F'])
    index_sort = index_sort[::-1]
    cur_result_dict['F'] = cur_result_dict['F'][index_sort]
    cur_result_dict['E'] = cur_result_dict['E'][index_sort]
    cur_result_dict['time'] = cur_result_dict['time'][index_sort]
    cur_result_dict['MF'] = cur_result_dict['MF'][index_sort]
    cur_result_dict['term_code_pass'] = cur_result_dict['term_code_pass'][index_sort]
    cur_result_dict['tv'] = cur_result_dict['tv'][index_sort]
    cur_result_dict['MF_tv'] = cur_result_dict['MF_tv'][index_sort]
    cur_result_dict['iter'] = cur_result_dict['iter'][index_sort]
    cur_result_dict['index_sort'] = index_sort

def print_result(result_dict,total,rng_seed,maxfolding):
    dict_key = str(rng_seed) + maxfolding
    cur_result_dict = result_dict[dict_key]

    print("Time = {}".format(cur_result_dict['time']) )
    print("F obj = {}".format(cur_result_dict['F']))
    print("MF obj = {}".format(cur_result_dict['MF']))
    print("termination code = {}".format(cur_result_dict['term_code_pass']))
    print("total violation tvi + tve = {}".format(cur_result_dict['tv']))
    print("MF total violation tvi + tve = {}".format(cur_result_dict['MF_tv']))
    print('iterations = {}'.format(cur_result_dict['iter']))
    print("Error = {}".format(cur_result_dict['E']))
    print("index sort = {}".format(cur_result_dict['index_sort']))
    print("failed code: {}".format(cur_result_dict['term_code_fail']))

    arr_len = cur_result_dict['F'].shape[0]
    print("successful rate = {}".format(arr_len/total))
    return arr_len

def add_path(my_path,rng_seed, date_time, name_str):
    png_title =  "png/sorted_F_" +  date_time + '_seed_{}'.format(rng_seed) + name_str
    data_name =  'data/' + date_time + '_seed_{}'.format(rng_seed) + name_str +'.npy'
    data_name = os.path.join(my_path, data_name)
    png_title = os.path.join(my_path, png_title)
    return [data_name,png_title]