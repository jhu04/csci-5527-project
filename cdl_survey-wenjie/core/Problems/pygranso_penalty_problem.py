
import torch
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import numpy as np
import time
import copy

class pygranso_penalty_problem():
  def __init__(
      self,
      cfg,
      data,

      device,
      model,
      fn = ''
      ): # dataset is in

      self.device = device
      self.threshold = cfg.getfloat('EXP','THRESHOLD')

      self.model = model.to(device,dtype=torch.float64)
      

      self.train_X_0 = data['train_X_0'].to(device,dtype=torch.float64)
      self.train_y_0 = data['train_y_0'].to(device,dtype=torch.float64)
      self.train_X_1 = data['train_X_1'].to(device,dtype=torch.float64)
      self.train_y_1 = data['train_y_1'].to(device,dtype=torch.float64)
      self.test_X_0 = data['test_X_0'].to(device,dtype=torch.float64)
      self.test_y_0 = data['test_y_0'].to(device,dtype=torch.float64)
      self.test_X_1 = data['test_X_1'].to(device,dtype=torch.float64)
      self.test_y_1 = data['test_y_1'].to(device,dtype=torch.float64)

      self.fn = fn
      self.cfg = cfg


      self.output = {
        'epoch':-1,
        'f':[],
        'f_test':[],
        'ci':[],
        'ci_test':[],
        'diff':[],
        'diff_test':[],
        'time':[],

        'accuracy':[],
        'accuracy0':[],
        'accuracy1':[],

        'accuracy_test':[],
        'accuracy0_test':[],
        'accuracy1_test':[]
      }

  def user_fn(self,model,X_0,X_1,y_0,y_1):
    # objective function
    num_samples = X_0.size(0) + X_1.size(0)
    pred0 = model(X_0)
    pred1 = model(X_1)
    #print(pred0.size())
    #print(y_0.size())
    #print(pred0.size())
    #print(self.train_y_0.size())
    f = (self.obj_loss(pred0,y_0) \
         + self.obj_loss(pred1,y_1))/ num_samples
    #print(f)
    
    #ci.c1 = torch.abs(self.constraint_loss(pred0,self.train_y_0)\
    #      - self.constraint_loss(pred1,self.train_y_1))-self.threshold
    
    c1 = torch.relu(self.constr_loss(pred0,y_0)\
          - self.constr_loss(pred1,y_1)-self.threshold)
    c2 = torch.relu(- self.constr_loss(pred0,y_0)\
      + self.constr_loss(pred1,y_1)-self.threshold)
    f += self.rho * (c1 ** 2 + c2 ** 2)
    



    #print(ci)
    ci = None
    ce = None
    return [f,ci,ce]
  
  

  def train(self):
    self.print_summary_every = self.cfg.getint('OPTIMIZER','PRINT_SUMMARY_EVERY')
    
    # Construct Pygranso Problem
    self.rho = self.cfg.getfloat('OPTIMIZER','RHO')

    obj_loss = self.cfg.get('OPTIMIZER','OBJ_LOSS')
    constr_loss = self.cfg.get('OPTIMIZER','CONSTR_LOSS')
    if obj_loss == 'BCE':
      self.obj_loss = torch.nn.BCELoss(reduction='sum')
    elif obj_loss == 'MSE':
      self.obj_loss = torch.nn.MSELoss(reduction='sum')
    if constr_loss == 'BCE': 
      self.constr_loss = torch.nn.BCELoss()
    elif obj_loss == 'MSE':
      self.constr_loss = torch.nn.MSELoss()

    comb_fn = lambda model : self.user_fn(
       model,self.train_X_0,self.train_X_1,self.train_y_0,self.train_y_1)
    
    
    # Setup OPTIONS
    opts = pygransoStruct()
    opts.torch_device = self.device
    nvar = getNvarTorch(self.model.parameters())
    opts.x0 = torch.nn.utils.parameters_to_vector(
        self.model.parameters()
        ).detach().reshape(nvar,1)
    opts.opt_tol = self.cfg.getfloat('OPTIMIZER','OPT_TOL')
    #opts.viol_ineq_tol = self.cfg.getfloat('OPTIMIZER','VIOL_INEQ_TOL')
    opts.maxit = int(self.cfg.getfloat('OPTIMIZER','MAX_ITER'))
    opts.print_level = self.cfg.getint('OPTIMIZER','PRINT_LEVEL')
    opts.print_frequency = self.cfg.getint('OPTIMIZER','PRINT_FREQ')

    # opts.print_ascii = True
    opts.limited_mem_size = 20
    opts.halt_log_fn = lambda iteration, x, \
      penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, \
        alpha, n_gradients, stat_vec, stat_val, fallback_level: \
          self.summary_epoch(iteration, x, penaltyfn_parts,
                             d,get_BFGS_state_fn, H_regularized, 
                             ls_evals, alpha, n_gradients, 
                             stat_vec, stat_val, fallback_level)

    self.start_time = time.time()
    self.summary_epoch(None,None,None,None,None,None,None,None,None,None,None,None)

    soln = pygranso(var_spec= self.model, combined_fn = comb_fn, user_opts = opts)


  def summary_epoch(self, iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized,
                ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level):
    
    f, ci, diff, accuracy, accuracy0,accuracy1 =\
    self.eval_model(self.train_X_0,self.train_X_1,self.train_y_0,self.train_y_1)
    self.output['f'].append(f)
    self.output['ci'].append(ci)
    self.output['diff'].append(diff)
    self.output['accuracy'].append(accuracy)
    self.output['accuracy0'].append(accuracy0)
    self.output['accuracy1'].append(accuracy1)
    ft, cit, difft, accuracyt,accuracy0t,accuracy1t =\
    self.eval_model(self.test_X_0,self.test_X_1,self.test_y_0,self.test_y_1)
    self.output['f_test'].append(ft)
    self.output['ci_test'].append(cit)
    self.output['diff_test'].append(difft)
    self.output['accuracy_test'].append(accuracyt)
    self.output['accuracy0_test'].append(accuracy0t)
    self.output['accuracy1_test'].append(accuracy1t)
    current_time = time.time() - self.start_time
    self.saveLog()
    if self.output['epoch'] % self.print_summary_every == 0 or self.output['epoch'] == -1:
      for key,value in self.output.items():
        if isinstance(value, list) and len(value) > 0:  # Check if it's a non-empty list
           print(f'>>> {key}: {value[-1]}')
        else:
            print(f'>>> {key}: {value}')
    self.output['epoch'] += 1
    self.output['time'].append(current_time)


  def eval_model(self,X_0,X_1,y_0,y_1):
    #self.model.eval()
    with torch.no_grad():
        num_samples = X_0.size(0) + X_1.size(0)
        pred0 = self.model(X_0)
        pred1 = self.model(X_1)
        #print(pred0.size())
        #print(self.train_y_0.size())
        f = ((self.obj_loss(pred0,y_0) \
            + self.obj_loss(pred1,y_1))/ num_samples).item()
        
        c1 = self.constr_loss(pred0,y_0)\
            - self.constr_loss(pred1,y_1)-self.threshold
        c2 = - self.constr_loss(pred0,y_0)\
        + self.constr_loss(pred1,y_1)-self.threshold
        ci = [c1.item(),c2.item()]

        diff = abs((self.constr_loss(pred0,y_0)\
            - self.constr_loss(pred1,y_1)).item())
        
        pred0 = (pred0 >= 0.5)
        pred1 = (pred1 >= 0.5)
        accuracy0 = ((pred0 == y_0).sum()/X_0.size(0)).item()
        accuracy1 = ((pred1 == y_1).sum()/X_1.size(0)).item()
        num_correct = (pred0 == y_0).sum() + (pred1 == y_1).sum() 
        accuracy = (num_correct/num_samples).item()
    #self.model.train()
    return f, ci, diff, accuracy,accuracy0,accuracy1



  def saveLog(self):
    log = copy.deepcopy(self.output)
    for key in log:
      log[key] = np.array(log[key])
    np.savez(self.fn,**log)