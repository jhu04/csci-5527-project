from .pygranso_problem import pygranso_problem
from .tfco_problem import TFCO_problem
from .pygranso_penalty_problem import pygranso_penalty_problem
from .penalty import penalty_problem
from .problem_no_constr import problem_no_constr_pygranso
def init_problem(cfg,
                   data,
                   device,
                   model,
                   fn = ''):
    PROB_NAME = cfg.get('OPTIMIZER','NAME')
    if PROB_NAME == 'PyGRANSO':
        prob = pygranso_problem(
            cfg,data,device,model,fn=fn
        )
        return prob
    elif PROB_NAME == 'TFCO':
        prob = TFCO_problem(
                cfg,data,device,model,fn=fn
            )
        return prob
    elif PROB_NAME == 'PyGRANSO_PENALTY':
        prob = pygranso_penalty_problem(
            cfg,data,device,model,fn=fn
        )
        return prob
    elif PROB_NAME == 'PENALTY':
        prob = penalty_problem(
            cfg,data,device,model,fn=fn
        )
        return prob
    elif PROB_NAME == 'PLAIN':
        prob = problem_no_constr_pygranso(
            cfg,data,device,model,fn=fn
        )
        return prob
    else:
        raise ValueError(f"Optimizer <{PROB_NAME}> is not implemented")
    