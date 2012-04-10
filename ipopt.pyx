cimport numpy as npc
import numpy as np


cdef extern from "coin/IpStdCInterface.h":
    cdef struct IpoptProblemInfo:
        pass
    
    ctypedef double Number
    ctypedef int Index
    ctypedef int Int
    ctypedef int Bool
    ctypedef void *UserDataPtr
    ctypedef IpoptProblemInfo *IpoptProblem
    
    ctypedef Bool (*Eval_F_CB)(Index n, Number* x, Bool new_x,
                               Number* obj_value,
                               UserDataPtr user_data) except 0
    ctypedef Bool (*Eval_Grad_F_CB)(Index n, Number* x, Bool new_x,
                                    Number* grad_f,
                                    UserDataPtr user_data) except 0
    ctypedef Bool (*Eval_G_CB)(Index n, Number* x, Bool new_x,
                               Index m, Number* g,
                               UserDataPtr user_data) except 0
    ctypedef Bool (*Eval_Jac_G_CB)(Index n, Number *x, Bool new_x,
                                   Index m, Index nele_jac,
                                   Index *iRow, Index *jCol, Number *values,
                                   UserDataPtr user_data) except 0
    ctypedef Bool (*Eval_H_CB)(Index n, Number *x, Bool new_x,Number obj_factor,
                               Index m, Number *lambda_, Bool new_lambda,
                               Index nele_hess, Index *iRow, Index *jCol,
                               Number *values, UserDataPtr user_data) except 0
    ctypedef Bool (*Intermediate_CB)(Index alg_mod,
                                     Index iter_count, Number obj_value,
                                     Number inf_pr, Number inf_du,
                                     Number mu, Number d_norm,
                                     Number regularization_size,
                                     Number alpha_du, Number alpha_pr,
                                     Index ls_trials,
                                     UserDataPtr user_data) except 0
    
    cdef IpoptProblem CreateIpoptProblem(
        Index n, Number* x_L, Number* x_U, Index m, Number* g_L, Number* g_U,
        Index nele_jac, Index nele_hess, Index index_style, Eval_F_CB eval_f,
        Eval_G_CB eval_g, Eval_Grad_F_CB eval_grad_f, Eval_Jac_G_CB eval_jac_g,
        Eval_H_CB eval_h)


cdef class Problem:
    cdef IpoptProblem prob
    cdef readonly object merit, constr, merit_grad, constr_jac, hess
    
    def __cinit__(self, x_L, x_U, constr_L, constr_U, nele_jac, nele_hess,
                  merit, constr, merit_grad, constr_jac, hess=None):
        #Convert input to numpy arrays
        x_L = np.asarray(x_L, np.double).ravel()
        x_U = np.asarray(x_U, np.double).ravel()
        constr_L = np.asarray(constr_L, np.double).ravel()
        constr_U = np.asarray(constr_U, np.double).ravel()
        
        if len(x_L) != len(x_U) or len(constr_L) != len(constr_U):
            raise ValueError, 'Upper and lower bounds are of different sizes.'
        
        n = len(x_L)
        m = len(constr_L)

        self.merit = merit
        self.constr = constr
        self.merit_grad = merit_grad
        self.constr_jac = constr_jac
        self.hess = hess
        
        self.prob = CreateIpoptProblem(
            n, <Number*> npc.PyArray_DATA(x_L), <Number*> npc.PyArray_DATA(x_U),
            m, <Number*> npc.PyArray_DATA(constr_L),
            <Number*> npc.PyArray_DATA(constr_U), nele_jac, nele_hess, 0,
            eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)


cdef Bool eval_f(Index n, Number* x_ptr, Bool new_x, Number* obj_value,
                 UserDataPtr user_data) except 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *x_dims = [n]
    x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
    
    obj_value[0] = problem.merit(x, new_x)
    
    return 1


cdef Bool eval_grad_f(Index n, Number* x, Bool new_x, Number* grad_f,
                      UserDataPtr user_data) except 0:
    return 1


cdef Bool eval_g(Index n, Number* x, Bool new_x, Index m, Number* g,
                 UserDataPtr user_data) except 0:
    return 1


cdef Bool eval_jac_g(Index n, Number *x, Bool new_x, Index m, Index nele_jac,
                     Index *iRow, Index *jCol, Number *values,
                     UserDataPtr user_data) except 0:
    return 1


cdef Bool eval_h(Index n, Number *x, Bool new_x,Number obj_factor, Index m,
                 Number *lambda_, Bool new_lambda, Index nele_hess,
                 Index *iRow, Index *jCol, Number *values,
                 UserDataPtr user_data) except 0:
    return 1


cdef Bool intermediate_callback(Index alg_mod, Index iter_count,
                                Number obj_value, Number inf_pr, Number inf_du,
                                Number mu, Number d_norm,
                                Number regularization_size,  Number alpha_du,
                                Number alpha_pr, Index ls_trials,
                                UserDataPtr user_data) except 0:
    return 1


def create(*args):
    pass
