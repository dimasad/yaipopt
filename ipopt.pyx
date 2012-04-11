from libc.string cimport memcpy

cimport numpy as npc
import numpy as np


cdef extern from "coin/IpStdCInterface.h":
    cdef struct IpoptProblemInfo:
        pass

    cdef enum ApplicationReturnStatus:
        Solve_Succeeded, Solved_To_Acceptable_Level,
        Infeasible_Problem_Detected, Search_Direction_Becomes_Too_Small,
        Diverging_Iterates, User_Requested_Stop, Feasible_Point_Found,
        Maximum_Iterations_Exceeded, Restoration_Failed,
        Error_In_Step_Computation, Maximum_CpuTime_Exceeded,
        Not_Enough_Degrees_Of_Freedom, Invalid_Problem_Definition,
        Invalid_Option, Invalid_Number_Detected, Unrecoverable_Exception,
        NonIpopt_Exception_Thrown, Insufficient_Memory, Internal_Error
    
    ctypedef double Number
    ctypedef int Index
    ctypedef int Int
    ctypedef int Bool
    ctypedef void *UserDataPtr
    ctypedef IpoptProblemInfo *IpoptProblem
    
    ctypedef Bool (*Eval_F_CB)(Index n, Number* x, Bool new_x,
                               Number* obj_value,
                               UserDataPtr user_data) except? 0
    ctypedef Bool (*Eval_Grad_F_CB)(Index n, Number* x, Bool new_x,
                                    Number* grad_f,
                                    UserDataPtr user_data) except? 0
    ctypedef Bool (*Eval_G_CB)(Index n, Number* x, Bool new_x,
                               Index m, Number* g,
                               UserDataPtr user_data) except? 0
    ctypedef Bool (*Eval_Jac_G_CB)(Index n, Number *x, Bool new_x,
                                   Index m, Index nele_jac,
                                   Index *iRow, Index *jCol, Number *values,
                                   UserDataPtr user_data) except? 0
    ctypedef Bool (*Eval_H_CB)(Index n, Number *x, Bool new_x,Number obj_factor,
                               Index m, Number *lmult, Bool new_lmult,
                               Index nele_hess, Index *iRow, Index *jCol,
                               Number *values, UserDataPtr user_data) except? 0
    ctypedef Bool (*Intermediate_CB)(Index alg_mod,
                                     Index iter_count, Number obj_value,
                                     Number inf_pr, Number inf_du,
                                     Number mu, Number d_norm,
                                     Number regularization_size,
                                     Number alpha_du, Number alpha_pr,
                                     Index ls_trials,
                                     UserDataPtr user_data) except? 0
    
    cdef IpoptProblem CreateIpoptProblem(
        Index n, Number* x_L, Number* x_U, Index m, Number* g_L, Number* g_U,
        Index nele_jac, Index nele_hess, Index index_style, Eval_F_CB eval_f,
        Eval_G_CB eval_g, Eval_Grad_F_CB eval_grad_f, Eval_Jac_G_CB eval_jac_g,
        Eval_H_CB eval_h)

    cdef ApplicationReturnStatus IpoptSolve(
        IpoptProblem ipopt_problem, Number* x, Number* g, Number* obj_val,
        Number* mult_g, Number* mult_x_L, Number* mult_x_U,
        UserDataPtr user_data)


cdef class Problem:
    cdef IpoptProblem prob
    cdef readonly object merit, constr, merit_grad, constr_jac, hess
    cdef readonly object constr_jac_inds, hess_inds
    cdef readonly int m, n
    
    def __init__(self, x_bounds, constr_bounds, constr_jac_inds, hess_inds,
                 merit, constr, merit_grad, constr_jac, hess=None):
        #Convert input to numpy arrays
        x_L = np.require(x_bounds[0], np.double, 'AC')
        x_U = np.require(x_bounds[1], np.double, 'AC')
        constr_L = np.require(constr_bounds[0], np.double, 'AC')
        constr_U = np.require(constr_bounds[1], np.double, 'AC')
        
        if x_L.size != x_U.size or constr_L.size != constr_U.size:
            raise ValueError, 'Upper and lower bounds are of different sizes.'
        
        hess_inds = (np.ravel(hess_inds[0]), np.ravel(hess_inds[1]))
        constr_jac_inds = (np.ravel(constr_jac_inds[0]),
                           np.ravel(constr_jac_inds[1]))

        if (hess_inds[0].size != hess_inds[1].size or
            constr_jac_inds[0].size != constr_jac_inds[1].size):
            raise ValueError, 'Hessian column and row inds of different sizes.'
        
        self.n = x_L.size
        self.m = constr_L.size
        nele_hess = hess_inds[0].size
        nele_jac = constr_jac_inds[0].size
        
        self.merit = merit
        self.constr = constr
        self.merit_grad = merit_grad
        self.constr_jac = constr_jac
        self.hess = hess
        
        self.prob = CreateIpoptProblem(
            self.n, <double*> npc.PyArray_DATA(x_L),
            <double*> npc.PyArray_DATA(x_U), self.m,
            <double*> npc.PyArray_DATA(constr_L),
            <double*> npc.PyArray_DATA(constr_U), nele_jac, nele_hess, 0,
            eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

    def solve(self, start_x):
        x = np.array(start_x, np.double).ravel()
        if x.size != self.n:
            raise ValueError, 'Start point of invalid size.'


cdef Bool eval_f(Index n, Number* x_ptr, Bool new_x, Number* obj_value,
                 UserDataPtr user_data) except? 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *x_dims = [n]
    x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
    
    obj_value[0] = problem.merit(x, new_x)
    
    return 1


cdef Bool eval_grad_f(Index n, Number* x_ptr, Bool new_x, Number* grad_ptr,
                      UserDataPtr user_data) except? 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *x_dims = [n]
    x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
    grad = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, grad_ptr)
    
    grad[:] = np.ravel(problem.merit_grad(x, new_x))
    
    return 1


cdef Bool eval_g(Index n, Number* x_ptr, Bool new_x, Index m, Number* g_ptr,
                 UserDataPtr user_data) except? 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *x_dims = [n]
    cdef npc.npy_intp *g_dims = [m]
    x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
    g = npc.PyArray_SimpleNewFromData(1, g_dims, npc.NPY_DOUBLE, g_ptr)
    
    g[:] = problem.constr(x, new_x)
    
    return 1


cdef Bool eval_jac_g(Index n, Number *x_ptr, Bool new_x, Index m,
                     Index nele_jac, Index *i_ptr, Index *j_ptr,
                     Number *values_ptr, UserDataPtr user_data) except? 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *jac_dims = [nele_jac]
    cdef npc.npy_intp *x_dims = [n]
    
    if values_ptr == NULL:
        i = npc.PyArray_SimpleNewFromData(1, jac_dims, npc.NPY_DOUBLE, i_ptr)
        j = npc.PyArray_SimpleNewFromData(1, jac_dims, npc.NPY_DOUBLE, j_ptr)
        i[:], j[:] = problem.constr_jac_inds
    else:
        x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
        values = npc.PyArray_SimpleNewFromData(
            1, jac_dims, npc.NPY_DOUBLE, values_ptr)
        
        values[:] = problem.constr_jac(x, new_x)
    
    return 1


cdef Bool eval_h(Index n, Number *x_ptr, Bool new_x, Number obj_factor, Index m,
                 Number *lmult_ptr, Bool new_lmult, Index nele_hess,
                 Index *i_ptr, Index *j_ptr, Number *values_ptr,
                 UserDataPtr user_data) except? 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *hess_dims = [nele_hess]
    cdef npc.npy_intp *lmult_dims = [m]
    cdef npc.npy_intp *x_dims = [n]
    
    if problem.hess is None:
        return 0
    
    if values_ptr == NULL:
        i = npc.PyArray_SimpleNewFromData(1, hess_dims, npc.NPY_DOUBLE, i_ptr)
        j = npc.PyArray_SimpleNewFromData(1, hess_dims, npc.NPY_DOUBLE, j_ptr)
        i[:], j[:] = problem.hess_inds
    else:
        x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
        lmult = npc.PyArray_SimpleNewFromData(
            1, lmult_dims, npc.NPY_DOUBLE, lmult_ptr)
        values = npc.PyArray_SimpleNewFromData(
            1, hess_dims, npc.NPY_DOUBLE, values_ptr)
        
        values[:] = problem.hess(x, new_x, obj_factor, lmult, new_lmult)
    
    return 1


cdef Bool intermediate_callback(Index alg_mod, Index iter_count,
                                Number obj_value, Number inf_pr, Number inf_du,
                                Number mu, Number d_norm,
                                Number regularization_size,  Number alpha_du,
                                Number alpha_pr, Index ls_trials,
                                UserDataPtr user_data) except? 0:
    return 1


def create(*args):
    pass
