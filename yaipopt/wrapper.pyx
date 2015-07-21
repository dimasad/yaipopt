from libc.string cimport memcpy

cimport numpy as npc
import numpy as np

npc.import_array()


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
    ctypedef IpoptProblemInfo* IpoptProblem
    
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
    cdef void FreeIpoptProblem(IpoptProblem prob)
    cdef Bool AddIpoptStrOption(IpoptProblem prob, char* keyword, char* val)
    cdef Bool AddIpoptNumOption(IpoptProblem prob, char* keyword, Number val)
    cdef Bool AddIpoptIntOption(IpoptProblem prob, char* keyword, Int val)
    cdef Bool OpenIpoptOutputFile(IpoptProblem prob, char* file_name,
                                  Int print_level)
    cdef Bool SetIpoptProblemScaling(IpoptProblem prob, Number obj_scaling,
                                     Number* x_scaling, Number* g_scaling)
    cdef Bool SetIntermediateCallback(IpoptProblem ipopt_problem,
                                      Intermediate_CB intermediate_cb)


cdef class Problem:
    cdef IpoptProblem ipopt_problem
    cdef readonly object obj, constr, obj_grad, constr_jac, hess
    cdef readonly object constr_jac_inds, hess_inds
    cdef readonly int m, n
    
    def __init__(self, x_bounds, obj, obj_grad,
                 constr_bounds=None, constr=None, constr_jac=None,
                 constr_jac_inds=None, hess=None, hess_inds=None):
        
        # Unconstrained case
        if constr is None:
            constr_bounds = constr_jac_inds = np.zeros((2, 0))
        
        # Quasi-Newton case
        if hess is None:
            hess_inds = np.zeros((2, 0))
        
        # Check dimensions of input arguments
        if len(x_bounds) != 2 or len(constr_bounds) != 2:
            raise ValueError, 'Bounds of invalid lengths.'

        if len(hess_inds) != 2 or len(constr_jac_inds) != 2:
            raise ValueError, 'Indices of invalid lengths.'

        # Convert input to numpy arrays
        x_bounds = np.ascontiguousarray(x_bounds, np.double)
        constr_bounds = np.ascontiguousarray(constr_bounds, np.double)
        hess_inds = np.ascontiguousarray(hess_inds, np.int)
        constr_jac_inds = np.ascontiguousarray(constr_jac_inds, np.int)
        
        # Define problem dimensions
        self.n = x_bounds.size / 2
        self.m = constr_bounds.size / 2
        cdef int nele_hess = hess_inds.size / 2
        cdef int nele_jac = constr_jac_inds.size / 2
        
        # Check bounds of indices. Invalid bounds generate segfault
        if np.any(hess_inds < 0) or np.any(hess_inds >= self.n):
            raise ValueError, 'Hessian indices out of range.'
        
        if (np.any(constr_jac_inds < 0) or
            np.any(constr_jac_inds >= [[self.m], [self.n]])):
            raise ValueError, 'Constraint Jacobian indices out of range.'
        
        # Save problem data
        self.obj = obj
        self.constr = constr
        self.obj_grad = obj_grad
        self.constr_jac = constr_jac
        self.constr_jac_inds = constr_jac_inds
        self.hess = hess
        self.hess_inds = hess_inds
        
        # Create IPOPT's representation of the problem
        self.ipopt_problem = CreateIpoptProblem(
            self.n, <double*> npc.PyArray_DATA(x_bounds[0]),
            <double*> npc.PyArray_DATA(x_bounds[1]), self.m,
            <double*> npc.PyArray_DATA(constr_bounds[0]),
            <double*> npc.PyArray_DATA(constr_bounds[1]), nele_jac, nele_hess,
            0, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

        if nele_hess == 0 or hess is None:
            self.str_option(b'hessian_approximation', b'limited-memory')
    
    def __dealloc__(self):
        if self.ipopt_problem is not NULL:
            FreeIpoptProblem(self.ipopt_problem)

    def str_option(self, option, value):
        if isinstance(option, str):
            option = option.encode('ascii')
        if isinstance(value, str):
            value = value.encode('ascii')
        return bool(AddIpoptStrOption(self.ipopt_problem, option, value))
    
    def num_option(self, option, value):
        if isinstance(option, str):
            option = option.encode('ascii')
        return bool(AddIpoptNumOption(self.ipopt_problem, option, value))

    def int_option(self, option, value):
        if isinstance(option, str):
            option = option.encode('ascii')
        return bool(AddIpoptIntOption(self.ipopt_problem, option, value))
    
    def solve(self, start_x):
        if self.ipopt_problem is NULL:
            raise RuntimeError, 'Problem not initialized.'
        
        x = np.array(start_x, np.double).ravel()
        if x.size != self.n:
            raise ValueError, 'Start point of invalid size.'
        
        cdef double obj_val
        cdef npc.npy_intp* constr_dims = [self.m]
        cdef npc.npy_intp* x_dims = [self.n]
        
        x_mult_L = npc.PyArray_SimpleNew(1, x_dims, npc.NPY_DOUBLE)
        x_mult_U = npc.PyArray_SimpleNew(1, x_dims, npc.NPY_DOUBLE)
        constr = npc.PyArray_SimpleNew(1, constr_dims, npc.NPY_DOUBLE)
        constr_mult = npc.PyArray_SimpleNew(1, constr_dims, npc.NPY_DOUBLE)
        
        status = IpoptSolve(
            self.ipopt_problem, <double*> npc.PyArray_DATA(x),
            <double*> npc.PyArray_DATA(constr), &obj_val,
            <double*> npc.PyArray_DATA(constr_mult),
            <double*> npc.PyArray_DATA(x_mult_L),
            <double*> npc.PyArray_DATA(x_mult_U), <void*> self)
        
        info = {'status': status, 'obj_val': obj_val, 'constr': constr,
                'constr_mul': constr_mult, 'x_mult_L': x_mult_L,
                'x_mult_U': x_mult_U}
        
        return x, info


cdef Bool eval_f(Index n, Number* x_ptr, Bool new_x, Number* obj_value,
                 UserDataPtr user_data) except? 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *x_dims = [n]
    x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
    
    obj_value[0] = problem.obj(x, new_x)
    
    return 1


cdef Bool eval_grad_f(Index n, Number* x_ptr, Bool new_x, Number* grad_ptr,
                      UserDataPtr user_data) except? 0:
    cdef Problem problem = <Problem>  user_data
    cdef npc.npy_intp *x_dims = [n]
    x = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, x_ptr)
    grad = npc.PyArray_SimpleNewFromData(1, x_dims, npc.NPY_DOUBLE, grad_ptr)
    
    grad[:] = problem.obj_grad(x, new_x)
    
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
        i = npc.PyArray_SimpleNewFromData(1, jac_dims, npc.NPY_INT, i_ptr)
        j = npc.PyArray_SimpleNewFromData(1, jac_dims, npc.NPY_INT, j_ptr)
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
        i = npc.PyArray_SimpleNewFromData(1, hess_dims, npc.NPY_INT, i_ptr)
        j = npc.PyArray_SimpleNewFromData(1, hess_dims, npc.NPY_INT, j_ptr)
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
