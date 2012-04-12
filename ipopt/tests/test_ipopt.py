import numpy as np

import ipopt


def test_hs071():
    x_bounds = (np.ones(4), np.repeat(5.0, 4))
    constr_bounds = ([25, 40], [np.inf, 40])

    obj = lambda x, new_x: x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]
    obj_grad = lambda x, new_x: [x[0]*x[3] + x[3]*(x[0] + x[1] + x[2]),
                                 x[0]*x[3],
                                 x[0]*x[3] + 1.0,
                                 x[0]*(x[0] + x[1] + x[2])]
    constr = lambda x, new_x: [x[0]*x[1]*x[2]*x[3], 
                               x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]]
    
    constr_jac_inds = ([0, 0, 0, 0, 1, 1, 1, 1],
                       [0, 1, 2, 3, 0, 1, 2, 3])
    hess_inds = ([0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
                 [0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    
    constr_jac = lambda x, new_x: [x[1]*x[2]*x[3], 
                                   x[0]*x[2]*x[3], 
                                   x[0]*x[1]*x[3], 
                                   x[0]*x[1]*x[2],
                                   2.0*x[0],
                                   2.0*x[1],
                                   2.0*x[2],
                                   2.0*x[3]]
    hess = lambda x, new_x, obj_factor, lmult, new_lmult: [
        obj_factor*2*x[3] + lmult[1]*2,
        obj_factor*x[3] + lmult[0]*(x[2]*x[3]),
        lmult[1]*2,
        obj_factor*(x[3]) + lmult[0]*(x[1]*x[3]),
        lmult[0]*(x[0]*x[3]),
        lmult[1]*2,
        obj_factor*(2*x[0] + x[1] + x[2]) +  lmult[0]*x[1]*x[2],
        obj_factor*x[0] + lmult[0]*x[0]*x[2],
        obj_factor*x[0] + lmult[0]*x[0]*x[1],
        lmult[1]*2]

    problem = ipopt.Problem(x_bounds, constr_bounds, constr_jac_inds, hess_inds,
                            obj, constr, obj_grad, constr_jac, hess)

    x0 = [1.0, 5.0, 5.0, 1.0]
    xopt, info = problem.solve(x0)

    expected_xopt = [1, 4.743, 3.82115, 1.379408]
    np.testing.assert_almost_equal(xopt, expected_xopt, decimal=6)
