import numpy as np
from scipy.sparse import coo_matrix

def get_constraint_matrix(m):
    def get_expr_coos(expr, var_indices):
        for i in range(expr.size()):
            dvar = expr.getVar(i)
            yield expr.getCoeff(i), var_indices[dvar]

    dvars = m.getVars()
    constrs = m.getConstrs()
    var_indices = {v: i for i, v in enumerate(dvars)}
    rows = []
    cols = []
    coeffs = []
    for row_idx, constr in enumerate(constrs):
        for coeff, col_idx in get_expr_coos(m.getRow(constr), var_indices):
            coeffs.append(coeff)
            rows.append(row_idx)
            cols.append(col_idx)

    return coo_matrix((np.array(coeffs), (np.array(rows), np.array(cols))))
