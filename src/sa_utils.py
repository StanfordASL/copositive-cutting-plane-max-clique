import math
import numpy as np
import scipy.sparse
import scipy.optimize

TOL = 1e-6

def get_temps_qubo(Q):
    J = (Q - np.diag(np.diag(Q.A))) / 4.
    h = np.ones(Q.shape[0]) @ Q / 2.
    return get_temps_ising(J, h)

def get_temps_ising(J, h):
    min_temp = np.minimum(np.min(2 * abs(J[J != 0.])), np.min(abs(h[h != 0])))
    max_temp = np.max(2. * np.sum(abs(J), axis = 1) + abs(h))
    return min_temp, max_temp

def binary_mask(n_grid, n_vars):
    assert(n_grid >= 2)
    K = math.ceil(math.log2(n_grid)) - 1
    scale = (2 ** K) / (2 ** (K + 1) - 1)
    row_mask = [scale /(2**k) for k in range(K + 1)]
    mask_rows = [n for n in range(n_vars) for grid_count in range(K + 1) ]
    mask_cols = list(range(n_vars * (K + 1)))
    mask_vals = row_mask*n_vars
    return scipy.sparse.csr_matrix((mask_vals, (mask_rows, mask_cols)))
                                                
def unary_mask(n_grid, n_vars):
    assert(n_grid >= 2)
    N = n_grid - 1
    mask_rows = [n for n in range(n_vars) for grid_count in range(N) ]
    mask_cols = list(range(n_vars * N))
    mask_vals = [1/N] * N * n_vars
    return scipy.sparse.csr_matrix((mask_vals, (mask_rows, mask_cols)))

def expand_qubo_binary(A, n_grid):
    n_vars = A.shape[0]
    mask = binary_mask(n_grid, n_vars)
    return mask.T @ A @ mask

def expand_qubo_unary(A, n_grid):
    n_vars = A.shape[0]
    mask = unary_mask(n_grid, n_vars)
    return mask.T @ A @ mask

def expand_qubo(A, n_grid, encoding):
    if encoding == 'unary':
        return expand_qubo_unary(A, n_grid)
    elif encoding == 'binary':
        return expand_qubo_binary(A, n_grid)  
    else:
        raise ValueError('Encoding type not understood')

def lower_disc_encoding(vi, mask_i): 
    #assert that the mask is sorted in decreasing order
    assert(np.all(np.diff(mask_i) <= 0.))

    if len(mask_i) == 1:
        if vi > mask_i[0]:
            return [1]
        else:
            return [0]
    else:
        e0 = np.zeros(len(mask_i))
        e0[0] = 1.
        if e0 @ mask_i  <= vi:
            return [1] + lower_disc_encoding(vi - e0 @ mask_i, mask_i[1 : ])
        else:
            return [0] + lower_disc_encoding(vi, mask_i[1 : ])
    
def nearest_disc_element(vi, mask_i):
    enc = lower_disc_encoding(vi, mask_i)
    lb = enc @ mask_i
    ub = lb + mask_i[-1]
    assert(ub >= vi)
    assert(lb <= vi)
    if (ub - vi) >= (vi - lb):
        return lb, enc
    else:
        return nearest_disc_element(ub + 1e-5*mask_i[-1], mask_i)

def nearest_disc(v, mask):
    disc_v = np.zeros_like(v)
    disc_enc = []
    for i in range(len(v)):
        row_mask = mask.getrow(i)
        if v[i] > 0.:
            vi = v[i]
            _, nnz_col = row_mask.nonzero()
            mask_i = row_mask[:, nnz_col].A.flatten()
            disc_v[i], disc_enc_vi = nearest_disc_element(vi, mask_i)
            disc_enc.extend(disc_enc_vi)
        else:
            disc_enc.extend(row_mask.nnz * [0])
    return disc_v, np.array(disc_enc)

        
def min_unary_disc(prob, sol):
    bin_n_grid, bin_disc_sol = min_binary_disc(prob, sol)
    n_vars = len(sol)
    
    if bin_n_grid == 2:
        return bin_n_grid, bin_disc_sol
    offset = int(bin_n_grid / 2)
    lb = 0
    ub = bin_n_grid - offset
    
    while (ub - lb) > 1.:
        n_grid = int(np.floor((ub - lb) / 2) + offset)
        mask = unary_mask(n_grid, n_vars)
        disc_sol, _ = nearest_disc(sol, mask)
        if (disc_sol.T @ prob @ disc_sol < 0.):
            ub = int(np.floor((ub - lb) / 2))
        else:
            lb = int(np.floor((ub - lb) / 2))
    
    n_grid = ub + offset
    mask = unary_mask(n_grid, n_vars)
    disc_sol, disc_enc = nearest_disc(sol, mask)
    assert(disc_sol.T @ prob @ disc_sol < 0.)
    return n_grid, disc_sol, disc_enc

def min_binary_disc(prob, sol):
    assert(sol.T @ prob @ sol < 0)
    n_vars = len(sol)
    n_grid = 2
    found_disc = False
    while not found_disc:
        mask = binary_mask(n_grid, n_vars)
        disc_sol, disc_enc = nearest_disc(sol, mask)
        found_disc = (disc_sol.T @ prob @ disc_sol < 0.)
        n_grid *= 2
    return n_grid, disc_sol, disc_enc

def polish(prob, disc_sol):
    qubo_fcn = lambda x: x @ prob @ x
    lb = np.zeros_like(disc_sol)
    ub = np.ones_like(disc_sol)
    
    polished_sol = scipy.optimize.minimize(qubo_fcn, disc_sol,
                                          method='nelder-mead',
                                          bounds = scipy.optimize.Bounds(lb, ub)).x
    return qubo_fcn(polished_sol), polished_sol