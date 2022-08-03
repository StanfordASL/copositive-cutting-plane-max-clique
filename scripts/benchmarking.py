import argparse
import csv
import dill
import dwave_networkx as dnx
import glob
import gurobipy as gp
from gurobipy import GRB
from hyperopt import hp, tpe, fmin
import itertools
from importlib import reload
import json
import logging
import numpy as np
import neal
import networkx as nx
import scipy.sparse
from scipy.sparse import coo_matrix
from os.path import exists
import os
import sys
import time

sys.path.append('../src/')
import sa_utils
import read_qubo

data_path = '../data/'
log_suffix = 'result_cp_log_GT_anst.txt'
TOL = 0.05

n_reads = 1000
phot = 0.5
pcold = 0.01

class seen_result:
    def __init__(self):
        self.obj = np.inf
        self.idx_list = []
        self.runtime_list = []
        
def qubo2J(qubo):
    J = {}
    for (r, c) in zip(*qubo.nonzero()):
        r, c = sorted((r, c))
        if (r, c) in J:
            J[(r, c)] += qubo[r, c]
        else:
            J[(r, c)] = qubo[r, c]
    return J

def recover_graph(cop):
    constr_mat = cop.Ai[0]
    comp_adj = -(constr_mat + scipy.sparse.identity(constr_mat.shape[0]))
    A_bar = nx.from_scipy_sparse_array(comp_adj)
    A = nx.complement(A_bar)
    
    return A

def save_data(data, filename):
    with open(filename, 'w') as f:
        f_out=csv.writer(f)
        for row in data:
            f_out.writerow(row)

def save_filenames(instance_name, idx, ngrid, sweeps, data_dir):
    neal_dir = data_dir + instance_name + 'neal_tuning/'
    if not os.path.isdir(neal_dir):
        try:
            os.makedirs(neal_dir)
        except:
            pass
        
    suffix = 'iter_{}_ngrid_{}_swe_{}'.format(idx, int(ngrid), sweeps)
    pickle_file = neal_dir + suffix + '_neal.pkl'
    summary_file = neal_dir + suffix + '_summary.pkl'
    
    return pickle_file, summary_file
    
def sol_to_bitstring(sol):
    bitstring = ''.join([str(int(x)) for x in sol])
    return bitstring
def gp_additive(A):
    nv = A.number_of_nodes()
    A_bar = nx.complement(A)

    model = gp.Model('model')
    model.setParam(GRB.Param.Threads, 1)
    model.setParam('OutputFlag', 0 )

    x = model.addMVar(shape = nv, vtype = GRB.BINARY)
    for u, v in A_bar.edges():
        model.addConstr(x[u] + x[v] <= 1)

    model.setObjective(sum(x), GRB.MAXIMIZE)
    model.optimize()
    
    return int(model.ObjVal), model.runtime

def gp_mult(A):
    nv = A.number_of_nodes()
    A_bar = nx.complement(A)

    model = gp.Model('model')
    model.setParam(GRB.Param.Threads, 1)
    model.setParam('OutputFlag', 0 )

    x = model.addMVar(shape = nv, vtype = GRB.BINARY)
    for u, v in A_bar.edges():
        model.addConstr(x[u]*x[v] == 0.)

    model.setObjective(sum(x), GRB.MAXIMIZE)
    model.optimize()
    
    return int(model.ObjVal), model.runtime

def is_clique(A, sample):
    A_sub = A.subgraph(sample)
    n_edges = A_sub.number_of_edges()
    n = len(sample)
    
    return (n_edges == (n * (n - 1) / 2))

def sa_sampler(A, penalty):
    n_reads = 1000
    valid_clique_count = []
    sampler_sizes = []
    timing = []
    for read in range(n_reads):
        start_time = time.time()
        sample = dnx.maximum_clique(A, neal.SimulatedAnnealingSampler(), penalty)
        stop_time = time.time()
        
        sampler_sizes.append(len(sample))
        valid_clique_count.append(is_clique(A, sample))
        timing.append(stop_time - start_time)
        
    return valid_clique_count, sampler_sizes, timing


def process_neal(res, base_prob, ngrid, GT_obj):
    n_vars = base_prob.shape[0]
    mask = sa_utils.binary_mask(ngrid, n_vars)
    n_reads = len(res)
    seen = {}
    for read_num, rec in enumerate(res.record):
        neal_sol = rec[0]
        neal_obj = rec[1]
        bit_neal_sol = sol_to_bitstring(neal_sol)
        
        if bit_neal_sol in seen:
            seen[bit_neal_sol].idx_list.append(read_num)
        else:
            seen[bit_neal_sol] = seen_result()
            seen[bit_neal_sol].obj = neal_obj
            seen[bit_neal_sol].idx_list = [read_num]
    
    successes = []
    for k, v in seen.items():
        if v.obj >= 0.:
            successes.append(0)
        elif '1' not in k:
            successes.append(0) 
        else:
            p_succ = lambda x: min(1., x / GT_obj)
            successes.append(p_succ(v.obj) * len(v.idx_list))
    
    obj = np.sum(successes) / n_reads
    return obj, seen

def neal_tuning(args, base_prob, instance_name, idx, ngrid, GT_obj, data_dir):
    sweeps = int(args['sweeps'])    
    neal_res_file, summary_file = save_filenames(instance_name, idx, ngrid, sweeps, data_dir)
    
    if exists(summary_file):
        with open(summary_file, 'rb') as f:
            obj = dill.load(f)
            walltime = dill.load(f)
            seen = dill.load(f)

    else:
        qubo = sa_utils.expand_qubo_binary(base_prob, ngrid)
        min_temp, max_temp = sa_utils.get_temps_qubo(qubo)
        min_temp /= np.log(1. / pcold)
        max_temp /= np.log(1. / phot)
        
        J = qubo2J(qubo)
        sampler = neal.SimulatedAnnealingSampler()
        walltime_start = time.time()
        res = sampler.sample_qubo(J, beta_range = (1/max_temp, 1/min_temp), num_reads=n_reads, num_sweeps=sweeps)
        walltime_stop = time.time()
        walltime = walltime_stop - walltime_start

        obj, seen = process_neal(res, base_prob, ngrid, GT_obj)
        with open(summary_file, 'wb') as f:
            dill.dump(obj, f, dill.HIGHEST_PROTOCOL)
            dill.dump(walltime, f, dill.HIGHEST_PROTOCOL)
            dill.dump(seen, f, dill.HIGHEST_PROTOCOL)
    
    mean_time = walltime / n_reads
    s = 0.99
    if obj <=  0.:
        tts = 1e15
    elif obj >= 1.:
        tts = mean_time
    else:
        tts = mean_time * np.log(1 - s) / np.log(1 - obj)
    logger=logging.getLogger()
    logger.info(str(args) + 'tts: ' + str(tts))
    
    return tts

def run_gurobi(base_prob, data_dir, instance_name, idx):
    csv_name = data_dir + instance_name + 'iter_{}_gurobi_bds.csv'.format(idx)
    qubo_size = base_prob.shape[0]
    
    data = []
    def data_cb(model, where):
        if where == gp.GRB.Callback.MIP:
            cur_obj = model.cbGet(gp.GRB.Callback.MIP_OBJBST)
            cur_bd = model.cbGet(gp.GRB.Callback.MIP_OBJBND)
            cur_time = model.cbGet(gp.GRB.Callback.RUNTIME)

            data.append((cur_time, cur_bd, cur_obj))
            
    m = np.ones(qubo_size) + np.sum(np.clip(base_prob.A, 0, None), axis= 1)
    model = gp.Model('model')
    model.setParam(GRB.Param.Threads, 1)
    model.setParam('LogFile', '')
    gamma = model.addMVar(shape=1, lb = 0, vtype=GRB.CONTINUOUS)
    y = model.addMVar(shape=qubo_size, vtype=GRB.BINARY, name='y')
    x = model.addMVar(shape=qubo_size, vtype=GRB.CONTINUOUS, name = 'x')
    model.addConstr(x <= y)
    model.addConstr(sum(y) >= 1)

    for i in range(qubo_size):
        model.addConstr(base_prob[i, :]@ x<= -gamma + m[i]*(1 - y[i]))
    model.setObjective(gamma, GRB.MAXIMIZE)
    model.optimize(callback=data_cb)
    data.append((model.Runtime, model.objBound, model.objVal))
    
    save_data(data,  csv_name)
    
    return 
    
def run_neal(base_prob, data_dir, instance_name, GT_obj, idx):
    sweeps = 100
    
    neal_res_file, summary_file = save_filenames(instance_name, idx, ngrid, sweeps, data_dir)

    qubo = sa_utils.expand_qubo_binary(base_prob, ngrid)
    min_temp, max_temp = sa_utils.get_temps_qubo(qubo)
    min_temp /= np.log(1. / pcold)
    max_temp /= np.log(1. / phot)

    J = qubo2J(qubo)
    sampler = neal.SimulatedAnnealingSampler()
    walltime_start = time.time()
    res = sampler.sample_qubo(J, beta_range = (1/max_temp, 1/min_temp), num_reads=n_reads, num_sweeps=sweeps)
    walltime_stop = time.time()
    walltime = walltime_stop - walltime_start

    obj, seen = process_neal(res, base_prob, ngrid, GT_obj)
    with open(summary_file, 'wb') as f:
        dill.dump(obj, f, dill.HIGHEST_PROTOCOL)
        dill.dump(walltime, f, dill.HIGHEST_PROTOCOL)
        dill.dump(seen, f, dill.HIGHEST_PROTOCOL)
    return


def main(param, args):
    spaceVar = {'sweeps': hp.qloguniform('sweeps', 0, 9, 1)}
    N = param['N']
    n = param['n']
    p = param['p']
    
    global ngrid
    ngrid = param['ngrid']
    
    global instance_name 
    instance_name = 'N={}_n={}_p={}./'.format(N, n, p)
    
    global data_dir
    data_dir = data_path + args.data_dir + '/'
    
    cop = read_qubo.COP()
    cop.read_all(data_dir+instance_name, log_suffix)
    A = recover_graph(cop)
    
    if args.gp_mip:
        GT_val, gp_add_time = gp_additive(A)
        _, gp_mult_time = gp_mult(A)
    
        gb_bl_file = data_dir + instance_name + 'gp_baselines.txt'
        with open(gb_bl_file, 'w') as f:
            f.write(str(GT_val) + '\n')
            f.write(str(gp_add_time) + '\n')
            f.write(str(gp_mult_time) + '\n')
            
    if args.dw_mc:
        for penalty_exp in range(-1, 5):
            penalty = 2**penalty_exp
            valid_clique_count, sampler_sizes, timing = sa_sampler(A, penalty)
            dw_bl_file = data_dir + instance_name + 'dwave_baselines_{}.csv'.format(penalty)

            with open(dw_bl_file, 'w') as f:
                f_writer = csv.writer(f, delimiter=',')
                for entry in zip(valid_clique_count, sampler_sizes, timing):
                    f_writer.writerow(entry)
        
    if p == 0.75 and n >= 71:
        return
    
    for idx in cop.get_infeas_idx():
        reload(logging)
        base_prob = cop.get_qubo(idx)
        GT_obj, GT_sol = cop.get_scaled_cop_min(idx)
        
        # Only run gurobi profiling once
        if args.gp_cop and ngrid == 2: 
            run_gurobi(base_prob, data_dir, instance_name, idx)
        
        # Run dwave neal with fixed # of sweeps first
        if args.fixed:
            run_neal(base_prob, data_dir, instance_name, GT_obj, idx)
        
        # Run full hyperparameter tuning experiment
        if args.hpo:
            log_name = data_dir + instance_name + 'iter_{}_ngrid_{}.log'.format(idx, int(ngrid))
            logging.basicConfig(filename=log_name, format='%(asctime)s %(message)s',filemode='w', force=True)
            logger=logging.getLogger()
            logger.setLevel(logging.INFO)

            objective = lambda args : neal_tuning(args, base_prob, instance_name, idx, ngrid, GT_obj, data_dir)
            best = fmin(fn = objective, space=spaceVar, algo=tpe.suggest, max_evals = 25)
            
            
if __name__ == '__main__':
    # Set up parameters dictionary
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='a positional argument, directory to save data in')
    parser.add_argument('--param', type=str, help='parameter file number: parameters{#}.json')
    parser.add_argument('--hpo', action='store_true', help='run hyperopt parameter tuning')
    parser.add_argument('--fixed', action='store_true', help='run fixed neal with 100 sweeps')
    parser.add_argument('--gp_cop', action='store_true', help='run gurobi copositivity check experiment')
    parser.add_argument('--gp_mip', action='store_true', help='run gurobi mip formulations')
    parser.add_argument('--dw_mc', action='store_true', help='run dwave maximum clique sampler')
    
    args = parser.parse_args()
    fname = '../scripts/parameters/parameters'+args.param+'.json'
    f = open(fname)
    params = json.load(f)
    keys = list(params)
    
    for values in itertools.product(*map(params.get, keys)):
        params = dict(zip(keys, values))
        main(params, args)
