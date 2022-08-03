import csv
import dill
import glob
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
import pandas as pd
import sys
sys.path.append("../src/")
import time
import read_qubo

n_reads = 1000

data_dir = '../data/exp_raw/'
log_suffix = 'result_cp_log_GT_anst.txt'
neal_log_suffix = 'result_cp_log_GT_neal.txt'

def read_gb_baselines(gb_bl_file):
    if exists(gb_bl_file):
        with open(gb_bl_file) as f:
            GT_val = int(f.readline().strip())
            gp_add_time = float(f.readline().strip())
            gp_mult_time = float(f.readline().strip())
    else:
        GT_val = None
        gp_add_time = None
        gp_mult_time = None    
    return GT_val, gp_add_time, gp_mult_time

def read_neal_baselines(neal_bl_file):
    valid_clique_count = []
    sampler_sizes = []
    timing = []

    cc_convert = lambda cc : 1 if cc=='True' else 0
    
    if exists(neal_bl_file):
        with open(neal_bl_file) as f:
            f_reader = csv.reader(f, delimiter=',')
            for cc, ss, t in f_reader:
                valid_clique_count.append(cc_convert(cc))
                sampler_sizes.append(int(ss))
                timing.append(float(t))

    return np.array(valid_clique_count), np.array(sampler_sizes), np.array(timing)

def get_final_oa(instance_name):
    cop = read_qubo.COP()
    cop.read_all(data_dir+instance_name, neal_log_suffix)
    return cop.final_oa

def read_baselines(N, n, p, ngrid):
    instance_name = 'N={}_n={}_p={}./'.format(N, n, p)
    gb_bl_file = data_dir + instance_name + 'gp_baselines.txt'
    GT_val, gp_add_time, gp_mult_time = read_gb_baselines(gb_bl_file)
    
    vcc_list = []
    ss_list = []
    t_list = []
    
    for penalty_exp in range(-1, 5):
        penalty = 2**penalty_exp
        neal_bl_file = data_dir + instance_name + 'dwave_baselines_{}.csv'.format(penalty)
        vcc, ss, t = read_neal_baselines(neal_bl_file)
        vcc_list.append(vcc)
        ss_list.append(ss)
        t_list.append(t)
    return GT_val, gp_add_time, gp_mult_time, vcc_list, ss_list, t_list

def read_gurobi_bds(gp_summary_file):
    gp_bds = []
    with open(gp_summary_file, 'r') as file:
        line = file.readline()
        start_time = time.time()
        while line:
            row = line.strip().split(',')
            gp_bds.append([float(r) for r in row])
            if time.time() - start_time >= 30.:
                break
            line = file.readline()
    gp_t = [float(v) for v,_,_ in gp_bds]
    gp_ub = [float(v) for _,v,_ in gp_bds]
    gp_best = [float(v) for _,_,v in gp_bds]
    
    return gp_t, gp_ub, gp_best

def rescale_gp(gp_t, gp_best, gp_ub):
    if len(gp_t) == 0:
        return gp_t, gp_best, gp_ub
    
    scale_fact = gp_best[-1]
    if scale_fact == 0.:
        return gp_t, gp_best, gp_ub
    
    scaled_best = [s / scale_fact for s in gp_best]
    scaled_ub = [s / scale_fact for s in gp_ub]
    return gp_t,  scaled_best, scaled_ub

def read_hyperopt_log(tuning_log):
#     print(tuning_log)
    TTS = []
    best_TTS = []
    swe_list = []
    running_min_tts = np.inf
    with open(tuning_log, 'r') as log:
        line = log.readline() #'build posterio wrapper line'
        while line:
            line = log.readline() # TPE info
            line = log.readline() # swe, tts line

            start_idx = line.find('{')
            end_idx = line.find('}') + 1
            try:
                param_dict = json.loads(line[start_idx:end_idx].replace("'", '"'))
            except:
                break
        #             replicas = int(param_dict['replicas'])
            replicas = 0
            sweeps = int(param_dict['sweeps'])
            try:
                tts = float(line[line.find('tts') + 5:].strip())
            except:
                print(tuning_log)
                print(line)
            running_min_tts = np.minimum(running_min_tts, tts)
            swe_list.append(sweeps)
            best_TTS.append(running_min_tts)
            TTS.append(tts)
            line = log.readline() #'build posterior wrapper line'
    return best_TTS, swe_list, TTS

def read_summary(summary_file):
    if exists(summary_file):
        with open(summary_file, 'rb') as summary:
            obj = dill.load(summary)
            walltime = dill.load(summary)
            seen = dill.load(summary)
    else:
        obj, walltime, seen = (None, None, None)
    return obj, walltime, seen

def get_gp_single_time(instance_name, idx):
    cop = read_qubo.COP()
    cop.read_all(data_dir+instance_name, log_suffix)
    if idx >= len(cop.cop_info_list):
        gp_time = None
    else:
        gp_time = cop.cop_info_list[idx].end_time - cop.cop_info_list[idx].start_time
    return gp_time

def get_profile_breakdown(instance_name, logname):
    M_file = data_dir + instance_name + 'result_M.txt'
    constr_file = data_dir + instance_name + 'result_constr.txt'
    log_file = data_dir + instance_name + logname
    
    if exists(M_file) and exists(constr_file) and exists(log_file):
        cop = read_qubo.COP()
        cop.read_all(data_dir+instance_name, logname)
        if len(cop.cop_info_list) == 0 or len(cop.cutting_plane_info_list) == 0:
            full_time = None
            cop_times  = None
        else:
            full_start = cop.cop_info_list[0].start_time
            if len(cop.cop_info_list) > len(cop.cutting_plane_info_list):
                println("Falling back")
                full_stop = cop.cop_info_list[-1].end_time
            else:
                full_stop = cop.cutting_plane_info_list[-1].curr_time
            cop_times = sum(c.end_time - c.start_time for c in cop.cop_info_list)
            full_time = full_stop - full_start
    else:
        full_time = None
        cop_times = None
    
    return full_time, cop_times

def partial_data_tuple(N, n, p, ngrid):
    instance_name = 'N={}_n={}_p={}./'.format(N, n, p)
    full_time, cop_times = get_partial_profile_breakdown(instance_name)
    rest_time = full_time - cop_times
    
    return cop_times, rest_time
    
def data_tuple(N, n, p, ngrid):
    instance_name = 'N={}_n={}_p={}./'.format(N, n, p)
    dir_list = glob.glob(data_dir+instance_name+'*gurobi_bds.csv')
    if len(dir_list) == 0:
        gp_t, gp_ub, gp_best = (None, None, None)
        best_TTS_list = []
        swe_list = []
        TTS = []
        best_TTS = None
        idx = None
        gp_infeas_time = None
    
    else:
        gp_summary_file = dir_list[0]
        gp_t, gp_ub, gp_best = rescale_gp(*read_gurobi_bds(gp_summary_file))
        idx = int(gp_summary_file[gp_summary_file.find('iter') + 5 :gp_summary_file.find('gurobi') - 1])
        tuning_log = data_dir + instance_name + 'iter_{}_ngrid_{}.log'.format(idx, ngrid)
        best_TTS_list, swe_list, TTS = read_hyperopt_log(tuning_log)
        best_TTS = best_TTS_list[-1]
        gp_infeas_time = get_gp_single_time(instance_name, idx)
    

    sweeps = 100
    fixed_swe_file = data_dir + instance_name +\
    'neal_tuning/iter_{}_ngrid_{}_swe_{}_summary.pkl'.format(idx, ngrid, sweeps)
    obj, walltime, seen = read_summary(fixed_swe_file)
    
    if obj == None:
        tts_99 = None
        tts_999 = None
    else:
        if obj == 0.:
            tts_99 = 1e6
            tts_999 = 1e6
        elif obj == 1.:
            tts_99 = walltime / n_reads
            tts_999 = walltime / n_reads
        else:
            tts_99 = walltime / n_reads * np.log(1 - 0.99) / np.log(1 - obj)
            tts_999 = walltime / n_reads * np.log(1 - 0.999) / np.log(1 - obj)
        
    
    full_time_gp, cop_times_gp = get_profile_breakdown(instance_name, log_suffix)
    full_time_neal, cop_times_neal = get_profile_breakdown(instance_name, neal_log_suffix)
    final_oa = get_final_oa(instance_name)
    return_tuple = (best_TTS, obj, tts_99, tts_999, gp_infeas_time,\
                    full_time_gp, cop_times_gp, full_time_neal, cop_times_neal,\
                    gp_t, gp_ub, gp_best, TTS, swe_list, final_oa)
    return return_tuple

def read_data(file):
    f = open(file)
    params = json.load(f)
    keys = list(params)

    unpack = lambda param : (param['N'], param['n'], param['p'], param['ngrid'])
    # print([(*unpack(dict(zip(keys, v))), pr.data_tuple(*unpack(dict(zip(keys, v)))))\
                                    # for v in itertools.product(*map(params.get, keys))])
    record = []
    for v in itertools.product(*map(params.get, keys)):
        v_dict = dict(zip(keys, v))
        record.append((*unpack(v_dict),\
                       *data_tuple(*unpack(v_dict)),\
                      *read_baselines(*unpack(v_dict))))
        
    cols = ['N', 'n', 'p', 'ngrid',\
            'best_TTS', 'swe_100_obj', 'tts_99', 'tts_999', 'gp_infeas_time',\
            'full_time_gp', 'cop_times_gp', 'full_time_neal', 'cop_times_neal',\
            'gp_t', 'gp_ub', 'gp_best', 'TTS', 'swe_list','final_oa',\
            'GT_val', 'gp_add_time','gp_mult_time',\
            'vcc_list', 'ss_list', 't_list']
    df = pd.DataFrame.from_records(record, columns = cols)
    
    return df