import math
import numpy as np
import scipy.sparse
import sys

TOL = 1e-5

def cop_block(block):
    obj = float(block[1][10 : ])
    x = np.array([float(x_i) for x_i in block[2][block[2].find('[') + 1:block[2].find(']')].split(', ')])
    start_time = float(block[3][17 : ])
    end_time = float(block[4][15 : ])
    
    return obj, x, start_time, end_time

def cutting_plane_block(block):
    iter_num = int(block[1][11 : ])
    y = [float(x_i) for x_i in block[2][block[2].find('[') + 1:block[2].find(']')].split(', ')]
    obj = block[3][10 : ]
    status = block[4][13 : ]
    relgap = float(block[5][13 : ])
    curr_time = float(block[6][16 : ])
    
    return iter_num, y, obj, status, relgap, curr_time

def rescale_inf_norm(x):
    if np.linalg.norm(x, np.inf) > TOL:
        return x / np.linalg.norm(x, np.inf)
    else:
        return x

class cop_info:
    def __init__(self, obj, x, start_time, end_time):
        self.obj = obj
        self.x = x
        self.start_time = start_time 
        self.end_time = end_time

class cutting_plane_info:
    def __init__(self, iter_num, y, obj, status, relgap, curr_time):
        self.iter_num = iter_num
        self.y = y
        self.obj = obj
        self.status = status
        self.relgap = relgap
        self.curr_time = curr_time
    
class COP:
    def __init__(self):
        self.M = np.zeros((0, 0))
        self.Ai = []
        self.b = []
        
        self.cop_info_list = []
        self.cutting_plane_info_list = []
        self.final_oa = None
        
        self.ys = []
        self.log_status = []
        
    def y2mat(self, y):
        nconstr = len(self.b)
        QUBO = self.M  - sum(y[i] * self.Ai[i] for i in range(nconstr))
        return QUBO
    
    def read_all(self, instance_dir, log_suffix):
        M_file = instance_dir + 'result_M.txt'
        constr_file = instance_dir + 'result_constr.txt'
        GT_log_file = instance_dir + log_suffix
        
        self.read_file(M_file, constr_file)
        self.read_GT_log(GT_log_file)
    
    def read_file(self, M_file_name, constr_file_name):
        M_file = open(M_file_name, 'r')
        M_rows = []
        M_cols = []
        M_vals = []

        M_line = M_file.readline()
        while M_line:
            row, col, val = M_line.strip().split(', ')
            M_rows.append(int(row) - 1)
            M_cols.append(int(col) - 1)
            M_vals.append(float(val))
            M_line = M_file.readline()
        M_file.close()
        
        

        constr_file  = open(constr_file_name, 'r')
        Ai_list = []

        constr_line = constr_file.readline()
        while constr_line:
            self.b.append(float(constr_line.strip()))
            constr_line = constr_file.readline()
            Ai_nnz = int(constr_line.strip())

            Aii_rows = []
            Aii_cols = []
            Aii_vals = []

            for i in range(Ai_nnz):
                constr_line = constr_file.readline()
                row, col, val = constr_line.strip().split(', ')
                Aii_rows.append(int(row) - 1)
                Aii_cols.append(int(col) - 1)
                Aii_vals.append(float(val))

            Ai_list.append((Aii_rows, Aii_cols, Aii_vals))
            constr_line = constr_file.readline()
        constr_file.close()

        ndims = max([max(Aii[0]) for Aii in Ai_list]) + 1
        self.M = scipy.sparse.coo_matrix((M_vals, (M_rows, M_cols)), (ndims, ndims))
        self.Ai = [scipy.sparse.coo_matrix((Aii[2], (Aii[0], Aii[1])), (ndims, ndims)) for Aii in Ai_list]
    
    def read_GT_log(self, GT_log_filename):
        logfile = open(GT_log_filename, "r")
        line = logfile.readline().strip()
        while line:
            block = []
            while line[0] != '└':
                block.append(line)
                line = logfile.readline().strip()
            block.append(line)
            
            if 'Copositivity check optimum' in block[0]:
                self.cop_info_list.append(cop_info(*cop_block(block)))
            elif 'Cutting-plane iteration information' in block[0]:
                self.cutting_plane_info_list.append(cutting_plane_info(*cutting_plane_block(block)))
            elif 'Final outer approximation' in block[0]:
                self.final_oa = (float(block[1][block[1].find('lb') + 5:]), float(block[2][block[2].find('ub') + 5:]))
            
            line = logfile.readline().strip()
        
    def read_ys(self, cp_log_file_name):
        log_file = open(cp_log_file_name, 'r')
        log_line = log_file.readline()
        while log_line:
            status = log_line.strip().split(', ')
            self.log_status.append((int(status[0]), status[1], float(status[2]), float(status[3])))
            
            log_line = log_file.readline()
            y = np.array([float(x) for x in log_line.strip().split(', ')])
            self.ys.append(y)
            log_line = log_file.readline()
            
    def get_infeas_idx(self):
        infeas_idx = [idx for (idx, c) in enumerate(self.cutting_plane_info_list) if c.status == 'infeas']
        return infeas_idx
    
    def get_qubo(self, idx):
        y = self.cutting_plane_info_list[idx].y
        return self.y2mat(y)
    
    def get_cop_min(self, idx):
        return (self.cop_info_list[idx].obj, self.cop_info_list[idx].x)
    
    def get_scaled_cop_min(self, idx):
        x_unscaled = self.cop_info_list[idx].x
        if np.linalg.norm(x_unscaled, np.inf) > TOL:
            scale = 1 / np.linalg.norm(x_unscaled, np.inf)
            return (scale**2 * self.cop_info_list[idx].obj, x_unscaled * scale)
        else:
            return self.get_cop_min(idx)
        
    
