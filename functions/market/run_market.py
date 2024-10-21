import os
import sys
import numpy as np
import pandas as pd
import gurobipy as gp
from pathlib import Path
from gurobipy import GRB
from tqdm.auto import tqdm
from functions.market.models import flex_general_model as gen_m
from functions.market.models import generator_model as gens_m
from functions.market.models import load_model as loads_m
from functions.market.models import storage_model as batt_m


class Lfm:
    def __init__(self, _root_out, _root_in, _matrix_root, _int_soc=False, _cap_cstr=True, _net_mrk=True, _dt=1, _thresh_sens_prob_size=10**15):
        self._out_root = _root_out
        self._in_root = _root_in
        self._matrix_root = _matrix_root
        self._list_fsp_cat = ['A', 'B', 'C']
        self._mrk_type = ['CM', 'VC']
        self.int_soc = _int_soc
        self.cap_cstr = _cap_cstr
        self.net_mrk = _net_mrk
        self.dt = _dt
        self.thresh_sens_prob_size = _thresh_sens_prob_size
        self.dVtol = None
        self.vmax = None
        self.vmin = None
        self.max_vch = None
        self.max_cmh = None
        self.pilot_bus = None
        self.hours2study = None
        self.simperiod = None
        self.service = None
        self.mrk_scenario = None
        self.mrk_tag = None
        self.sim_tag = None
        self.cong_needs = None
        self.beta_cost = None
        self.alpha_cost = None
        self.load_conv = None
        self.ess_conv = None
        self.list_general_vars = list()
        self.results = dict()
        self.n_fsp = {_k: 1 for _k in self._list_fsp_cat}
        self.fsp_contrib = {_k: 0 for _k in self._list_fsp_cat}
        self.fsp_file = {_k: None for _k in self._list_fsp_cat}
        self.fsp_data = {_k: dict() for _k in self._list_fsp_cat}
        self._vars = {_k: dict() for _k in self._list_fsp_cat}
        self.sens_matrix_cm = {_k: dict() for _k in self._list_fsp_cat}
        self.sens_matrix_vc = {_k: dict() for _k in self._list_fsp_cat}

    @staticmethod
    def _exception_factory(_exception, _message):
        """Function for managing Exceptions."""
        print('An error occurred: ', _exception, ' – ', _message)
        sys.exit()

    def _reshape_sens_matrix(self, _item, _t):
        """
        Reshape the sensitivity matrix in order to obtain the correct shape of the matrix.
        The correct shape is as follows:
        - Columns:  Number of FSP of that resource.
        - Rows:     Number of congested lines/pilot buses."""
        if self.mrk_scenario == 'CM':
            _correct_shape = (len(self.cong_needs), self.n_fsp[_item])
            if self.sens_matrix_cm[_item][_t]['W'].shape != _correct_shape:
                self.sens_matrix_cm[_item][_t]['W'] = self.sens_matrix_cm[_item][_t]['W'].T
            if self.sens_matrix_cm[_item][_t]['R'].shape != _correct_shape:
                self.sens_matrix_cm[_item][_t]['R'] = self.sens_matrix_cm[_item][_t]['R'].T
        elif self.mrk_scenario == 'VC':
            _correct_shape = (len(self.pilot_bus), self.n_fsp[_item])
            if self.sens_matrix_vc[_item][_t]['W'].shape != _correct_shape:
                self.sens_matrix_vc[_item][_t]['W'] = self.sens_matrix_vc[_item][_t]['W'].T
            if self.sens_matrix_vc[_item][_t]['R'].shape != _correct_shape:
                self.sens_matrix_vc[_item][_t]['R'] = self.sens_matrix_vc[_item][_t]['R'].T
        elif self.mrk_scenario == 'CMVC' or self.mrk_scenario == 'VCCM':
            _correct_shape_cm = (len(self.cong_needs), self.n_fsp[_item])
            if self.sens_matrix_cm[_item][_t]['W'].shape != _correct_shape_cm:
                self.sens_matrix_cm[_item][_t]['W'] = self.sens_matrix_cm[_item][_t]['W'].T
            if self.sens_matrix_cm[_item][_t]['R'].shape != _correct_shape_cm:
                self.sens_matrix_cm[_item][_t]['R'] = self.sens_matrix_cm[_item][_t]['R'].T
            _correct_shape_vc = (len(self.pilot_bus), self.n_fsp[_item])
            if self.sens_matrix_vc[_item][_t]['W'].shape != _correct_shape_vc:
                self.sens_matrix_vc[_item][_t]['W'] = self.sens_matrix_vc[_item][_t]['W'].T
            if self.sens_matrix_vc[_item][_t]['R'].shape != _correct_shape_vc:
                self.sens_matrix_vc[_item][_t]['R'] = self.sens_matrix_vc[_item][_t]['R'].T
        return True

    def _eval_asybid(self, _cat):
        """Evaluate the symmetry of bid for the Flexibility Service Provider type C (Storage)."""
        self.fsp_data[_cat]['DW_U_inj'] = np.zeros((self.n_fsp[_cat], len(self.simperiod)))
        self.fsp_data[_cat]['DW_D_inj'] = np.zeros((self.n_fsp[_cat], len(self.simperiod)))
        self.fsp_data[_cat]['DW_U_ads'] = np.zeros((self.n_fsp[_cat], len(self.simperiod)))
        self.fsp_data[_cat]['DW_D_ads'] = np.zeros((self.n_fsp[_cat], len(self.simperiod)))
        for t in self.simperiod:
            for i in range(self.n_fsp[_cat]):
                if self.fsp_data[_cat]['Winit'].iloc[i, t] >= 0:
                    # Generator
                    self.fsp_data[_cat]['DW_U_inj'][i, t] = self.fsp_data[_cat]['Wub'].iloc[i, t] - self.fsp_data[_cat]['Winit'].iloc[i, t]
                    self.fsp_data[_cat]['DW_D_inj'][i, t] = min([self.fsp_data[_cat]['Winit'].iloc[i, t], self.fsp_data[_cat]['Winit'].iloc[i, t] - self.fsp_data[_cat]['Wlb'].iloc[i, t]])
                    self.fsp_data[_cat]['DW_U_ads'][i, t] = 0
                    self.fsp_data[_cat]['DW_D_ads'][i, t] = abs(min([0, self.fsp_data[_cat]['Wlb'].iloc[i, t]]))
                else:
                    # Load
                    self.fsp_data[_cat]['DW_U_inj'][i, t] = max([self.fsp_data[_cat]['Wub'].iloc[i, t] - self.fsp_data[_cat]['Winit'].iloc[i, t], self.fsp_data[_cat]['Wub'].iloc[i, t]])
                    self.fsp_data[_cat]['DW_D_inj'][i, t] = 0
                    self.fsp_data[_cat]['DW_U_ads'][i, t] = abs(min([self.fsp_data[_cat]['Winit'].iloc[i, t], self.fsp_data[_cat]['Wub'].iloc[i, t] - self.fsp_data[_cat]['Winit'].iloc[i, t]]))
                    self.fsp_data[_cat]['DW_D_ads'][i, t] = abs(self.fsp_data[_cat]['Wlb'].iloc[i, t] - self.fsp_data[_cat]['Winit'].iloc[i, t])
        return True

    def _read_input_data(self, _filename, _root, _sheetname=None):
        """Check the filename in the input root and read the file."""
        _tmp_data = None
        _ext = Path(_filename).suffix
        if _ext == '.csv':
            _tmp_data = pd.read_csv(os.path.join(_root, _filename), index_col=0, header=0)
        elif _ext == '.xlsx':
            # if _sheetname is None:
            #     self._exception_factory(ValueError, 'No Sheetname has been passed.')
            _tmp_data = pd.read_excel(os.path.join(_root, _filename), index_col=0, header=0, sheet_name=_sheetname)
        return _tmp_data

    def _define_fsp_data(self, _filename, _cat=None):
        """Define the data of Flexibility Service Providers for category A, B and C."""
        if _cat is None:
            self._exception_factory(ValueError, 'No FSP Type have been passed.')
        if _cat not in self._list_fsp_cat:
            self._exception_factory(ValueError, 'The FSP Type "{category}" do not exist or it is not yet implemented.'.format(category=_cat))

        n_fsp = 0
        fsp_info = None
        fsp_data = None
        if _filename != 0:
            # fsp_info = self._read_input_data(_filename=_filename, _root=self._in_root, _sheetname='FSPinfo')
            # fsp_node = fsp_info.bus.to_numpy()
            fsp_data = self._read_input_data(_filename=_filename, _root=self._in_root)
            n_fsp = fsp_data['FSPinfo'].bus.to_numpy().shape[0]

        if n_fsp > 0:
            self.n_fsp[_cat] = n_fsp
            self.fsp_contrib[_cat] = 1

        if _cat == self._list_fsp_cat[0]:
            # load_fsp_data, load_conv = loads_m.load_inputs(_n_fsp=n_fsp, _simperiod=self.simperiod, _filename=_filename,
            #                                                _in_root=self._in_root)
            load_fsp_data, load_conv = loads_m.res_params(_n_fsp=n_fsp, _sim_period=self.simperiod, _fsp_data=fsp_data)
            self.fsp_data[_cat] = load_fsp_data
            self.load_conv = load_conv
        elif _cat == self._list_fsp_cat[1]:
            # gen_fsp_data = gens_m.gen_input(_n_fsp=n_fsp, _simperiod=self.simperiod, _fsp_info=fsp_info,
            #                                 _filename=_filename, _in_root=self._in_root)
            gen_fsp_data, gen_conv = gens_m.res_params(_n_fsp=n_fsp, _sim_period=self.simperiod, _fsp_data=fsp_data)
            self.fsp_data[_cat] = gen_fsp_data
        elif _cat == self._list_fsp_cat[2]:
            # ess_fsp_data, ess_conv = batt_m.storage_inputs(_n_fsp=n_fsp, _simperiod=self.simperiod, _fsp_info=fsp_info,
            #                                                _filename=_filename, _in_root=self._in_root)
            ess_fsp_data, ess_conv = batt_m.res_params(_n_fsp=n_fsp, _sim_period=self.simperiod, _fsp_data=fsp_data)
            self.fsp_data[_cat] = ess_fsp_data
            self.ess_conv = ess_conv
        elif _cat == self._list_fsp_cat[3]:
            pass
        elif _cat == self._list_fsp_cat[4]:
            pass
        return True

    def _load_sens_matrix(self, _cat, _cm_factor=-1):
        """Load Sensitivity Matrix from file."""
        if len(self.hours2study) * len(self.pilot_bus) > self.thresh_sens_prob_size:
            # Simplified Model
            if self.mrk_scenario == 'CM':
                _hours2check = self.max_cmh
            elif self.mrk_scenario == 'VC':
                _hours2check = self.max_vch
            else:
                # CMVC
                _hours2check = [self.max_vch, self.max_cmh]
        else:
            # Full Model
            _hours2check = self.hours2study

        self.sens_matrix_vc['all'] = pd.DataFrame()
        for _h in tqdm(_hours2check, desc='Loading Sensitivity Matrix for "{_model}" Market Model'.format(_model=self.mrk_scenario)):
            t_th = str(_h).zfill(2)
            if self.mrk_scenario == 'CM':
                self.sens_matrix_cm[_cat][_h] = dict()
                shape_zeros_matrix = len(self.cong_needs)
                w_name_tags = ['HCMdSdP_FSP' + _cat, 'h' + t_th, self.sim_tag]
                r_name_tags = ['HCMdSdQ_FSP' + _cat, 'h' + t_th, self.sim_tag]
                Wfile2read = '_'.join(filter(None, w_name_tags))
                Rfile2read = '_'.join(filter(None, r_name_tags))
                if self.fsp_contrib[_cat] > 0:
                    if self.service != 'P':
                        self.sens_matrix_cm[_cat][_h]['W'] = _cm_factor * self._read_input_data(_filename=Wfile2read, _root=self._matrix_root)
                        self.sens_matrix_cm[_cat][_h]['R'] = _cm_factor * self._read_input_data(_filename=Rfile2read, _root=self._matrix_root)
                    else:
                        self.sens_matrix_cm[_cat][_h]['W'] = _cm_factor * self._read_input_data(_filename=Wfile2read + '.csv', _root=self._matrix_root)
                        k_w_shape = self.sens_matrix_cm[_cat][_h]['W'].shape
                        self.sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                else:
                    self.sens_matrix_cm[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
                    self.sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
            elif self.mrk_scenario == 'VC':
                self.sens_matrix_vc[_cat][_h] = dict()
                # Upload Sensitivity Matrix that contains all
                file2read = 'HVMPii_h' + t_th + '.csv'
                HVPii_h = self._read_input_data(_filename=file2read, _root=self._matrix_root)
                self.sens_matrix_vc['all'] = pd.concat([self.sens_matrix_vc['all'], HVPii_h], axis=1)
                # Upload all the remaining Sensitivity matrix
                shape_zeros_matrix = len(self.pilot_bus)
                Wfile2read = 'HVMP_FSP' + _cat + '_h' + t_th + '.csv'
                Rfile2read = 'HVMQ_FSP' + _cat + '_h' + t_th + '.csv'
                if self.fsp_contrib[_cat] > 0:
                    if self.service != 'P':
                        self.sens_matrix_vc[_cat][_h]['W'] = self._read_input_data(_filename=Wfile2read, _root=self._matrix_root)
                        self.sens_matrix_vc[_cat][_h]['R'] = self._read_input_data(_filename=Rfile2read, _root=self._matrix_root)
                    else:
                        self.sens_matrix_vc[_cat][_h]['W'] = self._read_input_data(_filename=Wfile2read, _root=self._matrix_root)
                        k_w_shape = self.sens_matrix_vc[_cat][_h]['W'].shape
                        self.sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                else:
                    self.sens_matrix_vc[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
                    self.sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
            elif self.mrk_scenario == 'CMVC' or self.mrk_scenario == 'VCCM':
                self.sens_matrix_cm[_cat][_h] = dict()
                self.sens_matrix_vc[_cat][_h] = dict()
                # Upload Sensitivity Matrix that contains all
                file2read = 'HVMPii_h' + t_th + '.csv'
                HVPii_h = self._read_input_data(_filename=file2read, _root=self._matrix_root)
                self.sens_matrix_vc['all'] = pd.concat([self.sens_matrix_vc['all'], HVPii_h], axis=1)
                # Upload all the remaining Sensitivity matrix
                shape_zeros_matrix_cm = len(self.cong_needs)
                shape_zeros_matrix_vc = len(self.pilot_bus)
                w_name_tags = ['HCMdSdP_FSP' + _cat, 'h' + t_th, self.sim_tag]
                r_name_tags = ['HCMdSdQ_FSP' + _cat, 'h' + t_th, self.sim_tag]
                Wfile2read_cm = '_'.join(filter(None, w_name_tags))
                Rfile2read_cm = '_'.join(filter(None, r_name_tags))
                Wfile2read_vc = 'HVMP_FSP' + _cat + '_h' + t_th + '.csv'
                Rfile2read_vc = 'HVMQ_FSP' + _cat + '_h' + t_th + '.csv'
                if self.fsp_contrib[_cat] > 0:
                    if self.service != 'P':
                        self.sens_matrix_cm[_cat][_h]['W'] = _cm_factor * self._read_input_data(_filename=Wfile2read_cm, _root=self._matrix_root)
                        self.sens_matrix_cm[_cat][_h]['R'] = _cm_factor * self._read_input_data(_filename=Rfile2read_cm, _root=self._matrix_root)
                        self.sens_matrix_vc[_cat][_h]['W'] = self._read_input_data(_filename=Wfile2read_vc, _root=self._matrix_root)
                        self.sens_matrix_vc[_cat][_h]['R'] = self._read_input_data(_filename=Rfile2read_vc, _root=self._matrix_root)
                    else:
                        self.sens_matrix_cm[_cat][_h]['W'] = _cm_factor * self._read_input_data(_filename=Wfile2read_cm + '.csv', _root=self._matrix_root)
                        self.sens_matrix_vc[_cat][_h]['W'] = self._read_input_data(_filename=Wfile2read_vc, _root=self._matrix_root)
                        k_w_shape = self.sens_matrix_cm[_cat][_h]['W'].shape
                        self.sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                        k_w_shape = self.sens_matrix_vc[_cat][_h]['W'].shape
                        self.sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                else:
                    self.sens_matrix_cm[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix_cm))
                    self.sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix_cm))
                    self.sens_matrix_vc[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix_vc))
                    self.sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix_vc))
            else:
                self._exception_factory(ValueError, 'The Market Model "{model}" do not exist.'.format(model=self.mrk_scenario))
            # Reshape the matrix
            # self._reshape_sens_matrix(_cat, _h)
        try:
            self.sens_matrix_vc['all'].columns = self.hours2study
        except ValueError:
            print('\nSensitivity Matrix HVMPii not uploaded.')
        return True

    def load_data_input(self, _cm_factor=-1):
        """Define data input of Local Flexibility Market."""
        if _cm_factor is None:
            _cm_factor = -1

        for item in self._list_fsp_cat:
            if self.fsp_file[item] is None:
                self._exception_factory(ValueError, 'No FSP Type {type} file have been passed.'.format(type=item))
            try:
                self.fsp_file[item] = self.fsp_file[item] + '.xlsx'
            except TypeError:
                pass
            self._define_fsp_data(_filename=self.fsp_file[item], _cat=item)
            if self.net_mrk:
                self._load_sens_matrix(_cat=item, _cm_factor=_cm_factor)
        return True

    def _var_indexing(self, _m):
        """Define variables and save them into dictionary shared across multiple methods of the class."""
        # General Variables
        n_pb = len(self.pilot_bus)
        n_t = len(self.hours2study)
        n_cm = len(self.cong_needs)
        # # Slack Variable for Optim (per unit version)
        # shape_var_vc = tuplelist([(_f, _t) for _f in range(n_pb) for _t in range(n_t)])
        # alpha_DVpu = _m.addVars(shape_var_vc, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='alpha_DVpu')
        # self._vars['alpha_DVpu'] = alpha_DVpu
        # # Voltage After the market voltage magnitude on pilot busses VA
        # VA = _m.addVars(shape_var_vc, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='VA')
        # self._vars['VA'] = VA
        # ABS_alphaDVpu = _m.addVars(shape_var_vc, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_alphaDVpu')
        # self._vars['ABS_alpha_DVpu'] = ABS_alphaDVpu
        # # Congestion Management Variables
        # shape_var_cm = tuplelist([(_f, _t) for _f in range(n_cm) for _t in range(n_t)])
        # # slack variable for CM flex not-supplied
        # beta = _m.addVars(shape_var_cm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')
        # self._vars['beta'] = beta
        # # abs(slack variable for CM flex not-supplied)
        # ABS_beta = _m.addVars(shape_var_cm, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_beta')
        # self._vars['ABS_beta'] = ABS_beta
        # self.list_general_vars = ['alpha_DVpu', 'VA', 'ABS_alpha_DVpu', 'beta', 'ABS_beta']
        self._vars, self.list_general_vars = gen_m.general_variables(_m, n_pb, n_cm, n_t)

        for item in self._list_fsp_cat:
            _contr = self.fsp_contrib[item] * GRB.INFINITY
            if item == self._list_fsp_cat[0]:
                # Type A (load)
                self._vars[item] = loads_m.define_variables(_m, self.n_fsp[item], n_t, _contr, _type=self._list_fsp_cat[0])
            elif item == self._list_fsp_cat[1]:
                # Type B (generator)
                self._vars[item] = gens_m.define_variables(_m, self.n_fsp[item], n_t, _contr, _type=self._list_fsp_cat[1])
            elif item == self._list_fsp_cat[2]:
                # Type C (battery)
                self._vars[item] = batt_m.define_variables(_m, self.n_fsp[item], n_t, _contr, _type=self._list_fsp_cat[2])
            else:
                # Undefined Type
                self._exception_factory(ValueError, 'The FSP Type "{category}" do not exist.'.format(category=item))
        return True

    def _cons_indexing(self, _m):
        """Define constraints of the Local Flexibility Market."""
        # General Constraints of the Flexibility Market

        beta = self._vars['beta']
        # ABS_beta = self._vars['ABS_beta']
        va = self._vars['VA']
        alpha_dV = self._vars['alpha_DVpu']
        # ABS_alpha_DVpu = self._vars['ABS_alpha_DVpu']

        gen_m.general_constraints(_m, self._vars)
        # # Absolute value for beta (CM flexibility not-supplied)
        # for x, y in zip(beta.keys(), ABS_beta.keys()):
        #     _name_abs_beta = 'abs_beta_#' + str(x[0]) + '_t' + str(x[1])
        #     _m.addGenConstrAbs(ABS_beta[y], beta[x], name=_name_abs_beta)
        # # Absolute value for alpha (VC flexibility not-supplied)
        # for x, y in zip(alpha_dV.keys(), ABS_alpha_DVpu.keys()):
        #     _name_abs_alpha = 'abs_alpha_#' + str(x[0]) + '_t' + str(x[1])
        #     _m.addGenConstrAbs(ABS_alpha_DVpu[y], alpha_dV[x], name=_name_abs_alpha)

        # Specific Constraints for Resource Model
        for item in self._list_fsp_cat:
            if self.fsp_contrib[item] > 0:
                # Check if the Resource is available in the network.
                # If not, do not create constraints.
                for t in range(len(self.hours2study)):
                    for i in range(self.n_fsp[item]):
                        if item == self._list_fsp_cat[0]:
                            # Type A (Load)
                            _vars_load = self._vars[item]
                            _data_load = self.fsp_data[item]
                            loads_m.define_constraints(_m, _vars_load, _data_load, i, t, self.cap_cstr, item)
                        elif item == self._list_fsp_cat[1]:
                            # Type B (Generator)
                            _vars_gen = self._vars[item]
                            _data_gen = self.fsp_data[item]
                            gens_m.define_constraints(_m, _vars_gen, _data_gen, i, t, self.cap_cstr, item)
                        elif item == self._list_fsp_cat[2]:
                            # Type C (battery)
                            _vars_ess = self._vars[item]
                            _data_ess = self.fsp_data[item]
                            batt_m.define_constraints(_m, _vars_ess, _data_ess, i, t, self.cap_cstr, self.int_soc, self.dt, item)
                        elif item == self._list_fsp_cat[3]:
                            # Type D (future developments)
                            pass
                        elif item == self._list_fsp_cat[4]:
                            # Type E (future developments)
                            pass
                        else:
                            # Undefined Type
                            self._exception_factory(ValueError, 'The FSP Type "{category}" do not exist.'.format(category=item))

        # Network Constraints
        t_aux = 0
        for t in range(len(self.hours2study)):
            t_th = self.hours2study[t]
            if self.mrk_scenario == 'CM':
                # Constraints Congestion Management
                for _l in range(len(self.cong_needs)):
                    _name_cong = 'C#08_CM_l' + str(_l) + '_t' + str(t_th)
                    if self.net_mrk:
                        # Market with sensitivity factors
                        k_market = gp.quicksum(
                                        gp.quicksum((self.sens_matrix_cm[_item][t_th]['W'].iloc[_l, x] * self._vars[_item]['DW_' + _item][x, t_aux])
                                                    + (self.sens_matrix_cm[_item][t_th]['R'].iloc[_l, x] * self._vars[_item]['DR_' + _item][x, t_aux])
                                        for x in range(self.n_fsp[_item]))
                                    for _item in self._list_fsp_cat)
                    else:
                        # Market without sensitivity factors
                        _coeff_k = 1
                        k_market = gp.quicksum(
                                        gp.quicksum((_coeff_k * self._vars[item]['DW_' + item][x, t_aux])
                                                   + (_coeff_k * self._vars[item]['DR_' + item][x, t_aux])
                                        for x in range(self.n_fsp[item]))
                                    for item in self._list_fsp_cat)
                    # Create Constraints
                    _m.addConstr(self.cong_needs.iloc[_l, t_aux] - k_market - beta[_l, t_aux] <= 0, name=_name_cong)
                    # _m.addConstr(k_market + beta[_l, t_aux] - self.cong_needs.iloc[_l, t_aux] >= 0, name=_name_cong)
            elif self.mrk_scenario == 'VC':
                # Constraints Voltage Control
                vb = self.results['VB']
                for _l in range(len(self.pilot_bus)):
                    if self.net_mrk:
                        # Market with sensitivity factors
                        k_market = gp.quicksum(
                                        gp.quicksum((self.sens_matrix_vc[item][t_th]['W'].iloc[_l, x] * self._vars[item]['DW_' + item][x, t_aux])
                                                    + (self.sens_matrix_vc[item][t_th]['R'].iloc[_l, x] * self._vars[item]['DR_' + item][x, t_aux])
                                        for x in range(self.n_fsp[item]))
                                    for item in self._list_fsp_cat)
                    else:
                        # Market without sensitivity factors
                        _coeff_k = 1
                        k_market = gp.quicksum(
                                        gp.quicksum((_coeff_k * self._vars[item]['DW_' + item][x, t_aux])
                                                    + (_coeff_k * self._vars[item]['DR_' + item][x, t_aux])
                                        for x in range(self.n_fsp[item]))
                                    for item in self._list_fsp_cat)
                    # Create Constraints
                    _name_cong = 'C#02a_Vmax_Bus_n' + str(_l) + '_t' + str(t)
                    _m.addConstr(va[_l, t] - self.vmin - self.dVtol >= 0, name=_name_cong)
                    _name_cong = 'C#02b_Vmin_Bus_n' + str(_l) + '_t' + str(t)
                    _m.addConstr(va[_l, t] - self.vmax + self.dVtol <= 0, name=_name_cong)

                    if vb.iloc[_l, t_aux] > self.vmax:
                        _name_cong = 'C#01_VMAX_Bus_n' + str(_l) + '_t' + str(t)
                        _m.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market >= 0, name=_name_cong)
                    elif vb.iloc[_l, t_aux] < self.vmin:
                        _name_cong = 'C#01_VMIN_Bus_n' + str(_l) + '_t' + str(t)
                        _m.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market <= 0, name=_name_cong)
                    else:
                        _name_cong = 'C#01_V_Bus_n' + str(_l) + '_t' + str(t)
                        _m.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market == 0, name=_name_cong)
            elif self.mrk_scenario == 'CMVC' or self.mrk_scenario == 'VCCM':
                # Constraints Congestion Management
                for _l in range(len(self.cong_needs)):
                    _name_cong = 'C#08_CM_l' + str(_l) + '_t' + str(t_th)
                    if self.net_mrk:
                        # Market with sensitivity factors
                        k_market = gp.quicksum(
                            gp.quicksum((self.sens_matrix_cm[item][t_th]['W'].iloc[_l, x] * self._vars[item]['DW_' + item][x, t_aux])
                                        + (self.sens_matrix_cm[item][t_th]['R'].iloc[_l, x] * self._vars[item]['DR_' + item][x, t_aux])
                                        for x in range(self.n_fsp[item]))
                            for item in self._list_fsp_cat)
                    else:
                        # Market without sensitivity factors
                        _coeff_k = 1
                        k_market = gp.quicksum(
                            gp.quicksum((_coeff_k * self._vars[item]['DW_' + item][x, t_aux])
                                        + (_coeff_k * self._vars[item]['DR_' + item][x, t_aux])
                                        for x in range(self.n_fsp[item]))
                            for item in self._list_fsp_cat)
                    # Create Constraints
                    _m.addConstr(self.cong_needs.iloc[_l, t_aux] - k_market - beta[_l, t_aux] <= 0, name=_name_cong)
                # Constraints Voltage Control
                vb = self.results['VB']
                for _l in range(len(self.pilot_bus)):
                    if self.net_mrk:
                        # Market with sensitivity factors
                        k_market = gp.quicksum(
                            gp.quicksum((self.sens_matrix_vc[item][t_th]['W'].iloc[_l, x] * self._vars[item]['DW_' + item][x, t_aux])
                                        + (self.sens_matrix_vc[item][t_th]['R'].iloc[_l, x] * self._vars[item]['DR_' + item][x, t_aux])
                                        for x in range(self.n_fsp[item]))
                            for item in self._list_fsp_cat)
                    else:
                        # Market without sensitivity factors
                        _coeff_k = 1
                        k_market = gp.quicksum(
                            gp.quicksum((_coeff_k * self._vars[item]['DW_' + item][x, t_aux])
                                        + (_coeff_k * self._vars[item]['DR_' + item][x, t_aux])
                                        for x in range(self.n_fsp[item]))
                            for item in self._list_fsp_cat)
                    # Create Constraints
                    _name_cong = 'C#02a_Vmax_Bus_n' + str(_l) + '_t' + str(t)
                    _m.addConstr(va[_l, t] - self.vmin - self.dVtol >= 0, name=_name_cong)
                    _name_cong = 'C#02b_Vmin_Bus_n' + str(_l) + '_t' + str(t)
                    _m.addConstr(va[_l, t] - self.vmax + self.dVtol <= 0, name=_name_cong)

                    if vb.iloc[_l, t_aux] > self.vmax:
                        _name_cong = 'C#01_VMAX_Bus_n' + str(_l) + '_t' + str(t)
                        _m.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market >= 0, name=_name_cong)
                    elif vb.iloc[_l, t_aux] < self.vmin:
                        _name_cong = 'C#01_VMIN_Bus_n' + str(_l) + '_t' + str(t)
                        _m.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market <= 0, name=_name_cong)
                    else:
                        _name_cong = 'C#01_V_Bus_n' + str(_l) + '_t' + str(t)
                        _m.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market == 0, name=_name_cong)
            else:
                self._exception_factory(ValueError, 'The Market Model "{model}" do not exist.'.format(model=self.mrk_scenario))
            t_aux += 1
        return _m

    def _obj_indexing(self, _m):
        """Define Objective of the Local Flexibility Market."""
        # Extract Variable
        ABS_beta = self._vars['ABS_beta']
        obj_fcn = dict()

        if self.mrk_scenario == 'CM':
            # Beta Slack Variable - Objective Function Cost
            obj_fcn_beta = gp.quicksum(
                                gp.quicksum(ABS_beta[_l, t] * self.beta_cost for _l in range(len(self.cong_needs)))
                            for t in range(len(self.hours2study)))
            obj_fcn['beta'] = obj_fcn_beta
        elif self.mrk_scenario == 'VC':
            # Alpha Slack Variable - Objective Function Cost
            ABS_alpha_DVpu = self._vars['ABS_alpha_DVpu']
            obj_fcn_alpha = gp.quicksum(
                                    gp.quicksum((ABS_alpha_DVpu[j, t] / self.sens_matrix_vc['all'].iloc[j, t]) * self.alpha_cost
                                    for j in range(len(self.pilot_bus)))
                            for t in range(len(self.hours2study)))
            obj_fcn['alpha'] = obj_fcn_alpha
        elif self.mrk_scenario == 'CMVC' or self.mrk_scenario == 'VCCM':
            # Beta Slack Variable - Objective Function Cost
            obj_fcn_beta = gp.quicksum(
                gp.quicksum(ABS_beta[_l, t] * self.beta_cost for _l in range(len(self.cong_needs))) for t in range(len(self.hours2study)))
            obj_fcn['beta'] = obj_fcn_beta
            # Alpha Slack Variable - Objective Function Cost
            ABS_alpha_DVpu = self._vars['ABS_alpha_DVpu']
            obj_fcn_alpha = gp.quicksum(
                gp.quicksum((ABS_alpha_DVpu[j, t] / self.sens_matrix_vc['all'].iloc[j, t]) * self.alpha_cost for j in range(len(self.pilot_bus)))
                for t in range(len(self.hours2study)))
            obj_fcn['alpha'] = obj_fcn_alpha

        if self.net_mrk:
            pass
        else:
            # Alpha Slack Variable for Network Constraints not included in the Market - Objective Function Cost
            ABS_alpha_DVpu = self._vars['ABS_alpha_DVpu']
            obj_fcn_alpha_net_mrk = gp.quicksum(
                gp.quicksum(ABS_alpha_DVpu[j, t] * self.alpha_cost * 1000 for j in range(len(self.pilot_bus)))
                for t in range(len(self.hours2study)))
            obj_fcn['alpha_net_mrk'] = obj_fcn_alpha_net_mrk

        # Contribution of Resource - Objective Function Cost
        for item in self._list_fsp_cat:
            if item == self._list_fsp_cat[0]:
                # Type A (Load)
                # Extract Variable
                DW_A_U = self._vars[item]['DW_' + item + '_U']
                DW_A_D = self._vars[item]['DW_' + item + '_D']
                DR_A_U = self._vars[item]['DR_' + item + '_U']
                DR_A_D = self._vars[item]['DR_' + item + '_D']
                obj_fcn[item] = gp.quicksum(
                    gp.quicksum(DW_A_U[i, t] * self.fsp_data[item]['WU_cost'].iloc[i, t]
                        + DW_A_D[i, t] * self.fsp_data[item]['WD_cost'].iloc[i, t]
                        + DR_A_U[i, t] * self.fsp_data[item]['RU_cost'].iloc[i, t]
                        + DR_A_D[i, t] * self.fsp_data[item]['RD_cost'].iloc[i, t] for i in range(self.n_fsp[item]))
                    for t in range(len(self.hours2study)))
            elif item == self._list_fsp_cat[1]:
                # Type B (Generator)
                # Extract Variable
                DW_B_U = self._vars[item]['DW_' + item + '_U']
                DW_B_D = self._vars[item]['DW_' + item + '_D']
                DR_B_U = self._vars[item]['DR_' + item + '_U']
                DR_B_D = self._vars[item]['DR_' + item + '_D']
                obj_fcn[item] = gp.quicksum(
                    gp.quicksum(DW_B_U[i, t] * self.fsp_data[item]['WU_cost'].iloc[i, t]
                        + DW_B_D[i, t] * self.fsp_data[item]['WD_cost'].iloc[i, t]
                        + DR_B_U[i, t] * self.fsp_data[item]['RU_cost'].iloc[i, t]
                        + DR_B_D[i, t] * self.fsp_data[item]['RD_cost'].iloc[i, t] for i in range(self.n_fsp[item]))
                    for t in range(len(self.hours2study)))
            elif item == self._list_fsp_cat[2]:
                # Type C (Storage)
                # Extract Variable
                DW_C_U_inj = self._vars[item]['DW_' + item + '_U_inj']
                DW_C_D_inj = self._vars[item]['DW_' + item + '_D_inj']
                DW_C_U_ads = self._vars[item]['DW_' + item + '_U_ads']
                DW_C_D_ads = self._vars[item]['DW_' + item + '_D_ads']
                DR_C_U = self._vars[item]['DR_' + item + '_U']
                DR_C_D = self._vars[item]['DR_' + item + '_D']
                obj_fcn[item] = gp.quicksum(
                    gp.quicksum(DW_C_U_inj[i, t] * self.fsp_data[item]['WU_inj_cost'].iloc[i, t]
                        + DW_C_D_inj[i, t] * self.fsp_data[item]['WD_inj_cost'].iloc[i, t]
                        + DW_C_U_ads[i, t] * self.fsp_data[item]['WU_ads_cost'].iloc[i, t]
                        + DW_C_D_ads[i, t] * self.fsp_data[item]['WD_ads_cost'].iloc[i, t]
                        + DR_C_U[i, t] * self.fsp_data[item]['RU_cost'].iloc[i, t]
                        + DR_C_D[i, t] * self.fsp_data[item]['RD_cost'].iloc[i, t] for i in range(self.n_fsp[item]))
                    for t in range(len(self.hours2study)))
            elif item == self._list_fsp_cat[3]:
                pass
            elif item == self._list_fsp_cat[4]:
                pass

        # Set Objective Function
        _m.setObjective(gp.quicksum(obj_fcn[_k] for _k in obj_fcn.keys()), GRB.MINIMIZE)
        return _m

    def flex_market(self, _filename='model_results_', _mip_focus=2, _mip_gap=.00001, _nnvonvex=2, _method=5,
                    _flag_lp=False, _flag_mps=False):
        """Run the Local Flexibility Market according to the type of service, market etc."""
        lfm_model = gp.Model(self.mrk_scenario + self.service + '_MarketModel')

        # Define Variables
        self._var_indexing(lfm_model)

        # Define Constraints
        lfm_model = self._cons_indexing(lfm_model)

        # Objective Function
        lfm_model = self._obj_indexing(lfm_model)

        lfm_model.setParam(GRB.Param.MIPFocus, _mip_focus)
        lfm_model.setParam(GRB.Param.MIPGap, _mip_gap)
        lfm_model.setParam(GRB.Param.NonConvex, _nnvonvex)
        lfm_model.setParam(GRB.Param.Method, _method)

        # Save model
        dir2check = os.path.join(self._out_root, 'GReports')
        if _flag_lp:
            if not os.path.isdir(dir2check):
                os.makedirs(dir2check)

            _name_lp = 'report_case_lp_' + self.mrk_tag + '_MG.lp'
            lfm_model.write(os.path.join(dir2check, _name_lp))

        if _flag_mps:
            if not os.path.isdir(dir2check):
                os.makedirs(dir2check)

            _name_mps = 'report_case_mps_' + self.mrk_tag + '_MG.mps'
            lfm_model.write(os.path.join(dir2check, _name_mps))

        # Solve multi-scenario model
        lfm_model.Params.LogToConsole = 0
        lfm_model.optimize()

        # Check model status
        self._diagnostic(lfm_model)

        # Save results into Results Dictionary
        # self._save_vars(lfm_model)
        self._save_vars_tuple(lfm_model)

        # Save results into Excel file
        filename = _filename + self.mrk_tag
        self.save_excel(_filename=filename)
        return [os.path.join(self._out_root, filename + '.xlsx'), lfm_model]

    def _save_vars(self, _m, _round_vars=3):
        """Save variables of optimisation model into Dataframes."""
        self.results['ObjValue'] = pd.DataFrame({'ObjValue': [_m.ObjVal]})
        _load_str = [_val + self._list_fsp_cat[0] for _val in self.load_conv]
        for _k, _i in self._vars.items():
            try:
                self.results[_k] = pd.DataFrame(_i.X, columns=self.hours2study)
            except AttributeError:
                for _k1, _i1 in self._vars[_k].items():
                    if _k1 in _load_str:
                        self.results[_k1] = pd.DataFrame(-_i1.X, columns=self.hours2study)
                    elif _k1.split('_')[0] == 'qSMVA':
                        W = self.results['W_' + _k]
                        R = self.results['R_' + _k]
                        smva_calc = np.sqrt(W.to_numpy() ** 2 + R.to_numpy() ** 2)
                        self.results['SMVA_' + _k] = pd.DataFrame(np.sqrt(_i1.X), columns=self.hours2study)
                        self.results['SMVA_' + _k + '_calc'] = pd.DataFrame(smva_calc, columns=self.hours2study)
                    elif _k1 == 'RATIO_' + self._list_fsp_cat[1] or _k1 == 'RATIO_' + self._list_fsp_cat[2]:
                        RATIO = _i1.X
                        self.results[_k1] = pd.DataFrame(RATIO, columns=self.hours2study)
                        cos_phi = np.cos(np.arctan(RATIO))
                        self.results['COSPHI_' + _k] = pd.DataFrame(cos_phi, columns=self.hours2study)
                    else:
                        self.results[_k1] = pd.DataFrame(_i1.X, columns=self.hours2study)
            except gp.GurobiError:
                pass
        if self.fsp_contrib[self._list_fsp_cat[2]] > 0:
            # Converting the Storage from GENERATION to LOAD Convention
            for _val in self.ess_conv:
                self.results[_val + self._list_fsp_cat[2]] = -self.results[_val + self._list_fsp_cat[2]]
        # Save alpha_W
        # alpha_w_idx = self.results['alpha_DVpu'].index
        # self.results['alpha_W'] = pd.DataFrame(np.zeros(shape=(len(alpha_w_idx), len(self.hours2study))), index=alpha_w_idx, columns=self.hours2study)
        self.results['alpha_W'] = self.results['alpha_DVpu']
        return True

    def _save_vars_tuple(self, _m, _round_vars=3):
        """Save variables of optimisation model into Dataframes."""
        self.results['ObjValue'] = pd.DataFrame({'ObjValue': [_m.ObjVal]})
        _load_str = [_val + self._list_fsp_cat[0] for _val in self.load_conv]
        for _k, _i in self._vars.items():
            try:
                if _k in self.list_general_vars:
                    if 'beta' not in _k:
                        # Extract the results like a matrix (Voltage Control General variables)
                        _matrix = np.zeros(shape=(len(self.pilot_bus), len(self.hours2study)))
                        for _np in range(len(self.pilot_bus)):
                            for _t in range(len(self.hours2study)):
                                _matrix[_np, _t] = _i[(_np, _t)].X
                    else:
                        # Extract the results like a matrix (Congestion Management General variables)
                        _matrix = np.zeros(shape=(len(self.cong_needs), len(self.hours2study)))
                        for _c in range(len(self.cong_needs)):
                            for _t in range(len(self.hours2study)):
                                _matrix[_c, _t] = _i[(_c, _t)].X
                    # Save as Dataframe
                    self.results[_k] = pd.DataFrame(_matrix, columns=self.hours2study)
                else:
                    for _k1, _i1 in self._vars[_k].items():
                        # Extract the results like a matrix
                        _matrix = np.zeros(shape=(self.n_fsp[_k], len(self.hours2study)))
                        for _fsp in range(self.n_fsp[_k]):
                            for _t in range(len(self.hours2study)):
                                _matrix[_fsp, _t] = _i1[(_fsp, _t)].X

                        if _k1 in _load_str:
                            self.results[_k1] = pd.DataFrame(-_matrix, columns=self.hours2study)
                        elif _k1.split('_')[0] == 'qSMVA':
                            W = self.results['W_' + _k]
                            R = self.results['R_' + _k]
                            smva_calc = np.sqrt(W.to_numpy() ** 2 + R.to_numpy() ** 2)
                            self.results['SMVA_' + _k] = pd.DataFrame(np.sqrt(_matrix), columns=self.hours2study)
                            self.results['SMVA_' + _k + '_calc'] = pd.DataFrame(smva_calc, columns=self.hours2study)
                        elif _k1 == 'RATIO_' + self._list_fsp_cat[1] or _k1 == 'RATIO_' + self._list_fsp_cat[2]:
                            RATIO = _matrix
                            self.results[_k1] = pd.DataFrame(RATIO, columns=self.hours2study)
                            cos_phi = np.cos(np.arctan(RATIO))
                            self.results['COSPHI_' + _k] = pd.DataFrame(cos_phi, columns=self.hours2study)
                        else:
                            self.results[_k1] = pd.DataFrame(_matrix, columns=self.hours2study)
            except gp.GurobiError:
                pass
        if self.fsp_contrib[self._list_fsp_cat[2]] > 0:
            # Converting the Storage from GENERATION to LOAD Convention
            for _val in self.ess_conv:
                self.results[_val + self._list_fsp_cat[2]] = -self.results[_val + self._list_fsp_cat[2]]
        # Save alpha_W
        alpha_w_idx = self.results['alpha_DVpu'].index
        self.results['alpha_W'] = pd.DataFrame(np.zeros(shape=(len(alpha_w_idx), len(self.hours2study))), index=alpha_w_idx, columns=self.hours2study)
        return True

    def save_excel(self, _filename, _extension='.xlsx'):
        """Save results into Excel files"""
        with pd.ExcelWriter(os.path.join(self._out_root, _filename + _extension)) as writer:
            if type(self.results) is dict:
                for _k, _v in self.results.items():
                    try:
                        _v.to_excel(writer, sheet_name=_k)
                    except AttributeError:
                        pd.DataFrame(_v).to_excel(writer, sheet_name=_k)
            else:
                self._exception_factory(ValueError, 'Format type "{_out}" not supported.'.format(_out=str(type(self.results))))
        return True

    def _diagnostic(self, _m):
        """Check the model results"""
        sc = gp.StatusConstClass
        d = {sc.__dict__[k]: k for k in sc.__dict__.keys() if 'A' <= k[0] <= 'Z'}
        status = _m.status
        if status == GRB.UNBOUNDED:
            print(_m.UnbdRay)
            raise Warning('The model cannot be solved because it is unbounded.')
        elif status == GRB.OPTIMAL:
            print('\nThe optimal objective is %g' % _m.ObjVal)
            print('------------------------------------------------------------------------\n')
        elif status == GRB.INF_OR_UNBD:
            # In order to determine if the model is infeasible or unbounded, set the "DualReductions" parameter to 0,
            # call "reset" on the model, and optimize once again.
            _m.Params.DualReductions = 0
            _m.reset()
            _m.Params.InfUnbdInfo = 1
            _m.optimize()
            self._diagnostic(_m)
        elif status == GRB.INFEASIBLE:
            print('\nOptimization was stopped with status: ', d[status])
            _m.computeIIS()
            if _m.IISMinimal:
                print('IIS is minimal\n')
            else:
                print('IIS is not minimal\n')
            print('\nThe following constraint(s) cannot be satisfied:')
            for c in _m.getConstrs():
                if c.IISConstr:
                    print('%s' % c.ConstrName)
            raise Warning('The model cannot be solved because it is infeasible.')
        elif status == GRB.NUMERIC:
            raise Warning('Optimization was stopped with status: {_stat}'.format(_stat=d[status]))
        return True
