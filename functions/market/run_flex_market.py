import os
import importlib
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from tqdm.auto import tqdm
from functions import auxiliary as aux
from functions import file_io as io_file
from functions.market.models import flex_general_model as flx_gen


def _diagnostic(_m):
    """
    Check the model diagnostics.

    :param _m: Model representation by Gurobi.py.
    """
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
        _diagnostic(_m)
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


def _upload_res_params(_fsp_file, _in_root, _sim_period):
    """
    Upload resource parameters from resource models.
    :param _fsp_file: Dictionary of the names of the fsp filex per resource category.
    :param _in_root: Input root for reading fspInfo file.
    :param _sim_period: Number of hours to simulate. Example: hours2study: [5, 11] -> _sim_period: [0, 1]
    """

    # Dictionary of the model library of the fsp resource
    _fsp_lib = aux.set_resources()

    # Dictionary where to save the resource parameters per category
    res_parameters = {_k: dict() for _k in _fsp_file.keys()}
    for _key, _value in tqdm(_fsp_file.items(), desc='Loading Flexibility Service Provider Data'):
        if _value is None:
            raise ValueError('No FSP Type {type} file have been passed.'.format(type=_key))

        # Even when there are no resource of type "_key", put at least one resource with everything as zero.
        res_parameters[_key]['n_fsp'] = 1
        res_parameters[_key]['contrib'] = 0

        n_fsp = 0
        fsp_data = None
        if _value != 0:
            fsp_data = io_file.import_data(_filename=_value + '.xlsx', _root=_in_root, _index_col=0)
            n_fsp = fsp_data['FSPinfo'].bus.to_numpy().shape[0]

        if n_fsp > 0:
            res_parameters[_key]['n_fsp'] = n_fsp
            res_parameters[_key]['contrib'] = 1

        res_lib = importlib.import_module(f'functions.market.models.{_fsp_lib[_key]}')
        res_parameters[_key]['data'], res_parameters[_key]['conv'] = res_lib.res_params(_n_fsp=n_fsp, _sim_period=_sim_period, _fsp_data=fsp_data)
    return res_parameters


def _load_sens_matrix(_res_parameters, _hours2study, _n_cong, _n_pilot_bus, _matrix_root, _mrk_scenario, _service,
                      _sim_tag, _cm_factor=-1):
    """
    Upload Sensitivity Matrix from file.

    :param _res_parameters: Dictionary of the parameters of the fsp per resource category.

    :param _hours2study: List of hours to study.

    :param _n_cong: Number of congestions.

    :param _n_pilot_bus: Number of pilot buses.

    :param _matrix_root: Root to where the sensitivity matrix file are saved.

    :param _mrk_scenario: String that represents the tag for market model.

    :param _service: String that represents the tag for product in the market.

    :param _sim_tag: String that represents the specific name of the simulation.

    :param _cm_factor: Congestion Management Factor that change sign to the sensitivity matrix.
    """
    # fixme: we will put this evaluation into a separate function that will return the list of hours to study.
    # if len(self.hours2study) * len(self.pilot_bus) > self.thresh_sens_prob_size:
    #     # Simplified Mode
    #     if self.mrk_scenario == 'CM':
    #         _hours2check = self.max_cmh
    #     elif self.mrk_scenario == 'VC':
    #         _hours2check = self.max_vch
    #     else:
    #         # CMVC
    #         _hours2check = [self.max_vch, self.max_cmh]
    # else:
    #     # Full Model
    #     _hours2check = self.hours2study

    sens_matrix_cm = {_k: dict() for _k in _res_parameters.keys()}
    sens_matrix_vc = {_k: dict() for _k in _res_parameters.keys()}
    sens_matrix_vc['all'] = pd.DataFrame()
    _message = 'Loading Sensitivity Matrix for "{_model}" Market Model'.format(_model=_mrk_scenario)
    for _h in tqdm(_hours2study, desc=_message):
        t_th = str(_h).zfill(2)
        for _cat in _res_parameters.keys():
            if _mrk_scenario == 'CM':
                sens_matrix_cm[_cat][_h] = dict()
                shape_zeros_matrix = _n_cong
                w_name_tags = ['HCMdSdP_FSP' + _cat, 'h' + t_th, _sim_tag]
                r_name_tags = ['HCMdSdQ_FSP' + _cat, 'h' + t_th, _sim_tag]
                Wfile2read = '_'.join(filter(None, w_name_tags)) + '.csv'
                Rfile2read = '_'.join(filter(None, r_name_tags)) + '.csv'
                if _res_parameters[_cat]['contrib'] > 0:
                    if _service != 'P':
                        sens_matrix_cm[_cat][_h]['W'] = _cm_factor * io_file.import_data(_filename=Wfile2read, _root=_matrix_root, _index_col=0)
                        sens_matrix_cm[_cat][_h]['R'] = _cm_factor * io_file.import_data(_filename=Rfile2read, _root=_matrix_root, _index_col=0)
                    else:
                        sens_matrix_cm[_cat][_h]['W'] = _cm_factor * io_file.import_data(_filename=Wfile2read, _root=_matrix_root, _index_col=0)
                        k_w_shape = sens_matrix_cm[_cat][_h]['W'].shape
                        sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                else:
                    sens_matrix_cm[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
                    sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
            elif _mrk_scenario == 'VC':
                sens_matrix_vc[_cat][_h] = dict()
                # Upload all the remaining Sensitivity matrix
                shape_zeros_matrix = _n_pilot_bus
                w_name_tags_vc = ['HVMP_FSP' + _cat, 'h' + t_th, _sim_tag]
                r_name_tags_vc = ['HVMQ_FSP' + _cat, 'h' + t_th, _sim_tag]
                Wfile2read = '_'.join(filter(None, w_name_tags_vc)) + '.csv'
                Rfile2read = '_'.join(filter(None, r_name_tags_vc)) + '.csv'
                # Wfile2read = 'HVMP_FSP' + _cat + '_h' + t_th + '.csv'
                # Rfile2read = 'HVMQ_FSP' + _cat + '_h' + t_th + '.csv'
                if _res_parameters[_cat]['contrib'] > 0:
                    if _service != 'P':
                        sens_matrix_vc[_cat][_h]['W'] = io_file.import_data(_filename=Wfile2read, _root=_matrix_root, _index_col=0)
                        sens_matrix_vc[_cat][_h]['R'] = io_file.import_data(_filename=Rfile2read, _root=_matrix_root, _index_col=0)
                    else:
                        sens_matrix_vc[_cat][_h]['W'] = io_file.import_data(_filename=Wfile2read, _root=_matrix_root, _index_col=0)
                        k_w_shape = sens_matrix_vc[_cat][_h]['W'].shape
                        sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                else:
                    sens_matrix_vc[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
                    sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix))
            elif _mrk_scenario == 'CMVC' or _mrk_scenario == 'VCCM':
                sens_matrix_cm[_cat][_h] = dict()
                sens_matrix_vc[_cat][_h] = dict()
                # Upload all the remaining Sensitivity matrix
                shape_zeros_matrix_cm = _n_cong
                shape_zeros_matrix_vc = _n_pilot_bus
                w_name_tags = ['HCMdSdP_FSP' + _cat, 'h' + t_th, _sim_tag]
                r_name_tags = ['HCMdSdQ_FSP' + _cat, 'h' + t_th, _sim_tag]
                Wfile2read_cm = '_'.join(filter(None, w_name_tags)) + '.csv'
                Rfile2read_cm = '_'.join(filter(None, r_name_tags)) + '.csv'
                w_name_tags_vc = ['HVMP_FSP' + _cat, 'h' + t_th, _sim_tag]
                r_name_tags_vc = ['HVMQ_FSP' + _cat, 'h' + t_th, _sim_tag]
                Wfile2read_vc = '_'.join(filter(None, w_name_tags_vc)) + '.csv'
                Rfile2read_vc = '_'.join(filter(None, r_name_tags_vc)) + '.csv'
                # Wfile2read_vc = 'HVMP_FSP' + _cat + '_h' + t_th + '.csv'
                # Rfile2read_vc = 'HVMQ_FSP' + _cat + '_h' + t_th + '.csv'
                if _res_parameters[_cat]['contrib'] > 0:
                    if _service != 'P':
                        sens_matrix_cm[_cat][_h]['W'] = _cm_factor * io_file.import_data(_filename=Wfile2read_cm, _root=_matrix_root, _index_col=0)
                        sens_matrix_cm[_cat][_h]['R'] = _cm_factor * io_file.import_data(_filename=Rfile2read_cm, _root=_matrix_root, _index_col=0)
                        sens_matrix_vc[_cat][_h]['W'] = io_file.import_data(_filename=Wfile2read_vc, _root=_matrix_root, _index_col=0)
                        sens_matrix_vc[_cat][_h]['R'] = io_file.import_data(_filename=Rfile2read_vc, _root=_matrix_root, _index_col=0)
                    else:
                        sens_matrix_cm[_cat][_h]['W'] = _cm_factor * io_file.import_data(_filename=Wfile2read_cm, _root=_matrix_root, _index_col=0)
                        sens_matrix_vc[_cat][_h]['W'] = io_file.import_data(_filename=Wfile2read_vc, _root=_matrix_root, _index_col=0)
                        k_w_shape = sens_matrix_cm[_cat][_h]['W'].shape
                        sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                        k_w_shape = sens_matrix_vc[_cat][_h]['W'].shape
                        sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(k_w_shape))
                else:
                    sens_matrix_cm[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix_cm))
                    sens_matrix_cm[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix_cm))
                    sens_matrix_vc[_cat][_h]['W'] = pd.DataFrame(np.zeros(shape_zeros_matrix_vc))
                    sens_matrix_vc[_cat][_h]['R'] = pd.DataFrame(np.zeros(shape_zeros_matrix_vc))
            else:
                raise ValueError('The Market Model "{model}" do not exist.'.format(model=_mrk_scenario))
        if _mrk_scenario == 'VC' or _mrk_scenario == 'CMVC' or _mrk_scenario == 'VCCM':
            # Upload Sensitivity Matrix that contains all
            file2read = 'HVMPii_h' + t_th + '.csv'
            HVPii_h = io_file.import_data(_filename=file2read, _root=_matrix_root, _index_col=0)
            sens_matrix_vc['all'] = pd.concat([sens_matrix_vc['all'], HVPii_h], axis=1)
    # Update columns with hours to study
    sens_matrix_vc['all'].columns = _hours2study
    return sens_matrix_cm, sens_matrix_vc


def _var_indexing(_grb_model, _res_parameters, _n_pb, _n_cm, _n_t):
    """
    Define variables and save them into dictionary shared across multiple methods of the class.
    :param _grb_model: Gurobipy model.
    :param _res_parameters: Dictionary of all resource parameters
    :param _n_pb: Number of Pilot Buses.
    :param _n_cm: Number of congestions.
    :param _n_t: Number of hours to study.
    """
    # Dictionary where to save the variables
    variables = dict()

    # General Variables
    _vars, list_general_vars = flx_gen.general_variables(_grb_model, _n_pb, _n_cm, _n_t)
    variables.update(_vars)

    # Dictionary of the model library of the fsp resource
    _fsp_lib = aux.set_resources()

    for _key in tqdm(_fsp_lib.keys(), desc='Loading Flexibility Service Provider Variables'):
        res_contr = _res_parameters[_key]['contrib'] * GRB.INFINITY
        n_fsp = _res_parameters[_key]['n_fsp']

        res_lib = importlib.import_module(f'functions.market.models.{_fsp_lib[_key]}')
        variable_dict = res_lib.define_variables(_grb_model=_grb_model, _n_fsp=n_fsp, _n_t=_n_t, _contr=res_contr, _type=_key)
        variables[_key] = variable_dict
    return variables, list_general_vars


def _cons_indexing(_grb_model, _vars, _res_parameters, _sens_matrix_cm, _sens_matrix_vc, _hours, _cong_needs,
                   _pilot_bus, _v_before, _vmin, _vmax, _dVtol, _mrk_scenario, _net_mrk, _cap_cstr, _int_soc, _dt):
    """
    Define constraints of the Local Flexibility Market.
    :param _grb_model: Gurobu py model.
    :param _vars: Dictionary of the variables of the model.
    :param _res_parameters: Dictionary of the resource parameters.
    :param _sens_matrix_cm: Sensitivity Coefficient matrix for congestion management.
    :param _sens_matrix_vc: Sensitivity Coefficient matrix for voltage control.
    :param _hours: List of hours to study.
    :param _cong_needs: Dataframe of the congestions.
    :param _pilot_bus: List of Pilot buses.
    :param _v_before: Dataframe of the oltage before the market clearing.
    :param _vmin: Minimum Voltage in p.u.
    :param _vmax: Maximum Voltage in p.u.
    :param _dVtol: Delta Voltage tolerance.
    :param _mrk_scenario: String representing the market scenario
    :param _net_mrk: Boolean value for including network constraints in the market model.
    :param _cap_cstr: Boolean value for including resource capability constraints in the market model.
    :param _int_soc: Boolean value for including battery inter-temporal constraints in the market model.
    :param _dt: Delta time of the model.
    """
    # General Constraints of the Flexibility Market
    beta = _vars['beta']
    va = _vars['VA']
    alpha_dV = _vars['alpha_DVpu']

    vb = _v_before

    flx_gen.general_constraints(_grb_model=_grb_model, _vars_dict=_vars)

    # Dictionary of the model library of the fsp resource
    _fsp_lib = aux.set_resources()

    n_t = len(_hours)
    # Specific Constraints for Resource Model
    for _key in tqdm(_fsp_lib.keys(), desc='Loading Flexibility Service Provider Constraints'):
        res_contr = _res_parameters[_key]['contrib']
        n_fsp = _res_parameters[_key]['n_fsp']
        if res_contr > 0:
            # Check if the Resource is available in the network.
            # If not, do not create constraints.
            for t in range(n_t):
                for i in range(n_fsp):
                    res_lib = importlib.import_module(f'functions.market.models.{_fsp_lib[_key]}')
                    res_lib.define_constraints(_grb_model=_grb_model, _vars_dict=_vars[_key], _idx_res=i, _interval=t,
                                               _fsp_data=_res_parameters[_key]['data'], _cap_cstr=_cap_cstr, _dt=_dt,
                                               _int_soc=_int_soc, _type=_key)

    # Network Constraints
    t_aux = 0
    for _t in _hours:
        if _mrk_scenario == 'CM':
            # Constraints Congestion Management
            for _l in range(len(_cong_needs)):
                _name_cong = 'C#08_CM_l' + str(_l) + '_t' + str(_t)
                if _net_mrk:
                    # Market with sensitivity factors
                    k_market = gp.quicksum(
                                    gp.quicksum((_sens_matrix_cm[_key][_t]['W'].iloc[_l, x] * _vars[_key]['DW_' + _key][x, t_aux])
                                                + (_sens_matrix_cm[_key][_t]['R'].iloc[_l, x] * _vars[_key]['DR_' + _key][x, t_aux])
                                    for x in range(_res_parameters[_key]['n_fsp']))
                                for _key in _fsp_lib.keys())
                else:
                    # Market without sensitivity factors
                    _coeff_k = 1
                    k_market = gp.quicksum(
                        gp.quicksum(
                            (_coeff_k * _vars[_key]['DW_' + _key][x, t_aux]) + (_coeff_k * _vars[_key]['DR_' + _key][x, t_aux])
                            for x in range(_res_parameters[_key]['n_fsp']))
                        for _key in _fsp_lib.keys())
                # Create Constraints
                _grb_model.addConstr(_cong_needs.iloc[_l, t_aux] - k_market - beta[_l, t_aux] <= 0, name=_name_cong)
        elif _mrk_scenario == 'VC':
            # Constraints Voltage Control
            for _l in range(len(_pilot_bus)):
                if _net_mrk:
                    # Market with sensitivity factors
                    k_market = gp.quicksum(
                        gp.quicksum(
                            (_sens_matrix_vc[_key][_t]['W'].iloc[_l, x] * _vars[_key]['DW_' + _key][x, t_aux])
                            + (_sens_matrix_vc[_key][_t]['R'].iloc[_l, x] * _vars[_key]['DR_' + _key][x, t_aux])
                            for x in range(_res_parameters[_key]['n_fsp']))
                        for _key in _fsp_lib.keys())
                else:
                    # Market without sensitivity factors
                    _coeff_k = 1
                    k_market = gp.quicksum(
                        gp.quicksum(
                            (_coeff_k * _vars[_key]['DW_' + _key][x, t_aux]) + (_coeff_k * _vars[_key]['DR_' + _key][x, t_aux])
                            for x in range(_res_parameters[_key]['n_fsp']))
                        for _key in _fsp_lib.keys())
                # Create Constraints
                _name_cong = 'C#02a_Vmax_Bus_n' + str(_l) + '_t' + str(t_aux)
                _grb_model.addConstr(va[_l, t_aux] - _vmin - _dVtol >= 0, name=_name_cong)
                _name_cong = 'C#02b_Vmin_Bus_n' + str(_l) + '_t' + str(t_aux)
                _grb_model.addConstr(va[_l, t_aux] - _vmax + _dVtol <= 0, name=_name_cong)

                if vb.iloc[_l, t_aux] > _vmax:
                    _name_cong = 'C#01_VMAX_Bus_n' + str(_l) + '_t' + str(t_aux)
                    _grb_model.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market >= 0, name=_name_cong)
                elif vb.iloc[_l, t_aux] < _vmin:
                    _name_cong = 'C#01_VMIN_Bus_n' + str(_l) + '_t' + str(t_aux)
                    _grb_model.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market <= 0, name=_name_cong)
                else:
                    _name_cong = 'C#01_V_Bus_n' + str(_l) + '_t' + str(t_aux)
                    _grb_model.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market == 0, name=_name_cong)
        elif _mrk_scenario == 'CMVC' or _mrk_scenario == 'VCCM':
            # Constraints Congestion Management
            for _l in range(len(_cong_needs)):
                _name_cong = 'C#08_CM_l' + str(_l) + '_t' + str(_t)
                if _net_mrk:
                    # Market with sensitivity factors
                    k_market = gp.quicksum(
                        gp.quicksum(
                            (_sens_matrix_cm[_key][_t]['W'].iloc[_l, x] * _vars[_key]['DW_' + _key][x, t_aux])
                            + (_sens_matrix_cm[_key][_t]['R'].iloc[_l, x] * _vars[_key]['DR_' + _key][x, t_aux])
                            for x in range(_res_parameters[_key]['n_fsp']))
                        for _key in _fsp_lib.keys())
                else:
                    # Market without sensitivity factors
                    _coeff_k = 1
                    k_market = gp.quicksum(
                        gp.quicksum(
                            (_coeff_k * _vars[_key]['DW_' + _key][x, t_aux]) + (_coeff_k * _vars[_key]['DR_' + _key][x, t_aux])
                            for x in range(_res_parameters[_key]['n_fsp']))
                        for _key in _fsp_lib.keys())
                # Create Constraints
                _grb_model.addConstr(_cong_needs.iloc[_l, t_aux] - k_market - beta[_l, t_aux] <= 0, name=_name_cong)
            for _l in range(len(_pilot_bus)):
                if _net_mrk:
                    # Market with sensitivity factors
                    k_market = gp.quicksum(
                        gp.quicksum(
                            (_sens_matrix_vc[_key][_t]['W'].iloc[_l, x] * _vars[_key]['DW_' + _key][x, t_aux])
                            + (_sens_matrix_vc[_key][_t]['R'].iloc[_l, x] * _vars[_key]['DR_' + _key][x, t_aux])
                            for x in range(_res_parameters[_key]['n_fsp']))
                        for _key in _fsp_lib.keys())
                else:
                    # Market without sensitivity factors
                    _coeff_k = 1
                    k_market = gp.quicksum(
                        gp.quicksum(
                            (_coeff_k * _vars[_key]['DW_' + _key][x, t_aux]) + (_coeff_k * _vars[_key]['DR_' + _key][x, t_aux])
                            for x in range(_res_parameters[_key]['n_fsp']))
                        for _key in _fsp_lib.keys())
                # Create Constraints
                _name_cong = 'C#02a_Vmax_Bus_n' + str(_l) + '_t' + str(t_aux)
                _grb_model.addConstr(va[_l, t_aux] - _vmin - _dVtol >= 0, name=_name_cong)
                _name_cong = 'C#02b_Vmin_Bus_n' + str(_l) + '_t' + str(t_aux)
                _grb_model.addConstr(va[_l, t_aux] - _vmax + _dVtol <= 0, name=_name_cong)

                if vb.iloc[_l, t_aux] > _vmax:
                    _name_cong = 'C#01_VMAX_Bus_n' + str(_l) + '_t' + str(t_aux)
                    _grb_model.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market >= 0, name=_name_cong)
                elif vb.iloc[_l, t_aux] < _vmin:
                    _name_cong = 'C#01_VMIN_Bus_n' + str(_l) + '_t' + str(t_aux)
                    _grb_model.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market <= 0, name=_name_cong)
                else:
                    _name_cong = 'C#01_V_Bus_n' + str(_l) + '_t' + str(t_aux)
                    _grb_model.addConstr(va[_l, t_aux] - vb.iloc[_l, t_aux] - alpha_dV[_l, t_aux] - k_market == 0, name=_name_cong)
        else:
            raise ValueError('The Market Model "{model}" do not exist.'.format(model=_mrk_scenario))
        t_aux += 1
    return _grb_model


def _obj_indexing(_grb_model, _vars, _res_parameters, _alpha_cost, _beta_cost, _sens_matrix_vc, _n_pb, _n_cm, _n_t,
                  _mrk_scenario, _net_mrk):
    """
    Define Objective of the Local Flexibility Market.
    :param _grb_model: Gurobi py model.
    :param _vars: Dictionary of the variables of the model.
    :param _res_parameters: Dictionary of the resource parameters.
    :param _alpha_cost: Cost of Voltage Control Slack Variables.
    :param _beta_cost: Cost of Congestion Management Slack Variables.
    :param _sens_matrix_vc: Sensitivity Coefficient matrix for voltage control.
    :param _n_t: Number of hours to study.
    :param _n_cm: Number of the congestions.
    :param _n_pb: Number of Pilot buses.
    :param _mrk_scenario: String representing the market scenario
    :param _net_mrk: Boolean value for including network constraints in the market model.
    """
    # Extract Variable
    ABS_beta = _vars['ABS_beta']
    obj_fcn = dict()

    if _mrk_scenario == 'CM':
        # Beta Slack Variable - Objective Function Cost
        obj_fcn_beta = gp.quicksum(
            gp.quicksum(
                ABS_beta[_l, t] * _beta_cost
                for _l in range(_n_cm))
            for t in range(_n_t))
        obj_fcn['beta'] = obj_fcn_beta
    elif _mrk_scenario == 'VC':
        # Alpha Slack Variable - Objective Function Cost
        ABS_alpha_DVpu = _vars['ABS_alpha_DVpu']
        obj_fcn_alpha = gp.quicksum(
            gp.quicksum(
                (ABS_alpha_DVpu[j, t] / _sens_matrix_vc['all'].iloc[j, t]) * _alpha_cost
                for j in range(_n_pb))
            for t in range(_n_t))
        obj_fcn['alpha'] = obj_fcn_alpha
    elif _mrk_scenario == 'CMVC' or _mrk_scenario == 'VCCM':
        # Beta Slack Variable - Objective Function Cost
        obj_fcn_beta = gp.quicksum(
            gp.quicksum(
                ABS_beta[_l, t] * _beta_cost
                for _l in range(_n_cm))
            for t in range(_n_t))
        obj_fcn['beta'] = obj_fcn_beta
        # Alpha Slack Variable - Objective Function Cost
        ABS_alpha_DVpu = _vars['ABS_alpha_DVpu']
        obj_fcn_alpha = gp.quicksum(
            gp.quicksum(
                (ABS_alpha_DVpu[j, t] / _sens_matrix_vc['all'].iloc[j, t]) * _alpha_cost
                for j in range(_n_pb))
            for t in range(_n_t))
        obj_fcn['alpha'] = obj_fcn_alpha

    if _net_mrk:
        pass
    else:
        # Alpha Slack Variable for Network Constraints not included in the Market - Objective Function Cost
        ABS_alpha_DVpu = _vars['ABS_alpha_DVpu']
        obj_fcn_alpha_net_mrk = gp.quicksum(
            gp.quicksum(
                ABS_alpha_DVpu[j, t] * _alpha_cost * 1000
                for j in range(_n_pb))
            for t in range(_n_t))
        obj_fcn['alpha_net_mrk'] = obj_fcn_alpha_net_mrk

    # Dictionary of the model library of the fsp resource
    _fsp_lib = aux.set_resources()

    # Contribution of Resource - Objective Function Cost
    for _key in tqdm(_fsp_lib.keys(), desc='Loading Flexibility Service Provider Objective Functions'):
        n_fsp = _res_parameters[_key]['n_fsp']
        dw_u_list = aux.get_list(_startswith_key='DW_' + _key + '_U', _comparing_list=_vars[_key])
        dw_d_list = aux.get_list(_startswith_key='DW_' + _key + '_D', _comparing_list=_vars[_key])
        dr_u_list = aux.get_list(_startswith_key='DR_' + _key + '_U', _comparing_list=_vars[_key])
        dr_d_list = aux.get_list(_startswith_key='DR_' + _key + '_D', _comparing_list=_vars[_key])

        wu_cost_list = aux.get_list(_startswith_key='WU_', _comparing_list=_res_parameters[_key]['data'])
        wd_cost_list = aux.get_list(_startswith_key='WD_', _comparing_list=_res_parameters[_key]['data'])
        ru_cost_list = aux.get_list(_startswith_key='RU_', _comparing_list=_res_parameters[_key]['data'])
        rd_cost_list = aux.get_list(_startswith_key='RD_', _comparing_list=_res_parameters[_key]['data'])
        obj_fcn[_key] = gp.quicksum(
            gp.quicksum(
                gp.quicksum(_vars[_key][dw_u_list[_j]][i, t] * _res_parameters[_key]['data'][wu_cost_list[_j]].iloc[i, t] for _j in range(len(dw_u_list)))
                + gp.quicksum(_vars[_key][dw_d_list[_j]][i, t] * _res_parameters[_key]['data'][wd_cost_list[_j]].iloc[i, t] for _j in range(len(dw_d_list)))
                + gp.quicksum(_vars[_key][dr_u_list[_j]][i, t] * _res_parameters[_key]['data'][ru_cost_list[_j]].iloc[i, t] for _j in range(len(dr_u_list)))
                + gp.quicksum(_vars[_key][dr_d_list[_j]][i, t] * _res_parameters[_key]['data'][rd_cost_list[_j]].iloc[i, t] for _j in range(len(dr_d_list)))
                for i in range(n_fsp))
            for t in range(_n_t))

    # Set Objective Function
    _grb_model.setObjective(gp.quicksum(obj_fcn[_k] for _k in obj_fcn.keys()), GRB.MINIMIZE)
    return _grb_model
    
    
def _save_vars_tuple(_grb_model, _vars, _res_parameters, _list_general_vars, _hours, _cong_needs, _pilot_bus, _round_vars=3):
    """
    Save tuple variables of optimisation model into Dataframes.
    """
    # Dictionary of the model library of the fsp resource
    _fsp_lib = aux.set_resources()
    # Dictionary of the results
    _results = dict()
    _results['ObjValue'] = pd.DataFrame({'ObjValue': [_grb_model.ObjVal]})
    for _k, _i in _vars.items():
        try:
            if _k in _list_general_vars:
                if 'beta' not in _k:
                    # Extract the results like a matrix (Voltage Control General variables)
                    _matrix = np.zeros(shape=(len(_pilot_bus), len(_hours)))
                    for _np in range(len(_pilot_bus)):
                        for _t in range(len(_hours)):
                            _matrix[_np, _t] = _i[(_np, _t)].X
                else:
                    # Extract the results like a matrix (Congestion Management General variables)
                    _matrix = np.zeros(shape=(len(_cong_needs), len(_hours)))
                    for _c in range(len(_cong_needs)):
                        for _t in range(len(_hours)):
                            _matrix[_c, _t] = _i[(_c, _t)].X
                # Save as Dataframe
                _results[_k] = pd.DataFrame(_matrix, columns=_hours)
            else:
                n_fsp = _res_parameters[_k]['n_fsp']
                res_conv = [_val + _k for _val in _res_parameters[_k]['conv']]
                for _k1, _i1 in _vars[_k].items():
                    # Extract the results like a matrix
                    _matrix = np.zeros(shape=(n_fsp, len(_hours)))
                    for _fsp in range(n_fsp):
                        for _t in range(len(_hours)):
                            _matrix[_fsp, _t] = _i1[(_fsp, _t)].X

                    if _k1 in res_conv:
                        _results[_k1] = pd.DataFrame(-_matrix, columns=_hours)
                    # elif _k1.split('_')[0] == 'qSMVA':
                    elif 'qSMVA' in _k1:
                        W = _results['W_' + _k]
                        R = _results['R_' + _k]
                        smva_calc = np.sqrt(W.to_numpy() ** 2 + R.to_numpy() ** 2)
                        _results['SMVA_' + _k] = pd.DataFrame(np.sqrt(_matrix), columns=_hours)
                        _results['SMVA_' + _k + '_calc'] = pd.DataFrame(smva_calc, columns=_hours)
                    # elif _k1 == 'RATIO_' + _list_fsp_cat[1] or _k1 == 'RATIO_' + _list_fsp_cat[2]:
                    elif 'RATIO_' in res_conv:
                        RATIO = _matrix
                        _results[_k1] = pd.DataFrame(RATIO, columns=_hours)
                        cos_phi = np.cos(np.arctan(RATIO))
                        _results['COSPHI_' + _k] = pd.DataFrame(cos_phi, columns=_hours)
                    else:
                        _results[_k1] = pd.DataFrame(_matrix, columns=_hours)
        except gp.GurobiError:
            pass
    return _results


def _save_mrk_res_excel(_results, _out_root, _filename, _extension='.xlsx'):
    """
    Save market results into Excel files.

    :param _results: Dictionary market results to be saved.

    :param _out_root: Root to where the market results will be saved.

    :param _filename: Name of the file excel that contains the market results.

    :param _extension: Extension of the file excel.
    """
    with pd.ExcelWriter(os.path.join(_out_root, _filename + _extension)) as writer:
        if type(_results) is dict:
            for _k, _v in _results.items():
                try:
                    _v.to_excel(writer, sheet_name=_k)
                except AttributeError:
                    pd.DataFrame(_v).to_excel(writer, sheet_name=_k)
        else:
            raise ValueError('Format type "{_out}" not supported.'.format(_out=str(type(_results))))
    return True


def flex_market(_fsp_file, _hours, _cong_needs, _pilot_bus, _in_root, _out_root, _matrix_root, _v_before, _vmin, _vmax,
                _dVtol, _alpha_cost, _beta_cost, _mrk_tag, _sim_tag, _filename='model_results_', _cm_factor=-1,
                _mip_focus=2, _mip_gap=.00001, _nnvonvex=2, _method=5, _net_mrk=True, _cap_cstr=True, _int_soc=True,
                _dt=1, _flag_lp=False, _flag_mps=False):
    """
    Run the Local Flexibility Market.
    """
    # Define the Gurobi Model
    _mrk_scenario = _mrk_tag.split('_')[1]
    _service = _mrk_tag.split('_')[2]
    lfm_model = gp.Model(_mrk_scenario + _service + '_MarketModel')

    # Upload resource parameters
    _idx_hours = list(range(0, len(_hours)))
    n_t = len(_hours)
    n_cong = len(_cong_needs)
    n_pilot_bus = len(_pilot_bus)
    params_fps = _upload_res_params(_fsp_file, _in_root, _idx_hours)

    # Upload Sensitivity Matrix for specific hours
    if _net_mrk:
        cm_matrix, vc_matrix = _load_sens_matrix(_res_parameters=params_fps, _hours2study=_hours, _n_cong=n_cong,
                                                 _n_pilot_bus=n_pilot_bus, _matrix_root=_matrix_root, _mrk_scenario=_mrk_scenario,
                                                 _service=_service, _sim_tag=_sim_tag, _cm_factor=_cm_factor)
    else:
        cm_matrix, vc_matrix = 0, 0

    # Define Variables
    vars_dict, _list_general_vars = _var_indexing(_grb_model=lfm_model, _res_parameters=params_fps, _n_pb=n_pilot_bus,
                                                  _n_cm=n_cong, _n_t=n_t)

    # Define Constraints
    lfm_model = _cons_indexing(_grb_model=lfm_model, _vars=vars_dict, _res_parameters=params_fps, _sens_matrix_cm=cm_matrix,
                               _sens_matrix_vc=vc_matrix, _hours=_hours, _cong_needs=_cong_needs, _pilot_bus=_pilot_bus,
                               _v_before=_v_before, _vmin=_vmin, _vmax=_vmax, _dVtol=_dVtol, _mrk_scenario=_mrk_scenario,
                               _net_mrk=_net_mrk, _cap_cstr=_cap_cstr, _int_soc=_int_soc, _dt=_dt)

    # # Objective Function
    lfm_model = _obj_indexing(_grb_model=lfm_model, _vars=vars_dict, _res_parameters=params_fps, _alpha_cost=_alpha_cost,
                              _beta_cost=_beta_cost, _sens_matrix_vc=vc_matrix, _n_pb=n_pilot_bus, _n_cm=n_cong, _n_t=n_t,
                              _mrk_scenario=_mrk_scenario, _net_mrk=_net_mrk)

    lfm_model.setParam(GRB.Param.MIPFocus, _mip_focus)
    lfm_model.setParam(GRB.Param.MIPGap, _mip_gap)
    lfm_model.setParam(GRB.Param.NonConvex, _nnvonvex)
    lfm_model.setParam(GRB.Param.Method, _method)

    # Save model
    dir2check = os.path.join(_out_root, 'GReports')
    if _flag_lp:
        if not os.path.isdir(dir2check):
            os.makedirs(dir2check)

        _name_lp = 'report_case_lp_' + _mrk_tag + '.lp'
        lfm_model.write(os.path.join(dir2check, _name_lp))

    if _flag_mps:
        if not os.path.isdir(dir2check):
            os.makedirs(dir2check)

        _name_mps = 'report_case_mps_' + _mrk_tag + '.mps'
        lfm_model.write(os.path.join(dir2check, _name_mps))

    # Solve multi-scenario model
    lfm_model.Params.LogToConsole = 0
    lfm_model.optimize()

    # Check model status
    _diagnostic(lfm_model)

    # Save results into Results Dictionary
    _res_data = _save_vars_tuple(_grb_model=lfm_model, _vars=vars_dict, _res_parameters=params_fps,
                                 _list_general_vars=_list_general_vars, _hours=_hours, _cong_needs=_cong_needs,
                                 _pilot_bus=_pilot_bus, _round_vars=3)

    # Save results into Excel file
    filename = _filename + _mrk_tag
    _save_mrk_res_excel(_results=_res_data, _out_root=_out_root, _filename=filename)
    return [os.path.join(_out_root, filename + '.xlsx'), lfm_model]
