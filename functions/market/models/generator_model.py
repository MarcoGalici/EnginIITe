import math
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from gurobipy import GRB, tuplelist
from functions import file_io as io_file


def res_params(_n_fsp, _sim_period, _fsp_data):
    """
    Define Input Data of the Load Resource.
    :param _n_fsp: Number of FSP of type Load resource.
    :param _sim_period: List of hours to consider for the study.
    :param _fsp_data: Dictionary of the Data associated
    """
    fsp_data = dict()
    # Save Data that are Impose to GENERATION Convention.
    # Being in the generator model the list is empty since all variables are under the GENERATION Convention
    gen_convention = ['RATIO_']
    if _n_fsp > 0:
        # Reactive Power
        fsp_data['Rinit'] = _fsp_data['Rinit'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Active Power
        fsp_data['Winit'] = _fsp_data['Winit'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Nameplate Apparent Power of the Resources [MVA]
        fsp_data['mva'] = _fsp_data['FSPinfo'].mva.to_numpy()
        # Cost for Reactive Power UPWARD Support Provision [€/MVArh] for each Resource
        fsp_data['RU_cost'] = _fsp_data['R_U_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Cost for Reactive Power DOWNWARD Support Provision [€/MVArh] for each Resource
        fsp_data['RD_cost'] = _fsp_data['R_D_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]

        # Lower Bound Active Power (Minimum Active Power ABSORPTION Value admitted by the Resource)
        fsp_data['Wub'] = _fsp_data['WUB_ULim'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Upper Bound Active Power (Maximum Active Power INJECTION Value admitted by the Resource)
        fsp_data['Wlb'] = _fsp_data['WLB_DLim'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Upper Bound Reactive Power Value (Maximum Reactive Power INJECTION Value admitted by the Resource)
        fsp_data['Rub'] = _fsp_data['RUB_ULim'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Lower Bound Reactive Power Value (Minimum Reactive Power ABSORPTION Value admitted by the Resource)
        fsp_data['Rlb'] = _fsp_data['RUB_DLim'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Cost for Active Power UPWARD Support Provision [€/MVAh] for each Resource
        fsp_data['WU_cost'] = _fsp_data['W_U_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Cost for Active Power DOWNWARD Support Provision [€/MVAh] for each Resource
        fsp_data['WD_cost'] = _fsp_data['W_D_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Lower Bound Cosphi Value (Minimum Cosphi Value admitted by the Resource)
        fsp_data['COSPHILB'] = _fsp_data['COSPHILB'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Upper Bound Cosphi Value (Maximum Cosphi Value admitted by the Resource)
        fsp_data['COSPHIUB'] = _fsp_data['COSPHIUB'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Delta Active Power Upward
        fsp_data['DW_U'] = fsp_data['Winit'] - fsp_data['Wlb']
        # Delta Active Power Downward
        fsp_data['DW_D'] = fsp_data['Wub'] - fsp_data['Winit']
        # Delta Reactive Power Upward
        fsp_data['DR_U'] = fsp_data['Rinit'] - fsp_data['Rlb']
        # Delta Reactive Power Downward
        fsp_data['DR_D'] = fsp_data['Rub'] - fsp_data['Rinit']
        # Lower Bound Tanphi
        fsp_data['TANPHI_LB'] = np.tan(np.arccos(fsp_data['COSPHIUB']))
        # Upper Bound Tanphi
        fsp_data['TANPHI_UB'] = np.tan(np.arccos(fsp_data['COSPHILB']))
    else:
        fsp_data['mva'] = np.zeros(shape=(1, 1))
        _zero_df = pd.DataFrame(np.zeros(shape=(1, len(_sim_period))))
        fsp_data['Rinit'] = _zero_df
        fsp_data['Winit'] = _zero_df
        fsp_data['RD_cost'] = _zero_df
        fsp_data['RU_cost'] = _zero_df
        fsp_data['Wub'] = _zero_df
        fsp_data['Wlb'] = _zero_df
        fsp_data['Rub'] = _zero_df
        fsp_data['Rlb'] = _zero_df
        fsp_data['COSPHILB'] = _zero_df
        fsp_data['COSPHIUB'] = _zero_df
        fsp_data['TANPHI_LB'] = _zero_df
        fsp_data['TANPHI_UB'] = _zero_df
        fsp_data['DW_U'] = _zero_df
        fsp_data['DW_D'] = _zero_df
        fsp_data['DR_U'] = _zero_df
        fsp_data['DR_D'] = _zero_df
        fsp_data['WU_cost'] = _zero_df
        fsp_data['WD_cost'] = _zero_df
    return fsp_data, gen_convention


def define_variables(_grb_model, _n_fsp, _n_t, _contr, _type='B'):
    """
    Define Variables of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _n_fsp: Number of flexibility service providers.
    :param _n_t: Number of intervals.
    :param _contr: Contribution of these resource type.
    :param _type: Type of resource.
    """
    _vars_dict = gen_variables_v1(grb_model=_grb_model, n_fsp=_n_fsp, n_t=_n_t, _contr=_contr, _type=_type)
    return _vars_dict


def gen_variables_v0(grb_model, n_fsp, n_t, _contr, _type='B'):
    """Define Variables of the Generator Resource."""
    shape_var = (n_fsp, n_t)
    _vars_dict = dict()
    # Delta Reactive Power Exchange [p.u.]
    DR_B = grb_model.addMVar(shape=shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DR_' + _type)
    _vars_dict['DR_' + _type] = DR_B
    # After the Market Reactive Power Exchange
    R_B = grb_model.addMVar(shape=shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='R_' + _type)
    _vars_dict['R_' + _type] = R_B
    # Delta Active Power [p.u.]
    DW_B = grb_model.addMVar(shape=shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DW_' + _type)
    _vars_dict['DW_' + _type] = DW_B
    # After the Market Active Power
    W_B = grb_model.addMVar(shape=shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='W_' + _type)
    _vars_dict['W_' + _type] = W_B
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_B = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='qSMVA_' + _type)
    _vars_dict['qSMVA_' + _type] = qSMVA_B
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_B = grb_model.addMVar(shape=shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='RATIO_' + _type)
    _vars_dict['RATIO_' + _type] = RATIO_B
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_DR_B = grb_model.addMVar(shape=shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_DR_' + _type)
    _vars_dict['ABS_DR_' + _type] = ABS_DR_B
    ABS_DW_B = grb_model.addMVar(shape=shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_DW_' + _type)
    _vars_dict['ABS_DW_' + _type] = ABS_DW_B
    ABS_RATIO_B = grb_model.addMVar(shape=shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_RATIO_' + _type)
    _vars_dict['ABS_RATIO_' + _type] = ABS_RATIO_B
    # Variables for the Asymmetry of the Bids
    # variable: delta active power upwards contribution
    DW_B_U = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U')
    _vars_dict['DW_' + _type + '_U'] = DW_B_U
    # variable: delta active power downwards contribution
    DW_B_D = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D')
    _vars_dict['DW_' + _type + '_D'] = DW_B_D
    # variable: delta reactive power upwards contribution
    DR_B_U = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_U')
    _vars_dict['DR_' + _type + '_U'] = DR_B_U
    # variable: delta reactive power downwards contribution
    DR_B_R = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_D')
    _vars_dict['DR_' + _type + '_D'] = DR_B_R
    return _vars_dict


def gen_variables_v1(grb_model, n_fsp, n_t, _contr, _type='B'):
    """Define Variables of the Generator Resource."""
    shape_var = tuplelist([(_f, _t) for _f in range(n_fsp) for _t in range(n_t)])
    _vars_dict = dict()
    # Delta Reactive Power Exchange [p.u.]
    DR_B = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DR_' + _type)
    _vars_dict['DR_' + _type] = DR_B
    # After the Market Reactive Power Exchange
    R_B = grb_model.addVars(shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='R_' + _type)
    _vars_dict['R_' + _type] = R_B
    # Delta Active Power [p.u.]
    DW_B = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DW_' + _type)
    _vars_dict['DW_' + _type] = DW_B
    # After the Market Active Power
    W_B = grb_model.addVars(shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='W_' + _type)
    _vars_dict['W_' + _type] = W_B
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_B = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='qSMVA_' + _type)
    _vars_dict['qSMVA_' + _type] = qSMVA_B
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_B = grb_model.addVars(shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='RATIO_' + _type)
    _vars_dict['RATIO_' + _type] = RATIO_B
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_DR_B = grb_model.addVars(shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_DR_' + _type)
    _vars_dict['ABS_DR_' + _type] = ABS_DR_B
    ABS_DW_B = grb_model.addVars(shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_DW_' + _type)
    _vars_dict['ABS_DW_' + _type] = ABS_DW_B
    ABS_RATIO_B = grb_model.addVars(shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_RATIO_' + _type)
    _vars_dict['ABS_RATIO_' + _type] = ABS_RATIO_B
    # Variables for the Asymmetry of the Bids
    # variable: delta active power upwards contribution
    DW_B_U = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U')
    _vars_dict['DW_' + _type + '_U'] = DW_B_U
    # variable: delta active power downwards contribution
    DW_B_D = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D')
    _vars_dict['DW_' + _type + '_D'] = DW_B_D
    # variable: delta reactive power upwards contribution
    DR_B_U = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_U')
    _vars_dict['DR_' + _type + '_U'] = DR_B_U
    # variable: delta reactive power downwards contribution
    DR_B_R = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_D')
    _vars_dict['DR_' + _type + '_D'] = DR_B_R
    return _vars_dict


def define_constraints(_grb_model, _vars_dict, _fsp_data, _idx_res, _interval, _cap_cstr, _int_soc=False, _dt=1, _type='B'):
    """
    Define Constraints of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _vars_dict: Dictionary containing the variables of the gurobi model.
    :param _fsp_data: Dictionary containing the resource parameters.
    :param _idx_res: Index of the resource to which the constraints must be added.
    :param _interval: Index of the hour to which the constraints must be added.
    :param _cap_cstr: Boolean value to add the capability curve constraint of the resource.
    :param _int_soc: Boolean value to add the inter-temporal constraint of the resource.
    :param _dt: Delta time of the model
    :param _type: Type of resource.
    """
    grb_model = gen_constraints_v1(grb_model=_grb_model, _vars_dict=_vars_dict, fsp_data=_fsp_data, i=_idx_res,
                                   t=_interval, _cap_cstr=_cap_cstr, _type=_type)
    return grb_model


def gen_constraints_v0(grb_model, _vars_dict, fsp_data, i, t, _cap_cstr, _type='B'):
    """Define Constraints of the Generator Resource."""
    # Extract Variables
    # Delta Reactive Power Exchange [p.u.]
    DR_B = _vars_dict['DR_' + _type]
    # Absolute values of Delta Reactive Power Exchange
    ABS_DR_B = _vars_dict['ABS_DR_' + _type]
    # After the Market Reactive Power Exchange
    R_B = _vars_dict['R_' + _type]
    # Delta Active Power [p.u.]
    DW_B = _vars_dict['DW_' + _type]
    # Absolute values of Delta Active Power Exchange
    ABS_DW_B = _vars_dict['ABS_DW_' + _type]
    # After the Market Active Power
    W_B = _vars_dict['W_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_B = _vars_dict['qSMVA_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_B = _vars_dict['RATIO_' + _type]
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_RATIO_B = _vars_dict['ABS_RATIO_' + _type]
    # Variables for the Asymmetry of the Bids
    # variable: delta active power upwards contribution
    DW_B_U = _vars_dict['DW_' + _type + '_U']
    # variable: delta active power downwards contribution
    DW_B_D = _vars_dict['DW_' + _type + '_D']
    # variable: delta reactive power upwards contribution
    DR_B_U = _vars_dict['DR_' + _type + '_U']
    # variable: delta reactive power downwards contribution
    DR_B_D = _vars_dict['DR_' + _type + '_D']

    _idx_dr = 0
    for x, y in zip(DR_B.tolist(), ABS_DR_B.tolist()):
        _idx_dr_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_dr = 'abs_dr_' + _type + '_#' + str(_idx_dr) + '_t' + str(_idx_dr_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_dr)
            _idx_dr_t += 1
        _idx_dr += 1

    _idx_dw = 0
    for x, y in zip(DW_B.tolist(), ABS_DW_B.tolist()):
        _idx_dw_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_dw = 'abs_dw_' + _type + '_#' + str(_idx_dw) + '_t' + str(_idx_dw_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_dw)
            _idx_dw_t += 1
        _idx_dw += 1

    _idx_ratio = 0
    for x, y in zip(RATIO_B.tolist(), ABS_RATIO_B.tolist()):
        _idx_ratio_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_ratio = 'abs_ratio_' + _type + '_#' + str(_idx_ratio) + '_t' + str(_idx_ratio_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_ratio)
            _idx_ratio_t += 1
        _idx_ratio += 1

    _name_W = 'C#04a_W_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DW_B[i, t] + fsp_data['Winit'].iloc[i, t] - W_B[i, t] == 0, name=_name_W)
    _name_R = 'C#04a_R_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DR_B[i, t] + fsp_data['Rinit'].iloc[i, t] - R_B[i, t] == 0, name=_name_R)

    if _cap_cstr:
        _name_q_cstr = 'C#04a_' + _type + '_NORM(SMVA)_' + str(i) + '_t' + str(t)
        grb_model.addConstr(W_B[i, t] ** 2 + R_B[i, t] ** 2 - qSMVA_B[i, t] == 0, name=_name_q_cstr)
        _name_q_cstr2 = 'C#04a_' + _type + '_SMAX_limit' + str(i) + '_t' + str(t)
        grb_model.addConstr(qSMVA_B[i, t] <= fsp_data['mva'][i] ** 2, name=_name_q_cstr2)
        # Power Factor limits
        _name_pf_limit = "C#04b_TANPHI_B_" + str(i) + "_t" + str(t)
        grb_model.addConstr((W_B[i, t] * RATIO_B[i, t]) - R_B[i, t] == 0, name=_name_pf_limit)
        _name_tan_lb = "C#04b_TANPHI_B_LowerLimit_" + str(i) + "_t" + str(t)
        grb_model.addConstr(ABS_RATIO_B[i, t] - fsp_data['TANPHI_LB'].iloc[i, t] >= 0, name=_name_tan_lb)
        _name_tan_ub = "C#04b_TANPHI_B_UpperLimit_" + str(i) + "_t" + str(t)
        grb_model.addConstr(ABS_RATIO_B[i, t] - fsp_data['TANPHI_UB'].iloc[i, t] <= 0, name=_name_tan_ub)

    # Model with Upper and Lower Bounds for Q/P=TanPhi Ratio
    _name_w_ub = 'C#04b_W_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_B[i, t] - fsp_data['Wub'].iloc[i, t] <= 0, name=_name_w_ub)
    _name_w_lb = 'C#04b_W_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_B[i, t] - fsp_data['Wlb'].iloc[i, t] >= 0, name=_name_w_lb)
    _name_r_ub = 'C#04b_R_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_B[i, t] - fsp_data['Rub'].iloc[i, t] <= 0, name=_name_r_ub)
    _name_r_lb = 'C#04b_R_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_B[i, t] - fsp_data['Rlb'].iloc[i, t] >= 0, name=_name_r_lb)

    # Constrains for the Asymmetry of Bids
    # Upward and Downward Delta W contributions
    _name_dw = 'C#07a_DW_' + _type + '_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DW_B_U[i, t] - DW_B_D[i, t] - DW_B[i, t] == 0, name=_name_dw)
    # Upward and Downward Delta R contributions
    _name_dr = 'C#07a_DR_' + _type + '_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DR_B_U[i, t] - DR_B_D[i, t] - DR_B[i, t] == 0, name=_name_dr)
    return grb_model


def gen_constraints_v1(grb_model, _vars_dict, fsp_data, i, t, _cap_cstr, _type='B'):
    """Define Constraints of the Generator Resource."""
    # Extract Variables
    # Delta Reactive Power Exchange [p.u.]
    DR_B = _vars_dict['DR_' + _type]
    # Absolute values of Delta Reactive Power Exchange
    ABS_DR_B = _vars_dict['ABS_DR_' + _type]
    # After the Market Reactive Power Exchange
    R_B = _vars_dict['R_' + _type]
    # Delta Active Power Exchange [p.u.]
    DW_B = _vars_dict['DW_' + _type]
    # Absolute values of Delta Active Power Exchange
    ABS_DW_B = _vars_dict['ABS_DW_' + _type]
    # After the Market Active Power
    W_B = _vars_dict['W_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_B = _vars_dict['qSMVA_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_B = _vars_dict['RATIO_' + _type]
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_RATIO_B = _vars_dict['ABS_RATIO_' + _type]
    # Variables for the Asymmetry of the Bids
    # variable: delta active power upwards contribution
    DW_B_U = _vars_dict['DW_' + _type + '_U']
    # variable: delta active power downwards contribution
    DW_B_D = _vars_dict['DW_' + _type + '_D']
    # variable: delta reactive power upwards contribution
    DR_B_U = _vars_dict['DR_' + _type + '_U']
    # variable: delta reactive power downwards contribution
    DR_B_D = _vars_dict['DR_' + _type + '_D']

    for x, y in zip(DR_B.keys(), ABS_DR_B.keys()):
        _name_abs_dr = 'abs_dr_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_DR_B[y], DR_B[x], name=_name_abs_dr)

    for x, y in zip(DW_B.keys(), ABS_DW_B.keys()):
        _name_abs_dw = 'abs_dw_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_DW_B[y], DW_B[x], name=_name_abs_dw)

    for x, y in zip(RATIO_B.keys(), ABS_RATIO_B.keys()):
        _name_abs_ratio = 'abs_ratio_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_RATIO_B[y], RATIO_B[x], name=_name_abs_ratio)

    _name_W = 'C#04a_W_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DW_B[i, t] + fsp_data['Winit'].iloc[i, t] - W_B[i, t] == 0, name=_name_W)
    _name_R = 'C#04a_R_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DR_B[i, t] + fsp_data['Rinit'].iloc[i, t] - R_B[i, t] == 0, name=_name_R)

    if _cap_cstr:
        _name_q_cstr = 'C#04a_' + _type + '_NORM(SMVA)_' + str(i) + '_t' + str(t)
        grb_model.addConstr(W_B[i, t] ** 2 + R_B[i, t] ** 2 - qSMVA_B[i, t] == 0, name=_name_q_cstr)
        _name_q_cstr2 = 'C#04a_' + _type + '_SMAX_limit' + str(i) + '_t' + str(t)
        grb_model.addConstr(qSMVA_B[i, t] <= fsp_data['mva'][i] ** 2, name=_name_q_cstr2)
        # Power Factor limits
        _name_pf_limit = "C#04b_TANPHI_B_" + str(i) + "_t" + str(t)
        grb_model.addConstr((W_B[i, t] * RATIO_B[i, t]) - R_B[i, t] == 0, name=_name_pf_limit)
        _name_tan_lb = "C#04b_TANPHI_B_LowerLimit_" + str(i) + "_t" + str(t)
        grb_model.addConstr(ABS_RATIO_B[i, t] - fsp_data['TANPHI_LB'].iloc[i, t] >= 0, name=_name_tan_lb)
        _name_tan_ub = "C#04b_TANPHI_B_UpperLimit_" + str(i) + "_t" + str(t)
        grb_model.addConstr(ABS_RATIO_B[i, t] - fsp_data['TANPHI_UB'].iloc[i, t] <= 0, name=_name_tan_ub)

    # Model with Upper and Lower Bounds for Q/P=TanPhi Ratio
    _name_w_ub = 'C#04b_W_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_B[i, t] - fsp_data['Wub'].iloc[i, t] <= 0, name=_name_w_ub)
    _name_w_lb = 'C#04b_W_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_B[i, t] - fsp_data['Wlb'].iloc[i, t] >= 0, name=_name_w_lb)
    _name_r_ub = 'C#04b_R_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_B[i, t] - fsp_data['Rub'].iloc[i, t] <= 0, name=_name_r_ub)
    _name_r_lb = 'C#04b_R_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_B[i, t] - fsp_data['Rlb'].iloc[i, t] >= 0, name=_name_r_lb)

    # Constrains for the Asymmetry of Bids
    # Upward and Downward Delta W contributions
    _name_dw = 'C#07a_DW_' + _type + '_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DW_B_U[i, t] - DW_B_D[i, t] - DW_B[i, t] == 0, name=_name_dw)
    # Upward and Downward Delta R contributions
    _name_dr = 'C#07a_DR_' + _type + '_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DR_B_U[i, t] - DR_B_D[i, t] - DR_B[i, t] == 0, name=_name_dr)
    return grb_model


def fsp_converter(_data, _k_opt=1, _class_fsp='generation'):
    """
    Adapt the fsp generation info into usable data for the local flexibility market optimization.


    :param _data: Dictionary containing all the information useful for the conversion into usable data. The useful data
        are the pandapower network (net), the dictionary with the timeseries simulation (ts_res), the dataframe with the fsp
        data involved in the simulation (fsp_data), the output root where to save files (out_root), the list of hours with
        network constraint violations (hours), the bidding strategy of the fsp element (bid_strat), the multiplier
        coefficient of the load (scen_factor_load), the multiplier coefficient of the generation (scen_factor_gen),
        the active power bid up coefficient (kbid_p_up), the active power bid down coefficient (kbid_p_dwn),
        the reactive power bid up coefficient (kbid_q_up), the reactive power down up coefficient (kbid_q_dwn) and
        the simulation tag (sim_tag).


    :param _k_opt: Margin for bidding below the full capacity - Safe for optimization convergence.


    :param _class_fsp: String that define the name of the fsp typology.
    """
    warning_fsp = False

    # Pandapower network
    _net = _data['net']
    # Dictionary with the timeseries simulation.
    _ts_res = _data['ts_res']
    # Dataframe with the fsp data involved in the simulation.
    _fsp_data = _data['fsp_data']
    # Output root where to save files.
    _out_root = _data['out_root']
    # List of hours with network constraint violations.
    _hours = _data['hours']
    # Bidding strategy of the fsp element.
    _bid_strat = _data['bid_strat']
    # Multiplier coefficient of the generation.
    _scen_factor_gen = _data['scen_factor_sgen']
    # Active power bid up coefficient.
    _kbid_p_up = _data['kbid_p_up']
    # Active power bid down coefficient.
    _kbid_p_dwn = _data['kbid_p_dwn']
    # Reactive power bid up coefficient.
    _kbid_q_up = _data['kbid_q_up']
    # Reactive power bid down coefficient.
    _kbid_q_dwn = _data['kbid_q_dwn']
    # Simulation tag.
    _sim_tag = _data['sim_tag']

    fsp_filename = 0

    sgen_p_premkt = _ts_res['sgen_p_mw'].T
    sgen_q_premkt = _ts_res['sgen_q_mvar'].T
    class_fsp_df = _fsp_data.loc[_fsp_data.type == _class_fsp].reset_index(drop=True)

    n_sgen_fsp = len(class_fsp_df.id)

    class_fsp_df.mva = class_fsp_df.mva * _scen_factor_gen
    class_fsp_df.Pup = class_fsp_df.Pup * _kbid_p_up
    class_fsp_df.Pdown = class_fsp_df.Pdown * _kbid_p_dwn
    class_fsp_df.Qup = class_fsp_df.Qup * _kbid_q_up
    class_fsp_df.Qdown = class_fsp_df.Qdown * _kbid_q_dwn

    _n_hours = len(_hours)
    shape_init = (n_sgen_fsp, _n_hours)
    cosphi_init = np.zeros(shape=shape_init)
    cosphi_lb = 0.2 * np.ones(shape=shape_init)
    cosphi_ub = 1.0 * np.ones(shape=shape_init)

    r_ub_dlim = np.zeros(shape=shape_init)
    r_ub_ulim = np.zeros(shape=shape_init)
    w_lb_dlim = np.zeros(shape=shape_init)
    w_ub_ulim = np.zeros(shape=shape_init)

    fsp_p_up_cost = class_fsp_df.Pup_cost.to_numpy().repeat(_n_hours).reshape(shape_init)
    fsp_p_dwn_cost = class_fsp_df.Pdown_cost.to_numpy().repeat(_n_hours).reshape(shape_init)
    fsp_q_up_cost = class_fsp_df.Qup_cost.to_numpy().repeat(_n_hours).reshape(shape_init)
    fsp_q_dwn_cost = class_fsp_df.Qdown_cost.to_numpy().repeat(_n_hours).reshape(shape_init)

    fsp_w_init = class_fsp_df.mw.to_numpy().repeat(_n_hours).reshape(shape_init)
    fsp_r_init = class_fsp_df.mvar.to_numpy().repeat(_n_hours).reshape(shape_init)

    for i in tqdm(range(n_sgen_fsp), desc='Initialising Generator bids (Generators follow the generation convention)'):
        _idx = _net.sgen[_net.sgen.name == class_fsp_df.name.reset_index(drop=True)[i]].index
        fsp_w_init[i, :] = sgen_p_premkt.loc[_idx, _hours]
        fsp_r_init[i, :] = sgen_q_premkt.loc[_idx, _hours]

        if _bid_strat == 'initial_value':
            w_ref = fsp_w_init
        else:
            # _bid_strat == 'mva_value':
            w_ref = class_fsp_df.mva

        for h in range(_n_hours):
            # Initial Cosphi
            if fsp_w_init[i, h] == 0:
                cosphi_init[i, h] = 0
            else:
                cosphi_init[i, h] = math.cos(math.atan(fsp_r_init[i, h] / fsp_w_init[i, h]))

            # Reactive power upper bound downward limit
            if class_fsp_df.Qdown.iloc[i] > 0:
                r1 = math.sqrt(class_fsp_df.mva.iloc[i] ** 2 - fsp_w_init[i, h] ** 2)
                r_ub_dlim[i, h] = -(class_fsp_df.Qdown.iloc[i] * (r1 - abs(fsp_r_init[i, h]))) + abs(fsp_r_init[i, h])
            else:
                r_ub_dlim[i, h] = fsp_r_init[i, h]

            # Reactive power upper bound upward limit
            if class_fsp_df.Qup.iloc[i] > 0:
                r1 = math.sqrt(class_fsp_df.mva.iloc[i] ** 2 - fsp_w_init[i, h] ** 2)
                r_ub_ulim[i, h] = (class_fsp_df.Qup.iloc[i] * (r1 - abs(fsp_r_init[i, h]))) + abs(fsp_r_init[i, h])
            else:
                r_ub_ulim[i, h] = fsp_r_init[i, h]

            # Active power upper bound upward limit
            if class_fsp_df.Pup.iloc[i] > 0:
                w_ub_ulim[i, h] = fsp_w_init[i, h] + (class_fsp_df.Pup.iloc[i] * w_ref[i, h])
            else:
                w_ub_ulim[i, h] = fsp_w_init[i, h]

            # Active power lower bound downward limit
            if class_fsp_df.Pdown.iloc[i] > 0:
                w_lb_dlim[i, h] = fsp_w_init[i, h] - ((class_fsp_df.Pdown.iloc[i]) * w_ref[i, h])
            else:
                w_lb_dlim[i, h] = fsp_w_init[i, h]

            if w_ub_ulim[i, h] > (class_fsp_df.mva.iloc[i] * cosphi_ub[i, h]):
                warning_fsp = True
                w_ub_ulim[i, h] = class_fsp_df.mva.iloc[i] * cosphi_ub[i, h] * _k_opt

            if r_ub_ulim[i, h] > (class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h])))):
                warning_fsp = True
                r_ub_ulim[i, h] = class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h]))) * _k_opt

        if warning_fsp:
            warnings.warn('Static Generator capacity saturated for some item')

    if n_sgen_fsp > 0:
        fsp_info = pd.concat([class_fsp_df.id, class_fsp_df.name, class_fsp_df.bus, class_fsp_df.mva], axis=1)
        fsp_filename = 'FSPB_init' + _sim_tag
        io_file.save_excel(_data=fsp_info, _outroot=_out_root, _filename=fsp_filename, _sheetname='FSPinfo')

        io_file.save_excel(_data=pd.DataFrame(fsp_p_up_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='W_U_Cost', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(fsp_p_dwn_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='W_D_Cost', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(fsp_q_up_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='R_U_Cost', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(fsp_q_dwn_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='R_D_Cost', _mode='a')

        io_file.save_excel(_data=pd.DataFrame(fsp_w_init, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='Winit', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(fsp_r_init, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='Rinit', _mode='a')

        io_file.save_excel(_data=pd.DataFrame(cosphi_init, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='COSPHIinit', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(cosphi_lb, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='COSPHILB', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(cosphi_ub, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='COSPHIUB', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(r_ub_dlim, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='RUB_DLim', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(r_ub_ulim, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='RUB_ULim', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(w_ub_ulim, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='WUB_ULim', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(w_lb_dlim, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='WLB_DLim', _mode='a')
    return fsp_filename, class_fsp_df.bus
