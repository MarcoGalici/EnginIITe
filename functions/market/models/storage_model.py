import math
import warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from gurobipy import GRB, tuplelist
from functions import file_io as io_file


def _eval_asybid(_n_fsp, _simperiod, fsp_data):
    """Evaluate the symmetry of bid for the Flexibility Service Provider type C (Storage)."""
    fsp_data['DW_U_inj'] = np.zeros((_n_fsp, len(_simperiod)))
    fsp_data['DW_D_inj'] = np.zeros((_n_fsp, len(_simperiod)))
    fsp_data['DW_U_ads'] = np.zeros((_n_fsp, len(_simperiod)))
    fsp_data['DW_D_ads'] = np.zeros((_n_fsp, len(_simperiod)))
    for t in _simperiod:
        for i in range(_n_fsp):
            if fsp_data['Winit'].iloc[i, t] >= 0:
                # Generator
                fsp_data['DW_U_inj'][i, t] = fsp_data['Wub'].iloc[i, t] - fsp_data['Winit'].iloc[i, t]
                fsp_data['DW_D_inj'][i, t] = min([fsp_data['Winit'].iloc[i, t], fsp_data['Winit'].iloc[i, t] - fsp_data['Wlb'].iloc[i, t]])
                fsp_data['DW_U_ads'][i, t] = 0
                fsp_data['DW_D_ads'][i, t] = abs(min([0, fsp_data['Wlb'].iloc[i, t]]))
            else:
                # Load
                fsp_data['DW_U_inj'][i, t] = max([fsp_data['Wub'].iloc[i, t] - fsp_data['Winit'].iloc[i, t], fsp_data['Wub'].iloc[i, t]])
                fsp_data['DW_D_inj'][i, t] = 0
                fsp_data['DW_U_ads'][i, t] = abs(min([fsp_data['Winit'].iloc[i, t], fsp_data['Wub'].iloc[i, t] - fsp_data['Winit'].iloc[i, t]]))
                fsp_data['DW_D_ads'][i, t] = abs(fsp_data['Wlb'].iloc[i, t] - fsp_data['Winit'].iloc[i, t])
    return fsp_data


def res_params(_n_fsp, _sim_period, _fsp_data):
    """
    Define Input Data of the Load Resource.
    :param _n_fsp: Number of FSP of type Load resource.
    :param _sim_period: List of hours to consider for the study.
    :param _fsp_data: Dictionary of the Data associated
    """
    fsp_data = dict()
    # Save Data that are Impose to GENERATION Convention and can be changed to LOAD Convention
    ess_conv = ['DR_', 'R_', 'DW_', 'W_', 'Winj_', 'Wads_', 'RATIO_']
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
        # Upper Value Quantity for the Upward Offer
        fsp_data['Rub'] = _fsp_data['RUB_ULim'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Upper Value Quantity for the Upward Offer
        fsp_data['Rlb'] = _fsp_data['RUB_DLim'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Lower Bound Cosphi Value (Minimum Cosphi Value admitted by the Resource)
        fsp_data['COSPHILB'] = _fsp_data['COSPHILB'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Upper Bound Cosphi Value (Maximum Cosphi Value admitted by the Resource)
        fsp_data['COSPHIUB'] = _fsp_data['COSPHIUB'].iloc[list(range(0, _n_fsp)), _sim_period]
        # SoC Initial Value
        fsp_data['SoCinit'] = _fsp_data['SoCinit'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Efficiency in Injective Active Power
        fsp_data['NI_inj'] = _fsp_data['NIinj'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Efficiency in Absorbing Active Power
        fsp_data['NI_ads'] = _fsp_data['NIads'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Efficiency in Reactive Power Exchange
        fsp_data['NI_q'] = _fsp_data['NIq'].iloc[list(range(0, _n_fsp)), _sim_period]
        # INJECTION Cost for Active Power UPWARD Support Provision [€/MVAh] for each Resource
        fsp_data['WU_inj_cost'] = _fsp_data['W_U_inj_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]
        # INJECTION Cost for Active Power DOWNWARD Support Provision [€/MVAh] for each Resource
        fsp_data['WD_inj_cost'] = _fsp_data['W_D_inj_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]
        # ABSORPTION Cost for Active Power UPWARD Support Provision [€/MVAh] for each Resource
        fsp_data['WU_ads_cost'] = _fsp_data['W_U_ads_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]
        # ABSORPTION Cost for Active Power DOWNWARD Support Provision [€/MVAh] for each Resource
        fsp_data['WD_ads_cost'] = _fsp_data['W_D_ads_Cost'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Lower Bound SoC Value
        fsp_data['SoC_lb'] = _fsp_data['SoCLB'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Upper Bound SoC Value
        fsp_data['SoC_ub'] = _fsp_data['SoCUB'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Lower Bound Tanphi
        fsp_data['TANPHI_LB'] = np.tan(np.arccos(fsp_data['COSPHIUB']))
        # Upper Bound Tanphi
        fsp_data['TANPHI_UB'] = np.tan(np.arccos(fsp_data['COSPHILB']))
        # Evaluate the Asymmetry of Bids
        fsp_data = _eval_asybid(_n_fsp, _sim_period, fsp_data)
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
        fsp_data['SoC_lb'] = _zero_df
        fsp_data['SoC_ub'] = _zero_df
        fsp_data['SoCinit'] = _zero_df
        fsp_data['NI_inj'] = _zero_df
        fsp_data['NI_ads'] = _zero_df
        fsp_data['NI_q'] = _zero_df
        fsp_data['WU_inj_cost'] = _zero_df
        fsp_data['WD_inj_cost'] = _zero_df
        fsp_data['WU_ads_cost'] = _zero_df
        fsp_data['WD_ads_cost'] = _zero_df
        fsp_data['DW_U_inj'] = _zero_df
        fsp_data['DW_D_inj'] = _zero_df
        fsp_data['DW_U_ads'] = _zero_df
        fsp_data['DW_D_ads'] = _zero_df
    return fsp_data, ess_conv


def define_variables(_grb_model, _n_fsp, _n_t, _contr, _type='C'):
    """
    Define Variables of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _n_fsp: Number of flexibility service providers.
    :param _n_t: Number of intervals.
    :param _contr: Contribution of these resource type.
    :param _type: Type of resource.
    """
    _vars_dict = storage_variables_v1(grb_model=_grb_model, n_fsp=_n_fsp, n_t=_n_t, _contr=_contr, _type=_type)
    return _vars_dict


def storage_variables_v0(grb_model, n_fsp, n_t, _contr, _type='C'):
    """Define Variables of the Storage Resource."""
    shape_var = (n_fsp, n_t)
    _vars_dict = dict()
    # Delta Reactive Power Exchange [p.u.]
    DR_C = grb_model.addMVar(shape=shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DR_' + _type)
    _vars_dict['DR_' + _type] = DR_C
    # After the Market Reactive Power Exchange
    R_C = grb_model.addMVar(shape=shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='R_' + _type)
    _vars_dict['R_' + _type] = R_C
    # Delta Active Power [p.u.]
    DW_C = grb_model.addMVar(shape=shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DW_' + _type)
    _vars_dict['DW_' + _type] = DW_C
    # Delta Active Power Absorption for type C (batteries) [p.u.]
    DWads_C = grb_model.addMVar(shape=shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DWads_' + _type)
    _vars_dict['DWads_' + _type] = DWads_C
    # Delta Active Power Injection for type C (batteries) [p.u.]
    DWinj_C = grb_model.addMVar(shape=shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DWinj_' + _type)
    _vars_dict['DWinj_' + _type] = DWinj_C
    # variable: active power adsorption (in per unit) for batteries (type C)
    Wads_C = grb_model.addMVar(shape=shape_var, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name='Wads_' + _type)
    _vars_dict['Wads_' + _type] = Wads_C
    # variable: active power injection (in per unit) for batteries (type C)
    Winj_C = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Winj_' + _type)
    _vars_dict['Winj_' + _type] = Winj_C
    # variable: after the market State of Charge for batteries (type C) [energy KWh]
    SOC_C = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='SOC_' + _type)
    _vars_dict['SOC_' + _type] = SOC_C
    # After the Market Active Power
    W_C = grb_model.addMVar(shape=shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='W_' + _type)
    _vars_dict['W_' + _type] = W_C
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_C = grb_model.addMVar(shape=shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='RATIO_' + _type)
    _vars_dict['RATIO_' + _type] = RATIO_C
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_C = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='qSMVA_' + _type)
    _vars_dict['qSMVA_' + _type] = qSMVA_C
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_DR_C = grb_model.addMVar(shape=shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_DR_' + _type)
    _vars_dict['ABS_DR_' + _type] = ABS_DR_C
    ABS_R_C = grb_model.addMVar(shape=shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_R_' + _type)
    _vars_dict['ABS_R_' + _type] = ABS_R_C
    ABS_DWads_C = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_DWads_' + _type)
    _vars_dict['ABS_DWads_' + _type] = ABS_DWads_C
    ABS_DWinj_C = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_DWinj_' + _type)
    _vars_dict['ABS_DWinj_' + _type] = ABS_DWinj_C
    ABS_RATIO_C = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_RATIO_' + _type)
    _vars_dict['ABS_RATIO_' + _type] = ABS_RATIO_C
    # Variables for the Asymmetry of the Bids
    DW_C_U = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U')
    _vars_dict['DW_' + _type + '_U'] = DW_C_U
    DW_C_D = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D')
    _vars_dict['DW_' + _type + '_D'] = DW_C_D
    DW_C_U_inj = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U_inj')
    _vars_dict['DW_' + _type + '_U_inj'] = DW_C_U_inj
    DW_C_D_inj = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D_inj')
    _vars_dict['DW_' + _type + '_D_inj'] = DW_C_D_inj
    DW_C_U_ads = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U_ads')
    _vars_dict['DW_' + _type + '_U_ads'] = DW_C_U_ads
    DW_C_D_ads = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D_ads')
    _vars_dict['DW_' + _type + '_D_ads'] = DW_C_D_ads
    DR_C_U = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_U')
    _vars_dict['DR_' + _type + '_U'] = DR_C_U
    DR_C_D = grb_model.addMVar(shape=shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_D')
    _vars_dict['DR_' + _type + '_D'] = DR_C_D
    # Binary Variables
    b1_C = grb_model.addMVar(shape=shape_var, vtype=GRB.BINARY, name='b1_' + _type)
    _vars_dict['b1_' + _type] = b1_C
    b2_C = grb_model.addMVar(shape=shape_var, vtype=GRB.BINARY, name='b2_' + _type)
    _vars_dict['b2_' + _type] = b2_C
    b3_C = grb_model.addMVar(shape=shape_var, vtype=GRB.BINARY, name='b3_' + _type)
    _vars_dict['b3_' + _type] = b3_C
    b4_C = grb_model.addMVar(shape=shape_var, vtype=GRB.BINARY, name='b4_' + _type)
    _vars_dict['b4_' + _type] = b4_C
    return _vars_dict


def storage_variables_v1(grb_model, n_fsp, n_t, _contr, _type='C'):
    """Define Variables of the Storage Resource."""
    shape_var = tuplelist([(_f, _t) for _f in range(n_fsp) for _t in range(n_t)])
    _vars_dict = dict()
    # Delta Reactive Power Exchange [p.u.]
    DR_C = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DR_' + _type)
    _vars_dict['DR_' + _type] = DR_C
    # After the Market Reactive Power Exchange
    R_C = grb_model.addVars(shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='R_' + _type)
    _vars_dict['R_' + _type] = R_C
    # Delta Active Power [p.u.]
    DW_C = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DW_' + _type)
    _vars_dict['DW_' + _type] = DW_C
    # Delta Active Power Absorption for type C (batteries) [p.u.]
    DWads_C = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DWads_' + _type)
    _vars_dict['DWads_' + _type] = DWads_C
    # Delta Active Power Injection for type C (batteries) [p.u.]
    DWinj_C = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DWinj_' + _type)
    _vars_dict['DWinj_' + _type] = DWinj_C
    # variable: active power adsorption (in per unit) for batteries (type C)
    Wads_C = grb_model.addVars(shape_var, lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS, name='Wads_' + _type)
    _vars_dict['Wads_' + _type] = Wads_C
    # variable: active power injection (in per unit) for batteries (type C)
    Winj_C = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Winj_' + _type)
    _vars_dict['Winj_' + _type] = Winj_C
    # variable: after the market State of Charge for batteries (type C) [energy KWh]
    SOC_C = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='SOC_' + _type)
    _vars_dict['SOC_' + _type] = SOC_C
    # After the Market Active Power
    W_C = grb_model.addVars(shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='W_' + _type)
    _vars_dict['W_' + _type] = W_C
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_C = grb_model.addVars(shape_var, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='RATIO_' + _type)
    _vars_dict['RATIO_' + _type] = RATIO_C
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_C = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='qSMVA_' + _type)
    _vars_dict['qSMVA_' + _type] = qSMVA_C
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_DR_C = grb_model.addVars(shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_DR_' + _type)
    _vars_dict['ABS_DR_' + _type] = ABS_DR_C
    ABS_R_C = grb_model.addVars(shape_var, lb=0, ub=_contr, vtype=GRB.CONTINUOUS, name='ABS_R_' + _type)
    _vars_dict['ABS_R_' + _type] = ABS_R_C
    ABS_DWads_C = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_DWads_' + _type)
    _vars_dict['ABS_DWads_' + _type] = ABS_DWads_C
    ABS_DWinj_C = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_DWinj_' + _type)
    _vars_dict['ABS_DWinj_' + _type] = ABS_DWinj_C
    ABS_RATIO_C = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_RATIO_' + _type)
    _vars_dict['ABS_RATIO_' + _type] = ABS_RATIO_C
    # Variables for the Asymmetry of the Bids
    DW_C_U = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U')
    _vars_dict['DW_' + _type + '_U'] = DW_C_U
    DW_C_D = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D')
    _vars_dict['DW_' + _type + '_D'] = DW_C_D
    DW_C_U_inj = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U_inj')
    _vars_dict['DW_' + _type + '_U_inj'] = DW_C_U_inj
    DW_C_D_inj = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D_inj')
    _vars_dict['DW_' + _type + '_D_inj'] = DW_C_D_inj
    DW_C_U_ads = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_U_ads')
    _vars_dict['DW_' + _type + '_U_ads'] = DW_C_U_ads
    DW_C_D_ads = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DW_' + _type + '_D_ads')
    _vars_dict['DW_' + _type + '_D_ads'] = DW_C_D_ads
    DR_C_U = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_U')
    _vars_dict['DR_' + _type + '_U'] = DR_C_U
    DR_C_D = grb_model.addVars(shape_var, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='DR_' + _type + '_D')
    _vars_dict['DR_' + _type + '_D'] = DR_C_D
    # Binary Variables
    b1_C = grb_model.addVars(shape_var, vtype=GRB.BINARY, name='b1_' + _type)
    _vars_dict['b1_' + _type] = b1_C
    b2_C = grb_model.addVars(shape_var, vtype=GRB.BINARY, name='b2_' + _type)
    _vars_dict['b2_' + _type] = b2_C
    b3_C = grb_model.addVars(shape_var, vtype=GRB.BINARY, name='b3_' + _type)
    _vars_dict['b3_' + _type] = b3_C
    b4_C = grb_model.addVars(shape_var, vtype=GRB.BINARY, name='b4_' + _type)
    _vars_dict['b4_' + _type] = b4_C
    return _vars_dict


def define_constraints(_grb_model, _vars_dict, _fsp_data, _idx_res, _interval, _cap_cstr, _int_soc=True, _dt=1, _type='C'):
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
    grb_model = storage_constraints_v1(grb_model=_grb_model, _vars_dict=_vars_dict, fsp_data=_fsp_data, i=_idx_res,
                                       t=_interval, _cap_cstr=_cap_cstr, _int_soc=_int_soc, _dt=_dt, _type=_type)
    return grb_model


def storage_constraints_v0(grb_model, _vars_dict, fsp_data, i, t, _cap_cstr, _int_soc, _dt=1, _type='C'):
    """Define Constraints of the Storage Resource."""
    # Delta Reactive Power Exchange [p.u.]
    DR_C = _vars_dict['DR_' + _type]
    # Absolute values of Delta Reactive Power Exchange
    ABS_DR_C = _vars_dict['ABS_DR_' + _type]
    # After the Market Reactive Power Exchange
    R_C = _vars_dict['R_' + _type]
    # Delta Active Power [p.u.]
    DW_C = _vars_dict['DW_' + _type]
    # After the Market Active Power
    W_C = _vars_dict['W_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_C = _vars_dict['qSMVA_' + _type]
    # variable: active power adsorption (in per unit) for batteries (type C)
    Wads_C = _vars_dict['Wads_' + _type]
    # variable: Delta active power adsorption (in per unit) for batteries (type C)
    DWads_C = _vars_dict['DWads_' + _type]
    # variable: Absolute values of Delta active power adsorption for batteries (type C)
    ABS_DWads_C = _vars_dict['ABS_DWads_' + _type]
    # variable: active power injection (in per unit) for batteries (type C)
    Winj_C = _vars_dict['Winj_' + _type]
    # variable: Delta active power injection (in per unit) for batteries (type C)
    DWinj_C = _vars_dict['DWinj_' + _type]
    # variable: Absolute values of Delta active power injection for batteries (type C)
    ABS_DWinj_C = _vars_dict['ABS_DWinj_' + _type]
    # variable: after the market State of Charge for batteries (type C) [energy KWh]
    SOC_C = _vars_dict['SOC_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_C = _vars_dict['RATIO_' + _type]
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_R_C = _vars_dict['ABS_R_' + _type]
    ABS_RATIO_C = _vars_dict['ABS_RATIO_' + _type]
    # Variables for the Asymmetry of the Bids
    DW_C_U = _vars_dict['DW_' + _type + '_U']
    DW_C_D = _vars_dict['DW_' + _type + '_D']
    DW_C_U_inj = _vars_dict['DW_' + _type + '_U_inj']
    DW_C_D_inj = _vars_dict['DW_' + _type + '_D_inj']
    DW_C_U_ads = _vars_dict['DW_' + _type + '_U_ads']
    DW_C_D_ads = _vars_dict['DW_' + _type + '_D_ads']
    DR_C_U = _vars_dict['DR_' + _type + '_U']
    DR_C_D = _vars_dict['DR_' + _type + '_D']
    # Binary Variables
    b1_C = _vars_dict['b1_' + _type]
    b2_C = _vars_dict['b2_' + _type]
    b3_C = _vars_dict['b3_' + _type]
    b4_C = _vars_dict['b4_' + _type]

    _idx_dr = 0
    for x, y in zip(DR_C.tolist(), ABS_DR_C.tolist()):
        _idx_dr_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_dr = 'abs_dr_' + _type + '_#' + str(_idx_dr) + '_t' + str(_idx_dr_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_dr)
            _idx_dr_t += 1
        _idx_dr += 1

    _idx_dWads = 0
    for x, y in zip(DWads_C.tolist(), ABS_DWads_C.tolist()):
        _idx_dWads_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_wads = 'abs_wads_' + _type + '_#' + str(_idx_dWads) + '_t' + str(_idx_dWads_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_wads)
            _idx_dWads_t += 1
        _idx_dWads += 1

    _idx_dWinj = 0
    for x, y in zip(DWinj_C.tolist(), ABS_DWinj_C.tolist()):
        _idx_dWinj_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_winj = 'abs_winj_' + _type + '_#' + str(_idx_dWinj) + '_t' + str(_idx_dWinj_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_winj)
            _idx_dWinj_t += 1
        _idx_dWinj += 1

    _idx_ratio = 0
    for x, y in zip(RATIO_C.tolist(), ABS_RATIO_C.tolist()):
        _idx_ratio_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_ratio = 'abs_ratio_' + _type + '_#' + str(_idx_ratio) + '_t' + str(_idx_ratio_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_ratio)
            _idx_ratio_t += 1
        _idx_ratio += 1

    _idx_r = 0
    for x, y in zip(R_C.tolist(), ABS_R_C.tolist()):
        _idx_r_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_r = 'abs_r_' + _type + '_#' + str(_idx_r) + '_t' + str(_idx_r_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_r)
            _idx_r += 1
        _idx_r += 1

    _name_W = 'C#05c_W_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_C[i, t] - DW_C[i, t] - fsp_data['Winit'].iloc[i, t] == 0, name=_name_W)
    # TODO: The following constraint can be rewritten as: R - Dr - Rinit = 0,
    #  since moving all terms from the left-hand side to the right-hand side will change their sign
    #  and the constraint result remains unchanged.
    _name_R = 'C#05_R_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(-R_C[i, t] + DR_C[i, t] + fsp_data['Rinit'].iloc[i, t] == 0, name=_name_R)

    if _cap_cstr:
        _name_q_cstr = 'C#05a_' + _type + '_NORM(SMVA)_' + str(i) + '_t' + str(t)
        grb_model.addConstr(W_C[i, t] ** 2 + R_C[i, t] ** 2 - qSMVA_C[i, t] == 0, name=_name_q_cstr)
        _name_q_cstr2 = 'C#05a_' + _type + '_SMAX_limit' + str(i) + '_t' + str(t)
        grb_model.addConstr(qSMVA_C[i, t] - fsp_data['mva'][i] ** 2 <= 0, name=_name_q_cstr2)
        # Power Factor limits
        _name_pf_limit = 'C#05b_TANPHI_' + _type + '_' + str(i) + '_t' + str(t)
        grb_model.addConstr(W_C[i, t] * RATIO_C[i, t] - R_C[i, t] == 0, name=_name_pf_limit)
        _name_tan_lb = 'C#05b_TANPHI_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
        grb_model.addConstr(ABS_RATIO_C[i, t] - fsp_data['TANPHI_LB'].iloc[i, t] >= 0, name=_name_tan_lb)
        _name_tan_ub = 'C#05b_TANPHI_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
        grb_model.addConstr(ABS_RATIO_C[i, t] - fsp_data['TANPHI_UB'].iloc[i, t] <= 0, name=_name_tan_ub)

    # Model with Upper and Lower Bounds for Q/P=TanPhi Ratio
    # FSPC_WUB stays on the far right in the x-axis - zero is included in FSPC_WLB < 0 < FSPC_WUB
    _name_w_ub = 'C#05b_W_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_C[i, t] - fsp_data['Wub'].iloc[i, t] <= 0, name=_name_w_ub)
    # FSPC_WUB stays on the fart left in the x-axis - zero is included in FSPC_WLB < 0 < FSPC_WUB
    _name_w_lb = 'C#05b_W_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_C[i, t] - fsp_data['Wlb'].iloc[i, t] >= 0, name=_name_w_lb)

    # Definition of Upward and Downward Delta R contributions
    _name_dr = 'C#07a_DR_C_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DR_C_U[i, t] - DR_C_D[i, t] - DR_C[i, t] == 0, name=_name_dr)
    # FSPC_WUB stays on the far right in the x-axis - zero is included in FSPC_WLB<0<FSPC_WUB
    _name_r_ub = 'C#05b_R_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_C[i, t] - fsp_data['Rub'].iloc[i, t] <= 0, name=_name_r_ub)
    # FSPC_WUB stays on the fart left in the x-axis - zero is included in FSPC_WLB<0<FSPC_WUB
    _name_r_lb = 'C#05b_R_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_C[i, t] - fsp_data['Rlb'].iloc[i, t] >= 0, name=_name_r_lb)

    if _int_soc:
        if t == 0:
            _name_soc_t0 = 'C#05_FSP_' + _type + '_SoC_update_tx_' + str(i) + '_t' + str(t)
            grb_model.addConstr((fsp_data['SoCinit'].iloc[i, t] + (-Wads_C[i, t] * fsp_data['NI_ads'].iloc[i, t] * b2_C[i, t]
                                  - b1_C[i, t] * Winj_C[i, t] / fsp_data['NI_inj'].iloc[i, t]
                                - ABS_R_C[i, t] * (1 - fsp_data['NI_q'].iloc[i, t])) * _dt) - SOC_C[i, t] == 0, name=_name_soc_t0)
        else:
            _name_soc_t = 'C#05_FSP_' + _type + '_SoC_update_tx_' + str(i) + '_t' + str(t)
            grb_model.addConstr((SOC_C[i, t - 1] + (-Wads_C[i, t] * fsp_data['NI_ads'].iloc[i, t] * b2_C[i, t]
                                  - b1_C[i, t] * Winj_C[i, t] / fsp_data['NI_inj'].iloc[i, t]
                                  - ABS_R_C[i, t] * (1 - fsp_data['NI_q'].iloc[i, t])) * _dt) - SOC_C[i, t] == 0, name=_name_soc_t)

        # TODO: this constraints can be removed if we set the upper and lower bounds of SoC
        #  at the moment when we create the variable SoC
        _name_soc_min = 'C#05a_FSP_' + _type + '_SoC_LB_' + str(i) + '_t' + str(t)
        grb_model.addConstr(SOC_C[i, t] - fsp_data['SoC_lb'].iloc[i, t] >= 0, name=_name_soc_min)
        _name_soc_max = 'C#05b_FSP_' + _type + '_SoC_UB_' + str(i) + '_t' + str(t)
        grb_model.addConstr(SOC_C[i, t] - fsp_data['SoC_ub'].iloc[i, t] <= 0, name=_name_soc_max)

    # Constrains for the Asymmetry of Bids
    # Definition of Upward and Downward DELTA W contributions
    _name_dw = 'C#07a_DW_' + _type + '_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DW_C_U[i, t] - DW_C_D[i, t] - DW_C[i, t] == 0, name=_name_dw)
    # Binary Constraints for W output
    _name_w_binary = 'C#05a_' + _type + '_Winjetting_' + str(i) + '_t' + str(t)
    grb_model.addConstr((Winj_C[i, t] * b1_C[i, t] + Wads_C[i, t] * b2_C[i, t]) - W_C[i, t] == 0, name=_name_w_binary)
    _name_b12 = 'C#05a_' + _type + '_b1+b2_' + str(i) + '_t' + str(t)
    grb_model.addConstr(b1_C[i, t] + b2_C[i, t] == 1, name=_name_b12)

    _name_winj_max = 'C#05a_Winj_' + _type + '_boundB01_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Winj_C[i, t] - b1_C[i, t] * fsp_data['Wub'].iloc[i, t] <= 0, name=_name_winj_max)
    _name_winj_min = 'C#05a_Winj_' + _type + '_boundB02_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Winj_C[i, t] >= 0, name=_name_winj_min)
    _name_wads_max = 'C#05a_Wads_' + _type + '_boundB03_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Wads_C[i, t] <= 0, name=_name_wads_max)
    _name_wads_min = 'C#05a_Wads_' + _type + '_boundB04_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Wads_C[i, t] - b2_C[i, t] * fsp_data['Wlb'].iloc[i, t] >= 0, name=_name_wads_min)

    # Electric Vehicle Asymmetry bid mode ON
    # Differentiated Cost of Downward and Upward Flexibility Between Generation and Consumption
    if fsp_data['Winit'].iloc[i, t] >= 0:
        # x1 - alternative
        _name_dw_d_cond = 'C#05a_DW_' + _type + '_D_Condition_' + str(i) + '_t' + str(t)
        grb_model.addConstr(
            (DW_C_D_inj[i, t] * b3_C[i, t] + (DW_C_D_ads[i, t] + fsp_data['Winit'].iloc[i, t]) * b4_C[i, t])
            - DW_C_D[i, t] == 0, name=_name_dw_d_cond)
        _name_b34 = 'C#05a_' + _type + '_b3+b4_' + str(i) + '_t' + str(t)
        grb_model.addConstr(b3_C[i, t] + b4_C[i, t] == 1, name=_name_b34)

        _name_dWinj_true = 'C#05a_DW_' + _type + '_D_inj_TRUE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], True, DW_C_D_inj[i, t], GRB.EQUAL, fsp_data['Winit'].iloc[i, t], name=_name_dWinj_true)
        _name_dWinj_false = 'C#05a_DW_' + _type + '_D_inj_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_D_inj[i, t] - DW_C_D[i, t], GRB.EQUAL, 0, name=_name_dWinj_false)
        _name_dWads_false = 'C#05a_DW_' + _type + '_D_ads_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_D_ads[i, t], GRB.EQUAL, 0, name=_name_dWads_false)

        _name_dWinj_eq = 'C#05a_DW_' + _type + '_U_inj_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_U_inj[i, t] - DW_C_U[i, t] == 0, name=_name_dWinj_eq)
        _name_dWads_eq = 'C#05a_DW_' + _type + '_U_ads_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_U_ads[i, t] == 0, name=_name_dWads_eq)
    else:
        # x1 - alternative
        _name_dw_u_cond = 'C#05a_DW_' + _type + '_U_Condition_' + str(i) + '_t' + str(t)
        grb_model.addConstr(
            (DW_C_U_ads[i, t] * b3_C[i, t] + (DW_C_U_inj[i, t] + abs(fsp_data['Winit'].iloc[i, t])) * b4_C[i, t])
            - DW_C_U[i, t] == 0, name=_name_dw_u_cond)
        _name_b34 = 'C#05a_' + _type + '_b3+b4_' + str(i) + '_t' + str(t)
        grb_model.addConstr(b3_C[i, t] + b4_C[i, t] == 1, name=_name_b34)

        _name_dWads_true = 'C#05a_DW_' + _type + '_U_ads_TRUE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], True, DW_C_U_ads[i, t], GRB.EQUAL, -fsp_data['Winit'].iloc[i, t], name=_name_dWads_true)
        _name_dWads_false = 'C#05a_DW_' + _type + '_U_ads_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_U_ads[i, t] - DW_C_U[i, t], GRB.EQUAL, 0, name=_name_dWads_false)
        _name_dWinj_false = 'C#05a_DW_' + _type + '_U_inj_c02_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_U_inj[i, t], GRB.EQUAL, 0, name=_name_dWinj_false)

        _name_dWads_eq = 'C#05a_DW_' + _type + '_U_ads_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_D_ads[i, t] - DW_C_D[i, t] == 0, name=_name_dWads_eq)
        _name_dWinj_eq = 'C#05a_DW_' + _type + '_U_inj_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_D_inj[i, t] == 0, name=_name_dWinj_eq)

    # fixme: 'TO FORCE THE SOLUTION WHEN ONLY P OR ONLY Q AND VOID FLAT SOLUTIONS'.
    #  Why only available for the VC market?
    if fsp_data['Wlb'].iloc[i, t] == fsp_data['Wub'].iloc[i, t]:
        _name_onlyQ_dW_u = 'C#06a_C_force_onlyQ_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_U[i, t] == 0, name=_name_onlyQ_dW_u)
        _name_onlyQ_dW_d = 'C#06b_C_force_onlyQ_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_D[i, t] == 0, name=_name_onlyQ_dW_d)
    if fsp_data['Rub'].iloc[i, t] == fsp_data['Rlb'].iloc[i, t]:
        _name_onlyP_dR_u = 'C#06a_C_force_onlyP_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DR_C_U[i, t] == 0, name=_name_onlyP_dR_u)
        _name_onlyP_dR_d = 'C#06b_C_force_onlyP_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DR_C_D[i, t] == 0, name=_name_onlyP_dR_d)
    return grb_model


def storage_constraints_v1(grb_model, _vars_dict, fsp_data, i, t, _cap_cstr, _int_soc, _dt=1, _type='C'):
    """Define Constraints of the Storage Resource."""
    # Delta Reactive Power Exchange [p.u.]
    DR_C = _vars_dict['DR_' + _type]
    # Absolute values Delta Reactive Power Exchange
    ABS_DR_C = _vars_dict['ABS_DR_' + _type]
    # After the Market Reactive Power Exchange
    R_C = _vars_dict['R_' + _type]
    # Delta Active Power [p.u.]
    DW_C = _vars_dict['DW_' + _type]
    # After the Market Active Power
    W_C = _vars_dict['W_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    qSMVA_C = _vars_dict['qSMVA_' + _type]
    # variable: active power adsorption (in per unit) for batteries (type C)
    Wads_C = _vars_dict['Wads_' + _type]
    # variable: Delta active power adsorption (in per unit) for batteries (type C)
    DWads_C = _vars_dict['DWads_' + _type]
    # variable: Absolute values of Delta active power adsorption for batteries (type C)
    ABS_DWads_C = _vars_dict['ABS_DWads_' + _type]
    # variable: active power injection (in per unit) for batteries (type C)
    Winj_C = _vars_dict['Winj_' + _type]
    # variable: Delta active power injection (in per unit) for batteries (type C)
    DWinj_C = _vars_dict['DWinj_' + _type]
    # variable: Absolute values of Delta active power injection for batteries (type C)
    ABS_DWinj_C = _vars_dict['ABS_DWinj_' + _type]
    # variable: after the market State of Charge for batteries (type C) [energy KWh]
    SOC_C = _vars_dict['SOC_' + _type]
    # Companion variables - necessary in gurobi to include the QUADRATIC constraints
    RATIO_C = _vars_dict['RATIO_' + _type]
    # Companion variables - necessary in gurobi to include ABSOLUTE values in the objective function
    ABS_R_C = _vars_dict['ABS_R_' + _type]
    ABS_RATIO_C = _vars_dict['ABS_RATIO_' + _type]
    # Variables for the Asymmetry of the Bids
    DW_C_U = _vars_dict['DW_' + _type + '_U']
    DW_C_D = _vars_dict['DW_' + _type + '_D']
    DW_C_U_inj = _vars_dict['DW_' + _type + '_U_inj']
    DW_C_D_inj = _vars_dict['DW_' + _type + '_D_inj']
    DW_C_U_ads = _vars_dict['DW_' + _type + '_U_ads']
    DW_C_D_ads = _vars_dict['DW_' + _type + '_D_ads']
    DR_C_U = _vars_dict['DR_' + _type + '_U']
    DR_C_D = _vars_dict['DR_' + _type + '_D']
    # Binary Variables
    b1_C = _vars_dict['b1_' + _type]
    b2_C = _vars_dict['b2_' + _type]
    b3_C = _vars_dict['b3_' + _type]
    b4_C = _vars_dict['b4_' + _type]

    for x, y in zip(DR_C.keys(), ABS_DR_C.keys()):
        _name_abs_dr = 'abs_dr_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_DR_C[y], DR_C[x], name=_name_abs_dr)

    for x, y in zip(DWads_C.keys(), ABS_DWads_C.keys()):
        _name_abs_wads = 'abs_wads_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_DWads_C[y], DWads_C[x], name=_name_abs_wads)

    for x, y in zip(DWinj_C.keys(), ABS_DWinj_C.keys()):
        _name_abs_winj = 'abs_winj_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_DWinj_C[y], DWinj_C[x], name=_name_abs_winj)

    for x, y in zip(RATIO_C.keys(), ABS_RATIO_C.keys()):
        _name_abs_ratio = 'abs_ratio_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_RATIO_C[y], RATIO_C[x], name=_name_abs_ratio)

    for x, y in zip(R_C.keys(), ABS_R_C.keys()):
        _name_abs_r = 'abs_r_' + _type + '_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_R_C[y], R_C[x], name=_name_abs_r)

    _name_W = 'C#05c_W_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_C[i, t] - DW_C[i, t] - fsp_data['Winit'].iloc[i, t] == 0, name=_name_W)
    # TODO: The following constraint can be rewritten as: R - Dr - Rinit = 0,
    #  since moving all terms from the left-hand side to the right-hand side will change their sign
    #  and the constraint result remains unchanged.
    _name_R = 'C#05_R_' + _type + '_' + str(i) + '_t' + str(t)
    grb_model.addConstr(-R_C[i, t] + DR_C[i, t] + fsp_data['Rinit'].iloc[i, t] == 0, name=_name_R)

    if _cap_cstr:
        _name_q_cstr = 'C#05a_' + _type + '_NORM(SMVA)_' + str(i) + '_t' + str(t)
        grb_model.addConstr(W_C[i, t] ** 2 + R_C[i, t] ** 2 - qSMVA_C[i, t] == 0, name=_name_q_cstr)
        _name_q_cstr2 = 'C#05a_' + _type + '_SMAX_limit' + str(i) + '_t' + str(t)
        grb_model.addConstr(qSMVA_C[i, t] - fsp_data['mva'][i] ** 2 <= 0, name=_name_q_cstr2)
        # Power Factor limits
        _name_pf_limit = 'C#05b_TANPHI_' + _type + '_' + str(i) + '_t' + str(t)
        grb_model.addConstr(W_C[i, t] * RATIO_C[i, t] - R_C[i, t] == 0, name=_name_pf_limit)
        _name_tan_lb = 'C#05b_TANPHI_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
        grb_model.addConstr(ABS_RATIO_C[i, t] - fsp_data['TANPHI_LB'].iloc[i, t] >= 0, name=_name_tan_lb)
        _name_tan_ub = 'C#05b_TANPHI_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
        grb_model.addConstr(ABS_RATIO_C[i, t] - fsp_data['TANPHI_UB'].iloc[i, t] <= 0, name=_name_tan_ub)

    # Model with Upper and Lower Bounds for Q/P=TanPhi Ratio
    # FSPC_WUB stays on the far right in the x-axis - zero is included in FSPC_WLB < 0 < FSPC_WUB
    _name_w_ub = 'C#05b_W_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_C[i, t] - fsp_data['Wub'].iloc[i, t] <= 0, name=_name_w_ub)
    # FSPC_WUB stays on the fart left in the x-axis - zero is included in FSPC_WLB < 0 < FSPC_WUB
    _name_w_lb = 'C#05b_W_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(W_C[i, t] - fsp_data['Wlb'].iloc[i, t] >= 0, name=_name_w_lb)

    # Definition of Upward and Downward Delta R contributions
    _name_dr = 'C#07a_DR_C_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DR_C_U[i, t] - DR_C_D[i, t] - DR_C[i, t] == 0, name=_name_dr)
    # FSPC_WUB stays on the far right in the x-axis - zero is included in FSPC_WLB<0<FSPC_WUB
    _name_r_ub = 'C#05b_R_' + _type + '_UpperLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_C[i, t] - fsp_data['Rub'].iloc[i, t] <= 0, name=_name_r_ub)
    # FSPC_WUB stays on the fart left in the x-axis - zero is included in FSPC_WLB<0<FSPC_WUB
    _name_r_lb = 'C#05b_R_' + _type + '_LowerLimit_' + str(i) + '_t' + str(t)
    grb_model.addConstr(R_C[i, t] - fsp_data['Rlb'].iloc[i, t] >= 0, name=_name_r_lb)

    if _int_soc:
        if t == 0:
            _name_soc_t0 = 'C#05_FSP_' + _type + '_SoC_update_tx_' + str(i) + '_t' + str(t)
            grb_model.addConstr((fsp_data['SoCinit'].iloc[i, t] + (-Wads_C[i, t] * fsp_data['NI_ads'].iloc[i, t] * b2_C[i, t]
                                  - b1_C[i, t] * Winj_C[i, t] / fsp_data['NI_inj'].iloc[i, t]
                                - ABS_R_C[i, t] * (1 - fsp_data['NI_q'].iloc[i, t])) * _dt) - SOC_C[i, t] == 0, name=_name_soc_t0)
        else:
            _name_soc_t = 'C#05_FSP_' + _type + '_SoC_update_tx_' + str(i) + '_t' + str(t)
            grb_model.addConstr((SOC_C[i, t - 1] + (-Wads_C[i, t] * fsp_data['NI_ads'].iloc[i, t] * b2_C[i, t]
                                  - b1_C[i, t] * Winj_C[i, t] / fsp_data['NI_inj'].iloc[i, t]
                                  - ABS_R_C[i, t] * (1 - fsp_data['NI_q'].iloc[i, t])) * _dt) - SOC_C[i, t] == 0, name=_name_soc_t)

        # TODO: this constraints can be removed if we set the upper and lower bounds of SoC
        #  at the moment when we create the variable SoC
        _name_soc_min = 'C#05a_FSP_' + _type + '_SoC_LB_' + str(i) + '_t' + str(t)
        grb_model.addConstr(SOC_C[i, t] - fsp_data['SoC_lb'].iloc[i, t] >= 0, name=_name_soc_min)
        _name_soc_max = 'C#05b_FSP_' + _type + '_SoC_UB_' + str(i) + '_t' + str(t)
        grb_model.addConstr(SOC_C[i, t] - fsp_data['SoC_ub'].iloc[i, t] <= 0, name=_name_soc_max)

    # Constrains for the Asymmetry of Bids
    # Definition of Upward and Downward DELTA W contributions
    _name_dw = 'C#07a_DW_' + _type + '_Up&Down_' + str(i) + '_t' + str(t)
    grb_model.addConstr(DW_C_U[i, t] - DW_C_D[i, t] - DW_C[i, t] == 0, name=_name_dw)
    # Binary Constraints for W output
    _name_w_binary = 'C#05a_' + _type + '_Winjetting_' + str(i) + '_t' + str(t)
    grb_model.addConstr((Winj_C[i, t] * b1_C[i, t] + Wads_C[i, t] * b2_C[i, t]) - W_C[i, t] == 0, name=_name_w_binary)
    _name_b12 = 'C#05a_' + _type + '_b1+b2_' + str(i) + '_t' + str(t)
    grb_model.addConstr(b1_C[i, t] + b2_C[i, t] == 1, name=_name_b12)

    _name_winj_max = 'C#05a_Winj_' + _type + '_boundB01_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Winj_C[i, t] - b1_C[i, t] * fsp_data['Wub'].iloc[i, t] <= 0, name=_name_winj_max)
    _name_winj_min = 'C#05a_Winj_' + _type + '_boundB02_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Winj_C[i, t] >= 0, name=_name_winj_min)
    _name_wads_max = 'C#05a_Wads_' + _type + '_boundB03_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Wads_C[i, t] <= 0, name=_name_wads_max)
    _name_wads_min = 'C#05a_Wads_' + _type + '_boundB04_' + str(i) + '_t' + str(t)
    grb_model.addConstr(Wads_C[i, t] - b2_C[i, t] * fsp_data['Wlb'].iloc[i, t] >= 0, name=_name_wads_min)

    # Electric Vehicle Asymmetry bid mode ON
    # Differentiated Cost of Downward and Upward Flexibility Between Generation and Consumption
    if fsp_data['Winit'].iloc[i, t] >= 0:
        # x1 - alternative
        _name_dw_d_cond = 'C#05a_DW_' + _type + '_D_Condition_' + str(i) + '_t' + str(t)
        grb_model.addConstr(
            (DW_C_D_inj[i, t] * b3_C[i, t] + (DW_C_D_ads[i, t] + fsp_data['Winit'].iloc[i, t]) * b4_C[i, t])
            - DW_C_D[i, t] == 0, name=_name_dw_d_cond)
        _name_b34 = 'C#05a_' + _type + '_b3+b4_' + str(i) + '_t' + str(t)
        grb_model.addConstr(b3_C[i, t] + b4_C[i, t] == 1, name=_name_b34)

        _name_dWinj_true = 'C#05a_DW_' + _type + '_D_inj_TRUE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], True, DW_C_D_inj[i, t], GRB.EQUAL, fsp_data['Winit'].iloc[i, t], name=_name_dWinj_true)
        _name_dWinj_false = 'C#05a_DW_' + _type + '_D_inj_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_D_inj[i, t] - DW_C_D[i, t], GRB.EQUAL, 0, name=_name_dWinj_false)
        _name_dWads_false = 'C#05a_DW_' + _type + '_D_ads_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_D_ads[i, t], GRB.EQUAL, 0, name=_name_dWads_false)

        _name_dWinj_eq = 'C#05a_DW_' + _type + '_U_inj_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_U_inj[i, t] - DW_C_U[i, t] == 0, name=_name_dWinj_eq)
        _name_dWads_eq = 'C#05a_DW_' + _type + '_U_ads_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_U_ads[i, t] == 0, name=_name_dWads_eq)
    else:
        # x1 - alternative
        _name_dw_u_cond = 'C#05a_DW_' + _type + '_U_Condition_' + str(i) + '_t' + str(t)
        grb_model.addConstr(
            (DW_C_U_ads[i, t] * b3_C[i, t] + (DW_C_U_inj[i, t] + abs(fsp_data['Winit'].iloc[i, t])) * b4_C[i, t])
            - DW_C_U[i, t] == 0, name=_name_dw_u_cond)
        _name_b34 = 'C#05a_' + _type + '_b3+b4_' + str(i) + '_t' + str(t)
        grb_model.addConstr(b3_C[i, t] + b4_C[i, t] == 1, name=_name_b34)

        _name_dWads_true = 'C#05a_DW_' + _type + '_U_ads_TRUE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], True, DW_C_U_ads[i, t], GRB.EQUAL, -fsp_data['Winit'].iloc[i, t], name=_name_dWads_true)
        _name_dWads_false = 'C#05a_DW_' + _type + '_U_ads_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_U_ads[i, t] - DW_C_U[i, t], GRB.EQUAL, 0, name=_name_dWads_false)
        _name_dWinj_false = 'C#05a_DW_' + _type + '_U_inj_c02_FALSE' + str(i) + '_t' + str(t)
        grb_model.addGenConstrIndicator(b4_C[i, t], False, DW_C_U_inj[i, t], GRB.EQUAL, 0, name=_name_dWinj_false)

        _name_dWads_eq = 'C#05a_DW_' + _type + '_U_ads_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_D_ads[i, t] - DW_C_D[i, t] == 0, name=_name_dWads_eq)
        _name_dWinj_eq = 'C#05a_DW_' + _type + '_U_inj_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_D_inj[i, t] == 0, name=_name_dWinj_eq)

    # fixme: 'TO FORCE THE SOLUTION WHEN ONLY P OR ONLY Q AND VOID FLAT SOLUTIONS'.
    #  Why only available for the VC market?
    if fsp_data['Wlb'].iloc[i, t] == fsp_data['Wub'].iloc[i, t]:
        _name_onlyQ_dW_u = 'C#06a_C_force_onlyQ_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_U[i, t] == 0, name=_name_onlyQ_dW_u)
        _name_onlyQ_dW_d = 'C#06b_C_force_onlyQ_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DW_C_D[i, t] == 0, name=_name_onlyQ_dW_d)
    if fsp_data['Rub'].iloc[i, t] == fsp_data['Rlb'].iloc[i, t]:
        _name_onlyP_dR_u = 'C#06a_C_force_onlyP_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DR_C_U[i, t] == 0, name=_name_onlyP_dR_u)
        _name_onlyP_dR_d = 'C#06b_C_force_onlyP_' + str(i) + '_t' + str(t)
        grb_model.addConstr(DR_C_D[i, t] == 0, name=_name_onlyP_dR_d)

    return grb_model


def fsp_converter(_data, _k_opt=1, _class_fsp='storage'):
    """
    Adapt the fsp storage info into usable data for the local flexibility market optimization.


    :param _data: Dictionary containing all the information useful for the conversion into usable data. The useful data
        are the pandapower network (net), the dictionary with the timeseries simulation (ts_res), the dataframe with the fsp
        data involved in the simulation (fsp_data), the output root where to save files (out_root), the list of hours with
        network constraint violations (hours), the bidding strategy of the fsp element (bid_strat), the multiplier
        coefficient of the load (scen_factor_load), the multiplier coefficient of the generation (scen_factor_gen),
        the nominal power of the storage (ess_kmva), the active power bid up coefficient (kbid_p_up),
        the active power bid down coefficient (kbid_p_dwn), the reactive power bid up coefficient (kbid_q_up),
        the reactive power down up coefficient (kbid_q_dwn) and the simulation tag (sim_tag).


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
    # Multiplier coefficient of the load.
    _scen_factor_load = _data['scen_factor_load']
    # Nominal power of the storage
    _ess_kmva = _data['ess_kmva']
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

    load_p_premkt = _ts_res['load_p_mw'].T
    load_q_premkt = _ts_res['load_q_mvar'].T
    # load_p_premkt = _ts_res['storage_p_mw'].T
    # load_q_premkt = _ts_res['storage_q_mvar'].T
    class_fsp_df = _fsp_data.loc[_fsp_data.type == _class_fsp].reset_index(drop=True)

    n_load_fsp = len(class_fsp_df.id)

    class_fsp_df.mva = class_fsp_df.mva * _scen_factor_load * _ess_kmva
    class_fsp_df.Pup = class_fsp_df.Pup * _kbid_p_up
    class_fsp_df.Pdown = class_fsp_df.Pdown * _kbid_p_dwn
    class_fsp_df.Qup = class_fsp_df.Qup * _kbid_q_up
    class_fsp_df.Qdown = class_fsp_df.Qdown * _kbid_q_dwn

    _n_hours = len(_hours)
    shape_init = (n_load_fsp, _n_hours)
    cosphi_init = np.zeros(shape=shape_init)
    cosphi_lb = 0.2 * np.ones(shape=shape_init)
    cosphi_ub = 1.0 * np.ones(shape=shape_init)
    # soc_init = 500 * np.ones(shape=shape_init)
    # soc_lb = 2 * np.ones(shape=shape_init)
    # soc_ub = 1000 * np.ones(shape=shape_init)
    soc_init = pd.Series((_net.storage['soc_percent']/100) * _net.storage['max_e_mwh']).to_numpy().repeat(_n_hours).reshape(shape_init)
    soc_lb = _net.storage['min_e_mwh'].to_numpy().repeat(_n_hours).reshape(shape_init)
    soc_ub = _net.storage['max_e_mwh'].to_numpy().repeat(_n_hours).reshape(shape_init)

    ni_inj = 0.99 * np.ones(shape=shape_init)
    ni_abs = 0.99 * np.ones(shape=shape_init)
    ni_q = 0.99 * np.ones(shape=shape_init)

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

    for i in tqdm(range(n_load_fsp), desc='Initialising Battery bids (Batteries follow the generation convention)'):
        _idx = _net.load[_net.load.bus == class_fsp_df.bus.reset_index(drop=True)[i]].index
        # _idx = _net.storage[_net.storage.bus == class_fsp_df.bus.reset_index(drop=True)[i]].index
        # Changing to generation convention
        fsp_w_init[i, :] = -load_p_premkt.loc[_idx, _hours]
        fsp_r_init[i, :] = -load_q_premkt.loc[_idx, _hours]

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

            if fsp_w_init[i, h] >= 0:
                if class_fsp_df.Pup.iloc[i] > 0:
                    w_ub_ulim[i, h] = fsp_w_init[i, h] + (class_fsp_df.Pup.iloc[i] * w_ref[i, h])
                else:
                    w_ub_ulim[i, h] = fsp_w_init[i, h]

                if class_fsp_df.Pdown.iloc[i] > 0:
                    w_lb_dlim[i, h] = fsp_w_init[i, h] - (class_fsp_df.Pdown.iloc[i] * w_ref[i, h])
                else:
                    w_lb_dlim[i, h] = fsp_w_init[i, h]
            else:
                if class_fsp_df.Pup.iloc[i] > 0:
                    w_ub_ulim[i, h] = fsp_w_init[i, h] - (class_fsp_df.Pup.iloc[i] * w_ref[i, h])
                else:
                    w_ub_ulim[i, h] = fsp_w_init[i, h]

                if class_fsp_df.Pdown.iloc[i] > 0:
                    w_lb_dlim[i, h] = fsp_w_init[i, h] + (class_fsp_df.Pdown.iloc[i] * w_ref[i, h])
                else:
                    w_lb_dlim[i, h] = fsp_w_init[i, h]

            if fsp_r_init[i, h] >= 0:
                # if storage is generating (generation convention is used)
                if class_fsp_df.Qdown.iloc[i] > 0:
                    r1 = math.sqrt(class_fsp_df.mva.iloc[i] ** 2 - fsp_w_init[i, h] ** 2)
                    r_ub_dlim[i, h] = -((class_fsp_df.Qdown.iloc[i] * (r1 - abs(fsp_r_init[i, h]))) + abs(fsp_r_init[i, h]))
                else:
                    r_ub_dlim[i, h] = fsp_r_init[i, h]

                if class_fsp_df.Qup.iloc[i] > 0:
                    r1 = math.sqrt(class_fsp_df.mva.iloc[i] ** 2 - fsp_w_init[i, h] ** 2)
                    r_ub_ulim[i, h] = (class_fsp_df.Qup.iloc[i] * (r1 - abs(fsp_r_init[i, h]))) + abs(fsp_r_init[i, h])
                else:
                    r_ub_ulim[i, h] = fsp_r_init[i, h]
            else:
                # if storage is consuming (generation convention is used)
                if class_fsp_df.Qup.iloc[i] > 0:
                    r1 = math.sqrt(class_fsp_df.mva.iloc[i] ** 2 - fsp_w_init[i, h] ** 2)
                    r_ub_ulim[i, h] = (class_fsp_df.Qup.iloc[i] * (r1 - abs(fsp_r_init[i, h]))) + abs(fsp_r_init[i, h])
                else:
                    r_ub_ulim[i, h] = fsp_r_init[i, h]

                if class_fsp_df.Qdown.iloc[i] > 0:
                    r1 = math.sqrt(class_fsp_df.mva.iloc[i] ** 2 - fsp_w_init[i, h] ** 2)
                    r_ub_dlim[i, h] = -((class_fsp_df.Qdown.iloc[i] * (r1 - abs(fsp_r_init[i, h]))) + abs(fsp_r_init[i, h]))
                else:
                    r_ub_dlim[i, h] = fsp_r_init[i, h]

            if abs(w_ub_ulim[i, h]) > (class_fsp_df.mva.iloc[i] * cosphi_ub[i, h]):
                warning_fsp = True
                if w_ub_ulim[i, h] > 0:
                    w_ub_ulim[i, h] = class_fsp_df.mva.iloc[i] * cosphi_ub[i, h] * _k_opt
                else:
                    w_ub_ulim[i, h] = -class_fsp_df.mva.iloc[i] * cosphi_ub[i, h] * _k_opt

            if abs(r_ub_ulim[i, h]) > (class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h])))):
                warning_fsp = True
                if r_ub_ulim[i, h] > 0:
                    r_ub_ulim[i, h] = class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h]))) * _k_opt
                else:
                    r_ub_ulim[i, h] = -class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h]))) * _k_opt

            if abs(w_lb_dlim[i, h]) > (class_fsp_df.mva.iloc[i] * cosphi_ub[i, h]):
                warning_fsp = True
                if w_lb_dlim[i, h] > 0:
                    w_lb_dlim[i, h] = class_fsp_df.mva.iloc[i] * cosphi_ub[i, h] * _k_opt
                else:
                    w_lb_dlim[i, h] = -class_fsp_df.mva.iloc[i] * cosphi_ub[i, h] * _k_opt

            if abs(r_ub_dlim[i, h]) > (class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h])))):
                warning_fsp = True
                if r_ub_dlim[i, h] > 0:
                    r_ub_ulim[i, h] = class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h]))) * _k_opt
                else:
                    r_ub_ulim[i, h] = -class_fsp_df.mva.iloc[i] * math.sin(math.atan(math.acos(cosphi_lb[i, h]))) * _k_opt

        if warning_fsp:
            warnings.warn('Storage Sn capacity saturated for some item')

    if n_load_fsp > 0:
        fsp_info = pd.concat([class_fsp_df.id, class_fsp_df.name, class_fsp_df.bus, class_fsp_df.mva], axis=1)
        fsp_filename = 'FSPC_init' + _sim_tag
        io_file.save_excel(_data=fsp_info, _outroot=_out_root, _filename=fsp_filename, _sheetname='FSPinfo')

        io_file.save_excel(_data=pd.DataFrame(fsp_p_up_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='W_U_inj_Cost', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(fsp_p_up_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='W_U_ads_Cost', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(fsp_p_dwn_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='W_D_inj_Cost', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(fsp_p_dwn_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='W_D_ads_Cost', _mode='a')
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

        io_file.save_excel(_data=pd.DataFrame(soc_init, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='SoCinit', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(soc_lb, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='SoCLB', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(soc_ub, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='SoCUB', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(ni_inj, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='NIinj', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(ni_abs, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='NIads', _mode='a')
        io_file.save_excel(_data=pd.DataFrame(ni_q, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='NIq', _mode='a')
    return fsp_filename, class_fsp_df.bus
