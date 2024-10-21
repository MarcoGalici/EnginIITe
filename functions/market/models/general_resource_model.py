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
    # Declare the variables that are Impose to GENERATION Convention and must be changed when saved to excel
    # Example: resource_conv =  ['DR_', 'R_', 'DW_', 'W_'], where 'DR_' represents the delta Reactive Power,
    # 'R_' represents the Reactive Power, 'DW_' represents the delta Active Power and 'W_' represents the Active Power
    resource_conv = []
    if _n_fsp > 0:
        # If number of flexibility service providers is greater than 0,
        # save the parameters useful for defining variables and constraints.

        # Example
        # Nameplate Apparent Power of the Resources [MVA]
        fsp_data['mva'] = _fsp_data['FSPinfo'].mva.to_numpy()
        # Initial Reactive Power
        fsp_data['Rinit'] = _fsp_data['Rinit'].iloc[list(range(0, _n_fsp)), _sim_period]
        # Initial Active Power
        fsp_data['Winit'] = _fsp_data['Winit'].iloc[list(range(0, _n_fsp)), _sim_period]
    else:
        # If number of flexibility service providers is equal to 0, set all to zero as Dataframe
        fsp_data['mva'] = np.zeros(shape=(1, 1))
        _zero_df = pd.DataFrame(np.zeros(shape=(1, len(_sim_period))))
        fsp_data['Rinit'] = _zero_df
        fsp_data['Winit'] = _zero_df
    return fsp_data, resource_conv


def define_variables(_grb_model, _n_fsp, _n_t, _contr, _type=''):
    """
    Define Variables of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _n_fsp: Number of flexibility service providers.
    :param _n_t: Number of intervals.
    :param _contr: Contribution of these resource type.
    :param _type: Type of resource.
    """
    _vars_dict = resource_variables(grb_model=_grb_model, n_fsp=_n_fsp, n_t=_n_t, _contr=_contr, _type=_type)
    return _vars_dict


def resource_variables(grb_model, n_fsp, n_t, _contr, _type=''):
    """Define Variables of the Load Resource."""
    # Shape Variables define the dimension of the tuple dict of Gurobipy
    shape_var = tuplelist([(_f, _t) for _f in range(n_fsp) for _t in range(n_t)])
    # Define dictionary for saving variables
    _vars_dict = dict()

    # Declare variables as follows using addVars function and saving the result into dictionary.
    # Example:
    # DR_A = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name='DR_' + _type)
    # _vars_dict['DR_' + _type] = DR_A
    _name_var = 'general_var_' + _type
    general_var_type = grb_model.addVars(shape_var, lb=-_contr, ub=_contr, vtype=GRB.CONTINUOUS, name=_name_var)
    _vars_dict[_name_var] = general_var_type
    return _vars_dict


def define_constraints(_grb_model, _vars_dict, _fsp_data, _idx_res, _interval, _cap_cstr, _int_soc=False, _dt=1, _type=''):
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
    _new_grb_model = resource_constraints(grb_model=_grb_model, _vars_dict=_vars_dict, fsp_data=_fsp_data, i=_idx_res,
                                         t=_interval, _cap_cstr=_cap_cstr, _type=_type)
    return _new_grb_model


def resource_constraints(grb_model, _vars_dict, fsp_data, i, t, _cap_cstr, _type=''):
    """Define Constraints of the Load Resource."""
    # Extract Variables for Constraints
    general_var_type = _vars_dict['general_var_' + _type]
    # Define Constraints
    pass
    return grb_model


def fsp_resource_converter(_data, _k_opt=1, _class_fsp=''):
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

    # Extract variables from dictionary
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
    _scen_factor_resource = _data['scen_factor_sgen']
    _scen_factor_resource = _data['scen_factor_load']
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

    # Extract Resource Profile from Timeseries
    resource_p_premkt = _ts_res['_p_mw'].T
    resource_q_premkt = _ts_res['_q_mvar'].T
    class_fsp_df = _fsp_data.loc[_fsp_data.type == _class_fsp].reset_index(drop=True)

    n_res_fsp = len(class_fsp_df.id)

    class_fsp_df.mva = class_fsp_df.mva * _scen_factor_resource
    class_fsp_df.Pup = class_fsp_df.Pup * _kbid_p_up
    class_fsp_df.Pdown = class_fsp_df.Pdown * _kbid_p_dwn
    class_fsp_df.Qup = class_fsp_df.Qup * _kbid_q_up
    class_fsp_df.Qdown = class_fsp_df.Qdown * _kbid_q_dwn

    _n_hours = len(_hours)
    shape_init = (n_res_fsp, _n_hours)

    fsp_p_up_cost = class_fsp_df.Pup_cost.to_numpy().repeat(_n_hours).reshape(shape_init)
    fsp_w_init = class_fsp_df.mw.to_numpy().repeat(_n_hours).reshape(shape_init)
    fsp_r_init = class_fsp_df.mvar.to_numpy().repeat(_n_hours).reshape(shape_init)

    for i in tqdm(range(n_res_fsp), desc='Initialising Resource bids (Resource follow the generator/load convention)'):
        _idx = _net.sgen[_net.sgen.name == class_fsp_df.name.reset_index(drop=True)[i]].index
        _idx = _net.load[_net.load.name == class_fsp_df.name.reset_index(drop=True)[i]].index
        fsp_w_init[i, :] = resource_p_premkt.loc[_idx, _hours]
        fsp_r_init[i, :] = resource_q_premkt.loc[_idx, _hours]
        # Define values useful for optimisation
        pass

        if warning_fsp:
            warnings.warn('Resource capacity saturated for some item')

    # Save values into fsp_filename as excel
    fsp_info = pd.concat([class_fsp_df.id, class_fsp_df.name, class_fsp_df.bus, class_fsp_df.mva], axis=1)
    fsp_filename = 'FSPB_init' + _sim_tag
    io_file.save_excel(_data=fsp_info, _outroot=_out_root, _filename=fsp_filename, _sheetname='FSPinfo')

    io_file.save_excel(_data=pd.DataFrame(fsp_p_up_cost, columns=_hours), _outroot=_out_root, _filename=fsp_filename, _sheetname='W_U_Cost', _mode='a')

    return fsp_filename, class_fsp_df.bus
