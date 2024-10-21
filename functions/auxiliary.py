import os
import warnings
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from functions import file_io as io_file


def set_resources():
    """
    Define resource considered in the platform.
    :return: Dictionary containing as keys the name of the resource type considered in the platform and as
     values the name of the python file in which the resource model for the market are written
     (this python file must be saved inside the functions.market.models folder).
    """
    # NOTE: The model works even if you delete the model that you do not want to evaluate in the flexibility market.
    _res_fcn = {
        'A': 'load_model',
        'B': 'generator_model',
        'C': 'storage_model'
    }
    return _res_fcn


def set_resource_profiles():
    """
    Define resource that require calculation of initial profile. The key of the dictionary must be the
     "element_name" column in the network Excel file. The element name must be the same as the "type" column in the
     fsp_data Excel file in which the provider are defined.
    :return: Dictionary containing as keys the name of the resource type considered in the platform and as
     values the name of the python file in which the resource model for the market are written
     (this python file must be saved inside the functions.market.models folder).
    """
    _res_fcn_profile = {
        'heatpump': 'heatpump_model'
    }
    return _res_fcn_profile


def fsp_converter(_net, _ts_res, _fsp_data, _out_root, _hours, _bid_strat, _scen_factor_load, _scen_factor_sgen,
                  _ess_kmva, _kbid_p_up, _kbid_p_dwn, _kbid_q_up, _kbid_q_dwn, _sim_tag, _k_opt=1):
    """
    Adapt the fsp resource info into usable data for the local flexibility market optimization.

    :param _net: Pandapower network.

    :param _ts_res: Dictionary with the timeseries simulation.

    :param _fsp_data: Dataframe with the fsp data involved in the simulation.

    :param _out_root: Output root where to save files.

    :param _hours: List of hours with network constraint violations.

    :param _bid_strat: Bidding strategy of the fsp element.

    :param _scen_factor_load: Multiplier coefficient of the load.

    :param _scen_factor_sgen: Multiplier coefficient of the generation.

    :param _ess_kmva: Nominal power of the storage.

    :param _kbid_p_up: Active power bid up coefficient.

    :param _kbid_p_dwn: Active power bid down coefficient.

    :param _kbid_q_up: Reactive power bid up coefficient.

    :param _kbid_q_dwn: Reactive power down up coefficient.

    :param _sim_tag: Simulation tag.

    :param _k_opt: Margin for bidding below the full capacity - Safe for optimization convergence
    """
    dict_res = set_resources()

    _data = dict()
    _data['net'] = _net
    _data['ts_res'] = _ts_res
    _data['fsp_data'] = _fsp_data
    _data['out_root'] = _out_root
    _data['hours'] = _hours
    _data['bid_strat'] = _bid_strat
    _data['scen_factor_load'] = _scen_factor_load
    _data['scen_factor_sgen'] = _scen_factor_sgen
    _data['ess_kmva'] = _ess_kmva
    _data['kbid_p_up'] = _kbid_p_up
    _data['kbid_p_dwn'] = _kbid_p_dwn
    _data['kbid_q_up'] = _kbid_q_up
    _data['kbid_q_dwn'] = _kbid_q_dwn
    _data['sim_tag'] = _sim_tag

    filename_out = dict()
    # bus_out = dict()
    bus_out = list()

    # Future development: Avoid running the for loop over all the resource types, but use only those resources
    # included in the _fsp_data.type --> for _i in _fsp_data.type.unique():
    # Future development: Put bus type in dictionary according to resource type. This mod will affect the sensitivity
    # factors evaluation.
    for _i in dict_res.keys():
        list_res = dict_res.get(_i, None)
        # fcn_res = list_res[0]
        # pos_res = list_res[1]
        if list_res is None:
            raise IOError('Resource of type {_class} is unknown.'.format(_class=_i))

        res_lib = importlib.import_module(f'functions.market.models.{list_res}')
        # filename_res, bus_res = fcn_res(_data=_data, _k_opt=_k_opt)
        filename_res, bus_res = res_lib.fsp_converter(_data=_data, _k_opt=_k_opt)
        filename_out[_i] = filename_res
        # bus_out[pos_res] = bus_res
        bus_out.extend(bus_res)

    return filename_out, bus_out


def eval_energy_support_res(_ts_res, _pilot_bus, _hours, _out_root, _fsp_file, _filename_lfm,
                            _filename_vm_pilot='pilot_bus_vm_post'):
    """
    Evaluate the final parameters. In particular the resource support in terms of energy.

    :param _ts_res: Dictionary of the timeseries simulation results.

    :param _pilot_bus: List of the selected pilot buses.

    :param _hours: List of hours with congestions.

   :param _out_root: Directory where to save the data.

   :param _fsp_file: Dictionary of file for each resource type.

   :param _filename_lfm: Name of the Local Flexibility Market file.

   :param _filename_vm_pilot: Name of the csv file for the bus voltage of the pilot buses.
    """
    # Save pilot bus voltage in per unit after the market.
    _bus_voltages_post = _ts_res['bus_vm_pu']
    pilot_bus_vm_post = _bus_voltages_post.loc[:, _pilot_bus]
    io_file.save_cvs(_data=pilot_bus_vm_post, _outroot=_out_root, _filename=_filename_vm_pilot)

    for _fsp_type in tqdm(_fsp_file.keys(), desc='Evaluating the support of resources'):
        if _fsp_file[_fsp_type] != 0:
            dw = io_file.import_data(_filename=Path(_filename_lfm).name, _root=_out_root, _sheetname='DW_' + _fsp_type, _index_col=0, _header=0)
            dr = io_file.import_data(_filename=Path(_filename_lfm).name, _root=_out_root, _sheetname='DR_' + _fsp_type, _index_col=0, _header=0)
            supp_p = np.zeros(len(dw.index))
            supp_q = np.zeros(len(dr.index))
            supp_s = np.zeros(len(dr.index))
            net_supp_p = np.zeros(len(dw.index))
            net_supp_q = np.zeros(len(dr.index))
            net_supp_s = np.zeros(len(dr.index))
            for i in range(len(dw.index)):
                supp_p[i] = sum(abs(dw.iloc[i, :]))
                supp_q[i] = sum(abs(dr.iloc[i, :]))
                supp_s[i] = np.sqrt((supp_p[i] * supp_p[i]) + (supp_q[i] * supp_q[i]))
                # Load: Positive if downward, Negative if upward
                # Sgen: Negative if downward, Positive if upward
                # Storage: Positive if downward, Negative if upward
                net_supp_p[i] = sum(dw.iloc[i, :])
                # Load: Positive if downward, Negatives if upward
                # Sgen: Negative if downward, Positive if upward
                # Storage: Positive if downward, Negative if upward
                net_supp_q[i] = sum(dr.iloc[i, :])
                net_supp_s[i] = np.sqrt((net_supp_p[i] * net_supp_p[i]) + (net_supp_q[i] * net_supp_q[i]))

            df_supp_p = pd.DataFrame(supp_p)
            df_supp_p.columns = ['SupportP_' + _fsp_type]
            df_supp_q = pd.DataFrame(supp_q)
            df_supp_q.columns = ['SupportQ_' + _fsp_type]
            df_supp_s = pd.DataFrame(supp_s)
            df_supp_s.columns = ['SupportS_' + _fsp_type]
            df_net_supp_p = pd.DataFrame(net_supp_p)
            df_net_supp_p.columns = ['NetSupportP_' + _fsp_type]
            df_net_supp_q = pd.DataFrame(net_supp_q)
            df_net_supp_q.columns = ['NetSupportQ_' + _fsp_type]
            df_net_supp_s = pd.DataFrame(net_supp_s)
            df_net_supp_s.columns = ['NetSupportS_' + _fsp_type]
            support_res = pd.concat([df_supp_p, df_supp_q, df_supp_s, df_net_supp_p, df_net_supp_q, df_net_supp_s], axis=1)
            io_file.save_excel(_data=support_res, _outroot=_out_root, _filename='Support_' + _fsp_type, _sheetname=_fsp_type)
    return True


def get_list(_startswith_key, _comparing_list):
    """
    Get the keys that starts with the value passed as input.

    :param _startswith_key: Starting string to which the values will be compared.

    :param _comparing_list: List of value to be compared.
    """
    list_keys = [i for i in _comparing_list.keys() if i.startswith(_startswith_key + '_')]
    if len(list_keys) == 0:
        # This value is essential for the "Storage" resource that adopt the "inj" and "ads" key in addition
        list_keys = [i for i in _comparing_list.keys() if i.startswith(_startswith_key)]
    return list_keys


def _check_root(_dir2check):
    """
    Check if root exist and in case it doesn't create it.
    :param _dir2check: Directory to check existence
    :return: True if the directory already exists, False otherwise (directory create it just now)
    """
    if not os.path.isdir(_dir2check):
        warnings.warn('Building {_name} path ...'.format(_name=_dir2check))
        os.makedirs(_dir2check)
        return False
    else:
        return True


def _def_dp(_cfg_file):
    """
    Check the set-point of the resources and define the dp and dq for the sensitivity factors evaluation,
     Increas by one the dp and dq to remain in the linear approximation of the sensitivity factors calculation.
    :param _cfg_file: Yaml configuration file
    :return: The dP and dQ for the sensitivity factor calculation
    """
    _net = io_file.import_network(_cfg_file['network'])
    _load_p = _net.load.p_mw
    _gen_p = _net.gen.p_mw
    _sgen_p = _net.sgen.p_mw
    _batt_p = _net.storage.p_mw

    _load_q = _net.load.q_mvar
    _sgen_q = _net.sgen.q_mvar
    _batt_q = _net.storage.q_mvar

    _min_p = [min(_load_p, default=10000), min(_gen_p, default=10000), min(_sgen_p, default=10000), min(_batt_p, default=10000)]
    _min_q = [min(_load_q, default=10000), min(_sgen_q, default=10000), min(_batt_q, default=10000)]
    _min_nnzero_p = [i for i in _min_p if i != 0]
    _min_nnzero_q = [j for j in _min_q if j != 0]
    _dp_net = 1 / (10 ** (int(('%e' % min(_min_nnzero_p)).partition('-')[2]) + 1))
    _dq_net = 1 / (10 ** (int(('%e' % min(_min_nnzero_q)).partition('-')[2]) + 1))
    return _dp_net, _dq_net


def set_general_path(_main_root, _name_network, _model_tag, _ess_tag, _sim_tag, _in_root='data_in', _out_root='data_out'):
    """Set the general paths of the project.
    :param _main_root: Root of the project.
    :param _name_network: Name of the network.
    :param _model_tag: Tag for the market model.
    :param _ess_tag: Tag for the Storage capacity flexibility provider.
    :param _sim_tag: Tag for the specific simulation name.
    :param _in_root: Name of the folder in which the input are saved.
    :param _out_root: Name of the folder in which the output will be saved.
    :return: Dictionary containing all the general paths.
    """

    _gpath_dict = dict()
    _gpath_dict['net_name'] = _name_network

    # INPUT PATH SETTINGS
    _input_path = os.path.join(_main_root, _in_root)
    if not os.path.exists(_input_path):
        raise IOError('The "{_name}" directory do not exists.'.format(_name=_input_path))
    _gpath_dict['in_root'] = _input_path

    _root2net = 'network'
    _net_input_path = os.path.join(_input_path, _name_network, _root2net)
    if not os.path.exists(_net_input_path):
        raise IOError('The "{_name}" directory do not exists.'.format(_name=_net_input_path))
    _gpath_dict['net_in_root'] = _net_input_path

    _root2fspfiles = 'fsp_files'
    _fsp_input_path = os.path.join(_input_path, _name_network, _root2fspfiles)
    if not os.path.exists(_fsp_input_path):
        raise IOError('The "{_name}" directory do not exists.'.format(_name=_fsp_input_path))
    _gpath_dict['fsp_in_root'] = _fsp_input_path

    _root2profiles = 'profile'
    _prof_input_path = os.path.join(_input_path, _name_network, _root2profiles)
    if not os.path.exists(_prof_input_path):
        raise IOError('The "{_name}" directory do not exists.'.format(_name=_prof_input_path))
    _gpath_dict['prof_in_root'] = _prof_input_path

    # Input market folder
    _gpath_dict['mrk_inputs'] = dict()
    _root2mrkinputs = 'market_inputs'
    _mrk_input_path = os.path.join(_input_path, _name_network, _root2mrkinputs)
    _check_root(_mrk_input_path)

    # Input sensitivity matrix folder
    _root2sensmatrix = 'h24SensitivityMatrixes'
    _sensmatrix_path = os.path.join(_mrk_input_path, _root2sensmatrix)
    _check_root(_sensmatrix_path)
    _gpath_dict['sens_matrix_root'] = _sensmatrix_path

    # OUTPUT PATH SETTINGS
    _output_path = os.path.join(_main_root, _out_root)

    _name_tags = [_name_network, _model_tag, _ess_tag, _sim_tag]
    _root2out_net = '_'.join(filter(None, _name_tags))
    _net_output_path = os.path.join(_output_path, _root2out_net)
    _gpath_dict['net_out_root'] = _net_output_path

    # # Pre Market output folders
    # _root2premrk_results = 'res_premarket'
    # _premrk_output_path = os.path.join(_net_output_path, _root2premrk_results)
    # _check_root(_premrk_output_path)
    # _gpath_dict['premrk_out_root'] = _premrk_output_path

    # Timeseries simulation output folders
    _root2tsnet_results = 'res_ts_net'
    _ts_output_path = os.path.join(_net_output_path, _root2tsnet_results)
    _check_root(_ts_output_path)
    _gpath_dict['ts_out_root'] = _ts_output_path

    _kpi_tags = ['KPI_results', _name_network, _model_tag, _ess_tag, _sim_tag]
    _root2kpi = '_'.join(filter(None, _kpi_tags))
    _kpi_output_path = os.path.join(_net_output_path, _root2kpi)
    _check_root(_kpi_output_path)
    _gpath_dict['kpi_out_root'] = _kpi_output_path
    return _gpath_dict


def set_general_params(_cfg_file, _path_dict, _ess_tag):
    """Set the general parameters of the simulation.
    :param _cfg_file: Yaml configuration file
    :param _path_dict: Dictionary containing all the paths.
    :param _ess_tag: Tag for the Storage capacity flexibility provider.
    :return: Dictionary containing all the general parameters."""
    _gparams_dict = dict()

    # Timestep
    time_step = _cfg_file['simulation_window']
    _gparams_dict['timestep'] = time_step
    _gparams_dict['ts_output_file_type'] = _cfg_file['network']['ts_output_file_type']

    # FSP File Definition
    _net_page = 'network'
    _fsp_page = 'fsp'

    _gparams_dict['fsp_file'] = dict()
    _fsp_file = _cfg_file[_net_page][_fsp_page]['fsp_input_filename']
    _fsp_file2read = os.path.join(_path_dict['fsp_in_root'], _fsp_file)
    _gparams_dict['fsp_file'] = _fsp_file2read

    _bidding_Strategy = _cfg_file[_net_page][_fsp_page]['fsp_bidding_strategy']
    _gparams_dict['bidding_strategy'] = _bidding_Strategy

    # Consider the sensitivity coefficient in the clearing of the market
    net_in_market = True
    _gparams_dict['net_in_market'] = net_in_market
    cap_curve_cons = True
    _gparams_dict['cap_curve_cons'] = cap_curve_cons
    soc_cons = False
    _gparams_dict['soc_cons'] = soc_cons

    # Forces the timeseries calculation even if the files already exist
    force_ts = _cfg_file['timeseries_force_1st_calculation']
    _gparams_dict['force_ts'] = force_ts

    # Forces the HVM matrices calculation even if the files already exist
    force_HVM = True
    _gparams_dict['force_HVM'] = force_HVM

    sens_factors_mode = _cfg_file['Sens_factors_mode']
    _gparams_dict['sens_factor_mode'] = sens_factors_mode

    # If ((nº hours x nº pilots) > Sens_factors_problem_size_threshold) use the (HVMP x HVMQ) matrix of the worst hour
    sens_factors_problem_size_threshold = _cfg_file['Sens_factors_problem_size_threshold']
    _gparams_dict['sens_factors_problem_size_threshold'] = sens_factors_problem_size_threshold

    # Tolerance on the delta voltage w.r.t. to limits defined by model constraints
    _gparams_dict['dv_tol'] = _cfg_file['tol_v']
    _gparams_dict['di_tol'] = _cfg_file['tol_i']

    # Time granularity interval, default value: 1 [hour]
    dt = 1
    _gparams_dict['dt'] = dt

    vc_delta_sens_coeff = .01
    _gparams_dict['vc_delta_sens_coeff'] = vc_delta_sens_coeff

    # Congestion Management
    # Forces the HCM matrixes calculation even if the files already exist
    force_HCM = True
    _gparams_dict['force_HCM'] = force_HCM

    # Check Minimum FSP MW value
    cm_deltaP_coeff, cm_deltaQ_coeff = _def_dp(_cfg_file)
    # cm_deltaP_coeff = 0.01
    _gparams_dict['cm_deltaP_coeff'] = cm_deltaP_coeff

    # Check Minimum FSP MVAr value
    # cm_deltaQ_coeff = 0.01
    _gparams_dict['cm_deltaQ_coeff'] = cm_deltaQ_coeff

    line_overload_limit = 100
    _gparams_dict['line_overload_limit'] = line_overload_limit
    trafo_overload_limit = 100
    _gparams_dict['trafo_overload_limit'] = trafo_overload_limit

    cm_over_procurement_coeff = 1
    _gparams_dict['cm_over_procurement_coeff'] = cm_over_procurement_coeff

    # Over procuring in terms of line rating: decreases the line (or trafo) rating from 100 to flex_limit value
    flex_limit_pre = 95
    _gparams_dict['flex_limit_pre'] = flex_limit_pre

    # Over procuring in terms of line rating: decreases the line (or trafo) rating from 100 to flex_limit value
    flex_limit_post = 100
    _gparams_dict['flex_limit_post'] = flex_limit_post

    # Cost of flexibility not supplied (VOLL)
    beta_cost = _cfg_file['BetaCost']
    _gparams_dict['beta_cost'] = beta_cost

    # Cost of the slack variable for the optimisation
    alpha_cost = _cfg_file['BetaCost']
    _gparams_dict['alpha_cost'] = alpha_cost

    _gparams_dict['only_pilot_busMV'] = _cfg_file['only_pilot_busMV']
    _gparams_dict['vn_kv_mv'] = _cfg_file['vn_kv_mv']

    # Storage size (MVA) parameters
    try:
        ess_k_mva = int(_ess_tag.split('SK')[-1])
    except ValueError:
        ess_k_mva = 0
    _gparams_dict['ess_k_MVA'] = ess_k_mva
    return _gparams_dict


def set_sim_path(_gpath, _model_tag, _product_tags, _vl_tags, _fsp_tags, _ess_tag, _sim_tag):
    """Set the simulation path of the project.
    :param _gpath: Dictionary containing all the general paths.
    :param _model_tag: Tag for the market model.
    :param _product_tags: Tags for the products of the market.
    :param _vl_tags: Tag for the voltage limitations of the market.
    :param _fsp_tags: Tag for the flexibility provided in the market.
    :param _ess_tag: Tag for the Storage capacity flexibility provider.
    :param _sim_tag: Tag for the specific simulation name.
    :return: Dictionary containing all the paths.
    """

    _input_path = _gpath['in_root']
    _name_network = _gpath['net_name']
    _net_output_path = _gpath['net_out_root']

    # Input market folder
    _gpath['mrk_inputs'] = dict()
    _root2mrkinputs = 'market_inputs'
    _mrk_input_path = os.path.join(_input_path, _name_network, _root2mrkinputs)
    _check_root(_mrk_input_path)

    # Output market folder
    _gpath['simulation_tag'] = dict()
    _gpath['mrk_outputs'] = dict()
    _gpath['mrk_outputs']['res_market'] = dict()
    _gpath['mrk_outputs']['ts_postmrk_out_root'] = dict()
    _root2postmrk_tsnet_results = 'FullTS_postMKT'

    _modeltag_input_path = os.path.join(_mrk_input_path, _model_tag)
    _check_root(_modeltag_input_path)
    for _prod in _product_tags:
        _producttag_input_path = os.path.join(_modeltag_input_path, _prod)
        _check_root(_producttag_input_path)
        _gpath['simulation_tag'][_prod] = dict()
        _gpath['mrk_inputs'][_prod] = dict()
        _gpath['mrk_outputs']['res_market'][_prod] = dict()
        _gpath['mrk_outputs']['ts_postmrk_out_root'][_prod] = dict()
        for _vl in _vl_tags:
            _vltag_input_path = os.path.join(_producttag_input_path, _vl)
            _check_root(_vltag_input_path)
            _gpath['simulation_tag'][_prod][_vl] = dict()
            _gpath['mrk_inputs'][_prod][_vl] = dict()
            _gpath['mrk_outputs']['res_market'][_prod][_vl] = dict()
            _gpath['mrk_outputs']['ts_postmrk_out_root'][_prod][_vl] = dict()
            for _fsp in _fsp_tags:
                _fsptag_input_path = os.path.join(_vltag_input_path, _fsp)
                _check_root(_fsptag_input_path)
                _gpath['mrk_inputs'][_prod][_vl][_fsp] = _fsptag_input_path
                # Simulation tag
                _list_tags = [_name_network, _model_tag, _prod, _vl, _fsp, _ess_tag, _sim_tag]
                _simulation_tag = '_'.join(filter(None, _list_tags))
                _gpath['simulation_tag'][_prod][_vl][_fsp] = _simulation_tag
                # Post Market output folders
                _postmrk_output_path = os.path.join(_net_output_path, _simulation_tag)
                _check_root(_postmrk_output_path)
                _gpath['mrk_outputs']['res_market'][_prod][_vl][_fsp] = _postmrk_output_path
                # Timeseries simulation post market output folders
                _ts_postmrk_output_path = os.path.join(_postmrk_output_path, _root2postmrk_tsnet_results)
                _check_root(_ts_postmrk_output_path)
                _gpath['mrk_outputs']['ts_postmrk_out_root'][_prod][_vl][_fsp] = _ts_postmrk_output_path
    return _gpath


def set_sim_params(_gparams, _vl_tags, _fsp_tags):
    """Set the simulation parameter of the simulation.
    :param _gparams: Yaml configuration file
    :param _vl_tags: Tag for the voltage limitations of the market.
    :param _fsp_tags: Tag for the flexibility provided in the market.
    :return: Dictionary containing all the general parameters."""
    # Forces saving the inverse Jacobian matrices (saving them is no necessary for market clearing)
    save_invJ = True
    _gparams['save_invJ'] = dict()
    _gparams['coeff_bid_Pup'] = dict()
    _gparams['coeff_bid_Pdown'] = dict()
    _gparams['coeff_bid_Qup'] = dict()
    _gparams['coeff_bid_Qdown'] = dict()
    for _fsp in _fsp_tags:
        if _fsp_tags == 'F01':
            warnings.warn('SaveInvJ = {_value}'.format(_value=save_invJ))
        else:
            save_invJ = False
        _gparams['save_invJ'][_fsp] = save_invJ

        # Bid coefficients
        try:
            coeff_bid = int(_fsp.split('F')[-1])
            coeff_bid_Pup = 1 * coeff_bid
            _gparams['coeff_bid_Pup'][_fsp] = coeff_bid_Pup
            coeff_bid_Pdown = 1 * coeff_bid
            _gparams['coeff_bid_Pdown'][_fsp] = coeff_bid_Pdown
            coeff_bid_Qup = 1 * coeff_bid
            _gparams['coeff_bid_Qup'][_fsp] = coeff_bid_Qup
            coeff_bid_Qdown = 1 * coeff_bid
            _gparams['coeff_bid_Qdown'][_fsp] = coeff_bid_Qdown
        except ValueError:
            raise UserWarning('The FSP Tag {_name} is wrongly declared.'.format(_name=_fsp))

    # deltaV = -0.01  # for test purposes (set the voltage limits to 0.96 & 1.04)
    _gparams['vbus_max'] = dict()
    _gparams['vbus_min'] = dict()
    deltaV = 0  # VL01
    for _vl in _vl_tags:
        if _vl == 'VL02':
            deltaV = 0.02
        elif _vl == 'VL03':
            deltaV = 0.05

        vbus_max = 1.05 + deltaV
        vbus_min = 0.95 - deltaV
        _gparams['vbus_max'][_vl] = vbus_max
        _gparams['vbus_min'][_vl] = vbus_min
    return _gparams
