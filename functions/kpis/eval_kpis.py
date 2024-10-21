import os
import math
import warnings
import copy as cp
import numpy as np
import pandas as pd
import pandapower as pp
from pathlib import Path
import matplotlib.pyplot as plt
from functions import auxiliary as aux
from functions import file_io as io_file
from pandapower.plotting import fuse_geodata
from functions.kpis import plot_kpis as plt_kpi
from pandapower.plotting import draw_collections
from pandapower.plotting.plotly import pf_res_plotly
from pandapower.plotting import create_bus_collection
from pandapower.plotting import create_line_collection
from pandapower.plotting import create_trafo_collection
from pandapower.plotting import create_generic_coordinates
warnings.filterwarnings('ignore')
plt.ion()


def _check_consis(_value1, _value2):
    """
    Check consistency between kpi inputs requested from the user and config file inputs.

    :param _value1: kpi inputs requested from the user.

    :param _value2: config file inputs.
    """
    _tmp = list()
    for _val in _value1:
        if _val in _value2:
            _tmp.append(_val)
        else:
            print('The value {_value} is not simulated. It will be omitted in the KPI results'.format(_value=_val))

    if len(_tmp) == 0:
        _tmp = _value2
        print('The KPI results will consider the following scenario: {_value}'.format(_value=_tmp))
    return _tmp


def _get_scenario_names(_name_net, _mrk_model, _mrk_products, _mrk_vl, _mrk_fsps, _mrk_ess, _sim_tag):
    """
    Extract the names of the scenarios that the user want to analyse. The names are extracted from the combination
    of network name, market model, market products, market voltage violation limits, market service providers,
    market power batteries and market simulation tag.

    :param _name_net: Network name (string)

    :param _mrk_model: Market model (string)

    :param _mrk_products: Market products (list of strings)

    :param _mrk_vl: Voltage Limits of the market (list of strings)

    :param _mrk_fsps: Product service availability (list of strings)

    :param _sim_tag: Simulation tag (string)

    :return _list_of_scenarios: List of possible scenarios that are going to be evaluated for the kpi.
    """
    _list_of_scenarios = list()
    for _prod in _mrk_products:
        for _vl in _mrk_vl:
            for _fsp in _mrk_fsps:
                _filename = '_'.join(filter(None, [_name_net, _mrk_model, _prod, _vl, _fsp, _mrk_ess, _sim_tag]))
                _list_of_scenarios.append(_filename)
    return _list_of_scenarios


def _set_kpi_params(_cfg_file, _paths, _fsp_file, _prods, _vls, _fsps):
    """
    Define parameters useful for the evaluation of the KPI.

    :param _cfg_file: Yaml configuration file.

    :param _paths: Dictionary with all the paths for each specific scenario.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _prods: List of products that the user want to analyse in the KPI evaluation.

    :param _vls: List of voltage limits that the user want to analyse in the KPI evaluation.

    :param _fsps: List of fsp availability that the user want to analyse in the KPI evaluation.
    """
    # KPIs configuration parameter
    _data_dict = dict()

    # Pandapower network
    net = io_file.import_network(_cfg_file['network'])

    # Network name under analysis
    name_net = _cfg_file['network']['net_folder']
    # Market Model under analysis
    mrk_model = _cfg_file['ModelTAG']
    # Products under analysis (Selected from the kpi settings dictionary)
    mrk_products = _check_consis(_prods, _cfg_file['ProductTAG'])
    # Voltage Limitations under analysis (Selected from the kpi settings dictionary)
    mrk_vl = _check_consis(_vls, _cfg_file['VLTAG'])
    # FSP under analysis (Selected from the kpi settings dictionary)
    mrk_fsps = _check_consis(_fsps, _cfg_file['FspTAG'])
    # Market ESS under analysis
    mrk_ess = _cfg_file['StorageTag']
    # Simulation tag
    sim_tag = _cfg_file['simulation_tag']

    # Scenarios that will be considered in the KPI results
    scenarios = _get_scenario_names(name_net, mrk_model, mrk_products, mrk_vl, mrk_fsps, mrk_ess, sim_tag)

    # Voltage Limitation Enlargement Gaps
    dV = {'VL01': 0, 'VL02': 0.02, 'VL03': 0.05}
    dV_color = {'VL01': 'red', 'VL02': 'blue', 'VL03': 'green'}
    # Maximum and Minimum Voltage Limitation of pre market
    vbus_min_pre = [0.95 - dV.get(_vl, 0) for _vl in mrk_vl]
    vbus_max_pre = [1.05 + dV.get(_vl, 0) for _vl in mrk_vl]
    # Maximum Current Limitation in percentage before the market
    i_max_perc_pre = 100
    # Tolerance for considering the voltage violations in the post market evaluation
    tol_v = _cfg_file['tol_v']
    # tolerance for considering the over currents in the post market evaluation
    tol_i = _cfg_file['tol_i']
    # Maximum Current Limitation in percentage after the market
    i_max_perc_post = 100 + tol_i
    # Maximum and Minimum Voltage Limitation of post market
    vbus_min_post = [_vl - tol_v for _vl in vbus_min_pre]
    vbus_max_post = [_vl + tol_v for _vl in vbus_max_pre]

    # Check if the FSP category exist
    for _fsp_type in _fsp_file.keys():
        if _fsp_file[_fsp_type] != 0:
            _data_dict['flat_' + _fsp_type] = True
        else:
            _data_dict['flat_' + _fsp_type] = False

    # Slack Variable Costs (Congestion Management - beta, and Voltage Control - alpha)
    _data_dict['cost_alpha'] = float(_cfg_file['BetaCost'])
    _data_dict['cost_beta'] = float(_cfg_file['BetaCost'])
    # Pandapower Network
    _data_dict['network'] = cp.deepcopy(net)
    # Scenarios under analysis
    _data_dict['scenarios_selected'] = scenarios
    # Voltage Limits of the market
    _data_dict['vl_lims'] = mrk_vl
    # Color for the Voltage Limits
    _data_dict['vl_colors'] = dV_color
    # Pre market
    _data_dict['vbus_max_pre'] = vbus_max_pre
    _data_dict['vbus_min_pre'] = vbus_min_pre
    _data_dict['i_max_perc_pre'] = i_max_perc_pre
    # Post market
    _data_dict['vbus_max_post'] = vbus_max_post
    _data_dict['vbus_min_post'] = vbus_min_post
    _data_dict['i_max_perc_post'] = i_max_perc_post

    # Paths for directories
    _data_dict['paths'] = _paths

    # Dots per Inch of KPI result figures
    dpi_img = 600
    _data_dict['dpi_img'] = dpi_img
    return _data_dict


def _load_res_net_pre(_in_dict, _scenarios):
    """
    Load data of the network before the market.

    :param _in_dict: Dictionary with all the information for evaluating the kpis.

    :param _scenarios: List of string of names of the scenarios that the user want to analyse.
    """
    _net = _in_dict['network']
    _kpi_path = _in_dict['paths']
    # _scenarios = _in_dict['scenarios_selected']
    _in_dict['lines_ldg_perc_pre'] = dict()
    _in_dict['trafos_ldg_perc_pre'] = dict()
    _in_dict['trafos3w_ldg_perc_pre'] = dict()
    _in_dict['bus_vmpu_pre'] = dict()

    for name_tab in _scenarios:
        _root2timeseries_res = _kpi_path['ts_out_root']

        # Lines Data
        path_file_line = os.path.join(_root2timeseries_res, 'res_line')
        loading_percent_line = io_file.import_data(_filename='loading_percent.csv', _root=path_file_line, _index_col=0)
        _in_dict['lines_ldg_perc_pre'][name_tab] = loading_percent_line.T

        # Trafos Data
        if not _net.trafo.empty:
            path_file_trafos = os.path.join(_root2timeseries_res, 'res_trafo')
            loading_percent_trafos = io_file.import_data(_filename='loading_percent.csv', _root=path_file_trafos,
                                                         _index_col=0)
            _in_dict['trafos_ldg_perc_pre'][name_tab] = loading_percent_trafos.T

        # Trafos 3Winding Data
        if not _net.trafo3w.empty:
            path_file_trafos3W = os.path.join(_root2timeseries_res, 'res_trafo3w')
            loading_percent_trafos3W = io_file.import_data(_filename='loading_percent.csv', _root=path_file_trafos3W,
                                                           _index_col=0)
            _in_dict['trafos3w_ldg_perc_pre'][name_tab] = loading_percent_trafos3W.T

        # Bus Data
        path_file_bus = os.path.join(_root2timeseries_res, 'res_bus')
        vm_pu_bus = io_file.import_data(_filename='vm_pu.csv', _root=path_file_bus, _index_col=0)
        _in_dict['bus_vmpu_pre'][name_tab] = vm_pu_bus.T
    return _in_dict


def _load_res_net_post(_in_dict, _scenarios):
    """
    Load data after the market.

    :param _in_dict: Dictionary with all the information for evaluating the kpis.

    :param _scenarios: List of string of names of the scenarios that the user want to analyse.
    """
    _net = _in_dict['network']
    _kpi_path = _in_dict['paths']
    # _scenarios = _in_dict['scenarios_selected']
    _in_dict['lines_ldg_perc_post'] = dict()
    _in_dict['trafos_ldg_perc_post'] = dict()
    _in_dict['trafos3w_ldg_perc_post'] = dict()
    _in_dict['bus_vmpu_post'] = dict()

    for name_tab in _scenarios:
        _prod = name_tab.split('_')[2]
        _vl = name_tab.split('_')[3]
        _fsp = name_tab.split('_')[4]
        _root2timeseries_postmrk_res = _kpi_path['mrk_outputs']['ts_postmrk_out_root'][_prod][_vl][_fsp]

        # Lines Data
        path_file_line = os.path.join(_root2timeseries_postmrk_res, 'res_line')
        loading_percent_line = io_file.import_data(_filename='loading_percent.csv', _root=path_file_line, _index_col=0)
        _in_dict['lines_ldg_perc_post'][name_tab] = loading_percent_line.T

        # Trafos Data
        if not _net.trafo.empty:
            path_file_trafos = os.path.join(_root2timeseries_postmrk_res, 'res_trafo')
            loading_percent_trafos = io_file.import_data(_filename='loading_percent.csv', _root=path_file_trafos,
                                                         _index_col=0)
            _in_dict['trafos_ldg_perc_post'][name_tab] = loading_percent_trafos.T

        if not _net.trafo3w.empty:
            path_file_trafos3w = os.path.join(_root2timeseries_postmrk_res, 'res_trafo3w')
            loading_percent_trafos3w = io_file.import_data(_filename='loading_percent.csv', _root=path_file_trafos3w,
                                                           _index_col=0)
            _in_dict['trafos3w_ldg_perc_post'][name_tab] = loading_percent_trafos3w.T

        # Bus Data
        path_file_bus = os.path.join(_root2timeseries_postmrk_res, 'res_bus')
        vm_pu_bus = io_file.import_data(_filename='vm_pu.csv', _root=path_file_bus, _index_col=0)
        _in_dict['bus_vmpu_post'][name_tab] = vm_pu_bus.T
    return _in_dict


def max_mat_df(_input_df):
    """
    Extract the maximum value, the row (node) and the column (hour) in the dataframe passed as input.

    :param _input_df: Dataframe with nodes as rows and hours as columns.
    """
    df = _input_df.fillna(value=0)
    if not pd.DataFrame(df).empty:
        max_val = df.values.max()
        # Coordinates of all maximum values
        max_coords = list(zip(*np.where(df.values == max_val)))
        # Row and column names corresponding to each coordinates
        max_names = [(df.index[r], df.columns[c]) for r, c in max_coords]
        max_val = max_val
        node = max_names[0][0]
        hour = max_names[0][1]
    else:
        max_val = 'Empty'
        node = 'Empty'
        hour = 'Empty'
    return float(max_val), int(node), int(hour)


def eval_hosting_capacity(_net, _data):
    """
    Evaluate the hosting capacity increase in the network.

    :param _net: Pandapower network

    :param _data: Dictionary with all the information for evaluating the hosting capacity.
    """
    # Data inputs
    vpu_table = _data['vpu_table']
    ip_line_table = _data['Ipline_table']
    ip_trafo_table = _data['Iptrafo_table']
    ip_trafo3w_table = _data['Iptrafo3w_table']
    hc_info = _data['HC_info']
    vmax = _data['vbus_max_V']
    imax = _data['i_max_perc']

    # Hosting Capacity from Voltage in buses
    if not pd.DataFrame(hc_info).empty:
        dvdp = float(hc_info.HV_m)
        v_base = _net.bus.iloc[hc_info.index[0]].vn_kv
        v_node = float(hc_info.maxV_val)
        hc_v = (vmax - v_node) * v_base / dvdp  # MVA
    else:
        hc_v = np.inf
        v_node = np.inf

    # Hosting Capacity from lines
    if not pd.DataFrame(ip_line_table).empty:
        line_loading_perc = ip_line_table.max().max()
        max_value_columns = ip_line_table.max()
        hours_max_loading_line = max_value_columns.idxmax()
        index_row_max = ip_line_table.idxmax(axis=0)
        lines = index_row_max[hours_max_loading_line]

        max_i_ka = _net.line.max_i_ka[int(lines)]
        df = _net.line.df[int(lines)]
        parallel = _net.line.parallel[int(lines)]
        vn_line = _net.bus.iloc[_net.line.from_bus[int(lines)]].vn_kv
        voltage_line = vn_line * vpu_table[int(hours_max_loading_line)][_net.line.from_bus[int(lines)]]
        hc_i = math.sqrt(3) * (((imax - line_loading_perc) / 100) * max_i_ka * df * parallel * voltage_line)
    else:
        hc_i = np.inf
        lines = np.inf
        hours_max_loading_line = np.inf
        line_loading_perc = np.inf

    # Hosting Capacity from Transformers 2 windings
    if not pd.DataFrame(ip_trafo_table).empty:
        trafo_loading_perc, trafos, hours_max_loading_trafo = max_mat_df(ip_trafo_table)
        hc_trafo = ((imax - trafo_loading_perc) / 100) * _net.trafo.iloc[int(trafos)].sn_mva
    else:
        hc_trafo = np.inf
        trafos = np.inf
        hours_max_loading_trafo = np.inf
        trafo_loading_perc = np.inf

    # Hosting Capacity from Transformers 3 windings
    if not pd.DataFrame(ip_trafo3w_table).empty:
        trafo3w_loading_perc, trafos3w, hours_max_loading_trafo3w = max_mat_df(ip_trafo3w_table)
        hc_trafo3w = ((imax - trafo3w_loading_perc) / 100) * _net.trafo3w.iloc[int(trafos3w)].sn_hv_mva
    else:
        hc_trafo3w = np.inf
        trafos3w = np.inf
        hours_max_loading_trafo3w = np.inf
        trafo3w_loading_perc = np.inf

    tmp_hc = [hc_v, hc_i, hc_trafo, hc_trafo3w]
    comp = [v_node, lines, trafos, trafos3w]
    type_column = ['Node', 'Line', 'Trafo', 'Trafo3W']
    ind = tmp_hc.index(min(tmp_hc))
    hosting_cap = min(tmp_hc)

    # Results
    hc_res = dict()
    hc_res['Hosting Capacity [MVA]'] = round(hosting_cap, 3)
    hc_res['Type component'] = type_column[ind]
    hc_res['ID component'] = comp[ind]

    if not pd.DataFrame(hc_info).empty:
        hc_res['VC HC [MVA]'] = round(hc_v, 3)
        hc_res['VC Vmax [pu]'] = round(v_node, 3)
        hc_res['VC Node'] = hc_info.index[0]
        hc_res['VC hour'] = int(hc_info.maxV_hour)

    if not pd.DataFrame(ip_line_table).empty:
        hc_res['CM HC_L [MVA]'] = round(hc_i, 3)
        hc_res['CM perc_Load_L [%]'] = round(line_loading_perc, 3)
        hc_res['CM ID line'] = int(lines)
        hc_res['CM line_hour'] = hours_max_loading_line

    if not pd.DataFrame(ip_trafo_table).empty:
        hc_res['CM HC_T [MVA]'] = round(hc_trafo, 3)
        hc_res['CM perc_Load_T [%]'] = round(trafo_loading_perc, 3)
        hc_res['CM ID trafo'] = int(trafos)
        hc_res['CM trafo_hour'] = hours_max_loading_trafo

    if not pd.DataFrame(ip_trafo3w_table).empty:
        hc_res['CM HC_T3W [MVA]'] = round(hc_trafo3w, 3)
        hc_res['CM perc_Load_T3W [%]'] = round(trafo3w_loading_perc, 3)
        hc_res['CM ID trafo3W'] = int(trafos3w)
        hc_res['CM trafo3W_hour'] = hours_max_loading_trafo3w
    return hc_res


def increase_hc(_in_dict):
    """
    Evaluate the increase in the hosting capacity for each scenario.
    It requires the function eval_hosting_capacity(_net, _data), to calculate the increase in Hosting Capacity
    for an entire study. Increase in Hosting Capacity is calculated as the difference between post-market
    and pre-market.

    :param _in_dict: Dictionary with all the information for evaluating the kpis.
    """
    net = _in_dict['network']
    v_limits = _in_dict['vl_lims']
    scenarios = _in_dict['scenarios_selected']
    # Pre-market data
    lines_ldg_perc_pre = _in_dict['lines_ldg_perc_pre']
    trafos_ldg_perc_pre = _in_dict['trafos_ldg_perc_pre']
    trafos3w_ldg_perc_pre = _in_dict['trafos3w_ldg_perc_pre']
    bus_vmpu_pre = _in_dict['bus_vmpu_pre']
    hc_pre = _in_dict['hc_pre']
    vbus_max_pre = _in_dict['vbus_max_pre']
    i_max_perc_pre = _in_dict['i_max_perc_pre']
    # Post-market data
    lines_ldg_perc_post = _in_dict['lines_ldg_perc_post']
    trafos_ldg_perc_post = _in_dict['trafos_ldg_perc_post']
    trafos3w_ldg_perc_post = _in_dict['trafos3w_ldg_perc_post']
    bus_vmpu_post = _in_dict['bus_vmpu_post']
    hc_post = _in_dict['hc_post']
    vbus_max_post = _in_dict['vbus_max_post']
    i_max_perc_post = _in_dict['i_max_perc_post']

    v_lims = {market: idx for idx, market in enumerate(v_limits)}
    hc_res_pre = dict()
    hc_res_post = dict()
    ihc_res = dict()
    pihc_res = dict()
    null_var = dict()

    for name_tab in scenarios:
        _vl_type = name_tab.split('_')[3]

        _data_pre = dict()
        _data_post = dict()

        # Before Market
        _data_pre['vpu_table'] = bus_vmpu_pre[name_tab]
        _data_pre['Ipline_table'] = lines_ldg_perc_pre[name_tab]
        _data_pre['Iptrafo_table'] = trafos_ldg_perc_pre[name_tab] if trafos_ldg_perc_pre else null_var
        _data_pre['Iptrafo3w_table'] = trafos3w_ldg_perc_pre[name_tab] if trafos3w_ldg_perc_pre else null_var
        _data_pre['HC_info'] = hc_pre[name_tab] if not pd.DataFrame(hc_pre).empty else hc_pre
        _data_pre['vbus_max_V'] = vbus_max_pre[v_lims[_vl_type] if _vl_type in v_lims else None]
        _data_pre['i_max_perc'] = i_max_perc_pre

        hc_res_pre[name_tab] = eval_hosting_capacity(_net=net, _data=_data_pre)

        # After Market
        _data_post['vpu_table'] = bus_vmpu_post[name_tab]
        _data_post['Ipline_table'] = lines_ldg_perc_post[name_tab]
        _data_post['Iptrafo_table'] = trafos_ldg_perc_post[name_tab] if trafos_ldg_perc_post else null_var
        _data_post['Iptrafo3w_table'] = trafos3w_ldg_perc_post[name_tab] if trafos3w_ldg_perc_post else null_var
        _data_post['HC_info'] = hc_post[name_tab] if not pd.DataFrame(hc_post).empty else hc_post
        _data_post['vbus_max_V'] = vbus_max_post[v_lims[_vl_type] if _vl_type in v_lims else None]
        _data_post['i_max_perc'] = i_max_perc_post

        hc_res_post[name_tab] = eval_hosting_capacity(_net=net, _data=_data_post)

    ihc_res['Increase Hosting Capacity [MVA]'] = pd.DataFrame(hc_res_post).T['Hosting Capacity [MVA]'] - \
                                                 pd.DataFrame(hc_res_pre).T['Hosting Capacity [MVA]']
    pihc_res['Percentage of IHC [%]'] = (ihc_res['Increase Hosting Capacity [MVA]'] / pd.DataFrame(hc_res_pre).T[
        'Hosting Capacity [MVA]'].abs()) * 100

    ihc_res = pd.DataFrame(ihc_res)
    pihc_res = pd.DataFrame(pihc_res)
    hc_res_post = pd.DataFrame(hc_res_post)
    hc_res_pre = pd.DataFrame(hc_res_pre)

    return hc_res_pre, hc_res_post, ihc_res, pihc_res


def eval_avoided_congestion(_data_pre, _data_post, _round_val=3):
    """
    Evaluate the avoided congestion in the network between the pre- and post-market.

    :param _data_pre: Dictionary with all the information pre-market for evaluating the avoided congestions.

    :param _data_post: Dictionary with all the information post-market for evaluating the avoided congestions.

    :param _round_val: Value to which all value are rounded.
    """
    ip_line_table_pre = _data_pre['Ipline_table']
    ip_trafo_table_pre = _data_pre['Iptrafo_table']
    ip_trafo3w_table_pre = _data_pre['Iptrafo3w_table']
    imax_pre = _data_pre['i_max_perc']

    ip_line_table_post = _data_post['Ipline_table']
    ip_trafo_table_post = _data_post['Iptrafo_table']
    ip_trafo3w_table_post = _data_post['Iptrafo3w_table']
    imax_post = _data_post['i_max_perc']

    # Congestion Management lines
    n_pre_lines = np.count_nonzero(ip_line_table_pre > imax_pre)
    n_post_lines = np.count_nonzero(ip_line_table_post > imax_post)

    # Congestion Management transformers
    n_pre_trafo = np.count_nonzero(ip_trafo_table_pre > imax_pre) if not pd.DataFrame(ip_trafo_table_pre).empty else 0
    n_post_trafo = np.count_nonzero(ip_trafo_table_post > imax_post) if not pd.DataFrame(
        ip_trafo_table_post).empty else 0

    # Congestion Management trafos3W
    n_pre_trafo3w = np.count_nonzero(ip_trafo3w_table_pre > imax_pre) if not pd.DataFrame(
        ip_trafo3w_table_pre).empty else 0
    n_post_trafo3w = np.count_nonzero(ip_trafo3w_table_post > imax_post) if not pd.DataFrame(
        ip_trafo3w_table_post).empty else 0
    n_pre_total = n_pre_lines + n_pre_trafo + n_pre_trafo3w
    n_post_total = n_post_lines + n_post_trafo + n_post_trafo3w
    try:
        avoided_congestion = ((n_pre_total - n_post_total) / n_pre_total) * 100
    except ZeroDivisionError:
        avoided_congestion = 0

    # Results
    avoided_congestion_res = dict()
    avoided_congestion_res['AvoidCongestions [%]'] = round(avoided_congestion, _round_val)
    avoided_congestion_res['n_pre_Total [u]'] = round(n_pre_total, _round_val)
    avoided_congestion_res['n_post_Total [u]'] = round(n_post_total, _round_val)

    avoided_congestion_res['n_pre_L [u]'] = round(n_pre_lines, _round_val)
    avoided_congestion_res['n_post_L [u]'] = round(n_post_lines, _round_val)
    avoided_congestion_res['AvoidCong_L[u]'] = round(n_pre_lines, _round_val) - round(n_post_lines, _round_val)

    avoided_congestion_res['n_pre_T [u]'] = round(n_pre_trafo, _round_val)
    avoided_congestion_res['n_post_T [u]'] = round(n_post_trafo, _round_val)
    avoided_congestion_res['AvoidCong_T[u]'] = round(n_pre_trafo, _round_val) - round(n_post_trafo, _round_val)

    avoided_congestion_res['n_pre_T3w [u]'] = round(n_pre_trafo3w, _round_val)
    avoided_congestion_res['n_post_T3w [u]'] = round(n_post_trafo3w, _round_val)
    avoided_congestion_res['AvoidCong_T3w[u]'] = round(n_pre_trafo3w, _round_val) - round(n_post_trafo3w, _round_val)

    return avoided_congestion_res


def avoided_cong(_in_dict, _scenarios):
    """
    Evaluate the Avoided Congestion Problems - Lines and Transformers (2 windings and 3 windings).

    :param _in_dict: Dictionary with all the information for evaluating the avoided congestions.

    :param _scenarios: List of string of names of the scenarios that the user want to analyse.
    """
    # _scenarios = _in_dict['scenarios_selected']

    lines_ldg_perc_pre = _in_dict['lines_ldg_perc_pre']
    trafos_ldg_perc_pre = _in_dict['trafos_ldg_perc_pre']
    trafos3w_ldg_perc_pre = _in_dict['trafos3w_ldg_perc_pre']
    i_max_perc_pre = _in_dict['i_max_perc_pre']

    lines_ldg_perc_post = _in_dict['lines_ldg_perc_post']
    trafos_ldg_perc_post = _in_dict['trafos_ldg_perc_post']
    trafos3w_ldg_perc_post = _in_dict['trafos3w_ldg_perc_post']
    i_max_perc_post = _in_dict['i_max_perc_post']

    avoided_cong_res = dict()
    null_var = dict()

    for name_tab in _scenarios:
        _data_pre = dict()
        _data_post = dict()

        # Before Market
        _data_pre['Ipline_table'] = lines_ldg_perc_pre[name_tab]
        _data_pre['Iptrafo_table'] = trafos_ldg_perc_pre[name_tab] if trafos_ldg_perc_pre else null_var
        _data_pre['Iptrafo3w_table'] = trafos3w_ldg_perc_pre[name_tab] if trafos3w_ldg_perc_pre else null_var
        _data_pre['i_max_perc'] = i_max_perc_pre

        # After Market
        _data_post['Ipline_table'] = lines_ldg_perc_post[name_tab]
        _data_post['Iptrafo_table'] = trafos_ldg_perc_post[name_tab] if trafos_ldg_perc_post else null_var
        _data_post['Iptrafo3w_table'] = trafos3w_ldg_perc_post[name_tab] if trafos3w_ldg_perc_post else null_var
        _data_post['i_max_perc'] = i_max_perc_post

        avoided_cong_res[name_tab] = eval_avoided_congestion(_data_pre=_data_pre, _data_post=_data_post, _round_val=3)

    avoided_cong_res = pd.DataFrame(avoided_cong_res).T
    return avoided_cong_res


def eval_avoided_voltage_violation(_data_pre, _data_post, _round_val=3):
    """
    Evaluate the avoided congestion in the network between the pre- and post-market.

    :param _data_pre: Dictionary with all the information pre-market for evaluating the avoided voltage violations.

    :param _data_post: Dictionary with all the information post-market for evaluating the avoided voltage violations.

    :param _round_val: Value to which all value are rounded.
    """
    # Data inputs
    vpu_table_pre = _data_pre['vpu_table']
    vmax_pre = _data_pre['vbus_max']
    vmin_pre = _data_pre['vbus_min']

    vpu_table_post = _data_post['vpu_table']
    vmax_post = _data_post['vbus_max']
    vmin_post = _data_post['vbus_min']

    # Voltage Violations from nodes
    # Pre-Market
    n_pre_under = np.count_nonzero((vpu_table_pre < vmin_pre) & (vpu_table_pre > 0.1))
    n_pre_over = np.count_nonzero(vpu_table_pre > vmax_pre)
    n_pre = n_pre_under + n_pre_over

    # Post-Market
    n_post_under = np.count_nonzero((vpu_table_post < vmin_post) & (vpu_table_post > 0.1))
    n_post_over = np.count_nonzero(vpu_table_post > vmax_post)
    n_post = n_post_under + n_post_over

    try:
        avoided_voltage_violation = ((n_pre - n_post) / n_pre) * 100
    except ZeroDivisionError:
        avoided_voltage_violation = 0

    # Results
    avoided_voltage_violation_res = dict()
    avoided_voltage_violation_res['AvoidVoltageViolations [%]'] = round(avoided_voltage_violation, _round_val)
    avoided_voltage_violation_res['AvoidVoltageViolations [u]'] = round(n_pre, _round_val) - round(n_post, _round_val)

    avoided_voltage_violation_res['n_pre_B [u]'] = round(n_pre, _round_val)
    avoided_voltage_violation_res['n_post_B [u]'] = round(n_post, _round_val)

    avoided_voltage_violation_res['n_pre_B_Under [u]'] = round(n_pre_under, _round_val)
    avoided_voltage_violation_res['n_post_B_Under [u]'] = round(n_post_under, _round_val)

    avoided_voltage_violation_res['n_pre_B_Over [u]'] = round(n_pre_over, _round_val)
    avoided_voltage_violation_res['n_post_B_Over [u]'] = round(n_post_over, _round_val)

    return avoided_voltage_violation_res


def avoided_voltage_violations(_in_dict):
    """
    Evaluate the Avoided Voltage Violations - Buses.

    :param _in_dict: Dictionary with all the information for evaluating the avoided voltage violations.
    """
    v_limits = _in_dict['vl_lims']
    scenarios = _in_dict['scenarios_selected']

    bus_vmpu_pre = _in_dict['bus_vmpu_pre']
    vbus_max_v_pre = _in_dict['vbus_max_pre']
    vbus_min_v_pre = _in_dict['vbus_min_pre']

    bus_vmpu_post = _in_dict['bus_vmpu_post']
    vbus_max_v_post = _in_dict['vbus_max_post']
    vbus_min_v_post = _in_dict['vbus_min_post']

    avoided_voltage_viol_res = dict()
    v_lims = {market: idx for idx, market in enumerate(v_limits)}

    for name_tab in scenarios:
        _vl_type = name_tab.split('_')[3]
        _data_pre = dict()
        _data_post = dict()

        # Before Market
        _data_pre['vpu_table'] = bus_vmpu_pre[name_tab]
        _data_pre['vbus_max'] = vbus_max_v_pre[v_lims[_vl_type] if _vl_type in v_lims else None]
        _data_pre['vbus_min'] = vbus_min_v_pre[v_lims[_vl_type] if _vl_type in v_lims else None]

        # After Market
        _data_post['vpu_table'] = bus_vmpu_post[name_tab]
        _data_post['vbus_max'] = vbus_max_v_post[v_lims[_vl_type] if _vl_type in v_lims else None]
        _data_post['vbus_min'] = vbus_min_v_post[v_lims[_vl_type] if _vl_type in v_lims else None]

        avoided_voltage_viol_res[name_tab] = eval_avoided_voltage_violation(_data_pre=_data_pre, _data_post=_data_post,
                                                                            _round_val=3)

    avoided_voltage_viol_res = pd.DataFrame(avoided_voltage_viol_res).T
    return avoided_voltage_viol_res


def load_fsp_result(_in_dict, _scenario, _prefix='model_results'):
    """
    Upload the flexibility market results from folders.

    :param _in_dict: Dictionary with all the paths for importing the flexibility market results.

    :param _scenario: String representing the specific scenario to read.

    :param _prefix: Prefix of the excel file with the flexibility market results.
    """
    path_market_out = _in_dict['path_MkOut']

    root2file = os.path.join(path_market_out, _prefix + '_' + _scenario + '.xlsx')
    xl_file = pd.ExcelFile(root2file)
    list_sheets = xl_file.sheet_names
    dict_res = {_k: dict() for _k in _in_dict['fsp_file'].keys()}

    for _fsp_type in _in_dict['fsp_file'].keys():
        if _in_dict['flat_' + _fsp_type]:
            # tabs = ['DW_' + _fsp_type + '_U', 'DW_' + _fsp_type + '_D', 'DR_' + _fsp_type + '_U', 'DR_' + _fsp_type + '_D']
            tabs = ['DW_' + _fsp_type + '_U', 'DW_' + _fsp_type + '_D', 'DW_' + _fsp_type, 'W_' + _fsp_type,
                    'DR_' + _fsp_type + '_U', 'DR_' + _fsp_type + '_D', 'DR_' + _fsp_type, 'R_' + _fsp_type]
            sheet2load = list()
            for j in tabs:
                sheet2load.extend(i for i in list_sheets if i.startswith(j))
            dict_res[_fsp_type] = io_file.import_data(_filename=_prefix + '_' + _scenario + '.xlsx',
                                                      _root=path_market_out, _sheetname=sheet2load, _index_col=0)

    # Load Flexibility not supplied,
    tabs2read = ['ABS_alpha_DVpu', 'ABS_beta', 'ObjValue']
    dict_others = io_file.import_data(_filename=_prefix + '_' + _scenario + '.xlsx', _root=path_market_out,
                                      _sheetname=tabs2read, _index_col=0)

    return dict_res, dict_others


def load_data_costs(_in_dict, _scenario, _prefix='_init'):
    """
    Upload the flexibility market costs from folders.

    :param _in_dict: Dictionary with all the paths for importing the flexibility market results.

    :param _scenario: String representing the specific scenario to read.

    :param _prefix: Prefix of the Excel file with the flexibility market results.
    """
    path_market_in = _in_dict['path_MkIn']

    dict_res = {_k: dict() for _k in _in_dict['fsp_file'].keys()}

    for _fsp_type in _in_dict['fsp_file'].keys():
        if _in_dict['flat_' + _fsp_type]:
            file2read = 'FSP' + _fsp_type + _prefix + _scenario + '.xlsx'
            root2file = os.path.join(path_market_in, file2read)
            xl_file = pd.ExcelFile(root2file)
            list_sheets = xl_file.sheet_names
            # tabs = ['W_U_', 'W_D_', 'R_U_', 'R_D_']
            tabs = ['FSPinfo',
                    'Winit', 'WLB_ULim', 'WLB_DLim', 'WUB_ULim', 'WUB_DLim', 'W_U_', 'W_D_',
                    'Rinit', 'RLB_ULim', 'RLB_DLim', 'RUB_ULim', 'RUB_DLim', 'R_U_', 'R_D_'
                    ]
            sheet2load = list()
            for j in tabs:
                sheet2load.extend(i for i in list_sheets if i.startswith(j))
            dict_res[_fsp_type] = io_file.import_data(_filename=file2read, _root=path_market_in, _sheetname=sheet2load,
                                                      _index_col=0)

    return dict_res


def eval_flexibility_costs(_in_dict, _dict_res, _dict_others, _dict_costs, _round_val=3):
    """
    Evaluate the costs for the request of flexibility.

    :param _in_dict: Dictionary with all the information pre-market for evaluating the avoided voltage violations.

    :param _dict_res: Dictionary with all the information about the flexibility provided by each resource.

    :param _dict_others: Dictionary with all the information about the general flexibility market parameter.

    :param _dict_costs: Dictionary with all the information about the costs for flexibility by each resource.

    :param _round_val: Value to which all value are rounded.
    """
    df_zeros_22 = pd.DataFrame(0, index=range(2), columns=range(2))
    fsp_costs = dict()
    ac_fm_w = dict()
    ac_fm_r = dict()
    for _fsp_type in _in_dict['fsp_file'].keys():
        if _in_dict['flat_' + _fsp_type]:
            ###############################
            # Upward Active Power
            _key2use_p_up = aux.get_list(_startswith_key='DW_' + _fsp_type + '_U', _comparing_list=_dict_res[_fsp_type])
            if len(_key2use_p_up) == 1:
                # Load [Type A] and Generator [Type B]
                fsp_costs['DW_' + _fsp_type + '_U_total'] = np.sum(
                    np.sum(_dict_res[_fsp_type][_key2use_p_up[0]], axis=0))
            elif len(_key2use_p_up) == 2:
                # Storage [Type C]
                fsp_costs['DW_' + _fsp_type + '_U_total'] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_up[1]], fill_value=0),
                    axis=0))

            # Downward Active Power
            _key2use_p_dwn = aux.get_list(_startswith_key='DW_' + _fsp_type + '_D', _comparing_list=_dict_res[_fsp_type])
            if len(_key2use_p_dwn) == 1:
                fsp_costs['DW_' + _fsp_type + '_D_total'] = np.sum(
                    np.sum(_dict_res[_fsp_type][_key2use_p_dwn[0]], axis=0))
            elif len(_key2use_p_dwn) == 2:
                # Storage [Type C]
                fsp_costs['DW_' + _fsp_type + '_D_total'] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_p_dwn[0]].add(_dict_res[_fsp_type][_key2use_p_dwn[1]], fill_value=0),
                    axis=0))

            if len(_key2use_p_up) == 1 and len(_key2use_p_dwn) == 1:
                ac_fm_w[_fsp_type] = _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_dwn[0]],
                                                                                fill_value=0)
            elif len(_key2use_p_up) == 2 and len(_key2use_p_dwn) == 1:
                ac_fm_w[_fsp_type] = _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_up[1]],
                                                                                fill_value=0).add(
                    _dict_res[_fsp_type][_key2use_p_dwn[0]], fill_value=0)
            elif len(_key2use_p_up) == 1 and len(_key2use_p_dwn) == 2:
                ac_fm_w[_fsp_type] = _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_dwn[0]],
                                                                                fill_value=0).add(
                    _dict_res[_fsp_type][_key2use_p_dwn[1]], fill_value=0)
            else:
                ac_fm_w[_fsp_type] = _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_up[1]],
                                                                                fill_value=0).add(
                    _dict_res[_fsp_type][_key2use_p_dwn[0]], fill_value=0).add(_dict_res[_fsp_type][_key2use_p_dwn[1]],
                                                                               fill_value=0)

            # Active Power
            if len(_key2use_p_up) == 1 and len(_key2use_p_dwn) == 1:
                fsp_costs['W_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_dwn[0]], fill_value=0),
                    axis=0))
            elif len(_key2use_p_up) == 2 and len(_key2use_p_dwn) == 1:
                fsp_costs['W_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_up[1]],
                                                               fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_p_dwn[0]], fill_value=0), axis=0))
            elif len(_key2use_p_up) == 1 and len(_key2use_p_dwn) == 2:
                fsp_costs['W_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_p_up[0]].add(_dict_res[_fsp_type][_key2use_p_dwn[0]],
                                                               fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_p_dwn[1]], fill_value=0), axis=0))
            else:
                fsp_costs['W_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_p_up[0]].add(
                        _dict_res[_fsp_type][_key2use_p_up[1]], fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_p_dwn[0]], fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_p_dwn[1]], fill_value=0),
                    axis=0))

            ###############################
            # Upward Reactive Power
            _key2use_q_up = aux.get_list(_startswith_key='DR_' + _fsp_type + '_U', _comparing_list=_dict_res[_fsp_type])
            if len(_key2use_q_up) == 1:
                # Load [Type A], Generator [Type B] and Storage [Type C]
                fsp_costs['DR_' + _fsp_type + '_U_total'] = np.sum(
                    np.sum(_dict_res[_fsp_type][_key2use_q_up[0]], axis=0))
            elif len(_key2use_q_up) == 2:
                # Future development for Storage (Reactive power upward and downward)
                fsp_costs['DR_' + _fsp_type + '_U_total'] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_up[1]], fill_value=0),
                    axis=0))

            # Downward Reactive Power
            _key2use_q_dwn = aux.get_list(_startswith_key='DR_' + _fsp_type + '_D', _comparing_list=_dict_res[_fsp_type])
            if len(_key2use_q_dwn) == 1:
                # Load [Type A], Generator [Type B] and Storage [Type C]
                fsp_costs['DR_' + _fsp_type + '_D_total'] = np.sum(
                    np.sum(_dict_res[_fsp_type][_key2use_q_dwn[0]], axis=0))
            elif len(_key2use_q_dwn) == 2:
                # Future development for Storage (Reactive power upward and downward)
                fsp_costs['DR_' + _fsp_type + '_D_total'] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_q_dwn[0]].add(_dict_res[_fsp_type][_key2use_q_dwn[1]], fill_value=0),
                    axis=0))

            if len(_key2use_q_up) == 1 and len(_key2use_q_dwn) == 1:
                ac_fm_r[_fsp_type] = _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_dwn[0]],
                                                                                fill_value=0)
            elif len(_key2use_q_up) == 2 and len(_key2use_q_dwn) == 1:
                ac_fm_r[_fsp_type] = _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_up[1]],
                                                                                fill_value=0).add(
                    _dict_res[_fsp_type][_key2use_q_dwn[0]], fill_value=0)
            elif len(_key2use_q_up) == 1 and len(_key2use_q_dwn) == 2:
                ac_fm_r[_fsp_type] = _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_dwn[0]],
                                                                                fill_value=0).add(
                    _dict_res[_fsp_type][_key2use_q_dwn[1]], fill_value=0)
            else:
                ac_fm_r[_fsp_type] = _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_up[1]],
                                                                                fill_value=0).add(
                    _dict_res[_fsp_type][_key2use_q_dwn[0]], fill_value=0).add(_dict_res[_fsp_type][_key2use_q_dwn[1]],
                                                                               fill_value=0)

            # Reactive Power
            if len(_key2use_q_up) == 1 and len(_key2use_q_dwn) == 1:
                # Load [Type A], Generator [Type B] and Storage [Type C]
                fsp_costs['R_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_dwn[0]], fill_value=0),
                    axis=0))
            elif len(_key2use_q_up) == 2 and len(_key2use_q_dwn) == 1:
                fsp_costs['R_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_up[1]],
                                                               fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_q_dwn[0]], fill_value=0), axis=0))
            elif len(_key2use_q_up) == 1 and len(_key2use_q_dwn) == 2:
                fsp_costs['R_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_q_up[0]].add(_dict_res[_fsp_type][_key2use_q_dwn[0]],
                                                               fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_q_dwn[1]], fill_value=0), axis=0))
            else:
                # Future development for Storage (Reactive power upward and downward)
                fsp_costs['R_' + _fsp_type] = np.sum(np.sum(
                    _dict_res[_fsp_type][_key2use_q_up[0]].add(
                        _dict_res[_fsp_type][_key2use_q_up[1]], fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_q_dwn[0]], fill_value=0).add(
                        _dict_res[_fsp_type][_key2use_q_dwn[1]], fill_value=0),
                    axis=0))

            ###############################
            # Active power Upward costs
            _key2use_p_up_cost = aux.get_list(_startswith_key='W_U_', _comparing_list=_dict_costs[_fsp_type])
            if len(_key2use_p_up_cost) == 1:
                res_dw_u = (_dict_res[_fsp_type][_key2use_p_up[0]] * _dict_costs[_fsp_type][
                    _key2use_p_up_cost[0]]).dropna(axis=1)
                fsp_costs['DW_' + _fsp_type + '_U_Cost'] = np.sum(np.sum(res_dw_u, axis=0))
            elif len(_key2use_p_up_cost) == 2:
                res_dw_u_1 = (_dict_res[_fsp_type][_key2use_p_up[0]] * _dict_costs[_fsp_type][
                    _key2use_p_up_cost[0]]).dropna(axis=1)
                res_dw_u_2 = (_dict_res[_fsp_type][_key2use_p_up[1]] * _dict_costs[_fsp_type][
                    _key2use_p_up_cost[1]]).dropna(axis=1)
                fsp_costs['DW_' + _fsp_type + '_U_Cost'] = np.sum(
                    np.sum(res_dw_u_1.add(res_dw_u_2, fill_value=0), axis=0))

            # Active power Downward costs
            _key2use_p_dwn_cost = aux.get_list(_startswith_key='W_D_', _comparing_list=_dict_costs[_fsp_type])
            if len(_key2use_p_dwn_cost) == 1:
                res_dw_d = (_dict_res[_fsp_type][_key2use_p_dwn[0]] * _dict_costs[_fsp_type][
                    _key2use_p_dwn_cost[0]]).dropna(axis=1)
                fsp_costs['DW_' + _fsp_type + '_D_Cost'] = np.sum(np.sum(res_dw_d, axis=0))
            elif len(_key2use_p_dwn_cost) == 2:
                res_dw_d_1 = (_dict_res[_fsp_type][_key2use_p_dwn[0]] * _dict_costs[_fsp_type][
                    _key2use_p_dwn_cost[0]]).dropna(axis=1)
                res_dw_d_2 = (_dict_res[_fsp_type][_key2use_p_dwn[1]] * _dict_costs[_fsp_type][
                    _key2use_p_dwn_cost[1]]).dropna(axis=1)
                fsp_costs['DW_' + _fsp_type + '_D_Cost'] = np.sum(
                    np.sum(res_dw_d_1.add(res_dw_d_2, fill_value=0), axis=0))

            # Reactive power Upward costs
            _key2use_q_up_cost = aux.get_list(_startswith_key='R_U_', _comparing_list=_dict_costs[_fsp_type])
            if len(_key2use_q_up_cost) == 1:
                res_dr_u = (_dict_res[_fsp_type][_key2use_q_up[0]] * _dict_costs[_fsp_type][
                    _key2use_q_up_cost[0]]).dropna(axis=1)
                fsp_costs['DR_' + _fsp_type + '_U_Cost'] = np.sum(np.sum(res_dr_u, axis=0))
            elif len(_key2use_q_up_cost) == 2:
                res_dr_u_1 = (_dict_res[_fsp_type][_key2use_q_up[0]] * _dict_costs[_fsp_type][
                    _key2use_q_up_cost[0]]).dropna(axis=1)
                res_dr_u_2 = (_dict_res[_fsp_type][_key2use_q_up[1]] * _dict_costs[_fsp_type][
                    _key2use_q_up_cost[1]]).dropna(axis=1)
                fsp_costs['DR_' + _fsp_type + '_U_Cost'] = np.sum(
                    np.sum(res_dr_u_1.add(res_dr_u_2, fill_value=0), axis=0))

            # Reactive power Downward costs
            _key2use_q_dwn_cost = aux.get_list(_startswith_key='R_D_', _comparing_list=_dict_costs[_fsp_type])
            if len(_key2use_q_dwn_cost) == 1:
                res_dr_d = (_dict_res[_fsp_type][_key2use_q_dwn[0]] * _dict_costs[_fsp_type][
                    _key2use_q_dwn_cost[0]]).dropna(axis=1)
                fsp_costs['DR_' + _fsp_type + '_D_Cost'] = np.sum(np.sum(res_dr_d, axis=0))
            elif len(_key2use_q_dwn_cost) == 2:
                res_dr_d_1 = (_dict_res[_fsp_type][_key2use_q_dwn[0]] * _dict_costs[_fsp_type][
                    _key2use_q_dwn_cost[0]]).dropna(axis=1)
                res_dr_d_2 = (_dict_res[_fsp_type][_key2use_q_dwn[1]] * _dict_costs[_fsp_type][
                    _key2use_q_dwn_cost[1]]).dropna(axis=1)
                fsp_costs['DR_' + _fsp_type + '_D_Cost'] = np.sum(
                    np.sum(res_dr_d_1.add(res_dr_d_2, fill_value=0), axis=0))

            ###############################
            # Active Power
            if len(_key2use_p_up_cost) == 1:
                fsp_costs['C_FM_' + _fsp_type + '_W_Total'] = np.sum(
                    np.sum(res_dw_u.add(res_dw_d, fill_value=0), axis=0))
            elif len(_key2use_p_up_cost) == 2:
                fsp_costs['C_FM_' + _fsp_type + '_W_Total'] = np.sum(np.sum(
                    res_dw_u_1.add(res_dw_u_2, fill_value=0).add(res_dw_d_1, fill_value=0).add(res_dw_d_2,
                                                                                               fill_value=0), axis=0))

            # Reactive Power
            if len(_key2use_q_up_cost) == 1:
                fsp_costs['C_FM_' + _fsp_type + '_R_Total'] = np.sum(
                    np.sum(res_dr_u.add(res_dr_d, fill_value=0), axis=0))
            elif len(_key2use_q_up_cost) == 2:
                fsp_costs['C_FM_' + _fsp_type + '_R_Total'] = np.sum(np.sum(
                    res_dr_u_1.add(res_dr_u_2, fill_value=0).add(res_dr_d_1, fill_value=0).add(res_dr_d_2,
                                                                                               fill_value=0), axis=0))

        else:
            # Power
            fsp_costs['DW_' + _fsp_type + '_U_total'] = 0
            fsp_costs['DW_' + _fsp_type + '_D_total'] = 0
            fsp_costs['W_' + _fsp_type] = 0

            fsp_costs['DR_' + _fsp_type + '_U_total'] = 0
            fsp_costs['DR_' + _fsp_type + '_D_total'] = 0
            fsp_costs['R_' + _fsp_type] = 0

            # Costs
            fsp_costs['DW_' + _fsp_type + '_U_Cost'] = 0
            fsp_costs['DW_' + _fsp_type + '_D_Cost'] = 0
            fsp_costs['DR_' + _fsp_type + '_U_Cost'] = 0
            fsp_costs['DR_' + _fsp_type + '_D_Cost'] = 0

            fsp_costs['C_FM_' + _fsp_type + '_W_Total'] = 0
            fsp_costs['C_FM_' + _fsp_type + '_R_Total'] = 0

            ac_fm_w[_fsp_type] = df_zeros_22
            ac_fm_r[_fsp_type] = df_zeros_22

        ###############################
        # Average Active Power Cost
        fsp_costs['DW_' + _fsp_type + '_Net_Cost_Avg'] = fsp_costs['C_FM_' + _fsp_type + '_W_Total'] / fsp_costs[
            'W_' + _fsp_type] if round(fsp_costs['W_' + _fsp_type], _round_val) != 0 else 0
        fsp_costs['DW_' + _fsp_type + '_U_Cost_Avg'] = fsp_costs['DW_' + _fsp_type + '_U_Cost'] / fsp_costs[
            'DW_' + _fsp_type + '_U_total'] if round(fsp_costs['DR_' + _fsp_type + '_U_total'], _round_val) != 0 else 0
        fsp_costs['DW_' + _fsp_type + '_D_Cost_Avg'] = fsp_costs['DW_' + _fsp_type + '_D_Cost'] / fsp_costs[
            'DW_' + _fsp_type + '_D_total'] if round(fsp_costs['DR_' + _fsp_type + '_D_total'], _round_val) != 0 else 0

        # Average Reactive Power Cost
        fsp_costs['DR_' + _fsp_type + '_Net_Cost_Avg'] = fsp_costs['C_FM_' + _fsp_type + '_R_Total'] / fsp_costs[
            'R_' + _fsp_type] if round(fsp_costs['R_' + _fsp_type], _round_val) != 0 else 0
        fsp_costs['DR_' + _fsp_type + '_U_Cost_Avg'] = fsp_costs['DR_' + _fsp_type + '_U_Cost'] / fsp_costs[
            'DR_' + _fsp_type + '_U_total'] if round(fsp_costs['DR_' + _fsp_type + '_U_total'], _round_val) != 0 else 0
        fsp_costs['DR_' + _fsp_type + '_D_Cost_Avg'] = fsp_costs['DR_' + _fsp_type + '_D_Cost'] / fsp_costs[
            'DR_' + _fsp_type + '_D_total'] if round(fsp_costs['DR_' + _fsp_type + '_D_total'], _round_val) != 0 else 0

    # Active Power of all resources
    w_total = sum([fsp_costs['W_' + _fsp_type] for _fsp_type in _in_dict['fsp_file'].keys()])
    # Reactive Power of all resources
    r_total = sum([fsp_costs['R_' + _fsp_type] for _fsp_type in _in_dict['fsp_file'].keys()])

    # Active Power total cost for flexibility market
    c_fm_w = sum([fsp_costs['C_FM_' + _fsp_type + '_W_Total'] for _fsp_type in _in_dict['fsp_file'].keys()])
    # Reactive Power total cost for flexibility market
    c_fm_r = sum([fsp_costs['C_FM_' + _fsp_type + '_R_Total'] for _fsp_type in _in_dict['fsp_file'].keys()])

    # Flexibility Activated
    res_actived_flex_w = pd.DataFrame()
    res_actived_flex_r = pd.DataFrame()
    for _fsp_type in _in_dict['fsp_file'].keys():
        res_actived_flex_w = res_actived_flex_w.add(ac_fm_w[_fsp_type], fill_value=0, )
        res_actived_flex_r = res_actived_flex_r.add(ac_fm_r[_fsp_type], fill_value=0)
    ac_fm_w_total = np.sum(np.sum(res_actived_flex_w, axis=0))
    ac_fm_r_total = np.sum(np.sum(res_actived_flex_r, axis=0))

    # Cost alpha Total
    alpha = np.sum(np.sum(np.abs(_dict_others['ABS_alpha_DVpu'])), axis=0)
    cost_alpha_total = alpha * _in_dict['cost_alpha']

    # Cost beta Total
    beta = np.sum(np.sum(_dict_others['ABS_beta']), axis=0)
    cost_beta_total = beta * _in_dict['cost_beta']

    # Totals
    fsp_costs['ActivePower_Total [MW]'] = round(w_total, _round_val)
    fsp_costs['ReactivePower_Total [MVAR]'] = round(r_total, _round_val)
    fsp_costs['C_FM_ActivePower [Eur]'] = round(c_fm_w, _round_val)
    fsp_costs['C_FM_ActivePower_Avg [Eur/MW]'] = round(c_fm_w / ac_fm_w_total, _round_val) if ac_fm_w_total.round(_round_val) != 0 else 0
    fsp_costs['C_FM_ReactivePower [Eur]'] = round(c_fm_r, _round_val)
    fsp_costs['C_FM_ReactivePower_Avg [Eur/MVAR]'] = round(c_fm_r / ac_fm_r_total, _round_val) if ac_fm_r_total.round(_round_val) != 0 else 0

    # Flexibility no supplied
    fsp_costs['Alpha [MW]'] = float(alpha)
    fsp_costs['Cost_Alpha_Total'] = cost_alpha_total
    fsp_costs['Cost_Alpha_Avg'] = cost_alpha_total / alpha if round(alpha, _round_val) != 0 else 0

    fsp_costs['Beta [MVA]'] = float(beta)
    fsp_costs['Cost_Beta_Total'] = cost_beta_total
    fsp_costs['Cost_Beta_Avg'] = cost_beta_total / beta if round(beta, _round_val) != 0 else 0

    fsp_costs['ObjValue'] = _dict_others['ObjValue'].ObjValue.round(_round_val)

    return fsp_costs


def cost_flexibility(_in_dict):
    """
    Evaluate the cost af all the flexibility service activations.

    :param _in_dict: Dictionary with all the information for evaluating the avoided voltage violations.
    """
    scenarios = _in_dict['scenarios_selected']
    kpi_path = _in_dict['paths']

    flex_activated = dict()
    dict_fsp = dict()
    dict_costs = dict()
    for name_tab in scenarios:
        _prod = name_tab.split('_')[2]
        _vl = name_tab.split('_')[3]
        _fsp = name_tab.split('_')[4]
        _in_dict['path_MkOut'] = kpi_path['mrk_outputs']['res_market'][_prod][_vl][_fsp]
        _in_dict['path_MkIn'] = kpi_path['mrk_inputs'][_prod][_vl][_fsp]

        dict_fsp[name_tab], dict_others = load_fsp_result(_in_dict=_in_dict, _scenario=name_tab)
        dict_costs[name_tab] = load_data_costs(_in_dict=_in_dict, _scenario=name_tab)

        flex_activated[name_tab] = eval_flexibility_costs(_in_dict=_in_dict, _dict_res=dict_fsp[name_tab],
                                                          _dict_others=dict_others, _dict_costs=dict_costs[name_tab])
    return flex_activated, dict_fsp, dict_costs


def build_final_table(_in_dict):
    """
    Build final results table.
    The objective value and the total cost are different since the slack variables of the voltage (alpha) is written
    differently in the market. In the market, it is written as:
        -> (ABS_alpha_DVpu / sens_matrix_vc['all']) * alpha_cost        (if net_in_market parameters is not active)
        -> ABS_alpha_DVpu * alpha_cost * 1000                           (if net_in_market parameters is active)

    :param _in_dict: Dictionary with all the information for building the final table in terms of costs and volumes.
    """

    final_res_table = {
        'ObjValue [Eur]': pd.DataFrame(_in_dict).T['ObjValue'],
        'CostBeta_Total [Eur]': pd.DataFrame(_in_dict).T['Cost_Beta_Total'],
        'CostAlpha_Total [Eur]': pd.DataFrame(_in_dict).T['Cost_Alpha_Total'],
        'TotalActivePower [MW]': pd.DataFrame(_in_dict).T['ActivePower_Total [MW]'],
        'ActivePowerCost [Eur]': pd.DataFrame(_in_dict).T['C_FM_ActivePower [Eur]'],
        'TotalReactivePower [MVAR]': pd.DataFrame(_in_dict).T['ReactivePower_Total [MVAR]'],
        'ReactivePowerCost [Eur]': pd.DataFrame(_in_dict).T['C_FM_ReactivePower [Eur]'],
        'TotalCost [Eur]': pd.DataFrame(_in_dict).T['Cost_Beta_Total'] + pd.DataFrame(_in_dict).T['Cost_Alpha_Total']
                           + pd.DataFrame(_in_dict).T['C_FM_ActivePower [Eur]'] + pd.DataFrame(_in_dict).T[
                               'C_FM_ReactivePower [Eur]']
    }
    return pd.DataFrame(final_res_table)


def eval_statistics_volume_flex(_in_dict, _dict_res, _dict_costs, _scenario, _round_val=6):
    """
    Evaluate the statistics behind the volume of flexibility offered.

    :param _in_dict: Dictionary with all the information pre-market for evaluating the avoided voltage violations.

    :param _dict_res: Dictionary with all the information about the flexibility provided by each resource.

    :param _dict_costs: Dictionary with all the information about the costs for flexibility by each resource.

    :param _scenario: String representing the specific scenario to read.

    :param _round_val: Value to which all value are rounded.
    """
    flex_res = dict()
    for _fsp_type in _in_dict['fsp_file'].keys():
        if _in_dict['flat_' + _fsp_type]:
            w_init = _dict_costs[_fsp_type]['Winit']
            r_init = _dict_costs[_fsp_type]['Rinit']

            w_ub_ulim = _dict_costs[_fsp_type].get('WUB_ULim', pd.DataFrame())
            w_ub_dlim = _dict_costs[_fsp_type].get('WUB_DLim', pd.DataFrame())
            w_lb_ulim = _dict_costs[_fsp_type].get('WLB_ULim', pd.DataFrame())
            w_lb_dlim = _dict_costs[_fsp_type].get('WLB_DLim', pd.DataFrame())
            r_ub_ulim = _dict_costs[_fsp_type].get('RUB_ULim', pd.DataFrame())
            r_ub_dlim = _dict_costs[_fsp_type].get('RUB_DLim', pd.DataFrame())
            r_lb_ulim = _dict_costs[_fsp_type].get('RLB_ULim', pd.DataFrame())
            r_lb_dlim = _dict_costs[_fsp_type].get('RLB_DLim', pd.DataFrame())

            dw_upward = _dict_res[_fsp_type]['DW_' + _fsp_type + '_U']
            dw_dwnward = _dict_res[_fsp_type]['DW_' + _fsp_type + '_D']

            dr_upward = _dict_res[_fsp_type]['DR_' + _fsp_type + '_U']
            dr_dwnward = _dict_res[_fsp_type]['DR_' + _fsp_type + '_U']

            # Active Power
            stat_w_init = w_init.T.aggregate('describe').T[['mean', 'std', 'max', 'min']]

            # fixme: to check for future developments.
            #  As of now, it consider that the reactive power is only upward for all resources.
            if w_lb_ulim.empty is False and w_ub_ulim.empty is True:
                # Load
                stat_w_offered_upward = (w_init - w_lb_ulim).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
                stat_r_offered_upward = (r_init - r_ub_ulim).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
            elif w_lb_ulim.empty is True and w_ub_ulim.empty is False:
                # Generator and Storage
                stat_w_offered_upward = (w_ub_ulim - w_init).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
                stat_r_offered_upward = (r_ub_ulim - r_init).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]

            if w_ub_dlim.empty is False and w_lb_dlim.empty is True:
                # Load
                stat_w_offered_dwnward = (w_ub_dlim - w_init).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
                stat_r_offered_dwnward = (r_ub_dlim - r_init).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
            elif w_ub_dlim.empty is True and w_lb_dlim.empty is False:
                # Generator and Storage
                stat_w_offered_dwnward = (w_init - w_lb_dlim).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
                stat_r_offered_dwnward = (r_init - r_ub_dlim).T.aggregate('describe').T[['mean', 'std', 'max', 'min']]

            stat_w_cleared_upward = dw_upward.T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
            stat_w_cleared_dwnward = dw_dwnward.T.aggregate('describe').T[['mean', 'std', 'max', 'min']]

            # Reactive Power
            stat_r_init = r_init.T.aggregate('describe').T[['mean', 'std', 'max', 'min']]

            stat_r_cleared_upward = dr_upward.T.aggregate('describe').T[['mean', 'std', 'max', 'min']]
            stat_r_cleared_dwnward = dr_dwnward.T.aggregate('describe').T[['mean', 'std', 'max', 'min']]

            flex_res['ST_Winit_' + _fsp_type] = round(stat_w_init, _round_val)
            flex_res['ST_WOffered_' + _fsp_type + '_U'] = round(stat_w_offered_upward, _round_val)
            flex_res['ST_WOffered_' + _fsp_type + '_D'] = round(stat_w_offered_dwnward, _round_val)
            flex_res['ST_WCleared_' + _fsp_type + '_U'] = round(stat_w_cleared_upward, _round_val)
            flex_res['ST_WCleared_' + _fsp_type + '_D'] = round(stat_w_cleared_dwnward, _round_val)

            flex_res['ST_Rinit_' + _fsp_type] = round(stat_r_init, _round_val)
            flex_res['ST_ROffered_' + _fsp_type + '_U'] = round(stat_r_offered_upward, _round_val)
            flex_res['ST_ROffered_' + _fsp_type + '_D'] = round(stat_r_offered_dwnward, _round_val)
            flex_res['ST_RCleared_' + _fsp_type + '_U'] = round(stat_r_cleared_upward, _round_val)
            flex_res['ST_RCleared_' + _fsp_type + '_D'] = round(stat_r_cleared_dwnward, _round_val)
    return flex_res


def volume_flex_offered(_in_dict, _dict_res, _dict_costs):
    """
    Evaluate the volume of flexibility offered.

    :param _in_dict: Dictionary with all the information for evaluating the volume of flexibility offered.

    :param _dict_res: Dictionary with all the information about the flexibility provided by each resource.

    :param _dict_costs: Dictionary with all the information about the costs for flexibility by each resource.
    """
    scenarios = _in_dict['scenarios_selected']

    flex_q = dict()
    for name_tab in scenarios:
        flex_q[name_tab] = eval_statistics_volume_flex(_in_dict=_in_dict, _dict_res=_dict_res[name_tab],
                                                       _dict_costs=_dict_costs[name_tab], _scenario=name_tab)

    return flex_q


def increase_hc_plots(_df_res, _scenarios, _title, _y_lim):
    """
    Prepare the plots for the Increase Hosting Capacity.
    The plots are subdivided per Voltage Violation Scenarios.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.

    :param _y_lim: Limits of the y-axis.
    """
    fig_size = (10, 8)
    filtered_df_res = _df_res.loc[_scenarios]
    titles = ['KPI: Increased Hosting Capacity', _title]
    name_net_fig = 'Cases'
    labels = [name_net_fig, filtered_df_res.columns[0]]
    fig = plt_kpi.bar_increase_hc(filtered_df_res, fig_size, titles, labels, _y_lim)
    return fig


def avoided_cong_plots(_df_res, _scenarios, _title, _y_lim):
    """
    Prepare the plots for the avoided congestions.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.

    :param _y_lim: Limits of the y-axis.
    """
    figsize = (10, 8)

    filtered_df_res = _df_res.loc[_scenarios]

    name_net_fig = 'Cases'
    titles = ['KPI: Avoided Congestions', _title]
    labels = [name_net_fig,
              'Number of cumulative restriction avoided [element x hour]',
              'Percentage of Cumulative congestions avoided [%]',
              'Overall [%]']
    df1 = filtered_df_res[['AvoidCong_L[u]', 'AvoidCong_T[u]', 'AvoidCong_T3w[u]']]
    df1.rename(columns={'AvoidCong_L[u]': 'Lines',
                        'AvoidCong_T[u]': 'Trafo_2windings',
                        'AvoidCong_T3w[u]': 'Trafo_3windings'}, inplace=True)

    df2 = filtered_df_res[['AvoidCongestions [%]']]
    df2.rename(columns={'AvoidCongestions [%]': 'Overall[%]'}, inplace=True)
    df2.rename(columns={'Overall[%]': 'y'}, inplace=True)

    fig = plt_kpi.bar_avoid_violations(df1, df2, figsize, titles, labels, _y_lim)
    return fig


def avoided_cong_3dplots(_df_res, _scenarios, _title):
    """
    Prepare the 3D plots for the avoided congestions.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.
    """
    figsize = (10, 8)
    titles = ['KPI: Avoided Congestions [%]', _title]

    filtered_df_res = _df_res.loc[_scenarios]

    z_values = filtered_df_res['AvoidCongestions [%]'].values.tolist()
    product_tag = np.unique([i.split('_')[2] for i in _scenarios])
    fsp_tag = list()
    # fsp_tag = np.unique([i.split('_')[4] for i in _scenarios])

    # Build dictionary for plot (Comparison of 3 market product: ['P', 'Q', 'PQ'])
    mrk_products = ['P', 'Q', 'PQ']
    x = {_prod: list() for _prod in mrk_products}
    y = {_prod: list() for _prod in mrk_products}
    dz = {_prod: list() for _prod in mrk_products}
    _1st_mrk_product = 0
    value_x = 0
    value_y = 0
    idx_z = 0
    label_pos = []
    for _tag in _scenarios:
        fsp_tag.append('_'.join(_tag.split('_')[3:]))
        if _1st_mrk_product != _tag.split('_')[2]:
            _1st_mrk_product = _tag.split('_')[2]
            value_x += 2
            value_y = 1

        _mrk_product = _tag.split('_')[2]
        x[_mrk_product].append(value_x)
        y[_mrk_product].append(value_y)
        dz[_mrk_product].append(z_values[idx_z])
        value_y += 2
        idx_z += 1
        label_pos.append(value_x + 0.5)

    label_pos = np.unique(label_pos)
    fsp_tag = np.unique(fsp_tag).tolist()
    fig = plt_kpi.bar3d_avoid_violations(_x=x, _y=y, _dz=dz, _figsize=figsize, _mrk_products=product_tag, _fsps=fsp_tag,
                                         _label_pos=label_pos, _zlabel='AvoidCongestions [%]', _titles=titles)
    return fig


def avoided_cong_element_3dplots(_df_res, _scenarios, _title):
    """
    Prepare the 3D plots for the number of avoided congestions.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.
    """
    figsize = (10, 8)
    titles = ['KPI: Avoided Congestions', _title]

    filtered_df_res = _df_res.loc[_scenarios]

    z_values = filtered_df_res[['AvoidCong_L[u]', 'AvoidCong_T[u]', 'AvoidCong_T3w[u]']].values
    product_tag = np.unique([i.split('_')[2] for i in _scenarios])
    fsp_tag = list()

    # Build dictionary for plot
    # (Comparison of 3 market product: ['P', 'Q', 'PQ'] and comparison of 3 elements: ['Line', 'Trafo', 'Trafo3w'])
    mrk_products = ['P', 'Q', 'PQ']
    keys_element = ['Line', 'Trafo', 'Trafo3w']
    x = {_prod: {_elem: list() for _elem in keys_element} for _prod in mrk_products}
    y = {_prod: {_elem: list() for _elem in keys_element} for _prod in mrk_products}
    dz = {_prod: {_elem: list() for _elem in keys_element} for _prod in mrk_products}
    _1st_mrk_product = 0
    value_x = 0
    value_y = 1
    idx_z = 0
    label_pos = []
    for _tag in _scenarios:
        fsp_tag.append('_'.join(_tag.split('_')[3:]))
        if _1st_mrk_product != _tag.split('_')[2]:
            _1st_mrk_product = _tag.split('_')[2]
            value_x += 5
            value_y = 1

        _mrk_product = _tag.split('_')[2]

        _value_elem = 0
        for _elem in keys_element:
            x[_mrk_product][_elem].append(value_x + _value_elem)
            y[_mrk_product][_elem].append(value_y)
            dz[_mrk_product][_elem].append(z_values[idx_z, _value_elem])
            _value_elem += 1

        value_y += 2
        idx_z += 1
        label_pos.append(value_x + 1.5)

    label_pos = np.unique(label_pos)
    fsp_tag = np.unique(fsp_tag).tolist()
    fig = plt_kpi.bar3d_avoid_violations_element(_x=x, _y=y, _dz=dz, _figsize=figsize, _mrk_products=product_tag,
                                                 _key_elem=keys_element, _fsps=fsp_tag, _label_pos=label_pos,
                                                 _zlabel='Number of restriction avoided [element x hour]', _titles=titles)
    return fig


def avoided_voltage_viol_plots(_df_res, _scenarios, _title, _y_lim):
    """
    Prepare Avoided Voltage Violations plot.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.

    :param _y_lim: Limits of the y-axis.
    """
    figsize = (10, 8)

    filtered_df_res = _df_res.loc[_scenarios]

    name_net_fig = 'Cases'
    titles = ['KPI: Avoided Voltage Violations', _title]
    labels = [name_net_fig, 'Number of cumulative restriction avoided [element x hour]',
              'Percentage of cumulative voltage violations avoided [%]',
              'Percentage of Cumulative restrictions avoided']
    df1 = filtered_df_res[['AvoidVoltageViolations [u]']]
    df1.rename(columns={'AvoidVoltageViolations [u]': 'Number of cumulative restriction avoided [u]'}, inplace=True)

    df2 = filtered_df_res[['AvoidVoltageViolations [%]']]
    df2.rename(columns={'AvoidVoltageViolations [%]': 'Percentage of cumulative restriction avoided [%]'}, inplace=True)
    df2.rename(columns={'Percentage of cumulative restriction avoided [%]': 'y'}, inplace=True)

    fig = plt_kpi.bar_avoid_violations(df1, df2, figsize, titles, labels, _y_lim)

    return fig


def plot_kpi(_in_dict, _kpi2plot, _path_kpi_res, _simulation_tag, _draw_plots=True, _draw_net=False):
    """
    Prepare the data to be plotted.

    :param _in_dict: Dictionary with all the information for plotting the kpi.

    :param _kpi2plot: Dictionary with the kpi to be plotted.

    :param _path_kpi_res: Root to the folder in which the kpi will be saved.

    :param _simulation_tag: String that represent the simulation tag, composed by the name od the network, the market
     model and the storage tag.

    :param _draw_plots: Boolean value (True/False) plot graphs.

    :param _draw_net: Boolean value (True/False) plot network and save.
    """
    scenarios = _in_dict['scenarios_selected']
    dpi_img = _in_dict['dpi_img']

    if _draw_plots:
        ihc_res = _kpi2plot['ihc_res']
        pihc_res = _kpi2plot['pihc_res']
        avoided_cong_res = _kpi2plot['avoided_cong_res']
        avoided_voltage_viol_res = _kpi2plot['avoided_voltage_viol_res']

        # _in_dict_plot = {'scenarios': name_tab, 'sim_tag': _simulation_tag}

        max_value = ihc_res[list(ihc_res.keys())[0]].max()
        ylim = [-0.1, max_value + 0.5]
        fig = increase_hc_plots(_df_res=ihc_res, _scenarios=scenarios, _title=_simulation_tag, _y_lim=ylim)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_IHC[MVA]_Results.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        ylim = [-20, 140]
        fig = increase_hc_plots(_df_res=pihc_res, _scenarios=scenarios, _title=_simulation_tag, _y_lim=ylim)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_IHC[%]_Results.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        # Avoided restrictions: Congestions in lines, trafos and Voltage Violations in Buses plots
        ylim = [[], [0, 100]]
        fig = avoided_cong_plots(_df_res=avoided_cong_res, _scenarios=scenarios, _title=_simulation_tag, _y_lim=ylim)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_AvCongest.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        # Avoided restrictions [%] 3D Plot: Congestions in lines, trafos and Voltage Violations in Buses plots
        fig = avoided_cong_3dplots(_df_res=avoided_cong_res, _scenarios=scenarios, _title=_simulation_tag)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_AvCongest_3DPlot.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        # Avoided restrictions [Number of element] 3D Plot: Congestions in lines, trafos and Voltage Violations in
        # Buses plots
        fig = avoided_cong_element_3dplots(_df_res=avoided_cong_res, _scenarios=scenarios, _title=_simulation_tag)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_AvCongest_element_3DPlot.png'), format='png',
                    dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        ylim = [[], [0, 100]]
        fig = avoided_voltage_viol_plots(_df_res=avoided_voltage_viol_res, _scenarios=scenarios, _title=_simulation_tag, _y_lim=ylim)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_AvVoltViolat.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        # Density and Histograms Plots
        fig = plt_kpi.density_plot_bus(_data=_in_dict, _scenarios=scenarios, _title=_simulation_tag)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_DensityBus.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        ylim = [[], []]
        xlim = [0.85, 1.15]
        fig = plt_kpi.histogram_plot_bus(_data=_in_dict, _scenarios=scenarios, _title=_simulation_tag, _xlim=xlim, _ylim=ylim)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_HistoBus.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        fig = plt_kpi.density_plot_lines(_data=_in_dict, _scenarios=scenarios, _title=_simulation_tag)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_DensityLines.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

        ylim = [[], []]
        xlim = []
        fig = plt_kpi.histogram_plot_lines(_data=_in_dict, _scenarios=scenarios, _title=_simulation_tag, _xlim=xlim, _ylim=ylim)
        fig.savefig(os.path.join(_path_kpi_res, _simulation_tag + '_HistoLines.png'), format='png', dpi=dpi_img)
        print(fig)
        plt.ion()
        # "interactive mode": Plotting no longer blocks: matplotlib.pyplot.ion()
        # https://stackoverflow.com/questions/64064016/dont-stop-at-plt-show-in-vs-code
        plt.close()

    if _draw_net:
        _net = _in_dict['network']
        pf_res_plotly(_net)
        # plot with topology and no power flow results
        pp.plotting.plotly.simple_plotly(_net, respect_switches=True, use_line_geodata=None, on_map=False,
                                         projection=None, map_style='basic', figsize=1, aspectratio='auto',
                                         line_width=1, bus_size=10, ext_grid_size=20.0, bus_color='blue',
                                         line_color='grey', trafo_color='green', ext_grid_color='yellow')
        # plot with no topology for lines and power flow results
        pp.plotting.plotly.simple_plotly(_net, respect_switches=True, use_line_geodata=False, on_map=False,
                                         projection=None, map_style='basic', figsize=1, aspectratio='auto',
                                         line_width=1, bus_size=10, ext_grid_size=20.0, bus_color='blue',
                                         line_color='grey', trafo_color='green', ext_grid_color='yellow')
        # plot with single line diagram of the line diagrams
        colors = ["b", "g", "r", "c", "y"]
        net_nogeo = _net.deepcopy()
        net_nogeo.bus_geodata.drop(net_nogeo.bus_geodata.index, inplace=True)
        net_nogeo.line_geodata.drop(net_nogeo.line_geodata.index, inplace=True)
        create_generic_coordinates(net_nogeo, respect_switches=True)
        fuse_geodata(net_nogeo)
        bc = create_bus_collection(net_nogeo, net_nogeo.bus.index, size=.08, color=colors[0], zorder=10)
        tlc, tpc = create_trafo_collection(net_nogeo, net_nogeo.trafo.index, color=colors[1])
        lcd = create_line_collection(net_nogeo, net_nogeo.line.index, color=colors[1], linewidths=0.5,
                                     use_bus_geodata=True)
        sc = create_bus_collection(net_nogeo, net_nogeo.ext_grid.bus.values, patch_type="rect", size=.3,
                                   color=colors[4], zorder=11)
        draw_collections([lcd, bc, tlc, tpc, sc], figsize=(12, 10))
    return True


def evaluate_kpis(_cfg_file, _paths, _fsp_file, _prods, _vls, _fsps, _draw_plots=True, _draw_net=False):
    """
    Function for evaluating the KPI.

    :param _cfg_file: Yaml configuration file.

    :param _paths: Dictionary with all the paths for each specific scenario.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _prods: List of products that the user want to analyse in the KPI evaluation.

    :param _vls: List of voltage limits that the user want to analyse in the KPI evaluation.

    :param _fsps: List of fsp availability that the user want to analyse in the KPI evaluation.

    :param _draw_plots: Boolean value (True/False) plot graphs.

    :param _draw_net: Boolean value (True/False) plot network and save.
    """

    _kpi_params = _set_kpi_params(_cfg_file=_cfg_file, _paths=_paths, _fsp_file=_fsp_file, _prods=_prods, _vls=_vls, _fsps=_fsps)
    _kpi_params['path_KPIresults'] = _paths['kpi_out_root']
    _kpi_params['draw_plots'] = _draw_plots
    _kpi_params['draw_net'] = _draw_net
    _kpi_params['fsp_file'] = _fsp_file

    # Load data of pre-market
    _kpi_params = _load_res_net_pre(_kpi_params, _kpi_params['scenarios_selected'])
    # Load data of post-market
    _kpi_params = _load_res_net_post(_kpi_params, _kpi_params['scenarios_selected'])

    _kpi_params['hc_pre'] = list()
    _kpi_params['hc_post'] = list()
    # Increased hosting capacity
    hc_res_pre, hc_res_post, ihc_res, pihc_res = increase_hc(_kpi_params)

    # Avoided Congestion Problems (Lines and Trafos)
    avoided_cong_res = avoided_cong(_kpi_params, _kpi_params['scenarios_selected'])

    # Avoided Voltage Violations (Buses)
    avoided_voltage_viol_res = avoided_voltage_violations(_kpi_params)

    # FSPs Flexibility Activated and Costs
    flex_all, dict_res, dict_cost = cost_flexibility(_kpi_params)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.float_format = '{:.2f}'.format

    # Model Results
    overall_res = build_final_table(flex_all)

    file_workbook = Path(_paths['kpi_out_root']).name.split('_', 2)[2]
    io_file.save_excel(_data=overall_res, _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='Overall_Results')
    io_file.save_excel(_data=pd.DataFrame(flex_all), _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='FlexALL', _mode='a')
    io_file.save_excel(_data=avoided_voltage_viol_res, _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='AvoidVoltageViol_Results', _mode='a')
    io_file.save_excel(_data=avoided_cong_res, _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='AvoidCongestions_Results', _mode='a')
    io_file.save_excel(_data=pihc_res, _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='pIHC_Results', _mode='a')
    io_file.save_excel(_data=ihc_res, _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='IHC_Results', _mode='a')
    io_file.save_excel(_data=hc_res_post, _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='HC_Results_post', _mode='a')
    io_file.save_excel(_data=hc_res_pre, _filename=file_workbook, _outroot=_paths['kpi_out_root'],
                       _sheetname='HC_Results_pre', _mode='a')

    # Volumes of flexibility for each FSP
    volume_flex_stats = volume_flex_offered(_kpi_params, dict_res, dict_cost)
    for _k in volume_flex_stats.keys():
        _idx = 0
        mode_write = 'w'
        for _h in volume_flex_stats[_k].keys():
            if _idx > 0:
                mode_write = 'a'
            io_file.save_excel(_data=volume_flex_stats[_k][_h], _filename='KPI_VolFlex_' + _k,
                               _outroot=_paths['kpi_out_root'], _sheetname=_h, _mode=mode_write)
            _idx += 1

    _kpi2plot = {
        'ihc_res': ihc_res,
        'pihc_res': pihc_res,
        'avoided_cong_res': avoided_cong_res,
        'avoided_voltage_viol_res': avoided_voltage_viol_res
    }
    plot_kpi(_in_dict=_kpi_params, _kpi2plot=_kpi2plot, _path_kpi_res=_paths['kpi_out_root'], _simulation_tag=file_workbook, _draw_plots=True, _draw_net=False)

    print('\nKPI calculation complete.')
    return True
