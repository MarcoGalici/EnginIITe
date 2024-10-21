import os
import math
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import functions.file_io as io_file


def flex_needs_buses(_ts_res, _v_max, _v_min, _folders, _filenames, _flag_vpilot=False):
    """Function that identify under and over bus voltages.
    :param _ts_res: Dictionary with the results of the timeseries simulation.
    :param _v_max: Maximum voltage limitation.
    :param _v_min: Minimum voltage limitation.
    :param _folders: List of folders where to save data.
    :param _filenames: List of file names used to name the data.
    :return vbus_outbounds: Dataframe with the buses that are with over and under voltage problems
    :return vbus_over_flex: Dataframe with the buses that are with over voltage problems
    :return vbus_under_flex: Dataframe with the buses that are with under voltage problems
    :return vc_hours2study: Numpy array with the over and under voltage hours.
    """
    _bus_voltages = _ts_res['bus_vm_pu']

    vbus_over = _bus_voltages[_bus_voltages >= _v_max]
    vbus_under = _bus_voltages[_bus_voltages <= _v_min]

    vbus_over_flex = pd.DataFrame()
    vbus_under_flex = pd.DataFrame()

    # populate vbus_over_flex dataframe
    vbus_over_flex['Bus'] = vbus_over.stack().index.get_level_values(1).values
    vbus_over_flex['TimeStep'] = vbus_over.stack().index.get_level_values(0).values
    vbus_over_flex['VMpu'] = vbus_over.stack().values

    # populate the dataframe
    vbus_under_flex['Bus'] = vbus_under.stack().index.get_level_values(1).values
    vbus_under_flex['TimeStep'] = vbus_under.stack().index.get_level_values(0).values
    vbus_under_flex['VMpu'] = vbus_under.stack().values

    vbus_outbounds = pd.concat([vbus_over_flex, vbus_under_flex])

    vc_hours2study = vbus_outbounds['TimeStep'].unique()

    vm_stats_premrk = pd.DataFrame({'Total_Bxh': vbus_outbounds.shape[0],
                                    'OverV_Bxh': vbus_over_flex.shape[0],
                                    'UnderV_Bxh': vbus_under_flex.shape[0]}, index=[0])
    io_file.save_excel(vm_stats_premrk, _folders[0], _filenames[0], _sheetname='Stats')
    io_file.save_excel(vbus_over_flex, _folders[0], _filenames[0], _sheetname='Vbus_over', _mode='a')
    io_file.save_excel(vbus_under_flex, _folders[0], _filenames[0], _sheetname='Vbus_under', _mode='a')

    io_file.save_cvs(pd.DataFrame(vc_hours2study), _folders[1], _filenames[1])

    return vbus_outbounds, vbus_over_flex, vbus_under_flex, vc_hours2study


def flex_needs_lines(_net, _line_loading, _flex_limit, _overload_limit=100):
    """Function that evaluate the flexibility needs for overloaded lines.
    :param _net: Pandapower network.
    :param _line_loading: Dataframe with the results of the timeseries simulation of the lines.
    :param _overload_limit: Transformers loading limit in percentage.
    :param _flex_limit: Maximum flexibility available to the dso in percentage.
    :return dso_flex: Dataframe with the information about the lines congested."""
    line_overload = _line_loading[_line_loading >= _overload_limit]

    cong_lines = np.unique(line_overload.stack().index.get_level_values(1).values)
    smax_lines_new = [math.sqrt(3) * _net.line.max_i_ka[_l] * _net.bus.vn_kv[_net.line.from_bus[_l]] for _l in cong_lines]

    dso_flex = (((line_overload.iloc[:, cong_lines] - _flex_limit) * smax_lines_new) / 100).fillna(0)
    dso_flex.columns = _net.line.name[cong_lines].index.tolist()

    return dso_flex.T


def flex_needs_trafos_2w(_net, _trafo_loading, _flex_limit, _overload_limit=100):
    """Function that evaluate the flexibility needs for overloaded transformers (2 windings).
    :param _net: Pandapower network.
    :param _trafo_loading: Dataframe with the results of the timeseries simulation of the trafos.
    :param _overload_limit: Transformers loading limit in percentage.
    :param _flex_limit: Maximum flexibility available to the dso in percentage.
    :return dso_flex: Dataframe with the information about the trafo 2 windings congested."""
    trafo_overload = _trafo_loading[_trafo_loading >= _overload_limit]

    cong_tr = np.unique(trafo_overload.stack().index.get_level_values(1).values)

    smax_tr = _net.trafo.sn_mva.values[cong_tr]

    dso_flex = (((trafo_overload.iloc[:, cong_tr] - _flex_limit) * smax_tr) / 100).fillna(0)
    dso_flex.columns = _net.trafo.name[cong_tr].index.tolist()

    return dso_flex.T


def flex_needs_trafos_3w(_net, _trafo_loading, _flex_limit, _overload_limit=100):
    """Function that evaluate the flexibility needs for overloaded transformers (3 windings).
    :param _net: Pandapower network.
    :param _trafo_loading: Dataframe with the results of the timeseries simulation of the trafos.
    :param _overload_limit: Transformers loading limit in percentage.
    :param _flex_limit: Maximum flexibility available to the dso in percentage.
    :return dso_flex: Dataframe with the information about the trafo 3 windings congested."""
    trafo_overload = _trafo_loading[_trafo_loading >= _overload_limit]

    cong_tr = np.unique(trafo_overload.stack().index.get_level_values(1).values)

    smax_tr = _net.trafo3w.sn_hv_mva.values[cong_tr]

    dso_flex = (((trafo_overload.iloc[:, cong_tr] - _flex_limit) * smax_tr) / 100).fillna(0)
    dso_flex.columns = _net.trafo.name[cong_tr].index.tolist()

    return dso_flex.T


def congestion_needs(_net, _ts_res, _flexlimit, _folders, _filenames, _llimit=100, _tlimit=100):
    """Function to evaluate the congestion needs summary.
    :param _net: Pandapower network.
    :param _ts_res: Dictionary with the results of the timeseries simulation.
    :param _flexlimit: Maximum flexibility available to the dso in percentage.
    :param _llimit: Line loading limit in percentage.
    :param _tlimit: Transformers loading limit in percentage.
    :return cong_needs_df: Dataframe with the information about all the congestions.
    :return lines_flex_df: Dataframe with the information about the line congested.
    :return trafos_flex_df: Dataframe with the information about the trafo 2 windings congested.
    :return trafos3w_flex_df: Dataframe with the information about the trafo 3 windings congested.
    :return cong_hours: Numpy array with the congested hours.
    """
    _lloading = _ts_res['line_loading_percent']
    _t2wloading = _ts_res['trafo_loading_percent']
    _t3wloading = _ts_res['trafo3w_loading_percent']

    # Lines
    lines_flex_df = flex_needs_lines(_net=_net, _line_loading=_lloading, _flex_limit=_flexlimit, _overload_limit=_llimit)

    # Trafos2W
    trafos_flex_df = flex_needs_trafos_2w(_net=_net, _trafo_loading=_t2wloading, _flex_limit=_flexlimit, _overload_limit=_tlimit)

    # Trafos 3W
    trafos3w_flex_df = flex_needs_trafos_3w(_net=_net, _trafo_loading=_t3wloading, _flex_limit=_flexlimit, _overload_limit=_tlimit)

    # Concat all congestion needs
    cong_needs_df = round(pd.concat([lines_flex_df, trafos_flex_df, trafos3w_flex_df]), 4)

    # Save File
    io_file.save_excel(cong_needs_df, _folders[0], _filenames[0], _sheetname='Congestion needs')

    cong_filtered = cong_needs_df.loc[:, (cong_needs_df > 0).any()]
    cong_hours = cong_filtered.columns.values

    # Save file
    io_file.save_cvs(pd.DataFrame(cong_hours), _folders[1], _filenames[1])

    return cong_needs_df, lines_flex_df, trafos_flex_df, trafos3w_flex_df, cong_hours
