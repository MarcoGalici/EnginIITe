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
    return True


def flex_needs_lines(_net, _line_loading, _flex_limit, _overload_limit=100):
    """Function that evaluate the flexibility needs for overloaded lines.
    :param _net: Pandapower network.
    :param _line_loading: Dataframe with the results of the timeseries simulation of the lines.
    :param _overload_limit: Transformers loading limit in percentage.
    :param _flex_limit: Maximum flexibility available to the dso in percentage.
    :return dso_flex: Dataframe with the information about the lines congested."""
    return True


def flex_needs_trafos_2w(_net, _trafo_loading, _flex_limit, _overload_limit=100):
    """Function that evaluate the flexibility needs for overloaded transformers (2 windings).
    :param _net: Pandapower network.
    :param _trafo_loading: Dataframe with the results of the timeseries simulation of the trafos.
    :param _overload_limit: Transformers loading limit in percentage.
    :param _flex_limit: Maximum flexibility available to the dso in percentage.
    :return dso_flex: Dataframe with the information about the trafo 2 windings congested."""
    return True


def flex_needs_trafos_3w(_net, _trafo_loading, _flex_limit, _overload_limit=100):
    """Function that evaluate the flexibility needs for overloaded transformers (3 windings).
    :param _net: Pandapower network.
    :param _trafo_loading: Dataframe with the results of the timeseries simulation of the trafos.
    :param _overload_limit: Transformers loading limit in percentage.
    :param _flex_limit: Maximum flexibility available to the dso in percentage.
    :return dso_flex: Dataframe with the information about the trafo 3 windings congested."""
    return True


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
    return True
