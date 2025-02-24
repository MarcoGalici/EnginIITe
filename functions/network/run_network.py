def run_pf_timeseries(_pp_net, _dict_profiles, _time_steps, _out_root, _out_filetype, _all_steps, **kwargs):
    """
    Run timeseries power flow from pandapower

    :param _pp_net: Pandapower network format
    :param _dict_profiles: Dictionary with the dataframe of the profiles
    :param _time_steps: Number of interval to be evaluated
    :param _out_root: Output directory
    :param _out_filetype: Format file type in which the timeseries results are saved
    :param _all_steps: Boolean value. If True evaluate all the steps from _time_steps[0] to _time_steps[-1].
                        If False evaluate only the steps included in _time_steps.
    :return _res_dict: Dictionary with the results set as input.

    EXAMPLE:
        import pandapower.networks as pn

        net = pn.case14()


        dict_profiles = {

        'load_prof_p': pd.Dataframe(shape=(_time_steps, net.load.index)),

        'load_prof_q': pd.Dataframe(shape=(_time_steps, net.load.index)),

        'sgen_prof_q': pd.Dataframe(shape=(_time_steps, net.sgen.index))
        }


        run_pf_timeseries(net, _dict_profiles, _time_steps, _out_root, _out_filetype, res_load=['p_mw', 'q_mvar'], res_sgen='q_mvar')
    """
    return True


def _create_output_writer(_net, _t_steps, _out_root='data_out', _output_file_type='.xlsx', **kwargs):
    """
    Initialise the Output Writer to save data into files.

    :param _net: Pandapower network
    :param _t_steps: Time steps of the simulation
    :param _out_root: Output directory
    :param _output_file_type: File format in which the results are saved
    :return: OutputWriter class

    EXAMPLE:
        create_output_writer(_net, _t_steps, 'my_result_folder', '.csv', res_load=['p_mw', 'q_mvar'], res_sgen=['q_mvar'])
    """
    return True


def read_timeseries(_out_root, **kwargs):
    """
    Read results of the timeseries analysis.

    :param _out_root: Output root in which the files are saved
    :return: Dictionary with the results
    """
    return True


def run_pf_timeseries_snapshot(_pp_net, _dict_profiles, _time_step, _out_root, _out_filetype, **kwargs):
    """
    Run a snapshot power flow from pandapower using timeseries engine.

    :param _pp_net: Pandapower network format
    :param _dict_profiles: Dictionary with the numpy array or pandas Series of the elements of the network
    :param _time_step: Single interval to be evaluated
    :param _out_root: Output directory
    :param _out_filetype: Format file type in which the timeseries results are going to be saved
    :return _res_dict: Dictionary with the results set in the input.

    EXAMPLE:
        import pandapower.networks as pn

        net = pn.case14()


        dict_profiles = {

        'load_p': np.array(, net.load.index),

        'load_q': np.array(, net.load.index),

        'sgen_q': pd.Series(, net.sgen.index)
        }


        run_pf_timeseries(net, _profiles, _time_step, _out_root, _out_filetype, res_load=['p_mw', 'q_mvar'], res_sgen='q_mvar')
        """
    return True


def plot_timeseries(_df2plot, _ylabel=None, _title=None, _figsize=(10, 5)):
    """
    Plot results of timeseries analysis

    :param _df2plot: Dataframe with results to plot
    :param _ylabel: Y-axis label name
    :param _title: Title of the plot
    :param _figsize: Size of the window diplaying the plot
    """
    return True


def timeseries_powerflow_v2(_net0, time_steps, dfL_p, dfL_q, dfG_p, dfG_q, output_dir, output_file_type, ForceTS,
                            plot=False, _all_steps=True):
    """Run Timeseries power flow from pandapower."""
    return True


def create_output_writer(net, t_steps, output_dir, output_file_type):
    """Initialising the output writer to save data to Excel files in the current folder"""
    return True


def change_net_fsp(_net, _hours2study, _fsp_file, _ts_load_p, _ts_load_q, _ts_sgen_p, _ts_sgen_q, _root_fsp,
                   _root_results, _simulation_tag):
    """
    Adapt network loads and generators profiles according to the results of the market.

    :param _net: Pandapower network.

    :param _hours2study: List of hours to study that present network constraint violations.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _ts_load_p: Timeseries results of active power for each load bus.

    :param _ts_load_q: Timeseries results of reactive power for each load bus.

    :param _ts_sgen_p: Timeseries results of active power for each sgen bus.

    :param _ts_sgen_q: Timeseries results of reactive power for each sgen bus.

    :param _root_fsp: Folder to the input fsp file.

    :param _root_results: Folder to the results of the market.

    :param _simulation_tag: Simulation tag.
    """
    return True


def change_net_fsp_v1(_net, _hours2study, _fsp_file, _ts_load_p, _ts_load_q, _ts_sgen_p, _ts_sgen_q,
                      _root_fsp, _root_results, _simulation_tag):
    """
    Adapt network loads and generators profiles according to the results of the market.

    :param _net: Pandapower network.

    :param _hours2study: List of hours to study that present network constraint violations.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _ts_load_p: Timeseries results of active power for each load bus.

    :param _ts_load_q: Timeseries results of reactive power for each load bus.

    :param _ts_sgen_p: Timeseries results of active power for each sgen bus.

    :param _ts_sgen_q: Timeseries results of reactive power for each sgen bus.

    :param _root_fsp: Folder to the input fsp file.

    :param _root_results: Folder to the results of the market.

    :param _simulation_tag: Simulation tag.
    """
    return True


def change_net_fsp_v2(_net, _hours2study, _fsp_file, _root_fsp, _root_results, _simulation_tag, **kwargs):
    """
    Adapt network loads and generators profiles according to the results of the market.
    (Version 2: Storage resource include in the resource representation).

    :param _net: Pandapower network.

    :param _hours2study: List of hours to study that present network constraint violations.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _root_fsp: Folder to the input fsp file.

    :param _root_results: Folder to the results of the market.

    :param _simulation_tag: Simulation tag.
    """
    return True


def change_net_fsp_v3(_net, _hours2study, _fsp_file, _ts_load_p, _ts_load_q, _ts_sgen_p, _ts_sgen_q, _ts_storage_p,
                      _ts_storage_q, _root_fsp, _root_results, _simulation_tag):
    """
    Adapt network loads and generators profiles according to the results of the market.
    (Version 2: Storage resource include in the resource representation).

    :param _net: Pandapower network.

    :param _hours2study: List of hours to study that present network constraint violations.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _ts_load_p: Timeseries results of active power for each load bus.

    :param _ts_load_q: Timeseries results of reactive power for each load bus.

    :param _ts_sgen_p: Timeseries results of active power for each sgen bus.

    :param _ts_sgen_q: Timeseries results of reactive power for each sgen bus.

    :param _ts_storage_p: Timeseries results of active power for each storage bus.

    :param _ts_storage_q: Timeseries results of reactive power for each storage bus.

    :param _root_fsp: Folder to the input fsp file.

    :param _root_results: Folder to the results of the market.

    :param _simulation_tag: Simulation tag.
    """
    return True


def select_pilot_bus(_net, _bus_voltages, _vbus_outbounds, _fsp_data, _hours, _out_root, _vn_kv_mv=15,
                     _flag_vpilot=False, _flag_only_mv=True):
    """Pilot busses selection. Defining the table of Voltage Magnitudes for all active busses and hours to study.
     Slack busses and FALSE busses are eliminated from the pilot buses; only the fsps busses and the busses with
     problems in the HoursToStudy are selected (not true). V pilots are the busses with FSPs and the one with problems.

     :param _net: Pandapower network.

     :param _bus_voltages: Dataframe with bus voltage during the hours considered.

     :param _vbus_outbounds: Dataframe with the buses that are with over and under voltage problems.

     :param _fsp_data: Dataframe with the fsp data involved in the simulation.

     :param _hours: List of hours with network constraint violations.

     :param _out_root: Output root where to save files.

     :param _vn_kv_mv: Reference voltage for Medium Voltage buses.

     :param _flag_vpilot: Boolean value for saving into csv file the bus voltages.

     :param _flag_only_mv: Boolean value. Consider only medium voltage buses.
     """
    return True


def eval_worst_hour_bus(_df, _vmax=1.05, _vmin=0.95):
    """
    Evaluate the worst voltage, node and hour of the pilot bus voltages.

    :param _df: Dataframe voltage of the pilot bus.

    :param _vmax: Maximum voltage.

    :param _vmin: Minimum voltage.
    """
    return True


def eval_worst_hour_lines(_df, _imax=100):
    """
    Evaluate the worst loading percent, line and hour of the line loading percents.

    :param _df: Dataframe of loading percentages of the lines.

    :param _imax: Maximum loading percent.
    """
    return True
