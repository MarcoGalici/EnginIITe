import os
import shutil
import copy as cp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from functions import file_io as io_file
from pandapower.timeseries import DFData
from pandapower.control import ConstControl
from pandapower.timeseries.output_writer import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries


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

    _outputs = {key: val for key, val in kwargs.items()}
    if _all_steps:
        t_steps = range(_time_steps[0], _time_steps[1])
    else:
        t_steps = _time_steps

    _elements = None
    _elements_idx = None
    _vars = None
    for _k in _dict_profiles.keys():
        ds = DFData(_dict_profiles[_k])
        if 'p' == _k.split('_')[-1]:
            _vars = 'p_mw'
            if 'load' in _k:
                _elements = 'load'
                _elements_idx = _pp_net.load.index
            elif 'sgen' in _k:
                _elements = 'sgen'
                _elements_idx = _pp_net.sgen.index
            elif 'gen' in _k:
                _elements = 'gen'
                _elements_idx = _pp_net.gen.index
            elif 'storage' in _k:
                _elements = 'storage'
                _elements_idx = _pp_net.storage.index
        elif 'q' == _k.split('_')[-1]:
            _vars = 'q_mvar'
            if 'load' in _k:
                _elements = 'load'
                _elements_idx = _pp_net.load.index
            elif 'sgen' in _k:
                _elements = 'sgen'
                _elements_idx = _pp_net.sgen.index
            elif 'gen' in _k:
                _elements = 'gen'
                _elements_idx = _pp_net.gen.index
            elif 'storage' in _k:
                _elements = 'storage'
                _elements_idx = _pp_net.storage.index
        try:
            ConstControl(_pp_net, element=_elements, variable=_vars, data_source=ds, element_index=_elements_idx,
                         profile_name=_dict_profiles[_k].columns, recycle=False)
        except Exception as _ex:
            raise UserWarning(type(_ex).__name__, _ex)

    _create_output_writer(_pp_net, t_steps, _out_root=_out_root, _output_file_type=_out_filetype, _res=_outputs)

    print('\n----- Time Series Calculation -----')
    run_timeseries(_pp_net, t_steps, continue_on_divergence=False)

    _res_dict = read_timeseries(_out_root=_out_root, _res=_outputs)
    return _res_dict


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

    ow = OutputWriter(_net, _t_steps, _out_root, _output_file_type, write_time=None, log_variables=list(),
                      csv_separator=',')

    _results_dict = kwargs.get('_res', {})
    for key, val in _results_dict.items():
        for _values in val:
            ow.log_variable(key, _values)
    return ow


def read_timeseries(_out_root, **kwargs):
    """
    Read results of the timeseries analysis.

    :param _out_root: Output root in which the files are saved
    :return: Dictionary with the results
    """

    _results_dict = kwargs.get('_res', {})
    _res_dict = dict()
    for key, val in _results_dict.items():
        for _values in val:
            _key_page = key.split('_')[-1] + '_' + _values
            try:
                _file2read = os.path.join(_out_root, key, _values + '.xlsx')
                _res_dict[_key_page] = pd.read_excel(_file2read, index_col=0)
            except FileNotFoundError:
                _file2read = os.path.join(_out_root, key, _values + '.csv')
                _res_dict[_key_page] = pd.read_csv(_file2read, index_col=0)
                _res_dict[_key_page].columns = _res_dict[_key_page].columns.astype(np.int64)
    return _res_dict


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

    _outputs = {key: val for key, val in kwargs.items()}
    t_steps = range(_time_step, _time_step + 1)

    _elements = None
    _elements_idx = None
    _vars = None
    for _k in _dict_profiles.keys():
        ds = DFData(_dict_profiles[_k])
        if 'p' == _k.split('_')[-1]:
            _vars = 'p_mw'
            if 'load' in _k:
                _elements = 'load'
                _elements_idx = _pp_net.load.index
            elif 'sgen' in _k:
                _elements = 'sgen'
                _elements_idx = _pp_net.sgen.index
            elif 'gen' in _k:
                _elements = 'gen'
                _elements_idx = _pp_net.gen.index
            elif 'storage' in _k:
                _elements = 'storage'
                _elements_idx = _pp_net.storage.index
        elif 'q' == _k.split('_')[-1]:
            _vars = 'q_mvar'
            if 'load' in _k:
                _elements = 'load'
                _elements_idx = _pp_net.load.index
            elif 'sgen' in _k:
                _elements = 'sgen'
                _elements_idx = _pp_net.sgen.index
            elif 'gen' in _k:
                _elements = 'gen'
                _elements_idx = _pp_net.gen.index
            elif 'storage' in _k:
                _elements = 'storage'
                _elements_idx = _pp_net.storage.index
        try:
            ConstControl(_pp_net, element=_elements, variable=_vars, data_source=ds, element_index=_elements_idx,
                         profile_name=_dict_profiles[_k].columns, recycle=False)
        except Exception as _ex:
            raise UserWarning(type(_ex).__name__, _ex)

    _final_out_root = _out_root + '_snapshot'
    _create_output_writer(_pp_net, t_steps, _out_root=_final_out_root, _output_file_type=_out_filetype, _res=_outputs)

    run_timeseries(_pp_net, t_steps, continue_on_divergence=False)

    _res_dict = read_timeseries(_out_root=_final_out_root, _res=_outputs)

    # # Remove additional directory
    # for key in tqdm(_outputs.keys(), desc='Removing additional directories'):
    #     shutil.rmtree(os.path.join(_final_out_root, key))
    shutil.rmtree(_final_out_root)

    return _res_dict


def plot_timeseries(_df2plot, _ylabel=None, _title=None, _figsize=(10, 5)):
    """
    Plot results of timeseries analysis

    :param _df2plot: Dataframe with results to plot
    :param _ylabel: Y-axis label name
    :param _title: Title of the plot
    :param _figsize: Size of the window diplaying the plot
    """
    plt.figure(figsize=_figsize)
    plt.plot(_df2plot)
    plt.xlabel('Timestep')
    plt.ylabel(_ylabel)
    plt.title(_title)
    plt.ylim([0, 150])
    plt.grid()
    plt.show()
    return True


def timeseries_powerflow_v2(_net0, time_steps, dfL_p, dfL_q, dfG_p, dfG_q, output_dir, output_file_type, ForceTS,
                            plot=False, _all_steps=True):
    """Run Timeseries power flow from pandapower."""
    # Create copy of the original network
    net = cp.deepcopy(_net0)
    _file2check = 'vm_pu' + output_file_type
    if not os.path.isfile(os.path.join(output_dir, 'res_bus', _file2check)) or ForceTS:
        print('\n------------------------- Time Series calculation -------------------------')
        # 1. Create data sources from profiles inputs
        # It converts the Dataframe to the required format for the controllers
        dsL_p = DFData(dfL_p)
        dsL_q = DFData(dfL_q)
        dsG_p = DFData(dfG_p)
        dsG_q = DFData(dfG_q)

        # 2. Create controllers (to control P and Q values of load and sgen)
        ConstControl(net, element='load', variable='p_mw', data_source=dsL_p, element_index=net.load.index,
                     profile_name=net.load.index, recycle=False)

        ConstControl(net, element='load', variable='q_mvar', data_source=dsL_q, element_index=net.load.index,
                     profile_name=net.load.index, recycle=False)

        ConstControl(net, element='sgen', variable='p_mw', data_source=dsG_p, element_index=net.sgen.index,
                     profile_name=net.sgen.index, recycle=False)

        ConstControl(net, element='sgen', variable='q_mvar', data_source=dsG_q, element_index=net.sgen.index,
                     profile_name=net.sgen.index, recycle=False)

        # 3. Create the output writer, check complementary function 1 of this script
        if _all_steps:
            t_steps = range(time_steps[0], time_steps[-1])
        else:
            t_steps = time_steps

        # The output writer with the desired results to be stored to files
        _ = create_output_writer(net, t_steps, output_dir, output_file_type)

        # 4. Execute timeseries power flow according to the previous steps
        run_timeseries(net, t_steps, continue_on_divergence=False, desc='Running Timeseries')
    else:
        print('\n----------------- WARNING: Using Previous Timeseries Results -----------------')

    # 5. Read the timeseries results
    # Lines
    ll_file = os.path.join(output_dir, 'res_line', 'loading_percent' + output_file_type)
    # Transformer 2 Windings
    tt_file = os.path.join(output_dir, 'res_trafo', 'loading_percent' + output_file_type)
    # Transformer 3 Windings
    tt3w_file = os.path.join(output_dir, 'res_trafo3w', 'loading_percent' + output_file_type)
    # Bus
    bb_file = os.path.join(output_dir, 'res_bus', 'vm_pu' + output_file_type)

    if output_file_type == '.xlsx':
        # Line loading results
        line_loading = pd.read_excel(ll_file, index_col=0)
        # Trafo loading results
        trafo_loading = pd.read_excel(tt_file, index_col=0)
        # Trafo3w loading results
        trafo3w_loading = pd.read_excel(tt3w_file, index_col=0)
        # Bus voltage results
        bus_voltages = pd.read_excel(bb_file, index_col=0)

    elif output_file_type == '.csv':
        # Line loading results
        line_loading = pd.read_csv(ll_file, index_col=0)
        # Trafo loading results
        trafo_loading = pd.read_csv(tt_file, index_col=0)
        # Trafo3w loading results
        trafo3w_loading = pd.read_csv(tt3w_file, index_col=0)
        # Bus voltage results
        bus_voltages = pd.read_csv(bb_file, index_col=0)

    else:
        raise UserWarning('Output file type {_filetype} not recognize.'.format(_filetype=output_file_type))

    # 6. Plot the results if plot=1
    if plot:
        figsize = (10, 5)
        _ylim = [0, 150]
        # Line loading results
        plt.figure(figsize=figsize)
        plt.plot(line_loading)
        plt.xlabel("time step")
        plt.ylabel("line loading [%]")
        plt.title("Line Loading")
        plt.ylim(_ylim)
        plt.grid()
        plt.show()

        if len(net.trafo.index) > 0:
            # Trafo 2w loading results
            plt.figure(figsize=figsize)
            plt.plot(trafo_loading)
            plt.xlabel("time step")
            plt.ylabel("trafo loading [%]")
            plt.title("Trafo Loading")
            plt.ylim(_ylim)
            plt.grid()
            plt.show()

        if len(net.trafo3w.index) > 0:
            # Trafo 3w loading results
            plt.figure(figsize=figsize)
            plt.plot(trafo3w_loading)
            plt.xlabel("time step")
            plt.ylabel("trafo 3w loading [%]")
            plt.title("Trafo 3w Loading")
            plt.ylim(_ylim)
            plt.grid()
            plt.show()

        # Bus voltage results
        plt.figure(figsize=figsize)
        plt.plot(bus_voltages)
        plt.xlabel("time step")
        plt.ylabel("Bus Voltage [p.u]")
        plt.title("Bus Voltage")
        plt.grid()
        plt.show()

    return line_loading, trafo_loading, trafo3w_loading, bus_voltages


def create_output_writer(net, t_steps, output_dir, output_file_type):
    """Initialising the output writer to save data to Excel files in the current folder"""
    ow = OutputWriter(net, t_steps, output_dir, output_file_type, write_time=None, log_variables=list(),
                      csv_separator=",")
    # Select the outputs to save in the output writer
    # Loads
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_load', 'q_mvar')
    # Static Generators
    ow.log_variable('res_sgen', 'p_mw')
    ow.log_variable('res_sgen', 'q_mvar')
    # Bus
    ow.log_variable('res_bus', 'vm_pu')
    # Lines
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'p_from_mw')
    ow.log_variable('res_line', 'p_to_mw')
    ow.log_variable('res_line', 'q_from_mvar')
    ow.log_variable('res_line', 'q_to_mvar')
    # Transformers 2 Windings
    ow.log_variable('res_trafo', 'loading_percent')
    ow.log_variable('res_trafo', 'p_hv_mw')
    ow.log_variable('res_trafo', 'p_lv_mw')
    ow.log_variable('res_trafo', 'q_hv_mvar')
    ow.log_variable('res_trafo', 'q_lv_mvar')
    # Transformer 3 windings
    ow.log_variable('res_trafo3w', 'loading_percent')
    ow.log_variable('res_trafo3w', 'p_hv_mw')
    ow.log_variable('res_trafo3w', 'p_mv_mw')
    ow.log_variable('res_trafo3w', 'p_lv_mw')
    ow.log_variable('res_trafo3w', 'q_hv_mvar')
    ow.log_variable('res_trafo3w', 'q_mv_mvar')
    ow.log_variable('res_trafo3w', 'q_lv_mvar')
    return ow


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
    _net_post = cp.deepcopy(_net)
    _fsp_load_p = cp.deepcopy(_ts_load_p)
    _fsp_load_q = cp.deepcopy(_ts_load_q)
    _fsp_sgen_p = cp.deepcopy(_ts_sgen_p)
    _fsp_sgen_q = cp.deepcopy(_ts_sgen_q)
    for _fsp_type in _fsp_file.keys():
        if _fsp_file[_fsp_type] != 0:
            filename_fsp = 'FSP' + _fsp_type + '_init' + _simulation_tag + '.xlsx'
            FSPinfo = io_file.import_data(_filename=filename_fsp, _root=_root_fsp, _sheetname='FSPinfo', _index_col=0)

            filename_res = 'model_results_' + _simulation_tag + '.xlsx'
            FSP_P = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname='W_' + _fsp_type, _index_col=0)
            FSP_Q = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname='R_' + _fsp_type, _index_col=0)
            for i in tqdm(FSPinfo.index, desc='FSP {_letter} processing'.format(_letter=_fsp_type)):
                if _fsp_type == 'A' or _fsp_type == 'C':
                    if _fsp_type == 'A':
                        # Loads
                        index_res = _net_post.load.index[_net_post.load.bus == FSPinfo.bus[i]]
                        index_name_res = _net_post.load.index[_net_post.load.name == FSPinfo.name[i]]
                    else:
                        # Batteries
                        index_res = _net_post.load.index[_net_post.load.bus == FSPinfo.bus[i]]
                        index_name_res = index_res
                    index_res = index_res.intersection(index_name_res)
                    for h in _hours2study:
                        # fixme: Questa metodologia funziona se in ogni bus abbiamo una sola risorsa fsp per categoria.
                        #  Nel nostro modello storage e load sono considerati come "load" quindi se in un caso avessimo
                        #  sia carico che storage dobbiamo fare la somma dei profili (change_net_fsp_v1). Tuttavia se
                        #  avessimo due risorse fsp della stessa categoria (esempio: load & heatpump considerati
                        #  entrambi load) nello stesso bus allora non dovremmo farlo perchè i profili sono per
                        #  categoria di risorsa; infatti tali risorse avrebbero ognuna la propria colonna (indice).
                        #  Bisogna valutare se esiste la possibilità che si possano sviluppare casistiche in cui sullo
                        #  stesso nodo ci siano due risorse della stessa categoria e stesso modello
                        #  (esempio: load & load; ma non come load & heatpump).
                        _fsp_load_p.loc[index_res, h] = FSP_P.loc[i, h]
                        _fsp_load_q.loc[index_res, h] = FSP_Q.loc[i, h]
                else:
                    # Generators
                    index_res = _net_post.sgen.index[_net_post.sgen.bus == FSPinfo.bus[i]]
                    index_name_res = _net_post.sgen.index[_net_post.sgen.name == FSPinfo.name[i]]
                    index_res = index_res.intersection(index_name_res)
                    for h in _hours2study:
                        _fsp_sgen_p.loc[index_res, h] = FSP_P.loc[i, h]
                        _fsp_sgen_q.loc[index_res, h] = FSP_Q.loc[i, h]

        else:
            print('No FSP {_letter} type identified.'.format(_letter=_fsp_type))
    return _fsp_load_p, _fsp_load_q, _fsp_sgen_p, _fsp_sgen_q


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
    _net_post = cp.deepcopy(_net)
    _fsp_load_p = cp.deepcopy(_ts_load_p)
    _fsp_load_q = cp.deepcopy(_ts_load_q)
    _fsp_sgen_p = cp.deepcopy(_ts_sgen_p)
    _fsp_sgen_q = cp.deepcopy(_ts_sgen_q)
    # This parameter check if in a single bus are available more than one resource.
    # If yes, the profiles from the second resource in the bus are sum to the first profile of the resource
    # on the bus.
    load_bus_1st = [[0, 0] for _ in _net.bus.index]
    sgen_bus_1st = [[0, 0] for _ in _net.bus.index]
    for _b_load in _net.load.bus:
        load_bus_1st[_b_load][0] += 1
        load_bus_1st[_b_load][1] += 1
    for _b_sgen in _net.sgen.bus:
        sgen_bus_1st[_b_sgen][0] += 1
        sgen_bus_1st[_b_sgen][1] += 1
    for _b_storage in _net.storage.bus:
        load_bus_1st[_b_storage][0] += 1
        load_bus_1st[_b_storage][1] += 1

    for _fsp_type in _fsp_file.keys():
        sheet2read_p = 'W_'
        sheet2read_q = 'R_'
        if _fsp_file[_fsp_type] != 0:
            filename_fsp = 'FSP' + _fsp_type + '_init' + _simulation_tag + '.xlsx'
            FSPinfo = io_file.import_data(_filename=filename_fsp, _root=_root_fsp, _sheetname='FSPinfo', _index_col=0)

            filename_res = 'model_results_' + _simulation_tag + '.xlsx'
            if _fsp_type == 'C':
                # For Storage
                sheet2read_p = 'DW_'
                sheet2read_q = 'DR_'
            FSP_P = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname=sheet2read_p + _fsp_type, _index_col=0)
            FSP_Q = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname=sheet2read_q + _fsp_type, _index_col=0)
            for i in tqdm(FSPinfo.index, desc='FSP {_letter} processing'.format(_letter=_fsp_type)):
                if _fsp_type == 'A' or _fsp_type == 'C':
                    if _fsp_type == 'A':
                        # Loads
                        index_res_bus = _net_post.load.index[_net_post.load.bus == FSPinfo.bus[i]]
                        index_name_res = _net_post.load.index[_net_post.load.name == FSPinfo.name[i]]
                    else:
                        # Batteries
                        index_res_bus = _net_post.load.index[_net_post.load.bus == FSPinfo.bus[i]]
                        index_name_res = index_res_bus
                    index_res = index_res_bus.intersection(index_name_res)
                    for h in _hours2study:
                        if load_bus_1st[FSPinfo.bus[i]][0] - load_bus_1st[FSPinfo.bus[i]][1] == 0:
                            _fsp_load_p.loc[index_res, h] = FSP_P.loc[i, h]
                            _fsp_load_q.loc[index_res, h] = FSP_Q.loc[i, h]
                        else:
                            _fsp_load_p.loc[index_res, h] += FSP_P.loc[i, h]
                            _fsp_load_q.loc[index_res, h] += FSP_Q.loc[i, h]
                    load_bus_1st[FSPinfo.bus[i]][1] -= 1
                else:
                    # Generators
                    index_res_bus = _net_post.sgen.index[_net_post.sgen.bus == FSPinfo.bus[i]]
                    index_name_res = _net_post.sgen.index[_net_post.sgen.name == FSPinfo.name[i]]
                    index_res = index_res_bus.intersection(index_name_res)
                    for h in _hours2study:
                        if sgen_bus_1st[FSPinfo.bus[i]][0] - sgen_bus_1st[FSPinfo.bus[i]][1] == 0:
                            _fsp_sgen_p.loc[index_res, h] = FSP_P.loc[i, h]
                            _fsp_sgen_q.loc[index_res, h] = FSP_Q.loc[i, h]
                        else:
                            _fsp_sgen_p.loc[index_res, h] += FSP_P.loc[i, h]
                            _fsp_sgen_q.loc[index_res, h] += FSP_Q.loc[i, h]
                    sgen_bus_1st[FSPinfo.bus[i]][1] -= 1

        else:
            print('No FSP {_letter} type identified.'.format(_letter=_fsp_type))
            if _fsp_type == 'A' or _fsp_type == 'C':
                for _b_load in _net.load.bus:
                    load_bus_1st[_b_load][1] -= 1
            else:
                for _b_sgen in _net.sgen.bus:
                    sgen_bus_1st[_b_sgen][1] -= 1

    return _fsp_load_p, _fsp_load_q, _fsp_sgen_p, _fsp_sgen_q


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
    _net_post = cp.deepcopy(_net)
    dict_profiles_postmkt_all = {_k: cp.deepcopy(_value) for _k, _value in kwargs.items()}
    for _fsp_type in _fsp_file.keys():
        sheet2read_p = 'W_'
        sheet2read_q = 'R_'
        if _fsp_file[_fsp_type] != 0:
            if _fsp_type == 'A':
                _res_type = 'load'
                _net_res = cp.deepcopy(_net_post.load)
            elif _fsp_type == 'B':
                _res_type = 'sgen'
                _net_res = cp.deepcopy(_net_post.sgen)
            elif _fsp_type == 'C':
                _res_type = 'storage'
                _net_res = cp.deepcopy(_net_post.storage)
                sheet2read_p = 'DW_'
                sheet2read_q = 'DR_'
                # sheet2read_p and sheet2read_q sono definiti su DW e DR perché la batteria si basa sui dati iniziali
                # dei carichi sullo stesso nodo. Quindi il risultato di consumo/iniezione finale di energia al bus deve
                # essere basata su W_C batteria. Tuttavia in questo caso abbiamo diviso tra load e storage quindi la
                # batteria prende DW e DR
            else:
                raise UserWarning('FSP Type {_type} not defined.'.format(_type=_fsp_type))

            _fsp_res_p = dict_profiles_postmkt_all.get(_res_type + '_prof_p', None)
            _fsp_res_q = dict_profiles_postmkt_all.get(_res_type + '_prof_q', None)

            filename_fsp = 'FSP' + _fsp_type + '_init' + _simulation_tag + '.xlsx'
            FSPinfo = io_file.import_data(_filename=filename_fsp, _root=_root_fsp, _sheetname='FSPinfo', _index_col=0)

            filename_res = 'model_results_' + _simulation_tag + '.xlsx'
            FSP_P = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname=sheet2read_p + _fsp_type, _index_col=0)
            FSP_Q = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname=sheet2read_q + _fsp_type, _index_col=0)
            for i in tqdm(FSPinfo.index, desc='FSP {_letter} processing'.format(_letter=_fsp_type)):
                index_res = _net_res.index[_net_res.bus == FSPinfo.bus[i]]
                index_name_res = _net_res.index[_net_res.name == FSPinfo.name[i]]
                index_res = index_res.intersection(index_name_res)
                for h in _hours2study:
                    _fsp_res_p.loc[index_res, h] = FSP_P.loc[i, h]
                    _fsp_res_q.loc[index_res, h] = FSP_Q.loc[i, h]
        else:
            print('No FSP {_letter} type identified.'.format(_letter=_fsp_type))

    dict_profiles_postmkt = {_k: cp.deepcopy(_value.loc[:, _hours2study].T) for _k, _value in dict_profiles_postmkt_all.items()}
    return dict_profiles_postmkt


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
    _net_post = cp.deepcopy(_net)
    _fsp_load_p = cp.deepcopy(_ts_load_p)
    _fsp_load_q = cp.deepcopy(_ts_load_q)
    _fsp_sgen_p = cp.deepcopy(_ts_sgen_p)
    _fsp_sgen_q = cp.deepcopy(_ts_sgen_q)
    _fsp_storage_p = cp.deepcopy(_ts_storage_p)
    _fsp_storage_q = cp.deepcopy(_ts_storage_q)
    for _fsp_type in _fsp_file.keys():
        if _fsp_file[_fsp_type] != 0:
            sheet2read_p = 'W_'
            sheet2read_q = 'R_'
            if _fsp_file[_fsp_type] != 0:
                filename_fsp = 'FSP' + _fsp_type + '_init' + _simulation_tag + '.xlsx'
                FSPinfo = io_file.import_data(_filename=filename_fsp, _root=_root_fsp, _sheetname='FSPinfo', _index_col=0)

                filename_res = 'model_results_' + _simulation_tag + '.xlsx'
                if _fsp_type == 'C':
                    # For Storage
                    sheet2read_p = 'DW_'
                    sheet2read_q = 'DR_'
                FSP_P = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname=sheet2read_p + _fsp_type, _index_col=0)
                FSP_Q = io_file.import_data(_filename=filename_res, _root=_root_results, _sheetname=sheet2read_q + _fsp_type, _index_col=0)
                for i in tqdm(FSPinfo.index, desc='FSP {_letter} processing'.format(_letter=_fsp_type)):
                    if _fsp_type == 'A':
                        # Loads
                        index_res_bus = _net_post.load.index[_net_post.load.bus == FSPinfo.bus[i]]
                        index_name_res = _net_post.load.index[_net_post.load.name == FSPinfo.name[i]]
                        index_res = index_res_bus.intersection(index_name_res)
                        for h in _hours2study:
                            _fsp_load_p.loc[index_res, h] = FSP_P.loc[i, h]
                            _fsp_load_q.loc[index_res, h] = FSP_Q.loc[i, h]
                    elif _fsp_type == 'B':
                        # Generators
                        index_res_bus = _net_post.sgen.index[_net_post.sgen.bus == FSPinfo.bus[i]]
                        index_name_res = _net_post.sgen.index[_net_post.sgen.name == FSPinfo.name[i]]
                        index_res = index_res_bus.intersection(index_name_res)
                        for h in _hours2study:
                            _fsp_sgen_p.loc[index_res, h] = FSP_P.loc[i, h]
                            _fsp_sgen_q.loc[index_res, h] = FSP_Q.loc[i, h]
                    elif _fsp_type == 'C':
                        # Batteries
                        index_res_bus = _net_post.storage.index[_net_post.storage.bus == FSPinfo.bus[i]]
                        index_name_res = _net_post.storage.index[_net_post.storage.name == FSPinfo.name[i]]
                        index_res = index_res_bus.intersection(index_name_res)
                        for h in _hours2study:
                            _fsp_storage_p.loc[index_res, h] = FSP_P.loc[i, h]
                            _fsp_storage_q.loc[index_res, h] = FSP_Q.loc[i, h]
        else:
            print('No FSP {_letter} type identified.'.format(_letter=_fsp_type))

    return _fsp_load_p, _fsp_load_q, _fsp_sgen_p, _fsp_sgen_q, _fsp_storage_p, _fsp_storage_q


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

    if _flag_vpilot:
        io_file.save_cvs(_data=_bus_voltages.T, _outroot=_out_root, _filename='VPilotBus_FULLH_FULLBUS')

    # Slack bus index
    _slack = list(_net.ext_grid.bus)
    # Index of not in service buses (sort automatically, if possible)
    _inactive_bus = list(_net.bus.index[_net.bus.in_service == False])
    # Union of "slack bus index" and "index of not in service buses"
    _drop_idx = _inactive_bus + _slack
    # List of index of bus that are over or under voltage of the limits
    _bus_outbounds = [x for x in np.unique(_vbus_outbounds.Bus)]

    if _flag_only_mv:
        _idx_bus_not_mv = _net.bus.index[_net.bus.vn_kv != _vn_kv_mv].to_list()
        # fixme: Se i bus con violazione dei vincoli fossero non MV, è giusto che non vengano considerati come pilotbus?
        _ok_bus = list(set(_bus_outbounds) - set(_drop_idx) - set(_idx_bus_not_mv))

        # fixme: ho provato così, verificare con Matteo se è giusto.
        #  (potrebbe avere senso, ma il codice non funziona con questi calcoli
        #   - errore nel calcolo delle matrici di sensitività per Voltage Control)
        # drop_bus = set(IndexDrop) - set(IndexBusNOTMV)
        # Ok_bus = list(set(bus_outbounds) - drop_bus)
        # print('\nNot MV nodes removed')
    else:
        _ok_bus = list(set(_bus_outbounds) - set(_drop_idx))

    fsp_bus = list(_fsp_data.bus)

    _pilot_bus = _ok_bus + fsp_bus
    _pilot_bus_hts = _bus_voltages.loc[_hours, _pilot_bus]
    _pilot_bus_hts.to_csv(os.path.join(_out_root, 'VPilotBus_HTS.csv'), header=True, index=True)
    # only for retro-compatibility
    _pilot_bus_hts.to_csv(os.path.join(_out_root, 'VPilotBus24h.csv'), header=True, index=True)
    return _pilot_bus, _pilot_bus_hts


def eval_worst_hour_bus(_df, _vmax=1.05, _vmin=0.95):
    """
    Evaluate the worst voltage, node and hour of the pilot bus voltages.

    :param _df: Dataframe voltage of the pilot bus.

    :param _vmax: Maximum voltage.

    :param _vmin: Minimum voltage.
    """
    df = _df.fillna(value=0)
    dv_node = pd.DataFrame(np.where(df >= _vmax, df - _vmax, 0)) + pd.DataFrame(np.where(df <= _vmin, _vmin - df, 0))
    worst_val = dv_node.values.max()
    # coordinates of all maximum value
    max_coords = list(zip(*np.where(dv_node.values == worst_val)))
    # row and col names corresponding to each coord
    max_names = [(dv_node.index[r], dv_node.columns[c]) for r, c in max_coords]
    worst_val = worst_val
    node = df.index[max_names[0][0]]
    hour = df.columns[max_names[0][1]]
    # TODO: add a control that sends a worst_val==empty,
    #  this implies major changes in the code that follows this function
    if worst_val == 0:
        print("WARNING: NO VOLTAGE PROBLEMS DETECTED."
              " The worst hour for voltage control is set to HoursToStudy[0], that is :", df.columns[0])
    return worst_val, int(node), int(hour)


def eval_worst_hour_lines(_df, _imax=100):
    """
    Evaluate the worst loading percent, line and hour of the line loading percents.

    :param _df: Dataframe of loading percentages of the lines.

    :param _imax: Maximum loading percent.
    """
    df_CM = pd.DataFrame(np.clip(_df - _imax, 0, None))
    df_CM.index = _df.index
    df_CM.columns = _df.columns
    oveload_lines_perc = df_CM.loc[(df_CM != 0).any(axis=1)]
    oveload_lines_perc = oveload_lines_perc.fillna(value=0)
    if not oveload_lines_perc.empty:
        worst_val = float(oveload_lines_perc.values.max())
        # coordinates of all maximum value
        max_coords = list(zip(*np.where(oveload_lines_perc.values == worst_val)))
        # row and col names corresponding to each coord
        max_names = [(oveload_lines_perc.index[r], oveload_lines_perc.columns[c]) for r, c in max_coords]
        # fixme: why we check df_Cm if we already know the line and the hour from max_names?
        # Row
        line = int(df_CM.index[int(max_names[0][0])])
        # Column
        hour = int(list(df_CM.columns[df_CM.columns == int(max_names[0][1])])[0])
    else:
        worst_val = -1  # imposed -1 to avoid errors in the comparison with 0
        line = None
        hour = None
    return worst_val, line, hour
