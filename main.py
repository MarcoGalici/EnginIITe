import os
import time
import copy as cp
import numpy as np
import pandas as pd
import functions.auxiliary as aux
import functions.file_io as io_file
import functions.kpis.eval_kpis as kpi_fcn
import functions.network.flex_eval as feval
import functions.network.run_network as rnet
import functions.market.run_flex_market as flx_mrk
import functions.network.sensitivity_factors as calc_sens


if __name__ == '__main__':
    start_time = time.time()
    exit()

    file_cfg_name = 'config_Portugal_scen1.yaml'

    # Import Yaml Config File
    cfg_file = io_file.import_yaml(_yaml_filename=file_cfg_name, _folder='config')
    simulation_tag = cfg_file['simulation_tag']

    # Define directories
    path_dict = aux.set_general_path(_main_root=os.getcwd(), _name_network=cfg_file['network']['net_folder'],
                                     _model_tag=cfg_file['ModelTAG'], _ess_tag=cfg_file['StorageTag'],
                                     _sim_tag=simulation_tag, _in_root='data_in', _out_root='data_out')

    # Define Parameters
    params_dict = aux.set_general_params(_cfg_file=cfg_file, _path_dict=path_dict, _ess_tag=cfg_file['StorageTag'])

    # Load Pandapower Network
    file_net = os.path.join(path_dict['net_in_root'], cfg_file['network']['net_filename'])
    net = io_file.import_network(cfg_file['network'])
    net_post_pre = cp.deepcopy(net)

    # Load FSP Data
    fsp_file = os.path.split(params_dict['fsp_file'])[1]
    fsp_data = io_file.import_data(_filename=fsp_file, _root=path_dict['fsp_in_root'], _header=1,
                                   _sheetname=cfg_file['network']['fsp']['fsp_sheetname'])

    # Import Generator/Consumer Profiles
    dict_profiles_s01 = io_file.import_profiles_s01(_cfg_file=cfg_file, _pp_net=net, _timestep=params_dict['timestep'])

    # Run Timeseries Power Flow
    ts_results = rnet.run_pf_timeseries(_pp_net=net, _dict_profiles=dict_profiles_s01, _time_steps=params_dict['timestep'],
                                        _out_root=path_dict['ts_out_root'], _out_filetype=params_dict['ts_output_file_type'],
                                        _all_steps=True, res_bus=['vm_pu'], res_line=['loading_percent'], res_load=['p_mw', 'q_mvar'],
                                        res_sgen=['p_mw', 'q_mvar'], res_storage=['p_mw', 'q_mvar'],
                                        res_trafo=['loading_percent'], res_trafo3w=['loading_percent'])

    for vl in cfg_file['VLTAG']:
        for prod in cfg_file['ProductTAG']:
            for fsp in cfg_file['FspTAG']:
                net_post = cp.deepcopy(net_post_pre)
                print('\n----- Simulating Scenario: {a} {b} {c} {d} {e} -----'
                      .format(a=cfg_file['ModelTAG'], b=prod, c=vl, d=fsp, e=cfg_file['StorageTag']))

                path_dict = aux.set_sim_path(_gpath=path_dict, _model_tag=cfg_file['ModelTAG'],
                                             _product_tags=cfg_file['ProductTAG'], _vl_tags=cfg_file['VLTAG'],
                                             _fsp_tags=cfg_file['FspTAG'], _ess_tag=cfg_file['StorageTag'],
                                             _sim_tag=simulation_tag)
                output_root = path_dict['mrk_outputs']['res_market'][prod][vl][fsp]
                output_ts_root = path_dict['mrk_outputs']['ts_postmrk_out_root'][prod][vl][fsp]
                input_root = path_dict['mrk_inputs'][prod][vl][fsp]
                market_tag = path_dict['simulation_tag'][prod][vl][fsp]

                params_dict = aux.set_sim_params(_gparams=params_dict, _vl_tags=cfg_file['VLTAG'],
                                                 _fsp_tags=cfg_file['FspTAG'])
                v_max = params_dict['vbus_max'][vl]
                v_min = params_dict['vbus_min'][vl]

                vbus_outbounds, vbus_over, vbus_under,\
                VCHoursToStudy = feval.flex_needs_buses(_ts_res=ts_results, _v_max=v_max, _v_min=v_min,
                                                        _folders=[output_root, input_root],
                                                        _filenames=['VM_Problems_Stats_preMKT', 'VCHoursToStudy'])

                total_cm, lines_cm, trafos2w_cm,\
                trafos3w_cm, CMHoursToStudy = feval.congestion_needs(_net=net, _ts_res=ts_results,
                                                                     _llimit=params_dict['line_overload_limit'],
                                                                     _tlimit=params_dict['trafo_overload_limit'],
                                                                     _flexlimit=params_dict['flex_limit_pre'],
                                                                     _folders=[output_root, input_root],
                                                                     _filenames=['total_cm_PreMKT', 'CMHoursToStudy'])

                if cfg_file['ModelTAG'] == 'CMVC':
                    # Congestion Management & Voltage Control
                    HoursToStudy = np.unique(np.append(VCHoursToStudy, CMHoursToStudy))
                elif cfg_file['ModelTAG'] == 'VC':
                    # Voltage Control
                    HoursToStudy = VCHoursToStudy
                else:
                    # Congestion Management
                    HoursToStudy = CMHoursToStudy

                io_file.save_cvs(_data=pd.DataFrame(HoursToStudy), _outroot=input_root, _filename='HoursToStudy')

                fsp_file, fsp_bus = aux.fsp_converter(_net=net, _ts_res=ts_results, _fsp_data=fsp_data,
                                                      _out_root=input_root, _hours=HoursToStudy,
                                                      _bid_strat=params_dict['bidding_strategy'],
                                                      _scen_factor_load=cfg_file['scen_factor_load'],
                                                      _scen_factor_sgen=cfg_file['scen_factor_sgen'],
                                                      _ess_kmva=params_dict['ess_k_MVA'],
                                                      _kbid_p_up=params_dict['coeff_bid_Pup'][fsp],
                                                      _kbid_p_dwn=params_dict['coeff_bid_Pdown'][fsp],
                                                      _kbid_q_up=params_dict['coeff_bid_Qup'][fsp],
                                                      _kbid_q_dwn=params_dict['coeff_bid_Qdown'][fsp],
                                                      _sim_tag=market_tag, _k_opt=1)

                pilot_bus, pilot_bus_hts = rnet.select_pilot_bus(_net=net, _bus_voltages=ts_results['bus_vm_pu'],
                                                                 _fsp_data=fsp_data, _vbus_outbounds=vbus_outbounds,
                                                                 _hours=HoursToStudy, _vn_kv_mv=cfg_file['vn_kv_mv'],
                                                                 _out_root=input_root,
                                                                 _flag_vpilot=cfg_file['debug_save_VPilotBus_FULLH'],
                                                                 _flag_only_mv=cfg_file['only_pilot_busMV'])

                vc_worst_value, vc_worst_node,\
                vc_worst_hour = rnet.eval_worst_hour_bus(_df=pilot_bus_hts, _vmax=v_max, _vmin=v_min)

                (cm_trafo2w_worst_val,
                 cm_trafo2w_worst_node,
                 cm_trafo2w_worst_hour) = rnet.eval_worst_hour_lines(_df=ts_results['trafo_loading_percent'].T,
                                                                     _imax=params_dict['trafo_overload_limit'])
                (cm_line_worst_val,
                 cm_line_worst_node,
                 cm_line_worst_hour) = rnet.eval_worst_hour_lines(_df=ts_results['line_loading_percent'].T,
                                                                  _imax=params_dict['line_overload_limit'])
                (cm_trafo3w_worst_val,
                 cm_trafo3w_worst_node,
                 cm_trafo3w_worst_hour) = rnet.eval_worst_hour_lines(_df=ts_results['trafo3w_loading_percent'].T,
                                                                     _imax=params_dict['line_overload_limit'])
                list_max_value = [cm_line_worst_val, cm_trafo2w_worst_val, cm_trafo3w_worst_val]
                list_cm_hour = [cm_line_worst_hour, cm_trafo2w_worst_hour, cm_trafo3w_worst_hour]

                if cm_line_worst_val < 0 and cm_trafo2w_worst_val < 0 and cm_trafo3w_worst_val < 0:
                    cm_worst_hour = None
                else:
                    pos_hour = list_max_value.index(max(list_max_value))
                    cm_worst_hour = list_cm_hour[pos_hour]

                calc_sens.calc_sens_factors_vc(_net=net_post_pre, _fsp_file=fsp_file, _fsp_bus=fsp_bus, _ts_res=ts_results,
                                               _pilot_bus=pilot_bus, _hours=HoursToStudy, _sens_type='PQ',
                                               _delta_v=params_dict['vc_delta_sens_coeff'],
                                               _path_matrix=path_dict['sens_matrix_root'],
                                               _sim_tag=simulation_tag, _force_hvm=params_dict['force_HVM'],
                                               _save_inv_js=params_dict['save_invJ'])

                calc_sens.calc_sens_factors_cm(_net=net, _fsp_file=fsp_file, _fsp_bus=fsp_bus, _ts_res=ts_results,
                                               _clines=lines_cm, _ctrafo=trafos2w_cm, _ctrafo3w=trafos3w_cm,
                                               _hours=HoursToStudy, _dp=params_dict['cm_deltaP_coeff'],
                                               _dq=params_dict['cm_deltaQ_coeff'], _path_matrix=path_dict['sens_matrix_root'],
                                               _sens_type='PQ', _sim_tag=simulation_tag, _sens_mode='VSGEN',
                                               _version=1, _only_ds=True)

                # Run the Flexibility Market
                # fixme: how to improve?
                try:
                    cong_needs = total_cm.iloc[:, HoursToStudy] * params_dict['cm_over_procurement_coeff']
                except IndexError:
                    cong_needs = pd.DataFrame()

                cm_factor = -1
                mip_focus = 2
                nnvonvex = 2
                method = 5
                mip_gap = .00001
                flag_lp = True
                flag_mps = False
                ModelOut = [None]
                if cfg_file['ModelTAG'] == 'CMVC' or cfg_file['ModelTAG'] == 'VCCM':
                    cm_factor = -1
                    mip_gap = .00001
                elif cfg_file['ModelTAG'] == 'CM':
                    cm_factor = -1
                    mip_gap = .00001
                elif cfg_file['ModelTAG'] == 'VC':
                    cm_factor = 1
                    mip_gap = .000001
                ModelOut = flx_mrk.flex_market(_fsp_file=fsp_file, _hours=HoursToStudy, _cong_needs=cong_needs,
                                               _pilot_bus=pilot_bus, _mrk_tag=market_tag, _sim_tag=simulation_tag,
                                               _in_root=input_root, _out_root=output_root,
                                               _matrix_root=path_dict['sens_matrix_root'], _v_before=pilot_bus_hts.T,
                                               _vmin=v_min, _vmax=v_max, _dVtol=params_dict['dv_tol'],
                                               _alpha_cost=params_dict['alpha_cost'], _beta_cost=params_dict['beta_cost'],
                                               _net_mrk=params_dict['net_in_market'], _cap_cstr=params_dict['cap_curve_cons'],
                                               _dt=params_dict['dt'], _int_soc=params_dict['soc_cons'],
                                               _cm_factor=cm_factor, _mip_focus=mip_focus, _mip_gap=mip_gap,
                                               _nnvonvex=nnvonvex, _method=method, _flag_lp=flag_lp, _flag_mps=flag_mps)

                # Change net value
                Load_P_postMKT_TSin, Load_Q_postMKT_TSin, \
                Sgen_P_postMKT_TSin, Sgen_Q_postMKT_TSin = rnet.change_net_fsp_v1(_net=net_post, _hours2study=HoursToStudy,
                                                                                  _fsp_file=fsp_file, _ts_load_p=ts_results['load_p_mw'].T,
                                                                                  _ts_load_q=ts_results['load_q_mvar'].T,
                                                                                  _ts_sgen_p=ts_results['sgen_p_mw'].T,
                                                                                  _ts_sgen_q=ts_results['sgen_q_mvar'].T,
                                                                                  _root_fsp=input_root, _root_results=output_root,
                                                                                  _simulation_tag=market_tag)

                # Run Timeseries Power Flow
                # fixme: to improve. Maybe include in "change_net_fsp" function at the end.
                dict_profiles_postmkt = {
                    'load_prof_p': Load_P_postMKT_TSin.loc[:, HoursToStudy].T,
                    'load_prof_q': Load_Q_postMKT_TSin.loc[:, HoursToStudy].T,
                    'sgen_prof_p': Sgen_P_postMKT_TSin.loc[:, HoursToStudy].T,
                    'sgen_prof_q': Sgen_Q_postMKT_TSin.loc[:, HoursToStudy].T,
                }

                # Load_P_postMKT_TSin, Load_Q_postMKT_TSin, Sgen_P_postMKT_TSin, Sgen_Q_postMKT_TSin, \
                # Storage_P_postMKT_TSin, Storage_Q_postMKT_TSin = rnet.change_net_fsp_v3(_net=net_post, _hours2study=HoursToStudy,
                #                                                                         _fsp_file=fsp_file, _ts_load_p=ts_results['load_p_mw'].T,
                #                                                                         _ts_load_q=ts_results['load_q_mvar'].T,
                #                                                                         _ts_sgen_p=ts_results['sgen_p_mw'].T,
                #                                                                         _ts_sgen_q=ts_results['sgen_q_mvar'].T,
                #                                                                         _ts_storage_p=ts_results['storage_p_mw'].T,
                #                                                                         _ts_storage_q=ts_results['storage_q_mvar'].T,
                #                                                                         _root_fsp=input_root, _root_results=output_root,
                #                                                                         _simulation_tag=market_tag)
                # # Run Timeseries Power Flow
                # # fixme: to improve. Maybe include in "change_net_fsp" function at the end.
                # dict_profiles_postmkt = {
                #     'load_prof_p': Load_P_postMKT_TSin.loc[:, HoursToStudy].T,
                #     'load_prof_q': Load_Q_postMKT_TSin.loc[:, HoursToStudy].T,
                #     'sgen_prof_p': Sgen_P_postMKT_TSin.loc[:, HoursToStudy].T,
                #     'sgen_prof_q': Sgen_Q_postMKT_TSin.loc[:, HoursToStudy].T,
                #     'storage_prof_p': Storage_P_postMKT_TSin.loc[:, HoursToStudy].T,
                #     'storage_prof_q': Storage_Q_postMKT_TSin.loc[:, HoursToStudy].T
                # }

                ts_results_post = rnet.run_pf_timeseries(_pp_net=net_post, _dict_profiles=dict_profiles_postmkt,
                                                         _time_steps=HoursToStudy, _out_filetype=params_dict['ts_output_file_type'],
                                                         _out_root=output_ts_root, _all_steps=False, res_bus=['vm_pu', 'p_mw'],
                                                         res_line=['loading_percent'], res_load=['p_mw', 'q_mvar'],
                                                         res_sgen=['p_mw', 'q_mvar'], res_storage=['p_mw', 'q_mvar'],
                                                         res_trafo=['loading_percent'], res_trafo3w=['loading_percent'])

                vbus_outbounds_postMKT, vbus_over_postMKT, vbus_under_postMKT, \
                VCHoursToStudy_postMKT = feval.flex_needs_buses(_ts_res=ts_results_post, _v_max=v_max, _v_min=v_min,
                                                                _folders=[output_root, output_root],
                                                                _filenames=['VM_Problems_Stats_postMKT', 'VCHoursToStudy_post'])

                total_cm_postMKT, lines_cm_postMKT, trafos2w_cm_postMKT, \
                trafos3w_cm_postMKT, CMHoursToStudy_postMKT = feval.congestion_needs(_net=net_post, _ts_res=ts_results_post,
                                                                                     _llimit=params_dict['line_overload_limit'],
                                                                                     _tlimit=params_dict['trafo_overload_limit'],
                                                                                     _flexlimit=params_dict['flex_limit_pre'],
                                                                                     _folders=[output_root, output_root],
                                                                                     _filenames=['total_cm_postMKT', 'CMHoursToStudy_post'])

                aux.eval_energy_support_res(_ts_res=ts_results_post, _pilot_bus=pilot_bus, _hours=HoursToStudy,
                                            _out_root=output_root, _fsp_file=fsp_file, _filename_lfm=ModelOut[0])
    # Evaluate KPIs
    kpi_fcn.evaluate_kpis(_cfg_file=cfg_file, _paths=path_dict, _fsp_file=fsp_file,
                          _prods=cfg_file['ProductTAG'], _vls=cfg_file['VLTAG'], _fsps=cfg_file['FspTAG'],
                          _draw_plots=True, _draw_net=False)

    # Record the end time
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time_seconds = end_time - start_time
    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # print(f"Time elapsed: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    print('Time elapsed: {:0.2f} hours, {:0.2f} minutes, {:0.2f} seconds'.format(hours, minutes, seconds))
