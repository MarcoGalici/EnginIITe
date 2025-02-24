def _diagnostic(_m):
    """
    Check the model diagnostics.

    :param _m: Model representation by Gurobi.py.
    """
    return True


def _upload_res_params(_fsp_file, _in_root, _sim_period):
    """
    Upload resource parameters from resource models.
    :param _fsp_file: Dictionary of the names of the fsp filex per resource category.
    :param _in_root: Input root for reading fspInfo file.
    :param _sim_period: Number of hours to simulate. Example: hours2study: [5, 11] -> _sim_period: [0, 1]
    """
    return True


def _load_sens_matrix(_res_parameters, _hours2study, _n_cong, _n_pilot_bus, _matrix_root, _mrk_scenario, _service,
                      _sim_tag, _cm_factor=-1):
    """
    Upload Sensitivity Matrix from file.

    :param _res_parameters: Dictionary of the parameters of the fsp per resource category.

    :param _hours2study: List of hours to study.

    :param _n_cong: Number of congestions.

    :param _n_pilot_bus: Number of pilot buses.

    :param _matrix_root: Root to where the sensitivity matrix file are saved.

    :param _mrk_scenario: String that represents the tag for market model.

    :param _service: String that represents the tag for product in the market.

    :param _sim_tag: String that represents the specific name of the simulation.

    :param _cm_factor: Congestion Management Factor that change sign to the sensitivity matrix.
    """
    return True


def _var_indexing(_grb_model, _res_parameters, _n_pb, _n_cm, _n_t):
    """
    Define variables and save them into dictionary shared across multiple methods of the class.
    :param _grb_model: Gurobipy model.
    :param _res_parameters: Dictionary of all resource parameters
    :param _n_pb: Number of Pilot Buses.
    :param _n_cm: Number of congestions.
    :param _n_t: Number of hours to study.
    """
    return True


def _cons_indexing(_grb_model, _vars, _res_parameters, _sens_matrix_cm, _sens_matrix_vc, _hours, _cong_needs,
                   _pilot_bus, _v_before, _vmin, _vmax, _dVtol, _mrk_scenario, _net_mrk, _cap_cstr, _int_soc, _dt):
    """
    Define constraints of the Local Flexibility Market.
    :param _grb_model: Gurobu py model.
    :param _vars: Dictionary of the variables of the model.
    :param _res_parameters: Dictionary of the resource parameters.
    :param _sens_matrix_cm: Sensitivity Coefficient matrix for congestion management.
    :param _sens_matrix_vc: Sensitivity Coefficient matrix for voltage control.
    :param _hours: List of hours to study.
    :param _cong_needs: Dataframe of the congestions.
    :param _pilot_bus: List of Pilot buses.
    :param _v_before: Dataframe of the oltage before the market clearing.
    :param _vmin: Minimum Voltage in p.u.
    :param _vmax: Maximum Voltage in p.u.
    :param _dVtol: Delta Voltage tolerance.
    :param _mrk_scenario: String representing the market scenario
    :param _net_mrk: Boolean value for including network constraints in the market model.
    :param _cap_cstr: Boolean value for including resource capability constraints in the market model.
    :param _int_soc: Boolean value for including battery inter-temporal constraints in the market model.
    :param _dt: Delta time of the model.
    """
    return True


def _obj_indexing(_grb_model, _vars, _res_parameters, _alpha_cost, _beta_cost, _sens_matrix_vc, _n_pb, _n_cm, _n_t,
                  _mrk_scenario, _net_mrk):
    """
    Define Objective of the Local Flexibility Market.
    :param _grb_model: Gurobi py model.
    :param _vars: Dictionary of the variables of the model.
    :param _res_parameters: Dictionary of the resource parameters.
    :param _alpha_cost: Cost of Voltage Control Slack Variables.
    :param _beta_cost: Cost of Congestion Management Slack Variables.
    :param _sens_matrix_vc: Sensitivity Coefficient matrix for voltage control.
    :param _n_t: Number of hours to study.
    :param _n_cm: Number of the congestions.
    :param _n_pb: Number of Pilot buses.
    :param _mrk_scenario: String representing the market scenario
    :param _net_mrk: Boolean value for including network constraints in the market model.
    """
    return True
    
    
def _save_vars_tuple(_grb_model, _vars, _res_parameters, _list_general_vars, _hours, _cong_needs, _pilot_bus, _round_vars=3):
    """
    Save tuple variables of optimisation model into Dataframes.
    """
    return True


def _save_mrk_res_excel(_results, _out_root, _filename, _extension='.xlsx'):
    """
    Save market results into Excel files.

    :param _results: Dictionary market results to be saved.

    :param _out_root: Root to where the market results will be saved.

    :param _filename: Name of the file excel that contains the market results.

    :param _extension: Extension of the file excel.
    """
    return True


def flex_market(_fsp_file, _hours, _cong_needs, _pilot_bus, _in_root, _out_root, _matrix_root, _v_before, _vmin, _vmax,
                _dVtol, _alpha_cost, _beta_cost, _mrk_tag, _sim_tag, _filename='model_results_', _cm_factor=-1,
                _mip_focus=2, _mip_gap=.00001, _nnvonvex=2, _method=5, _net_mrk=True, _cap_cstr=True, _int_soc=True,
                _dt=1, _flag_lp=False, _flag_mps=False):
    """
    Run the Local Flexibility Market.
    """
    return True
