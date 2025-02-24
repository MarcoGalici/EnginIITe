def set_resources():
    """
    Define resource considered in the platform.
    :return: Dictionary containing as keys the name of the resource type considered in the platform and as
     values the name of the python file in which the resource model for the market are written
     (this python file must be saved inside the functions.market.models folder).
    """
    return True


def set_resource_profiles():
    """
    Define resource that require calculation of initial profile. The key of the dictionary must be the
     "element_name" column in the network Excel file. The element name must be the same as the "type" column in the
     fsp_data Excel file in which the provider are defined.
    :return: Dictionary containing as keys the name of the resource type considered in the platform and as
     values the name of the python file in which the resource model for the market are written
     (this python file must be saved inside the functions.market.models folder).
    """
    return True


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

    return True


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
    return True


def get_list(_startswith_key, _comparing_list):
    """
    Get the keys that starts with the value passed as input.

    :param _startswith_key: Starting string to which the values will be compared.

    :param _comparing_list: List of value to be compared.
    """
    return True


def _check_root(_dir2check):
    """
    Check if root exist and in case it doesn't create it.
    :param _dir2check: Directory to check existence
    :return: True if the directory already exists, False otherwise (directory create it just now)
    """
    return True


def _def_dp(_cfg_file):
    """
    Check the set-point of the resources and define the dp and dq for the sensitivity factors evaluation,
     Increas by one the dp and dq to remain in the linear approximation of the sensitivity factors calculation.
    :param _cfg_file: Yaml configuration file
    :return: The dP and dQ for the sensitivity factor calculation
    """
    return True


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
    return True


def set_general_params(_cfg_file, _path_dict, _ess_tag):
    """Set the general parameters of the simulation.
    :param _cfg_file: Yaml configuration file
    :param _path_dict: Dictionary containing all the paths.
    :param _ess_tag: Tag for the Storage capacity flexibility provider.
    :return: Dictionary containing all the general parameters."""
    return True


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
    return True


def set_sim_params(_gparams, _vl_tags, _fsp_tags):
    """Set the simulation parameter of the simulation.
    :param _gparams: Yaml configuration file
    :param _vl_tags: Tag for the voltage limitations of the market.
    :param _fsp_tags: Tag for the flexibility provided in the market.
    :return: Dictionary containing all the general parameters."""
    return True
