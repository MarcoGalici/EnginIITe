def _check_consis(_value1, _value2):
    """
    Check consistency between kpi inputs requested from the user and config file inputs.

    :param _value1: kpi inputs requested from the user.

    :param _value2: config file inputs.
    """
    return True


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
    return True


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
    return True


def _load_res_net_pre(_in_dict, _scenarios):
    """
    Load data of the network before the market.

    :param _in_dict: Dictionary with all the information for evaluating the kpis.

    :param _scenarios: List of string of names of the scenarios that the user want to analyse.
    """
    return True


def _load_res_net_post(_in_dict, _scenarios):
    """
    Load data after the market.

    :param _in_dict: Dictionary with all the information for evaluating the kpis.

    :param _scenarios: List of string of names of the scenarios that the user want to analyse.
    """
    return True


def max_mat_df(_input_df):
    """
    Extract the maximum value, the row (node) and the column (hour) in the dataframe passed as input.

    :param _input_df: Dataframe with nodes as rows and hours as columns.
    """
    return True


def eval_hosting_capacity(_net, _data):
    """
    Evaluate the hosting capacity increase in the network.

    :param _net: Pandapower network

    :param _data: Dictionary with all the information for evaluating the hosting capacity.
    """
    return True


def increase_hc(_in_dict):
    """
    Evaluate the increase in the hosting capacity for each scenario.
    It requires the function eval_hosting_capacity(_net, _data), to calculate the increase in Hosting Capacity
    for an entire study. Increase in Hosting Capacity is calculated as the difference between post-market
    and pre-market.

    :param _in_dict: Dictionary with all the information for evaluating the kpis.
    """
    return True


def eval_avoided_congestion(_data_pre, _data_post, _round_val=3):
    """
    Evaluate the avoided congestion in the network between the pre- and post-market.

    :param _data_pre: Dictionary with all the information pre-market for evaluating the avoided congestions.

    :param _data_post: Dictionary with all the information post-market for evaluating the avoided congestions.

    :param _round_val: Value to which all value are rounded.
    """
    return True


def avoided_cong(_in_dict, _scenarios):
    """
    Evaluate the Avoided Congestion Problems - Lines and Transformers (2 windings and 3 windings).

    :param _in_dict: Dictionary with all the information for evaluating the avoided congestions.

    :param _scenarios: List of string of names of the scenarios that the user want to analyse.
    """
    return True


def eval_avoided_voltage_violation(_data_pre, _data_post, _round_val=3):
    """
    Evaluate the avoided congestion in the network between the pre- and post-market.

    :param _data_pre: Dictionary with all the information pre-market for evaluating the avoided voltage violations.

    :param _data_post: Dictionary with all the information post-market for evaluating the avoided voltage violations.

    :param _round_val: Value to which all value are rounded.
    """
    return True


def avoided_voltage_violations(_in_dict):
    """
    Evaluate the Avoided Voltage Violations - Buses.

    :param _in_dict: Dictionary with all the information for evaluating the avoided voltage violations.
    """
    return True


def load_fsp_result(_in_dict, _scenario, _prefix='model_results'):
    """
    Upload the flexibility market results from folders.

    :param _in_dict: Dictionary with all the paths for importing the flexibility market results.

    :param _scenario: String representing the specific scenario to read.

    :param _prefix: Prefix of the excel file with the flexibility market results.
    """
    return True


def load_data_costs(_in_dict, _scenario, _prefix='_init'):
    """
    Upload the flexibility market costs from folders.

    :param _in_dict: Dictionary with all the paths for importing the flexibility market results.

    :param _scenario: String representing the specific scenario to read.

    :param _prefix: Prefix of the Excel file with the flexibility market results.
    """
    return True


def eval_flexibility_costs(_in_dict, _dict_res, _dict_others, _dict_costs, _round_val=3):
    """
    Evaluate the costs for the request of flexibility.

    :param _in_dict: Dictionary with all the information pre-market for evaluating the avoided voltage violations.

    :param _dict_res: Dictionary with all the information about the flexibility provided by each resource.

    :param _dict_others: Dictionary with all the information about the general flexibility market parameter.

    :param _dict_costs: Dictionary with all the information about the costs for flexibility by each resource.

    :param _round_val: Value to which all value are rounded.
    """
    return True


def cost_flexibility(_in_dict):
    """
    Evaluate the cost af all the flexibility service activations.

    :param _in_dict: Dictionary with all the information for evaluating the avoided voltage violations.
    """
    return True


def build_final_table(_in_dict):
    """
    Build final results table.
    The objective value and the total cost are different since the slack variables of the voltage (alpha) is written
    differently in the market. In the market, it is written as:
        -> (ABS_alpha_DVpu / sens_matrix_vc['all']) * alpha_cost        (if net_in_market parameters is not active)
        -> ABS_alpha_DVpu * alpha_cost * 1000                           (if net_in_market parameters is active)

    :param _in_dict: Dictionary with all the information for building the final table in terms of costs and volumes.
    """
    return True


def eval_statistics_volume_flex(_in_dict, _dict_res, _dict_costs, _scenario, _round_val=6):
    """
    Evaluate the statistics behind the volume of flexibility offered.

    :param _in_dict: Dictionary with all the information pre-market for evaluating the avoided voltage violations.

    :param _dict_res: Dictionary with all the information about the flexibility provided by each resource.

    :param _dict_costs: Dictionary with all the information about the costs for flexibility by each resource.

    :param _scenario: String representing the specific scenario to read.

    :param _round_val: Value to which all value are rounded.
    """
    return True


def volume_flex_offered(_in_dict, _dict_res, _dict_costs):
    """
    Evaluate the volume of flexibility offered.

    :param _in_dict: Dictionary with all the information for evaluating the volume of flexibility offered.

    :param _dict_res: Dictionary with all the information about the flexibility provided by each resource.

    :param _dict_costs: Dictionary with all the information about the costs for flexibility by each resource.
    """
    return True


def increase_hc_plots(_df_res, _scenarios, _title, _y_lim):
    """
    Prepare the plots for the Increase Hosting Capacity.
    The plots are subdivided per Voltage Violation Scenarios.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.

    :param _y_lim: Limits of the y-axis.
    """
    return True


def avoided_cong_plots(_df_res, _scenarios, _title, _y_lim):
    """
    Prepare the plots for the avoided congestions.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.

    :param _y_lim: Limits of the y-axis.
    """
    return True


def avoided_cong_3dplots(_df_res, _scenarios, _title):
    """
    Prepare the 3D plots for the avoided congestions.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.
    """
    return True


def avoided_cong_element_3dplots(_df_res, _scenarios, _title):
    """
    Prepare the 3D plots for the number of avoided congestions.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.
    """
    return True


def avoided_voltage_viol_plots(_df_res, _scenarios, _title, _y_lim):
    """
    Prepare Avoided Voltage Violations plot.

    :param _df_res: Dataframe of the Increase Hosting Capacity to be plotted.

    :param _scenarios: Scenario to be analyzed.

    :param _title: Title of the figure.

    :param _y_lim: Limits of the y-axis.
    """
    return True


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
    return True
