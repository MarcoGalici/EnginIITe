def import_yaml(_yaml_filename, _folder, **kwargs):
    """
    This function imports the yaml configuration file.

    Input:
        **_yaml_filename** (string) - Yaml network file input inside the config folder.

        **_folder** (string) - Folder in which the yaml config files are saved.

    Output:
        **_config_file** (dict) - config network file data.

    Example:

        >>> my_config_yaml_file = import_yaml('my_yaml_file.yaml', 'my_folder')

    """
    return True


def import_network(_net_data, _in_root='data_in', _folder='network', **kwargs):
    """
    This function imports the electrical network from Excel file according to pandapower format.

    Input:
        **_net_data** (dict) - Dictionary with the network input from Yaml file.

        **_in_root** (string, 'data_in') - Root in which the network file is saved.

        **_folder** (string, 'network') - Folder in which the .xlsx network file is saved.

    Output:
        **net** (attrdict) - Pandapower network.

    Example:

        >>> cfg_file = import_yaml('my_yaml_file.yaml', 'my_folder')
        >>> my_pandapower_network = import_network(cfg_file['network'])

    """
    return True


def read_file_profiles(_root2file, _timestep, **kwargs):
    """
    This function reads the file of profiles for specific time intervals.
    This function allows two file extensions: i) .csv and ii) .xlsx.
    In case of .xlsx, this function requires that the name of the file and the name of the sheet
    that contains the profiles are the same.

    Input:
        **_filename** (string) - Filename to be read.

        **_timestep** (list of integers) - Time intervals to be analysed.

    Output:
        **df_prof** (dict) - Dictionary with the profile values.

    Example:

        >>> import os
        >>> file2read = os.path.join(os.getcwd(), 'my_folder', 'my_profile_file.xlsx')
        >>> my_dictionary_profiles_from_file = read_file_profiles(file2read, [0, 24])

    """
    return True


def import_norm_profiles(_root2file, _timestep, _res_pp, _product, **kwargs):
    """
    This function imports the normalized profiles of specific resources.

    Input:
        **_root2file** (string) - Path to the file to be read.

        **_timestep** (list of integers) - Time intervals to be analysed.

        **_res_pp** (Dataframe) - Dataframe of the resources to which the normalized profiles are applied.
        The Dataframe must be configured like the resource Dataframe of pandapower network.

        **_product** (string) - Type of market product. It can be 'p' or 'q'.

    Output:
        **df_product** (Dataframe) - Dataframe of the profiles.

    Example:

        >>> import os
        >>> import pandapower.networks
        >>> file2read = os.path.join(os.getcwd(), 'my_folder', 'my_profile_file.xlsx')
        >>> net = pandapower.networks.example_multivoltage()
        >>> my_normalized_dataframe_profile = import_norm_profiles(file2read, [0, 24], net.load, 'p')

    """
    return True


def import_ext_profiles(_root2inputs, _timestep, _res_pp, **kwargs):
    """
    This function imports the profiles from an external module that evaluates the profile according to the user
    preferencies.

    Input:
        **_root2inputs** (string) - Path to the input parameters file to be read.

        **_timestep** (list of integers) - Time intervals to be analysed.

        **_res_pp** (Dataframe) - Dataframe of the resources to which the external module evaluation are applied.
        The Dataframe must be configured like the resource Dataframe of pandapower network.

    Output:
        **res_category_prof_p** (Dataframe) - Dataframe of the Active Power profiles.

        **res_category_prof_q** (Dataframe) - Dataframe of the Reactive Power profiles.

    Example:

        >>> import os
        >>> import pandapower.networks
        >>> file2read = os.path.join(os.getcwd(), 'my_folder', 'my_profile_file.xlsx')
        >>> net = pandapower.networks.example_multivoltage()
        >>> P_dataframe_profile, Q_dataframe_profile = import_ext_profiles(file2read, [0, 24], net.load)

    """
    return True


def import_profiles(_pp_net, _net_data, _timestep, _in_root='data_in', _folder_prof='profile', _folder_ext='fsp_files', **kwargs):
    """
    This function imports the network profiles for all the resources included in the network.
    In case a resource category is not present in the network file, this resource category will be skipped.

    Input:
        **_pp_net** (dict) - The pandapower format network.

        **_net_data** (dict) - Dictionary with the network input from Yaml file.

        **_timestep** (list of integers) - Time intervals to be analysed.

        **_in_root** (string, data_in) - Root in which the input files are saved.

        **_folder_prof** (string, profile) - Root in which the input profile files are saved.

        **_folder_ext** (string, fsp_files) - Root in which the input parameters of fsp, and the input parameters for
        the evaluation of the profiles from an external module are saved.

    Output:
        **dict_profiles** (dict) - Dictionary with the profiles.

    Example:

        >>> import pandapower.networks
        >>> net = pandapower.networks.example_multivoltage()
        >>> cfg_file = import_yaml('my_yaml_file.yaml', 'my_folder')
        >>> dictionary_profiles = import_profiles(net, cfg_file['network'], [0, 24], 'data_in', 'profile', 'fsp_files')

    """
    return True


def import_profiles_s01(_cfg_file, _pp_net, _timestep, _in_root='data_in', _folder='profile', **kwargs):
    """
    This function imports the network profiles for the initial scenario for all the resources categories included
    in the network. The resource categories are: 1) load, 2) gen, 3) sgen and 4) storage. It is important to point out
    that in the config file, the scenario factors for each category included in the network must be defined.
    The scenario factors are: 1) scen_factor_load, 2) scen_factor_gen, 3) scen_factor_sgen and 4) scen_factor_storage.

    Input:
        **_cfg_file** (dict) - Dictionary with the network input from Yaml file.

        **_pp_net** (dict) - The pandapower format network.

        **_timestep** (list of integers) - Time intervals to be analysed.

        **_in_root** (string, data_in) - Root in which the input files are saved.

        **_folder** (string, profile) - Folder in which the xlsx network files are saved.

    Output:
        **dict_profiles_s01** (dict) - Dictionary with the profile values.

    Example:

        >>> import pandapower.networks
        >>> net = pandapower.networks.example_multivoltage()
        >>> cfg_file = import_yaml('my_yaml_file.yaml', 'my_folder')
        >>> dictionary_profiles_scenario01 = import_profiles_s01(cfg_file['network'], net, [0, 24], 'data_in', 'profile')

    """
    return True


def import_data(_filename, _root, _sheetname=None, _index_col=None, _header=0):
    """
    This function imports the data from file passed as input.
    Two extension of files are available: i) .csv and ii) .xlsx.

    Input:
        **_filename** (string) - Name of the file to import (format .csv or .xlsx).

        **_root** (string) - Full directory in which the file is saved.

        **_sheetname** (string, None) - If .xlsx extension is selected, it represents the name of the sheet in which
         the information are saved (only xlsx file).

        **_index_col** (int, None) - Column(s) to use as row label(s), denoted either by column labels or column indices.

        **_header** (int, 0) - Row number(s) containing column labels and marking the start of the data.

    Output:
        **_tmp_data** (Dataframe) - Dataframe with the information contained in the file passed.

    Example:

        >>> import os
        >>> my_root = os.path.join(os.getcwd(), 'my_folder')
        >>> my_dataframe_data = import_data('my_file.xlsx', my_root, 'my_sheet', 0, 0)

    """
    return True


def save_excel(_data, _outroot, _filename, _sheetname, _mode='w', _extension='.xlsx'):
    """
    This function saves results into Excel files.

    Input:
        **_data** (Dataframe) - Dataframe to be saved.

        **_outroot** (string) - Output path to folder where to save input dataframe.

        **_filename** (string) - Name of the Excel file to be saved.

        **_sheetname** (string) - Name od the sheet of the Excel file to be saved.

        **_mode** (string, w) - File mode to use (write: 'w' or append: 'a').

        **_extension** (string, .xlsx) - Extension of the Excel file.

    Example:

        >>> import pandas as pd
        >>> my_empty_dataframe = pd.DataFrame()
        >>> save_excel(my_empty_dataframe, 'my_path_to_folder', 'my_filename', 'my_sheetname', 'w', '.xlsx')

    """
    return True


def save_cvs(_data, _outroot, _filename, _extension='.csv'):
    """
    This function saves results into Comma Separated Values files.

    Input:
        **_data** (Dataframe) - Dataframe to be saved.

        **_outroot** (string) - Output path to folder where to save input dataframe.

        **_filename** (string) - Name of the Excel file to be saved.

        **_extension** (string, .csv) - Extension of the CSV file.

    Example:

        >>> import pandas as pd
        >>> my_empty_dataframe = pd.DataFrame()
        >>> save_cvs(my_empty_dataframe, 'my_path_to_folder', 'my_filename', '.csv')

    """
    return True
