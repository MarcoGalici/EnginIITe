import os
import yaml
import importlib
import copy as cp
import pandas as pd
import pandapower as pp
from pathlib import Path
from functions import auxiliary as aux


def import_yaml(_yaml_filename, _folder, **kwargs):
    """
    This function imports the yaml configuration file.

    Input:
        **_yaml_filename** (string) - Yaml network file input inside the config folder.

        **_folder** (string) - Folder in which the yaml config files are saved.

    Output:
        **_config_file** () - config network file data.

    Example:
        import_yaml('my_yaml_file.yaml', 'my_folder')
    """

    additional_kwargs = {key: val for key, val in kwargs.items()}

    with open(os.path.join(os.getcwd(), _folder, _yaml_filename), 'r') as _filename:
        _config_file = yaml.safe_load(_filename)
    return _config_file


def import_network(_net_data, _in_root='data_in', _folder='network', **kwargs):
    """
    Import network file according to pandapower format.

    :param _net_data: Dictionary with the network input (Network Folder, Network filename, and Network Profiles)
    :param _in_root: Root in which the input files are saved
    :param _folder: Folder in which the xlsx network files are saved
    :return: network data according to pandapower
    """
    additional_kwargs = {key: val for key, val in kwargs.items()}
    file_extension = _net_data['net_filename'].split('.')[-1]
    _net_name = _net_data['net_folder']
    _net_filename = _net_data['net_filename']
    _root2network = os.path.join(os.getcwd(), _in_root, _net_name, _folder, _net_filename)
    if file_extension == 'xlsx':
        net = pp.from_excel(_root2network)
    else:
        raise ValueError('The format {0} is not supported.'.format(file_extension))
    return cp.deepcopy(net)


def import_denorm_profiles(_filename, _timestep, **kwargs):
    """
    Import denormalized network profiles
    :param _filename: Filename to be read
    :param _timestep: Time intervals to be analysed
    :return: dictionary with the profile values
    """
    additional_kwargs = {key: val for key, val in kwargs.items()}
    _file_ext = _filename.split('.')[1]
    if _file_ext == 'xlsx':
        try:
            _sheet = os.path.split(_filename)[1].split('.')[0]
        except ValueError:
            _sheet = os.path.split(_filename)[1].split('.')[0]
            _mess = 'The sheet of the file Excel must be equal to the name of the file. Correct the name "{_sheet}".'.format(_sheet=_sheet)
            raise ValueError(_mess)
        df_prof = pd.read_excel(_filename, sheet_name=_sheet, index_col=0).iloc[range(_timestep[0], _timestep[1])]
    elif _file_ext == 'csv':
        df_prof = pd.read_csv(_filename, index_col=0).iloc[range(_timestep[0], _timestep[1])]
    else:
        raise ValueError('The extension "{_ext}" is not supported.'.format(_ext=_file_ext))
    return df_prof


def import_norm_profiles(_pp_net, _filename, _timestep, _res, _product, _list_res, **kwargs):
    """
    Import network normalize profiles of specific resource.
    :param _pp_net: Pandapower network
    :param _filename: Filename to be read
    :param _timestep: Time intervals to be analysed
    :param _res: Type of resource from which the profile are uploaded
    :param _product: Type of product from which the profile are uploaded
    :param _list_res: List of resource available
    :return: dictionary with the profile values
    """
    additional_kwargs = {key: val for key, val in kwargs.items()}
    norm_df_prof = import_denorm_profiles(_filename=_filename, _timestep=_timestep, _add_args=additional_kwargs)

    dict_product = dict()
    _res_pp = None
    if _res == _list_res[0]:
        _res_pp = cp.deepcopy(_pp_net.load)
    elif _res == _list_res[1]:
        _res_pp = cp.deepcopy(_pp_net.sgen)
    elif _res == _list_res[2]:
        _res_pp = cp.deepcopy(_pp_net.gen)
    elif _res == _list_res[3]:
        _res_pp = cp.deepcopy(_pp_net.storage)

    for _i, _device in _res_pp.iterrows():
        _prof_cat = _device['prof_name']
        if _product == 'p':
            dict_product[_i] = norm_df_prof[_prof_cat] * _device.p_mw
        if _product == 'q':
            dict_product[_i] = norm_df_prof[_prof_cat] * _device.q_mvar

        # if isinstance(_prof_cat, str):
        #     if _product == 'p':
        #         dict_product[_i] = norm_df_prof[_prof_cat] * _device.p_mw
        #     if _product == 'q':
        #         dict_product[_i] = norm_df_prof[_prof_cat] * _device.q_mvar
        # else:
        #     # Here we call the Profile definition for the resource
        #     # I dati necessari ad eseguire la valutazione dei profili di input li leggiamo qui.
        #     dict_res_profiles = aux.set_resource_profiles()
        #     for _element in dict_res_profiles.keys():
        #         list_res = dict_res_profiles.get(_element, None)
        #         if list_res is None:
        #             raise IOError('Resource of type {_class} is unknown.'.format(_class=_i))
        #
        #         res_lib = importlib.import_module(f'functions.market.models.{list_res}')

    df_product = pd.DataFrame().from_dict(dict_product)
    return df_product


def import_profiles(_pp_net, _net_data, _timestep, _in_root='data_in', _folder='profile', **kwargs):
    """
    Import network profiles of specific resource.
    :param _pp_net: Pandapower network
    :param _net_data: Dictionary with the network input (Network Folder, Network filename, and Network Profiles)
    :param _timestep: Time intervals to be analysed
    :param _in_root: Root in which the input files are saved
    :param _folder: Folder in which the xlsx network files are saved
    :return: dictionary with the profile values
    """
    _list_res = ['load', 'sgen', 'gen', 'storage']
    additional_kwargs = {key: val for key, val in kwargs.items()}
    _net_name = _net_data['net_folder']
    _profile_data = _net_data['profile']
    _filenames = _profile_data['profile_filename']
    _norms = _profile_data['norm_profile']
    dict_profiles = dict()
    for _k, _v in _norms.items():
        _root2file = os.path.join(os.getcwd(), _in_root, _net_name, _folder, _filenames[_k])
        if _v:
            _res = _k.split('_')[0]
            _product = _k.split('_')[-1]
            dict_profiles[_k] = import_norm_profiles(_pp_net=_pp_net, _filename=_root2file, _timestep=_timestep,
                                                     _res=_res, _product=_product, _list_res=_list_res,
                                                     _add_args=additional_kwargs)
        else:
            dict_profiles[_k] = import_denorm_profiles(_filename=_root2file, _timestep=_timestep,
                                                       _add_args=additional_kwargs)
        # Transform column into integer
        dict_profiles[_k].columns = dict_profiles[_k].columns.astype(int)
    return dict_profiles


def import_profiles_s01(_cfg_file, _pp_net, _timestep, _in_root='data_in', _folder='profile',
                        _net_page='net_folder', _net_prof_page='profile', _prof_files_page='profile_filename',
                        _prof_norm_page='norm_profile', **kwargs):
    """
    Import network profiles of specific resource.
    :param _cfg_file: Configuration network file
    :param _pp_net: Pandapower network
    :param _timestep: Time intervals to be analysed
    :param _in_root: Root in which the input files are saved
    :param _folder: Folder in which the xlsx network files are saved
    :param _net_page: Dictionary page in which the network name is saved
    :param _net_prof_page: Dictionary page in which the network profile information are saved
    :param _prof_files_page: Dictionary page in which the network profile file names are saved
    :param _prof_norm_page: Dictionary page in which the info about the normalize data or not are saved
    :return: dictionary with the profile values
    """
    additional_kwargs = {key: val for key, val in kwargs.items()}
    dict_profiles = import_profiles(_pp_net=_pp_net, _net_data=_cfg_file['network'], _timestep=_timestep,
                                    _in_root=_in_root, _folder=_folder, _add_args=additional_kwargs)

    dict_profiles_s01 = cp.deepcopy(dict_profiles)
    for _key, _value in dict_profiles_s01.items():
        try:
            if 'load' in _key:
                dict_profiles_s01[_key] = _value * _cfg_file['scen_factor_load']
            elif 'sgen' in _key:
                dict_profiles_s01[_key] = _value * _cfg_file['scen_factor_sgen']
            elif 'gen' in _key:
                dict_profiles_s01[_key] = _value * _cfg_file['scen_factor_gen']
            elif 'storage' in _key:
                dict_profiles_s01[_key] = _value * _cfg_file['scen_factor_storage']
        except Exception as _:
            raise UserWarning('{_res} profiles or scenario factor not found.'.format(_res=_key))
    return dict_profiles_s01


def import_data(_filename, _root, _sheetname=None, _index_col=None, _header=0):
    """Check the filename in the input root and import the file.
    :param _filename: Name of the file to import (format csv or xlsx)
    :param _root: Full directory in which the file is saved
    :param _sheetname: Name of the sheet in which the information are saved (only xlsx file)
    :param _index_col: Column(s) to use as row label(s), denoted either by column labels or column indices
    :param _header: Row number(s) containing column labels and marking the start of the data
    :return: Dataframe with the information contained in the file passed.
    """
    if _filename is None:
        raise ValueError('No filename have been passed.')
    if _root is None:
        raise ValueError('No root to filename have been passed.')

    _tmp_data = None
    _ext = Path(_filename).suffix
    if _ext == '.csv':
        _tmp_data = pd.read_csv(os.path.join(_root, _filename), index_col=_index_col, header=_header)
    elif _ext == '.xlsx':
        _tmp_data = pd.read_excel(os.path.join(_root, _filename), index_col=_index_col, header=_header, sheet_name=_sheetname)
    return _tmp_data


def save_excel(_data, _outroot, _filename, _sheetname, _mode='w', _extension='.xlsx'):
    """Save results into Excel files"""
    if not isinstance(_data, pd.DataFrame):
        raise ValueError('Format type {0} not supported.'.format(str(type(_data))))
    root2file = os.path.join(_outroot, _filename + _extension)
    with pd.ExcelWriter(root2file, engine='openpyxl', mode=_mode) as writer:
        _data.to_excel(writer, sheet_name=_sheetname)
    return True


def save_cvs(_data, _outroot, _filename, _extension='.csv'):
    """Save results into Comma Separated Value files"""
    if not isinstance(_data, pd.DataFrame):
        raise ValueError('Format type {0} not supported.'.format(str(type(_data))))
    root2file = os.path.join(_outroot, _filename + _extension)
    _data.to_csv(root2file, encoding='utf-8')
    return True
