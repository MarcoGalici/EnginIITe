def _copy_net_essential(_pp_network):
    """
    Copy essential elements of a pandapower network.

    :param _pp_network: Pandapower network

    :return _new_net: Copy of the essential elements of the pandapower network.
    """
    return True


def create_vsgen(_net, _bus, _p_q, _delta):
    """
    Custom function to create a virtual generator on a specific bus in the network
    with an active/reactive power equal to delta.

    :param _net: Pandapower network.

    :param _bus: Bus in which the virtual generator will be included.

    :param _p_q: Flag to identify if the generator will inject active or reactive power.

    :param _delta: Active/Reactive power of the virtual generator.
    """
    return True


def sens_factors_cm(net0, Mode, FSP_data, FSPBranchArray, FSPTrafoArray, FSPTrafo3wArray, deltaP_coeff, deltaQ_coeff,
                    only_dS_factors=True):
    """Custom function for calculating the Sensitivity Function of the network"""
    return True


def sens_factors_cm_v1(_net, _fsp_bus, _br_array, _trafo_array, _trafo3w_array, _dp, _dq, _only_ds=True):
    """
    Custom function for calculating the Sensitivity factors of the network for congestion management.

    :param _net: Pandapower network.

    :param _fsp_bus: List of bus included as flexibility service provider.

    :param _br_array: Numpy array of the congested branch in the network.

    :param _trafo_array: Numpy array of the congested trafo 2 windings in the network.

    :param _trafo3w_array: Numpy array of the congested trafo 3 windings in the network.

    :param _dp: Active power deviation.

    :param _dq: Reactive power deviation.

    :param _only_ds: Boolean value (True/False) to evaluate only dSdP and dSdQ matrix
     or evaluate PTDF and QTDF matrix as well.
    """
    return True


def calc_sens_factors_cm(_net, _fsp_file, _fsp_bus, _ts_res, _clines, _ctrafo, _ctrafo3w, _hours, _dp, _dq,
                         _path_matrix, _sim_tag, _sens_type, _sens_mode='VSGEN', _version=1, _only_ds=True):
    """
    Evaluate the Congestion Management sensitivity factors (PTDF, QTDF & dSdP and dSdQ) for each congested hours.

    :param _net: Pandapower network.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _fsp_bus: List of bus included as flexibility service provider.

    :param _ts_res: Dictionary of the timeseries simulation results.

    :param _clines: Dataframe of the congested lines in the network.

    :param _ctrafo: Dataframe of the congested trafo 2 windings in the network.

    :param _ctrafo3w: Dataframe of the congested trafo 3 windings in the network.

    :param _hours: List of hours with congestions.

    :param _path_matrix: Directory where to save sensitivity factor files.

    :param _sim_tag: Simulation tag.

    :param _sens_type: Product from which the matrix are evaluated ('P', 'Q' or 'PQ').

    :param _sens_mode: Sensitivity mode on which evalute the matrix.

    :param _version: Function Version to which evaluate the sensitivity factors.

    :param _dp: Active power deviation.

    :param _dq: Reactive power deviation.

    :param _only_ds: Boolean value (True/False) to evaluate only dSdP and dSdQ matrix
     or evaluate PTDF and QTDF matrix as well.
    """
    return True


def fasor2complex(_magnitude, _angle):
    """
    Evaluate the real and imaginary part of the fasor value.

    :param _magnitude:

    :param _angle:
    """
    return True


def sens_factors_vc(_net, _fsp_bus, _pilot_bus, _dp, _path_matrix):
    """
    Custom function for calculating the Sensitivity factor of the network for voltage control.

    :param _net: Pandapower network.

    :param _fsp_bus: List of bus included as flexibility service provider.

    :param _pilot_bus: List of bus considered as pilot buses in the simulation.

    :param _dp: Active power deviation.

    :param _path_matrix: Directory where to save sensitivity factor files.
    """
    return True


def calc_sens_factors_vc(_net, _fsp_file, _fsp_bus, _ts_res, _pilot_bus, _hours, _delta_v, _path_matrix, _sim_tag,
                         _sens_type, _force_hvm, _save_inv_js):
    """
    Evaluate the Voltage Control sensitivity factors for each congested hours.

    :param _net: Pandapower network.

    :param _fsp_file: Dictionary of file for each resource type.

    :param _fsp_bus: List of bus included as flexibility service provider.

    :param _ts_res: Dictionary of the timeseries simulation results.

    :param _pilot_bus: List of the selected pilot buses.

    :param _hours: List of hours with congestions.

    :param _delta_v: Power Variation for each resource.

    :param _path_matrix: Directory where to save sensitivity factor files.

    :param _sim_tag: Simulation tag.

    :param _sens_type: Product from which the matrix are evaluated ('P', 'Q' or 'PQ').

    :param _force_hvm: Force evaluating hvm sensitivity matrix.

    :param _save_inv_js: Force saving the J22inv and J12inv to .csv.
    """
    return True
