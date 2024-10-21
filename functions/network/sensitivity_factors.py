import os
import gc
import math
import warnings
import copy as cp
import numpy as np
import pandas as pd
import pandapower as pp
from tqdm.auto import tqdm
from functions import file_io as io_file
import functions.network.run_network as rnet


gc.collect()
warnings.filterwarnings("ignore", category=FutureWarning)


def _copy_net_essential(_pp_network):
    """
    Copy essential elements of a pandapower network.

    :param _pp_network: Pandapower network

    :return _new_net: Copy of the essential elements of the pandapower network.
    """
    _new_net = pp.create_empty_network()
    _new_net.bus = _pp_network.bus.copy()
    _new_net.load = _pp_network.load.copy()
    _new_net.sgen = _pp_network.sgen.copy()
    _new_net.gen = _pp_network.gen.copy()
    _new_net.switch = _pp_network.switch.copy()
    _new_net.shunt = _pp_network.shunt.copy()
    _new_net.ext_grid = _pp_network.ext_grid.copy()
    _new_net.line = _pp_network.line.copy()
    _new_net.trafo = _pp_network.trafo.copy()
    _new_net.trafo3w = _pp_network.trafo3w.copy()
    return _new_net


def create_vsgen(_net, _bus, _p_q, _delta):
    """
    Custom function to create a virtual generator on a specific bus in the network
    with an active/reactive power equal to delta.

    :param _net: Pandapower network.

    :param _bus: Bus in which the virtual generator will be included.

    :param _p_q: Flag to identify if the generator will inject active or reactive power.

    :param _delta: Active/Reactive power of the virtual generator.
    """
    index = len(_net.sgen)
    nt = _net.sgen.type.iloc[index - 1]

    if _p_q == 'p':
        pp.create_sgen(_net, name='GVirt', index=index, bus=_bus, p_mw=_delta, q_mvar=0, type=nt, min_p_mw=0,
                       max_p_mw=_delta, min_q_mvar=0, max_q_mvar=0, controllable=True)
    elif _p_q == 'q':
        pp.create_sgen(_net, name='GVirt', index=index, bus=_bus, p_mw=0, q_mvar=_delta, type=nt, min_p_mw=0,
                       max_p_mw=0, min_q_mvar=0, max_q_mvar=_delta, controllable=True)
    else:
        raise UserWarning('Check p - q (p for active, q for reactive).')
    return True


def sens_factors_cm(net0, Mode, FSP_data, FSPBranchArray, FSPTrafoArray, FSPTrafo3wArray, deltaP_coeff, deltaQ_coeff,
                    only_dS_factors=True):
    """Custom function for calculating the Sensitivity Function of the network"""
    pp.toolbox.drop_elements_simple(net0, 'controller', net0.controller.index)
    pp.toolbox.clear_result_tables(net0)
    _ = gc.collect()

    net1 = _copy_net_essential(net0)

    # Identify fsp buses according FSP type: load, generation, and storage
    if Mode == 'FSP':
        FSPBusArray = []
        flatType = []

        for i in FSP_data.index:
            if FSP_data.bus[i] in net1.load.bus.tolist():
                FSPBusArray.append(int(-FSP_data.bus[i]))
                flatType.append('l')
            elif FSP_data.bus[i] in net1.sgen.bus.tolist():
                FSPBusArray.append(int(FSP_data.bus[i]))
                flatType.append('sg')
            elif FSP_data.bus[i] in net1.storage.bus.tolist():
                # Positive if the storage is injected, otherwise it should be updated to negative
                FSPBusArray.append(int(-FSP_data.bus[i]))
                flatType.append('stl')
            else:
                print("WARNING: Error of FSP type")

            # if FSP_data.type[i] == "load":
            #     FSPBusArray.append(int(-FSP_data.bus[i]))
            #     flatType.append('l')
            # elif FSP_data.type[i] == "generation":
            #     FSPBusArray.append(int(FSP_data.bus[i]))
            #     flatType.append('sg')
            # elif FSP_data.type[i] == "storage":
            #     # Positive if the storage is injected, otherwise it should be updated to negative
            #     FSPBusArray.append(int(-FSP_data.bus[i]))
            #     flatType.append('stl')
            # else:
            #     print("WARNING: Error of FSP type")
            #     # TODO: consider replacing print with raise UserWarning("Error of FSP type, types: load, generation or storage not defined in FSP file")
        FSPBusArray = np.array(FSPBusArray)
        flatType = np.array(flatType)
    else:
        FSPBusArray = FSP_data.bus.astype(int).tolist()

    # Power Flow zero
    pp.runpp(net0, algorithm='nr')

    TotalBusArray = net0.bus.index
    TotalBranchArray = net0.line.index
    TotalTrafoArray = net0.trafo.index
    TotalTrafo3wArray = net0.trafo3w.index

    if not only_dS_factors:
        shape_pqtdf = (len(FSPBranchArray) + len(FSPTrafoArray) + len(FSPTrafo3wArray), len(FSPBusArray))
        PTDFMatrix = pd.DataFrame(np.zeros(shape=shape_pqtdf))
        QTDFMatrix = pd.DataFrame(np.zeros(shape=shape_pqtdf))

    shape_dsdpq = (len(FSPBranchArray) + len(FSPTrafoArray) + len(FSPTrafo3wArray), len(FSPBusArray))
    dSdPMatrix = pd.DataFrame(np.zeros(shape=shape_dsdpq))
    dSdQMatrix = pd.DataFrame(np.zeros(shape=shape_dsdpq))

    colname = []
    rawname = []
    n = 0

    if (not all(i in TotalBusArray for i in map(abs, FSPBusArray))
            or not all(i in TotalBranchArray for i in map(abs, FSPBranchArray))
            or not all(i in TotalTrafoArray for i in map(abs, FSPTrafoArray))
            or not all(i in TotalTrafo3wArray for i in map(abs, FSPTrafo3wArray))):
        print("WARNING: FSP Bus or Branch or trafo vectors contain errors")
        # return

    # ---------------------------------- Obtaining PTDF and Sensitivity Factor dS/dP ----------------------------------
    for i in FSPBusArray:

        if Mode == 'VSGEN':
            create_vsgen(net1, abs(i), 'p', deltaP_coeff)
            pbus_zero = net0.res_bus.p_mw.iloc[abs(i)]
        elif Mode == 'FSP':
            if math.copysign(1, FSPBusArray[n]) == 1:
                if not net1.gen.empty and (
                        FSPBusArray[n] in set(net1.gen.bus)) and False:  # Deshabilitado para estre proyecto
                    ind = net1.gen[net1.gen['bus'] == abs(i)].index[0]
                    net1.gen.p_mw[ind] = net1.gen.p_mw[ind] + abs(deltaP_coeff)

                    pbus_zero = net0.res_bus.p_mw.iloc[abs(i)]

                elif not net1.sgen.empty and (FSPBusArray[n] in set(net1.sgen.bus)) and flatType[n] == 'sg':
                    ind = net1.sgen[net1.sgen['bus'] == abs(i)].index[0]
                    net1.sgen.p_mw[ind] = net1.sgen.p_mw[ind] + abs(deltaP_coeff)

                    pbus_zero = net0.res_bus.p_mw.iloc[abs(i)]

                elif not net1.storage.empty and (FSPBusArray[n] in set(
                        net1.storage.bus)) and False:  # deshabilitado hasta que se defina storage como gen
                    ind = net1.storage[net1.storage['bus'] == abs(i)].index[0]
                    net1.storage.p_mw[ind] = net1.storage.p_mw[ind] + abs(deltaP_coeff)

                    pbus_zero = net0.res_bus.p_mw.iloc[abs(i)]

                elif net1.gen.empty and net1.sgen.empty and net1.storage.empty:
                    print("WARNING: No Generators are connected to modify their power in the network")

                else:
                    print("WARNING: No Generators are connected to modify their power in the bus", FSPBusArray[n])
            else:
                if not net1.load.empty and (abs(FSPBusArray[n]) in set(net1.load.bus)) and flatType[n] == 'l':
                    ind = net1.load[net1.load['bus'] == abs(i)].index[0]
                    net1.load.p_mw[ind] = net1.load.p_mw[ind] - abs(deltaP_coeff)

                    pbus_zero = net0.res_bus.p_mw.iloc[abs(i)]

                elif not net1.storage.empty and (abs(FSPBusArray[n]) in set(net1.storage.bus)) and flatType[n] == 'stl':
                    ind = net1.storage[net1.storage['bus'] == abs(i)].index[0]
                    net1.storage.p_mw[ind] = net1.storage.p_mw[ind] - abs(deltaP_coeff)

                    pbus_zero = net0.res_bus.p_mw.iloc[abs(i)]

                elif net1.load.empty and net1.storage.empty:
                    print("WARNING: No Loads are connected to modify their power")

                else:
                    print("WARNING: No Loads are connected to modify their power in the bus", abs(FSPBusArray[n]))
        else:
            print("WARNING: Error in Mode")

        m = 0
        # print("running PF on net1")
        pp.runpp(net1, algorithm='nr')

        for j in FSPBranchArray:
            Pmn_zero = max(abs(net0.res_line.p_from_mw[j]), abs(net0.res_line.p_to_mw[j]))
            Pmn_plusone = max(abs(net1.res_line.p_from_mw[j]), abs(net1.res_line.p_to_mw[j]))

            pbus_uno = net1.res_bus.p_mw.iloc[abs(i)]

            if not only_dS_factors:
                PTDFMatrix[n][m] = (Pmn_plusone - Pmn_zero) / deltaP_coeff

            Qmn_zero = max(abs(net0.res_line.q_from_mvar[j]), abs(net0.res_line.q_from_mvar[j]))
            Qmn_plusone = max(abs(net1.res_line.q_from_mvar[j]), abs(net1.res_line.q_from_mvar[j]))

            Smn_zero = math.sqrt(Pmn_zero ** 2 + Qmn_zero ** 2)
            Smn_plusone = math.sqrt(Pmn_plusone ** 2 + Qmn_plusone ** 2)
            dSdPMatrix[n][m] = (Smn_plusone - Smn_zero) / deltaP_coeff

            m += 1
            rawname.append(net0.line.name[j])

        for k in FSPTrafoArray:
            Pmn_zero = max(abs(net0.res_trafo.p_hv_mw[k]), abs(net0.res_trafo.p_lv_mw[k]))
            Pmn_plusone = max(abs(net1.res_trafo.p_hv_mw[k]), abs(net1.res_trafo.p_lv_mw[k]))

            pbus_uno = net1.res_bus.p_mw.iloc[abs(i)]

            if not only_dS_factors:
                PTDFMatrix[n][m] = (Pmn_plusone - Pmn_zero) / deltaP_coeff

            Qmn_zero = max(abs(net0.res_trafo.q_hv_mvar[k]), abs(net0.res_trafo.q_lv_mvar[k]))
            Qmn_plusone = max(abs(net1.res_trafo.q_hv_mvar[k]), abs(net1.res_trafo.q_lv_mvar[k]))

            Smn_zero = math.sqrt(Pmn_zero ** 2 + Qmn_zero ** 2)
            Smn_plusone = math.sqrt(Pmn_plusone ** 2 + Qmn_plusone ** 2)
            dSdPMatrix[n][m] = (Smn_plusone - Smn_zero) / deltaP_coeff

            m += 1
            rawname.append(net0.trafo.name[k])

        for l in FSPTrafo3wArray:
            Pmn_zero = max(abs(net0.res_trafo3w.p_hv_mw[l]),
                           abs(net0.res_trafo3w.p_mv_mw[l] + net0.res_trafo3w.p_lv_mw[l]))
            Pmn_plusone = max(abs(net1.res_trafo3w.p_hv_mw[l]),
                              abs(net1.res_trafo3w.p_mv_mw[l] + net1.res_trafo3w.p_lv_mw[l]))

            pbus_uno = net1.res_bus.p_mw.iloc[abs(i)]

            if not only_dS_factors:
                PTDFMatrix[n][m] = (Pmn_plusone - Pmn_zero) / deltaP_coeff

            Qmn_zero = max(abs(net0.res_trafo3w.q_hv_mvar[l]),
                           abs(net0.res_trafo3w.q_mv_mvar[l] + net0.res_trafo3w.q_lv_mvar[l]))
            Qmn_plusone = max(abs(net1.res_trafo3w.q_hv_mvar[l]),
                              abs(net1.res_trafo3w.q_mv_mvar[l] + net1.res_trafo3w.q_lv_mvar[l]))

            Smn_zero = math.sqrt(Pmn_zero ** 2 + Qmn_zero ** 2)
            Smn_plusone = math.sqrt(Pmn_plusone ** 2 + Qmn_plusone ** 2)
            dSdPMatrix[n][m] = (Smn_plusone - Smn_zero) / deltaP_coeff

            m += 1
            rawname.append(net0.trafo3w.name[l])

        n += 1
        colname.append(net0.bus.index[abs(i)])
        del net1
        _ = gc.collect()
        net1 = _copy_net_essential(net0)

    # clean RAM after costly for-loop
    _ = gc.collect()

    # Initializing variables
    n = 0
    net1 = _copy_net_essential(net0)
    # Power Flow zero
    pp.runpp(net0, algorithm='nr')

    # ---------------------------------- Obtaining QTDF and Sensitivity Factor dS/dQ ----------------------------------
    for i in FSPBusArray:

        if Mode == 'VSGEN':
            create_vsgen(net1, abs(i), 'q', deltaQ_coeff)
            qbus_zero = net0.res_bus.q_mvar.iloc[abs(i)]

        elif Mode == 'FSP':
            if math.copysign(1, FSPBusArray[n]) == 1:
                if not net1.gen.empty and (FSPBusArray[n] in set(net1.gen.bus)) and False:
                    ind = net1.gen[net1.gen['bus'] == abs(i)].index[0]
                    net1.gen.q_mvar[ind] = net1.gen.q_mvar[ind] + abs(deltaQ_coeff)

                    qbus_zero = net0.res_bus.q_mvar.iloc[abs(i)]

                elif not net1.sgen.empty and (FSPBusArray[n] in set(net1.sgen.bus)) and flatType[n] == 'sg':
                    ind = net1.sgen[net1.sgen['bus'] == abs(i)].index[0]
                    net1.sgen.q_mvar[ind] = net1.sgen.q_mvar[ind] + abs(deltaQ_coeff)

                    qbus_zero = net0.res_bus.q_mvar.iloc[abs(i)]

                elif not net1.storage.empty and (FSPBusArray[n] in set(net1.storage.bus)) and False:
                    ind = net1.storage[net1.storage['bus'] == abs(i)].index[0]
                    net1.storage.q_mvar[ind] = net1.storage.q_mvar[ind] + abs(deltaQ_coeff)

                    qbus_zero = net0.res_bus.q_mvar.iloc[abs(i)]

                elif net1.gen.empty and net1.sgen.empty and net1.storage.empty:
                    print("WARNING: No Generators are connected to modify their power in the network")

                else:
                    print("WARNING: No Generators are connected to modify their power in the bus", FSPBusArray[n])

            else:
                if not net1.load.empty and (abs(FSPBusArray[n]) in set(net1.load.bus)) and flatType[n] == 'l':
                    ind = net1.load[net1.load['bus'] == abs(i)].index[0]
                    net1.load.q_mvar[ind] = net1.load.q_mvar[ind] - abs(deltaQ_coeff)

                    qbus_zero = net0.res_bus.q_mvar.iloc[abs(i)]

                elif not net1.storage.empty and (abs(FSPBusArray[n]) in set(net1.storage.bus)) and flatType[n] == 'stl':
                    ind = net1.storage[net1.storage['bus'] == abs(i)].index[0]
                    net1.storage.q_mvar[ind] = net1.storage.q_mvar[ind] - abs(deltaQ_coeff)

                    qbus_zero = net0.res_bus.q_mvar.iloc[abs(i)]

                elif net1.load.empty and net1.storage.empty:
                    print("WARNING: No Loads are connected to modify their power")

                else:
                    print("WARNING: No Loads are connected to modify their power in the bus", abs(FSPBusArray[n]))

        else:
            print("WARNING: Error in Mode")

        m = 0
        pp.runpp(net1, algorithm='nr')

        for j in FSPBranchArray:
            Qmn_zero = max(abs(net0.res_line.q_from_mvar[j]), abs(net0.res_line.q_to_mvar[j]))
            Qmn_plusone = max(abs(net1.res_line.q_from_mvar[j]), abs(net1.res_line.q_to_mvar[j]))

            qbus_uno = net1.res_bus.q_mvar.iloc[abs(i)]

            if not only_dS_factors:
                QTDFMatrix[n][m] = (Qmn_plusone - Qmn_zero) / deltaQ_coeff

            Pmn_zero = max(abs(net0.res_line.p_from_mw[j]), abs(net0.res_line.p_to_mw[j]))
            Pmn_plusone = max(abs(net1.res_line.p_from_mw[j]), abs(net1.res_line.p_to_mw[j]))

            Smn_zero = math.sqrt(Pmn_zero ** 2 + Qmn_zero ** 2)
            Smn_plusone = math.sqrt(Pmn_plusone ** 2 + Qmn_plusone ** 2)
            dSdQMatrix[n][m] = (Smn_plusone - Smn_zero) / deltaQ_coeff

            m += 1

        for k in FSPTrafoArray:
            Qmn_zero = max(abs(net0.res_trafo.q_hv_mvar[k]), abs(net0.res_trafo.q_lv_mvar[k]))
            Qmn_plusone = max(abs(net1.res_trafo.q_hv_mvar[k]), abs(net1.res_trafo.q_lv_mvar[k]))

            qbus_uno = net1.res_bus.q_mvar.iloc[abs(i)]

            if not only_dS_factors:
                QTDFMatrix[n][m] = (Qmn_plusone - Qmn_zero) / deltaQ_coeff

            Pmn_zero = max(abs(net0.res_trafo.p_hv_mw[k]), abs(net0.res_trafo.p_lv_mw[k]))
            Pmn_plusone = max(abs(net1.res_trafo.p_hv_mw[k]), abs(net1.res_trafo.p_lv_mw[k]))

            Smn_zero = math.sqrt(Pmn_zero ** 2 + Qmn_zero ** 2)
            Smn_plusone = math.sqrt(Pmn_plusone ** 2 + Qmn_plusone ** 2)
            dSdQMatrix[n][m] = (Smn_plusone - Smn_zero) / deltaQ_coeff

            m += 1

        for l in FSPTrafo3wArray:
            Qmn_zero = max(abs(net0.res_trafo3w.q_hv_mvar[l]),
                           abs(net0.res_trafo3w.q_mv_mvar[l] + net0.res_trafo3w.q_lv_mvar[l]))
            Qmn_plusone = max(abs(net1.res_trafo3w.q_hv_mvar[l]),
                              abs(net1.res_trafo3w.q_mv_mvar[l] + net1.res_trafo3w.q_lv_mvar[l]))

            qbus_uno = net1.res_bus.q_mvar.iloc[abs(i)]

            if not only_dS_factors:
                QTDFMatrix[n][m] = (Qmn_plusone - Qmn_zero) / deltaQ_coeff

            Pmn_zero = max(abs(net0.res_trafo3w.p_hv_mw[l]),
                           abs(net0.res_trafo3w.p_mv_mw[l] + net0.res_trafo3w.p_lv_mw[l]))
            Pmn_plusone = max(abs(net1.res_trafo3w.p_hv_mw[l]),
                              abs(net1.res_trafo3w.p_mv_mw[l] + net1.res_trafo3w.p_lv_mw[l]))

            Smn_zero = math.sqrt(Pmn_zero ** 2 + Qmn_zero ** 2)
            Smn_plusone = math.sqrt(Pmn_plusone ** 2 + Qmn_plusone ** 2)
            dSdQMatrix[n][m] = (Smn_plusone - Smn_zero) / deltaQ_coeff

            m += 1

        n += 1
        del net1
        _ = gc.collect()
        net1 = _copy_net_essential(net0)

    dSdPMatrix.columns = colname
    dSdPMatrix.index = pd.unique(rawname)
    # FIXME: using names from trafos2w, trafo3w and lines means that, it should be enforced in the loading of the input
    #  files that all these elements have individual names, otherwise an error is thrown here.
    #  Consider moving to using indices?

    if not only_dS_factors:
        QTDFMatrix.columns = dSdPMatrix.columns
        QTDFMatrix.index = dSdPMatrix.index
        PTDFMatrix.columns = dSdPMatrix.columns
        PTDFMatrix.index = dSdPMatrix.index

    dSdQMatrix.columns = dSdPMatrix.columns
    dSdQMatrix.index = dSdPMatrix.index

    if only_dS_factors:
        PTDFMatrix = None
        QTDFMatrix = None

    return PTDFMatrix, QTDFMatrix, dSdPMatrix, dSdQMatrix


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
    _ = gc.collect()

    # Starting power flow in initial conditions
    _net0 = cp.deepcopy(_net)
    pp.runpp(_net0)

    # Evaluation network - Active power
    _net1 = cp.deepcopy(_net)
    # Evaluation network - Reactive power
    _net2 = cp.deepcopy(_net)

    _tot_bus = _net.bus.index
    _tot_branch = _net.line.index
    _tot_trafo = _net.trafo.index
    _tot_trafo3w = _net.trafo3w.index

    _ptdf = [list() for _ in range(len(_fsp_bus))]
    _qtdf = [list() for _ in range(len(_fsp_bus))]
    _dsdp = [list() for _ in range(len(_fsp_bus))]
    _dsdq = [list() for _ in range(len(_fsp_bus))]
    rawname = []
    colname = []

    if (not all(i in _tot_bus for i in map(abs, _fsp_bus))
            or not all(i in _tot_branch for i in map(abs, _br_array))
            or not all(i in _tot_trafo for i in map(abs, _trafo_array))
            or not all(i in _tot_trafo3w for i in map(abs, _trafo3w_array))):
        print("WARNING: FSP Bus or Branch or trafo vectors contain errors")

    # ------------------------------- Obtaining PTDF, QTDF and Sensitivity Factor dS/dP -------------------------------
    for _i in range(len(_fsp_bus)):
        create_vsgen(_net1, abs(_fsp_bus[_i]), 'p', _dp)
        create_vsgen(_net2, abs(_fsp_bus[_i]), 'q', _dq)

        # Active Power Network
        pp.runpp(_net1)

        # Reactive Power Network
        pp.runpp(_net2)

        for _j in _br_array:
            p_zero = max(abs(_net0.res_line.p_from_mw[_j]), abs(_net0.res_line.p_to_mw[_j]))
            q_zero = max(abs(_net0.res_line.q_from_mvar[_j]), abs(_net0.res_line.q_to_mvar[_j]))
            p_plusone_1 = max(abs(_net1.res_line.p_from_mw[_j]), abs(_net1.res_line.p_to_mw[_j]))
            q_plusone_1 = max(abs(_net1.res_line.q_from_mvar[_j]), abs(_net1.res_line.q_to_mvar[_j]))
            p_plusone_2 = max(abs(_net2.res_line.p_from_mw[_j]), abs(_net2.res_line.p_to_mw[_j]))
            q_plusone_2 = max(abs(_net2.res_line.q_from_mvar[_j]), abs(_net2.res_line.q_to_mvar[_j]))

            if not _only_ds:
                _ptdf[_i].append((p_plusone_1 - p_zero) / _dp)
                _qtdf[_i].append((q_plusone_2 - q_zero) / _dq)

            smn_zero = math.sqrt(p_zero ** 2 + q_zero ** 2)

            smn_plusone_p = math.sqrt(p_plusone_1 ** 2 + q_plusone_1 ** 2)
            _dsdp[_i].append((smn_plusone_p - smn_zero) / _dp)

            smn_plusone_q = math.sqrt(p_plusone_2 ** 2 + q_plusone_2 ** 2)
            _dsdq[_i].append((smn_plusone_q - smn_zero) / _dq)

            rawname.append(_net.line.name[_j])

        for _k in _trafo_array:
            pmn_zero = max(abs(_net0.res_trafo.p_hv_mw[_k]), abs(_net0.res_trafo.p_lv_mw[_k]))
            pmn_plusone_p = max(abs(_net1.res_trafo.p_hv_mw[_k]), abs(_net1.res_trafo.p_lv_mw[_k]))
            pmn_plusone_q = max(abs(_net2.res_trafo.p_hv_mw[_k]), abs(_net2.res_trafo.p_lv_mw[_k]))

            qmn_zero = max(abs(_net0.res_trafo.q_hv_mvar[_k]), abs(_net0.res_trafo.q_lv_mvar[_k]))
            qmn_plusone_p = max(abs(_net1.res_trafo.q_hv_mvar[_k]), abs(_net1.res_trafo.q_lv_mvar[_k]))
            qmn_plusone_q = max(abs(_net2.res_trafo.q_hv_mvar[_k]), abs(_net2.res_trafo.q_lv_mvar[_k]))

            if not _only_ds:
                _ptdf[_i].append((pmn_plusone_p - pmn_zero) / _dp)
                _qtdf[_i].append((qmn_plusone_q - qmn_zero) / _dq)

            smn_zero = math.sqrt(pmn_zero ** 2 + qmn_zero ** 2)
            smn_plusone_p = math.sqrt(pmn_plusone_p ** 2 + qmn_plusone_p ** 2)
            _dsdp[_i].append((smn_plusone_p - smn_zero) / _dp)

            smn_plusone_q = math.sqrt(pmn_plusone_q ** 2 + qmn_plusone_q ** 2)
            _dsdq[_i].append((smn_plusone_q - smn_zero) / _dq)

            rawname.append(_net.trafo.name[_k])

        for _l in _trafo3w_array:
            pmn_zero = max(abs(_net0.res_trafo3w.p_hv_mw[_l]),
                           abs(_net0.res_trafo3w.p_mv_mw[_l] + _net0.res_trafo3w.p_lv_mw[_l]))
            pmn_plusone_p = max(abs(_net1.res_trafo3w.p_hv_mw[_l]),
                                abs(_net1.res_trafo3w.p_mv_mw[_l] + _net1.res_trafo3w.p_lv_mw[_l]))
            pmn_plusone_q = max(abs(_net2.res_trafo3w.p_hv_mw[_l]),
                                abs(_net2.res_trafo3w.p_mv_mw[_l] + _net2.res_trafo3w.p_lv_mw[_l]))

            qmn_zero = max(abs(_net0.res_trafo3w.q_hv_mvar[_l]),
                           abs(_net0.res_trafo3w.q_mv_mvar[_l] + _net0.res_trafo3w.q_lv_mvar[_l]))
            qmn_plusone_p = max(abs(_net1.res_trafo3w.q_hv_mvar[_l]),
                                abs(_net1.res_trafo3w.q_mv_mvar[_l] + _net1.res_trafo3w.q_lv_mvar[_l]))
            qmn_plusone_q = max(abs(_net2.res_trafo3w.q_hv_mvar[_l]),
                                abs(_net2.res_trafo3w.q_mv_mvar[_l] + _net2.res_trafo3w.q_lv_mvar[_l]))

            if not _only_ds:
                _ptdf[_i].append((pmn_plusone_p - pmn_zero) / _dp)
                _qtdf[_i].append((qmn_plusone_q - qmn_zero) / _dq)

            smn_zero = math.sqrt(pmn_zero ** 2 + qmn_zero ** 2)
            smn_plusone_p = math.sqrt(pmn_plusone_p ** 2 + qmn_plusone_p ** 2)
            _dsdp[_i].append((smn_plusone_p - smn_zero) / _dp)

            smn_plusone_q = math.sqrt(pmn_plusone_q ** 2 + qmn_plusone_q ** 2)
            _dsdq[_i].append((smn_plusone_q - smn_zero) / _dq)

            rawname.append(_net.trafo3w.name[_l])
        _ = gc.collect()
        # custom function to see it reduces the memory usage of the sensitivity factors calculation
        _net1 = cp.deepcopy(_net)
        _net2 = cp.deepcopy(_net)
        colname.append(_net0.bus.index[_fsp_bus[_i]])

    # clean RAM after costly for-loop
    _ = gc.collect()

    _dsdp = pd.DataFrame(_dsdp, index=colname, columns=pd.unique(rawname)).T
    _dsdq = pd.DataFrame(_dsdq, index=colname, columns=pd.unique(rawname)).T

    if not _only_ds:
        _ptdf = pd.DataFrame(_ptdf, index=colname, columns=pd.unique(rawname)).T
        _qtdf = pd.DataFrame(_qtdf, index=colname, columns=pd.unique(rawname)).T

    return _ptdf, _qtdf, _dsdp, _dsdq


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

    _cm_branch = _clines.index.tolist()
    _cm_trafo = _ctrafo.index.tolist()
    _cm_trafo3w = _ctrafo3w.index.tolist()

    load_p = _ts_res['load_p_mw'].T
    load_q = _ts_res['load_q_mvar'].T
    sgen_p = _ts_res['sgen_p_mw'].T
    sgen_q = _ts_res['sgen_q_mvar'].T

    for _fsp_type in _fsp_file.keys():
        if _fsp_file[_fsp_type] != 0:
            _ptdf, _qtdf, _dsdp, _dsdq = 0, 0, 0, 0
            _message = 'Building CM {h_type} sensitivity matrix - FSP type {fsp_type}'.format(fsp_type=_fsp_type, h_type=_sens_type)

            for h in tqdm(_hours, desc=_message):
                t_th = str(h).zfill(2)

                _list_tags_p = ['HCM' + 'dSdP', 'FSP' + _fsp_type, 'h' + t_th, _sim_tag]
                _list_tags_q = ['HCM' + 'dSdQ', 'FSP' + _fsp_type, 'h' + t_th, _sim_tag]
                _list_tags_ptdf = ['HCM' + 'PTDF', 'FSP' + _fsp_type, 'h' + t_th, _sim_tag]
                _list_tags_qtdf = ['HCM' + 'QTDF', 'FSP' + _fsp_type, 'h' + t_th, _sim_tag]
                filename_p = '_'.join(filter(None, _list_tags_p))
                filename_q = '_'.join(filter(None, _list_tags_q))
                filename_ptdf = '_'.join(filter(None, _list_tags_ptdf))
                filename_qtdf = '_'.join(filter(None, _list_tags_qtdf))
                file2sensp_cm = os.path.join(_path_matrix, filename_p + '.csv')
                file2sensq_cm = os.path.join(_path_matrix, filename_q + '.csv')
                if not (os.path.exists(file2sensq_cm) and os.path.exists(file2sensp_cm)):
                    # Update net according to each hour of study
                    pp.toolbox.drop_elements_simple(_net, 'controller', _net.controller.index)
                    pp.toolbox.clear_result_tables(_net)
                    _ = gc.collect()

                    # Custom function to see it reduces the memory usage of the sensitivity factors calculation
                    # _empty_ppnet = _copy_net_essential(_net)
                    _empty_ppnet = cp.deepcopy(_net)
                    _empty_ppnet.load.p_mw = list(load_p[h])
                    _empty_ppnet.load.q_mvar = list(load_q[h])
                    _empty_ppnet.sgen.p_mw = list(sgen_p[h])
                    _empty_ppnet.sgen.q_mvar = list(sgen_q[h])

                    if _version == 0:
                        _ptdf, _qtdf, _dsdp, _dsdq = sens_factors_cm(_empty_ppnet, _sens_mode, _fsp_bus, _cm_branch,
                                                                     _cm_trafo, _cm_trafo3w, _dp, _dq, _only_ds)
                    elif _version == 1:
                        _ptdf, _qtdf, _dsdp, _dsdq = sens_factors_cm_v1(_empty_ppnet, _fsp_bus, _cm_branch, _cm_trafo,
                                                                        _cm_trafo3w, _dp, _dq, _only_ds)

                    if not _only_ds:
                        io_file.save_cvs(_data=_ptdf, _outroot=_path_matrix, _filename=filename_ptdf)
                        io_file.save_cvs(_data=_qtdf, _outroot=_path_matrix, _filename=filename_qtdf)

                    io_file.save_cvs(_data=_dsdp, _outroot=_path_matrix, _filename=filename_p)
                    io_file.save_cvs(_data=_dsdq, _outroot=_path_matrix, _filename=filename_q)
                    _ = gc.collect()

            # print('HVCM {h_type} for FSP {fsp_type} writing complete.\n'.format(h_type=_sens_type, fsp_type=_fsp_type))
    return True


def fasor2complex(_magnitude, _angle):
    """
    Evaluate the real and imaginary part of the fasor value.

    :param _magnitude:

    :param _angle:
    """
    angle_rad = math.radians(_angle)
    part_real = _magnitude * math.cos(angle_rad)
    part_imag = _magnitude * math.sin(angle_rad)
    return complex(part_real, part_imag)


def sens_factors_vc(_net, _fsp_bus, _pilot_bus, _dp, _path_matrix):
    """
    Custom function for calculating the Sensitivity factor of the network for voltage control.

    :param _net: Pandapower network.

    :param _fsp_bus: List of bus included as flexibility service provider.

    :param _pilot_bus: List of bus considered as pilot buses in the simulation.

    :param _dp: Active power deviation.

    :param _path_matrix: Directory where to save sensitivity factor files.
    """
    _ = gc.collect()

    # Delta Q (Reactive Power)
    _dq = _dp

    # Starting power flow in initial conditions
    _net0 = cp.deepcopy(_net)
    pp.runpp(_net0)

    # Evaluation network - Active power
    _net1 = cp.deepcopy(_net)
    # Evaluation network - Reactive power
    _net2 = cp.deepcopy(_net)

    total_bus_array = _net.bus.index

    _hvp = [list() for _ in range(len(_fsp_bus))]
    _hvq = [list() for _ in range(len(_fsp_bus))]
    colname = []

    if not all(i in total_bus_array for i in map(abs, _fsp_bus)) \
            or not all(i in total_bus_array for i in map(abs, _pilot_bus)):
        raise UserWarning('FSP Bus or Branch or trafo vectors contain errors')

    for _i in range(len(_fsp_bus)):
        create_vsgen(_net1, abs(_fsp_bus[_i]), 'p', _dp)
        create_vsgen(_net2, abs(_fsp_bus[_i]), 'q', _dq)

        # Active Power Network
        pp.runpp(_net1)

        # Reactive Power Network
        pp.runpp(_net2)

        for _j in _pilot_bus:
            vmp_zero = fasor2complex(_net0.res_bus.vm_pu[_j], _net0.res_bus.va_degree[_j])
            vmp_uno = fasor2complex(_net1.res_bus.vm_pu[_j], _net1.res_bus.va_degree[_j])
            vmq_uno = fasor2complex(_net2.res_bus.vm_pu[_j], _net2.res_bus.va_degree[_j])

            _hvp[_i].append(abs(vmp_uno - vmp_zero) / _dp)
            _hvq[_i].append(abs(vmq_uno - vmp_zero) / _dq)

        _ = gc.collect()
        # custom function to see it reduces the memory usage of the sensitivity factors calculation
        _net1 = cp.deepcopy(_net)
        _net2 = cp.deepcopy(_net)
        colname.append(_net0.bus.index[_fsp_bus[_i]])

    # clean RAM after costly for-loop
    _ = gc.collect()

    _hvp = pd.DataFrame(_hvp, index=colname, columns=_pilot_bus).T
    _hvq = pd.DataFrame(_hvq, index=colname, columns=_pilot_bus).T
    return _hvp, _hvq


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

    # Get profiles according to the test scenario
    load_p = _ts_res['load_p_mw'].T
    load_q = _ts_res['load_q_mvar'].T
    sgen_p = _ts_res['sgen_p_mw'].T
    sgen_q = _ts_res['sgen_q_mvar'].T

    for _fsp_type in _fsp_file.keys():
        if _fsp_file[_fsp_type] != 0:
            _message = 'Building VC {_h_type} sensitivity matrix - FSP type {_fsp_type}'.format(_fsp_type=_fsp_type, _h_type=_sens_type)

            for h in tqdm(_hours, desc=_message):
                t_th = str(h).zfill(2)

                _list_tags_p = ['HVMP', 'FSP' + _fsp_type, 'h' + t_th, _sim_tag]
                # _list_tags_p = ['HVMP', 'FSP' + _fsp_type, 'h' + t_th]
                _list_tags_q = ['HVMQ', 'FSP' + _fsp_type, 'h' + t_th, _sim_tag]
                # _list_tags_q = ['HVMQ', 'FSP' + _fsp_type, 'h' + t_th]
                _list_hvm_fsp_p = ['HVMPii', 'h' + t_th, _sim_tag]
                # _list_hvm_fsp_p = ['HVMPii', 'h' + t_th]
                _list_hvm_fsp_q = ['HVMQii', 'h' + t_th, _sim_tag]
                # _list_hvm_fsp_q = ['HVMQii', 'h' + t_th]
                filename_p = '_'.join(filter(None, _list_tags_p))
                filename_q = '_'.join(filter(None, _list_tags_q))
                filename_hvm_p = '_'.join(filter(None, _list_hvm_fsp_p))
                filename_hvm_q = '_'.join(filter(None, _list_hvm_fsp_q))
                file2sens_vc_p = os.path.join(_path_matrix, filename_p + '.csv')
                # file2sens_vc_q = os.path.join(_path_matrix, filename_q + '.csv')
                file2hvm_p = os.path.join(_path_matrix, filename_hvm_p + '.csv')
                file2hvm_q = os.path.join(_path_matrix, filename_hvm_q + '.csv')

                if not os.path.exists(file2sens_vc_p) and _force_hvm:
                    # _empty_ppnet = _copy_net_essential(_net)
                    _empty_ppnet = cp.deepcopy(_net)
                    _empty_ppnet.load.p_mw = list(load_p[h])
                    _empty_ppnet.load.q_mvar = list(load_q[h])
                    _empty_ppnet.sgen.p_mw = list(sgen_p[h])
                    _empty_ppnet.sgen.q_mvar = list(sgen_q[h])

                    if not os.path.exists(file2hvm_p) or not os.path.exists(file2hvm_q):
                        _flagdiag = False
                        _hvp, _hvq = sens_factors_vc(_empty_ppnet, _pilot_bus, _pilot_bus, _delta_v, _path_matrix)
                    else:
                        _flagdiag = True
                        _hvp, _hvq = sens_factors_vc(_empty_ppnet, _fsp_bus, _pilot_bus, _delta_v, _path_matrix)

                    if _sens_type == 'Q':
                        raise UserWarning('H Type: P not yet implemented.')
                    elif _sens_type == 'P':
                        raise UserWarning('H Type: Q not yet implemented.')
                    else:
                        if not _flagdiag:
                            # _hvm_pii = pd.Series(np.diag(_hvp.T), index=_pilot_bus)
                            _hvm_pii = pd.DataFrame(np.diag(_hvp.T), index=_pilot_bus, columns=[h])
                            # _hvm_qii = pd.Series(np.diag(_hvq.T), index=_pilot_bus)
                            _hvm_qii = pd.DataFrame(np.diag(_hvq.T), index=_pilot_bus, columns=[h])
                            io_file.save_cvs(_data=pd.DataFrame(_hvm_pii), _outroot=_path_matrix, _filename=filename_hvm_p)
                            io_file.save_cvs(_data=pd.DataFrame(_hvm_qii), _outroot=_path_matrix, _filename=filename_hvm_q)

                        _hvmp = _hvp.T
                        _hvmp.index = _fsp_bus
                        _hvmp.columns = _pilot_bus

                        _hvmq = _hvq.T
                        _hvmq.index = _fsp_bus
                        _hvmq.columns = _pilot_bus

                        io_file.save_cvs(_data=_hvmp, _outroot=_path_matrix, _filename=filename_p)
                        io_file.save_cvs(_data=_hvmq, _outroot=_path_matrix, _filename=filename_q)

                        if _save_inv_js:
                            io_file.save_cvs(_data=_hvp, _outroot=_path_matrix, _filename='myJ22INV_h' + t_th)
                            io_file.save_cvs(_data=_hvq, _outroot=_path_matrix, _filename='myJ12INV_h' + t_th)

        # print('HVM {h_type} for FSP {fsp_type} writing complete.\n'.format(h_type=_sens_type, fsp_type=_fsp_type))
    return True
