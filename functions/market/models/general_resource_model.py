def res_params(_n_fsp, _sim_period, _fsp_data):
    """
    Define Input Data of the Load Resource.
    :param _n_fsp: Number of FSP of type Load resource.
    :param _sim_period: List of hours to consider for the study.
    :param _fsp_data: Dictionary of the Data associated
    """
    return True


def define_variables(_grb_model, _n_fsp, _n_t, _contr, _type=''):
    """
    Define Variables of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _n_fsp: Number of flexibility service providers.
    :param _n_t: Number of intervals.
    :param _contr: Contribution of these resource type.
    :param _type: Type of resource.
    """
    return True


def resource_variables(grb_model, n_fsp, n_t, _contr, _type=''):
    """Define Variables of the Load Resource."""
    return True


def define_constraints(_grb_model, _vars_dict, _fsp_data, _idx_res, _interval, _cap_cstr, _int_soc=False, _dt=1, _type=''):
    """
    Define Constraints of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _vars_dict: Dictionary containing the variables of the gurobi model.
    :param _fsp_data: Dictionary containing the resource parameters.
    :param _idx_res: Index of the resource to which the constraints must be added.
    :param _interval: Index of the hour to which the constraints must be added.
    :param _cap_cstr: Boolean value to add the capability curve constraint of the resource.
    :param _int_soc: Boolean value to add the inter-temporal constraint of the resource.
    :param _dt: Delta time of the model
    :param _type: Type of resource.
    """
    return True


def resource_constraints(grb_model, _vars_dict, fsp_data, i, t, _cap_cstr, _type=''):
    """Define Constraints of the Load Resource."""
    return True


def fsp_resource_converter(_data, _k_opt=1, _class_fsp=''):
    """
    Adapt the fsp generation info into usable data for the local flexibility market optimization.


    :param _data: Dictionary containing all the information useful for the conversion into usable data. The useful data
        are the pandapower network (net), the dictionary with the timeseries simulation (ts_res), the dataframe with the fsp
        data involved in the simulation (fsp_data), the output root where to save files (out_root), the list of hours with
        network constraint violations (hours), the bidding strategy of the fsp element (bid_strat), the multiplier
        coefficient of the load (scen_factor_load), the multiplier coefficient of the generation (scen_factor_gen),
        the active power bid up coefficient (kbid_p_up), the active power bid down coefficient (kbid_p_dwn),
        the reactive power bid up coefficient (kbid_q_up), the reactive power down up coefficient (kbid_q_dwn) and
        the simulation tag (sim_tag).


    :param _k_opt: Margin for bidding below the full capacity - Safe for optimization convergence.


    :param _class_fsp: String that define the name of the fsp typology.
    """
    return True
