from gurobipy import GRB, tuplelist


def general_variables(_grb_model, _n_pb, _n_cm, _n_t):
    """
    Define Variables of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _n_pb: Number of Pilot buses.
    :param _n_cm: Number of congestions.
    :param _n_t: Number of intervals.
    """
    _vars_dict, _list_general_vars = general_variables_v1(grb_model=_grb_model, n_pb=_n_pb, n_cm=_n_cm, n_t=_n_t)
    return _vars_dict, _list_general_vars


def general_variables_v0(grb_model, n_pb, n_cm, n_t):
    """Define General Variables of the Local Market Model."""
    _vars_dict = dict()
    # Slack Variable for Optim (per unit version)
    shape_var_vc = (n_pb, n_t)
    alpha_DVpu = grb_model.addMVar(shape=shape_var_vc, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='alpha_DVpu')
    _vars_dict['alpha_DVpu'] = alpha_DVpu
    # Voltage After the market voltage magnitude on pilot busses VA
    VA = grb_model.addMVar(shape=shape_var_vc, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='VA')
    _vars_dict['VA'] = VA
    ABS_alphaDVpu = grb_model.addMVar(shape=shape_var_vc, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_alphaDVpu')
    _vars_dict['ABS_alpha_DVpu'] = ABS_alphaDVpu
    # Congestion Management Variables
    shape_var_cm = (n_cm, n_t)
    # slack variable for CM flex not-supplied
    beta = grb_model.addMVar(shape=shape_var_cm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')
    _vars_dict['beta'] = beta
    # abs(slack variable for CM flex not-supplied)
    ABS_beta = grb_model.addMVar(shape=shape_var_cm, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_beta')
    _vars_dict['ABS_beta'] = ABS_beta
    list_general_vars = ['alpha_DVpu', 'VA', 'ABS_alpha_DVpu', 'beta', 'ABS_beta']
    return _vars_dict, list_general_vars


def general_variables_v1(grb_model, n_pb, n_cm, n_t):
    """Define General Variables of the Local Market Model."""
    _vars_dict = dict()
    # Slack Variable for Optim (per unit version)
    shape_var_vc = tuplelist([(_f, _t) for _f in range(n_pb) for _t in range(n_t)])
    alpha_DVpu = grb_model.addVars(shape_var_vc, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='alpha_DVpu')
    _vars_dict['alpha_DVpu'] = alpha_DVpu
    # Voltage After the market voltage magnitude on pilot busses VA
    VA = grb_model.addVars(shape_var_vc, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='VA')
    _vars_dict['VA'] = VA
    ABS_alphaDVpu = grb_model.addVars(shape_var_vc, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_alphaDVpu')
    _vars_dict['ABS_alpha_DVpu'] = ABS_alphaDVpu
    # Congestion Management Variables
    shape_var_cm = tuplelist([(_f, _t) for _f in range(n_cm) for _t in range(n_t)])
    # slack variable for CM flex not-supplied
    beta = grb_model.addVars(shape_var_cm, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='beta')
    _vars_dict['beta'] = beta
    # abs(slack variable for CM flex not-supplied)
    ABS_beta = grb_model.addVars(shape_var_cm, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='ABS_beta')
    _vars_dict['ABS_beta'] = ABS_beta
    list_general_vars = ['alpha_DVpu', 'VA', 'ABS_alpha_DVpu', 'beta', 'ABS_beta']
    return _vars_dict, list_general_vars


def general_constraints(_grb_model, _vars_dict):
    """Define Constraints of the Local Market Model selecting the more appropriate/updated function."""
    _new_grb_model = general_constraints_v1(grb_model=_grb_model, _vars_dict=_vars_dict)
    return _new_grb_model


def general_constraints_v0(grb_model, _vars_dict):
    """Define Constraints of the Local Market Model."""
    beta = _vars_dict['beta']
    ABS_beta = _vars_dict['ABS_beta']
    alpha_dV = _vars_dict['alpha_DVpu']
    ABS_alpha_DVpu = _vars_dict['ABS_alpha_DVpu']

    # Absolute value for beta (CM flexibility not-supplied)
    _idx_beta = 0
    for x, y in zip(beta.tolist(), ABS_beta.tolist()):
        _idx_beta_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_beta = 'abs_beta_#' + str(_idx_beta) + '_t' + str(_idx_beta_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_beta)
            _idx_beta_t += 1
        _idx_beta += 1
    # Absolute value for alpha (VC flexibility not-supplied)
    _idx_alpha = 0
    for x, y in zip(alpha_dV.tolist(), ABS_alpha_DVpu.tolist()):
        _idx_alpha_t = 0
        for x1, y1 in zip(x, y):
            _name_abs_alpha = 'abs_alpha_#' + str(_idx_alpha) + '_t' + str(_idx_alpha_t)
            grb_model.addGenConstrAbs(y1, x1, name=_name_abs_alpha)
            _idx_alpha_t += 1
        _idx_alpha += 1
    return grb_model


def general_constraints_v1(grb_model, _vars_dict):
    """Define Constraints of the Local Market Model."""
    beta = _vars_dict['beta']
    ABS_beta = _vars_dict['ABS_beta']
    alpha_dV = _vars_dict['alpha_DVpu']
    ABS_alpha_DVpu = _vars_dict['ABS_alpha_DVpu']

    # Absolute value for beta (CM flexibility not-supplied)
    for x, y in zip(beta.keys(), ABS_beta.keys()):
        _name_abs_beta = 'abs_beta_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_beta[y], beta[x], name=_name_abs_beta)
    # Absolute value for alpha (VC flexibility not-supplied)
    for x, y in zip(alpha_dV.keys(), ABS_alpha_DVpu.keys()):
        _name_abs_alpha = 'abs_alpha_#' + str(x[0]) + '_t' + str(x[1])
        grb_model.addGenConstrAbs(ABS_alpha_DVpu[y], alpha_dV[x], name=_name_abs_alpha)
    return grb_model
