def general_variables(_grb_model, _n_pb, _n_cm, _n_t):
    """
    Define Variables of the Load Resource selecting the more appropriate/updated function.
    :param _grb_model: Object that represents the gurobipy model.
    :param _n_pb: Number of Pilot buses.
    :param _n_cm: Number of congestions.
    :param _n_t: Number of intervals.
    """
    return True


def general_variables_v0(grb_model, n_pb, n_cm, n_t):
    """Define General Variables of the Local Market Model."""
    return True


def general_variables_v1(grb_model, n_pb, n_cm, n_t):
    """Define General Variables of the Local Market Model."""
    return True


def general_constraints(_grb_model, _vars_dict):
    """Define Constraints of the Local Market Model selecting the more appropriate/updated function."""
    return True


def general_constraints_v0(grb_model, _vars_dict):
    """Define Constraints of the Local Market Model."""
    return True


def general_constraints_v1(grb_model, _vars_dict):
    """Define Constraints of the Local Market Model."""
    return True
