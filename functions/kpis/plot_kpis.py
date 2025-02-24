def bar_increase_hc(_df, _figsize, _titles, _labels, _ylim):
    """
    Bar plot of the Increase Hosting Capacity in percentage.

    :param _df: Data to be plotted.

    :param _figsize: Size of the figure.

    :param _titles: Title of the figure.

    :param _labels: Labels of the figure.

    :param _ylim: Limits of the y-axis.
    """
    return True


def bar_avoid_violations(_df1, _df2, _figsize, _titles, _labels, _ylim):
    """
    Bar plot of the Avoided Violations.

    :param _df1: Data1 to be plotted.

    :param _df2: Data2 to be plotted.

    :param _figsize: Size of the figure.

    :param _titles: Title of the figure.

    :param _labels: Labels of the figure.

    :param _ylim: Limits of the y-axis.
    """
    return True


def bar3d_avoid_violations(_x, _y, _dz, _figsize, _mrk_products, _fsps, _label_pos, _zlabel, _titles):
    """
    Bar 3D Plot of the Avoided Violations.

    :param _x: Dictionary of x-axis position per market products as keys.

    :param _y: Dictionary of y-axis position per market products as keys.

    :param _dz: Dictionary of z-axis value per market products as keys.

    :param _figsize: Size of the figure.

    :param _mrk_products: List of the market products

    :param _fsps: List of the fsp capacity.

    :param _label_pos: List of x-axis position of the label.

    :param _zlabel: Title of the z-axis.

    :param _titles: Title of the figure.
    """
    return True


def bar3d_avoid_violations_element(_x, _y, _dz, _figsize, _mrk_products, _key_elem, _fsps, _label_pos, _zlabel, _titles):
    """
    Bar 3D Plot of the Number of Avoided Violations.

    :param _x: Dictionary of x-axis position per market products as keys.

    :param _y: Dictionary of y-axis position per market products as keys.

    :param _dz: Dictionary of z-axis value per market products as keys.

    :param _figsize: Size of the figure.

    :param _mrk_products: List of the market products.

    :param _key_elem: List of the Element that are compared by the number of avoided comngestion.

    :param _fsps: List of the fsp capacity.

    :param _label_pos: List of x-axis position of the label.

    :param _zlabel: Title of the z-axis.

    :param _titles: Title of the figure.
    """
    return True


def density_plot_bus(_data, _scenarios, _title):
    """
    Density plot of the bus voltage and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.
    """
    return True


def histogram_plot_bus(_data, _scenarios, _title, _xlim, _ylim):
    """
    Density plot of the bus voltage and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.

    :param _xlim: Limits of the x-axis.

    :param _ylim: Limits of the y-axis.
    """
    return True


def density_plot_lines(_data, _scenarios, _title):
    """
    Density plot of the Lines and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.
    """
    return True


def histogram_plot_lines(_data, _scenarios, _title, _xlim, _ylim):
    """
    Density plot of the lines and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.

    :param _xlim: Limits of the x-axis.

    :param _ylim: Limits of the y-axis.
    """
    return True
