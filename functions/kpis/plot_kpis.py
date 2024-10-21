import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()


def bar_increase_hc(_df, _figsize, _titles, _labels, _ylim):
    """
    Bar plot of the Increase Hosting Capacity in percentage.

    :param _df: Data to be plotted.

    :param _figsize: Size of the figure.

    :param _titles: Title of the figure.

    :param _labels: Labels of the figure.

    :param _ylim: Limits of the y-axis.
    """
    fig, ax1 = plt.subplots(figsize=_figsize)
    _df.plot(kind='bar', ax=ax1, color='blue', alpha=0.6, legend=True, width=0.8)

    ax1.set_xlabel(_labels[0], color='mediumblue', fontsize=14)
    ax1.set_ylabel(_labels[1], color='mediumblue', fontsize=14)

    fig.suptitle(_titles[0], fontsize=16, fontweight='bold')
    ax1.set_title(_titles[1], color='darkblue', fontsize=14)

    fig.set_facecolor('ghostwhite')
    ax1.set_facecolor('white')  # ghostwhite
    ax1.patch.set_alpha(0.9)

    if not _ylim:
        fa = 0.1
        l_inf = min(float(np.min(_df)) - (float(np.max(_df)) - float(np.min(_df))) * fa, 0)
        l_sup = max(float(np.max(_df)) + (float(np.max(_df)) - float(np.min(_df))) * fa, 0)
        _ylim = [l_inf, l_sup]

    ax1.set_ylim(_ylim)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=10)
    ax1.yaxis.set_label_coords(-0.06, 0.5)

    plt.grid(color='lightgray', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()
    plt.close()

    return fig


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
    _df2.reset_index(inplace=True)

    fig, ax1 = plt.subplots(figsize=_figsize)
    ax2 = ax1.twinx()

    _df1.plot(kind='bar', ax=ax1, alpha=0.6, legend=False, width=0.8)
    _df2.plot(x='index', y='y', kind='scatter', ax=ax2, marker='o', color='red', alpha=0.6, label=_labels[3], s=30)

    ax1.set_xlabel(_labels[0], color='mediumblue', fontsize=14)
    ax1.set_ylabel(_labels[1], color='mediumblue', fontsize=14)
    ax2.set_ylabel(_labels[2], color='mediumblue', fontsize=14)

    fig.suptitle(_titles[0], fontsize=16, fontweight='bold')
    ax1.set_title(_titles[1], color='darkblue', fontsize=14)

    fig.set_facecolor('ghostwhite')
    ax1.set_facecolor('white')
    ax1.patch.set_alpha(0.9)

    if not _ylim[0]:
        fa = 0.1
        l_inf1 = float(np.min(_df1.values)) - (float(np.max(_df1.values)) - float(np.min(_df1.values))) * fa
        l_sup1 = float(np.max(_df1.values)) + (float(np.max(_df1.values)) - float(np.min(_df1.values))) * fa
        if np.min(_df1.values) == 0:
            _ylim[0] = [0, l_sup1]
        else:
            _ylim[0] = [min(l_inf1, 0), max(l_sup1, 0)]

    if not _ylim[1]:
        fa = 0.1
        l_inf2 = float(np.min(_df2.y)) - (float(np.max(_df2.y)) - float(np.min(_df2.y))) * fa
        l_sup2 = float(np.max(_df2.y)) + (float(np.max(_df2.y)) - float(np.min(_df2.y))) * fa
        _ylim[1] = [min(l_inf2, 0), max(l_sup2, 0)]

    ax1.set_ylim(_ylim[0])
    ax2.set_ylim(_ylim[1])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='center', fontsize=10)

    bars1, labels1 = ax1.get_legend_handles_labels()
    bars2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(bars1 + bars2, labels1 + labels2, loc='best')

    plt.grid(color='lightgray', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.show()

    return fig


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
    _colors = ['yellow', 'red', 'green']

    fig = plt.figure(figsize=_figsize)
    ax1 = fig.add_subplot(111, projection='3d')

    fig.suptitle(_titles[0], fontsize=16, fontweight='bold')
    ax1.set_title(_titles[1], color='darkblue', fontsize=14)

    # plotting 3D bars
    _z = [0]
    idx_color = 0
    for _prod in _mrk_products:
        _dx = [1 for _ in range(len(_x[_prod]))]
        _dy = [1 for _ in range(len(_y[_prod]))]
        ax1.bar3d(_x[_prod], _y[_prod], _z, _dx, _dy, _dz[_prod], color=_colors[idx_color])
        plt.yticks(_y[_prod], _fsps)
        idx_color += 1

    plt.xticks(_label_pos, _mrk_products)

    ax1.set_zlabel(_zlabel)
    plt.show()
    return fig


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
    _colors = ['yellow', 'red', 'green']

    fig = plt.figure(figsize=_figsize)
    ax1 = fig.add_subplot(111, projection='3d')

    fig.suptitle(_titles[0], fontsize=16, fontweight='bold')
    ax1.set_title(_titles[1], color='darkblue', fontsize=14)

    # plotting 3D bars
    _z = [0]
    for _prod in _mrk_products:
        idx_color = 0
        for _elem in _key_elem:
            _dx = [1 for _ in range(len(_x[_prod][_elem]))]
            _dy = [1 for _ in range(len(_y[_prod][_elem]))]
            ax1.bar3d(_x[_prod][_elem], _y[_prod][_elem], _z, _dx, _dy, _dz[_prod][_elem], color=_colors[idx_color])
            plt.yticks(_y[_prod][_elem], _fsps)
            idx_color += 1

    plt.xticks(_label_pos, _mrk_products)

    ax1.legend(_key_elem)

    ax1.set_zlabel(_zlabel)
    plt.show()
    return fig


def density_plot_bus(_data, _scenarios, _title):
    """
    Density plot of the bus voltage and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.
    """
    bus_vmpu_pre = _data['bus_vmpu_pre']
    vbus_max_V_pre = _data['vbus_max_pre']
    vbus_min_V_pre = _data['vbus_min_pre']

    bus_vmpu_post = _data['bus_vmpu_post']
    vbus_max_V_post = _data['vbus_max_post']
    vbus_min_V_post = _data['vbus_min_post']

    _vl_list = _data['vl_lims']
    vl_colors = _data['vl_colors']
    vpu_above_max = dict()
    vpu_below_min = dict()

    vl_lims = {market: idx for idx, market in enumerate(_vl_list)}

    flat = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for name_tab in _scenarios:
        _vl = name_tab.split('_')[3]

        mrk_vl = vl_lims.get(_vl, None)

        if flat == 0:
            vpu_pre = bus_vmpu_pre[name_tab].T.loc[bus_vmpu_post[name_tab].T.index].T
            vpu_pre = vpu_pre.drop(vpu_pre[(vpu_pre == 0).any(axis=1)].index)
            vpu_pre_f = pd.DataFrame(vpu_pre.values.flatten())[0]

            Labelf = name_tab + '_pre'
            # Density Plots
            vpu_pre_f.plot(kind='density', ax=ax1, alpha=0.7, linewidth=2.5, label=Labelf, fontsize=12, linestyle='--')

            # Bar Plots
            vpu_above_max[Labelf] = max(np.count_nonzero(vpu_pre_f[vpu_pre_f > vbus_max_V_pre[mrk_vl]]), 0)
            vpu_below_min[Labelf] = max(np.count_nonzero(vpu_pre_f[vpu_pre_f < vbus_min_V_pre[mrk_vl]]), 0)

            flat = 1

        vpu_post = bus_vmpu_post[name_tab]
        vpu_post = vpu_post.drop(vpu_post[(vpu_post == 0).any(axis=1)].index)
        vpu_post_f = pd.DataFrame(vpu_post.values.flatten())[0]

        # Density Plots
        Labelf2 = name_tab + '_post'
        vpu_post_f.plot(kind='density', ax=ax1, alpha=0.9, linewidth=2, label=Labelf2, fontsize=12)

        # Bar Plots
        vpu_above_max[Labelf2] = max(np.count_nonzero(vpu_post_f[vpu_post_f > vbus_max_V_post[mrk_vl]]), 0)
        vpu_below_min[Labelf2] = max(np.count_nonzero(vpu_post_f[vpu_post_f < vbus_min_V_post[mrk_vl]]), 0)

    df1 = pd.DataFrame.from_dict(vpu_below_min, orient='index', columns=['Below Vmin'])
    df1.plot(kind='bar', ax=ax2, color='blue', alpha=0.5, legend=True, position=1.0, width=0.2)

    df2 = pd.DataFrame.from_dict(vpu_above_max, orient='index', columns=['Above Vmax'])
    df2.plot(kind='bar', ax=ax2, color='red', alpha=0.5, legend=True, position=0.0, width=0.2)

    for mrk_vl in vl_lims.keys():
        ax1.axvline(x=vbus_min_V_pre[vl_lims[mrk_vl]], color=vl_colors[mrk_vl], linestyle='--', label=mrk_vl + '_min')
        ax1.axvline(x=vbus_max_V_pre[vl_lims[mrk_vl]], color=vl_colors[mrk_vl], linestyle='--', label=mrk_vl + '_max')

    ax1.set_xlabel('Voltage Magnitude [p.u.]', color='mediumblue', fontsize=14)
    ax1.set_ylabel('Probability density of bus voltage magnitude values', color='mediumblue', fontsize=14)
    ax2.set_xlabel('Cases', color='mediumblue', fontsize=14)
    ax2.set_ylabel('Number of Deviations', color='mediumblue', fontsize=14)

    fig.suptitle(_title, fontsize=16, fontweight='bold')
    ax1.set_title('Density distribution function', color='darkblue', fontsize=14)
    ax2.set_title('Voltage Violations', color='darkblue', fontsize=14)
    ax1.legend()  # bbox_to_anchor=(1.4, 1)
    ax2.legend()  # bbox_to_anchor=(1.4, 1)

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=10, ha='center', fontsize=10)

    fig.set_facecolor('ghostwhite')
    ax1.set_facecolor('white')  # ghostwhite
    ax1.patch.set_alpha(0.9)
    ax1.grid(color='lightgray', alpha=0.5)  # lightgray

    ax2.set_facecolor('white')  # ghostwhite
    ax2.patch.set_alpha(0.9)
    ax2.grid(color='lightgray', alpha=0.5, axis='y')  # lightgray

    plt.tight_layout()
    plt.show()
    # time.sleep(5)
    # plt.close()
    return fig


def histogram_plot_bus(_data, _scenarios, _title, _xlim, _ylim):
    """
    Density plot of the bus voltage and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.

    :param _xlim: Limits of the x-axis.

    :param _ylim: Limits of the y-axis.
    """
    bus_vmpu_pre = _data['bus_vmpu_pre']
    vbus_max_V_pre = _data['vbus_max_pre']
    vbus_min_V_pre = _data['vbus_min_pre']

    bus_vmpu_post = _data['bus_vmpu_post']
    vbus_max_V_post = _data['vbus_max_post']
    vbus_min_V_post = _data['vbus_min_post']

    vl_colors = _data['vl_colors']
    _vl_list = _data['vl_lims']
    colours = ['orange', 'green', 'red', 'purple', 'brown', 'maroon']
    clr = 0
    vpu_above_max = dict()
    vpu_below_min = dict()
    n_lw = 1

    vl_lims = {market: idx for idx, market in enumerate(_vl_list)}

    flat = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for name_tab in _scenarios:
        _vl = name_tab.split('_')[3]

        mrk_vl = vl_lims.get(_vl, None)
        if flat == 0:
            vpu_pre = bus_vmpu_pre[name_tab].T.loc[bus_vmpu_post[name_tab].T.index].T
            vpu_pre = vpu_pre.drop(vpu_pre[(vpu_pre == 0).any(axis=1)].index)
            vpu_pre_f = pd.DataFrame(vpu_pre.values.flatten())[0]

            Labelf = name_tab + '_pre'
            # Histogram Plots
            counts, bins, _ = ax1.hist(vpu_pre_f, bins=20, alpha=0.0)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax1.plot(bin_centers, counts, color='lightblue', linewidth=3, linestyle='--', label=Labelf)

            # Bar Plots
            vpu_above_max[Labelf] = max(np.count_nonzero(vpu_pre_f[vpu_pre_f > vbus_max_V_pre[mrk_vl]]), 0)
            vpu_below_min[Labelf] = max(np.count_nonzero(vpu_pre_f[vpu_pre_f < vbus_min_V_pre[mrk_vl]]), 0)

            flat = 1

        vpu_post = bus_vmpu_post[name_tab]
        vpu_post = vpu_post.drop(vpu_post[(vpu_post == 0).any(axis=1)].index)
        vpu_post_f = pd.DataFrame(vpu_post.values.flatten())[0]

        # Histogram Plots
        Labelf2 = name_tab + '_post'
        counts, bins, _ = ax1.hist(vpu_post_f, bins=20, alpha=0.0)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax1.plot(bin_centers, counts, color=colours[clr], linewidth=n_lw, linestyle='-', label=Labelf2)
        n_lw = n_lw + 0.25
        clr = clr + 1
        if clr > (len(colours) - 1):
            clr = 0

        # Bar Plots
        vpu_above_max[Labelf2] = max(np.count_nonzero(vpu_post_f[vpu_post_f > vbus_max_V_post[mrk_vl]]), 0)
        vpu_below_min[Labelf2] = max(np.count_nonzero(vpu_post_f[vpu_post_f < vbus_min_V_post[mrk_vl]]), 0)

    df1 = pd.DataFrame.from_dict(vpu_below_min, orient='index', columns=['Below Vmin'])
    df1.plot(kind='bar', ax=ax2, color='blue', alpha=0.5, legend=True, position=1.0, width=0.2)

    df2 = pd.DataFrame.from_dict(vpu_above_max, orient='index', columns=['Above Vmax'])
    df2.plot(kind='bar', ax=ax2, color='red', alpha=0.5, legend=True, position=0.0, width=0.2)

    for mrk_vl in vl_lims.keys():
        ax1.axvline(x=vbus_min_V_pre[vl_lims[mrk_vl]], color=vl_colors[mrk_vl], linestyle='--', label=mrk_vl + '_min')
        ax1.axvline(x=vbus_max_V_pre[vl_lims[mrk_vl]], color=vl_colors[mrk_vl], linestyle='--', label=mrk_vl + '_max')

    ax1.set_xlabel('Voltage Magnitude [p.u.]', color='mediumblue', fontsize=14)
    ax1.set_ylabel('Number of occurrences [nº]', color='mediumblue', fontsize=14)
    ax2.set_xlabel('Cases', color='mediumblue', fontsize=14)
    ax2.set_ylabel('Number of Deviations', color='mediumblue', fontsize=14)

    fig.suptitle(_title, fontsize=16, fontweight='bold')
    ax1.set_title('Voltage for all Buses', color='darkblue', fontsize=14)
    ax2.set_title('Voltage Violations', color='darkblue', fontsize=14)
    ax1.legend()  # bbox_to_anchor=(1.4, 1)
    ax2.legend()  # bbox_to_anchor=(1.4, 1)

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=10, ha='center', fontsize=10)

    if _ylim[0]:
        ax1.set_ylim(_ylim[0])

    if _ylim[1]:
        ax2.set_ylim(_ylim[1])

    if _xlim:
        ax1.set_xlim(_xlim)

    fig.set_facecolor('ghostwhite')
    ax1.set_facecolor('white')  # ghostwhite
    ax1.patch.set_alpha(0.9)
    ax1.grid(color='lightgray', alpha=0.5)  # lightgray

    ax2.set_facecolor('white')  # ghostwhite
    ax2.patch.set_alpha(0.9)
    ax2.grid(color='lightgray', alpha=0.5, axis='y')  # lightgray

    plt.tight_layout()
    plt.show()
    return fig


def density_plot_lines(_data, _scenarios, _title):
    """
    Density plot of the Lines and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.
    """
    lines_loadPerc_pre = _data['lines_ldg_perc_pre']
    i_max_perc_pre = _data['i_max_perc_pre']

    lines_loadPerc_post = _data['lines_ldg_perc_post']
    i_max_perc_post = _data['i_max_perc_post']

    loadp_above_max = dict()

    flat = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for name_tab in _scenarios:

        if flat == 0:
            loadp_pre = lines_loadPerc_pre[name_tab].T.loc[lines_loadPerc_post[name_tab].T.index].T
            loadp_pre = loadp_pre.drop(loadp_pre[(loadp_pre == 0).any(axis=1)].index)
            loadp_pre_f = pd.DataFrame(loadp_pre.values.flatten())[0]

            Labelf = name_tab + '_pre'
            # Density Plots
            loadp_pre_f.plot(kind='density', ax=ax1, alpha=0.7, linewidth=2.5, label=Labelf, fontsize=12,
                             linestyle='--')

            # Bar Plots
            loadp_above_max[Labelf] = max(np.count_nonzero(loadp_pre_f[loadp_pre_f > i_max_perc_pre]), 0)

            flat = 1

        loadp_post = lines_loadPerc_post[name_tab]
        loadp_post = loadp_post.drop(loadp_post[(loadp_post == 0).any(axis=1)].index)
        loadp_post_f = pd.DataFrame(loadp_post.values.flatten())[0]

        # Density Plots
        Labelf2 = name_tab + '_post'
        loadp_post_f.plot(kind='density', ax=ax1, alpha=0.9, linewidth=2, label=Labelf2, fontsize=12)

        # Bar Plots
        loadp_above_max[Labelf2] = max(np.count_nonzero(loadp_post_f[loadp_post_f > i_max_perc_post]), 0)

    df2 = pd.DataFrame.from_dict(loadp_above_max, orient='index',
                                 columns=['Overloaded Lines'])  # @Eliana: only lines or elements?
    df2.rename(columns={'Overloaded Lines': 'Cumulative nº of overloaded lines'},
               inplace=True)  # @Eliana: only lines or elements?
    df2.plot(kind='bar', ax=ax2, color='red', alpha=0.5, legend=True, position=0.5, width=0.2)

    ax1.axvline(x=100, color='red', linestyle='--')  # 100% was written directly

    ax1.set_xlabel('Loading Percentage [%]', color='mediumblue', fontsize=14)
    ax1.set_ylabel('Probability density', color='mediumblue', fontsize=14)
    ax2.set_xlabel('Cases', color='mediumblue', fontsize=14)
    ax2.set_ylabel('Number of occurrences [nº]', color='mediumblue', fontsize=14)

    fig.suptitle(_title, fontsize=16, fontweight='bold')
    ax1.set_title('Density distribution function', color='darkblue', fontsize=14)
    ax2.set_title('Element operating status', color='darkblue', fontsize=14)
    ax1.legend()  # bbox_to_anchor=(1.4, 1)
    ax2.legend()  # bbox_to_anchor=(1.4, 1)

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=10, ha='center', fontsize=10)

    fig.set_facecolor('ghostwhite')
    ax1.set_facecolor('white')  # ghostwhite
    ax1.patch.set_alpha(0.9)
    ax1.grid(color='lightgray', alpha=0.5)  # lightgray

    ax2.set_facecolor('white')  # ghostwhite
    ax2.patch.set_alpha(0.9)
    ax2.grid(color='lightgray', alpha=0.5, axis='y')  # lightgray

    plt.tight_layout()
    plt.show()

    return fig


def histogram_plot_lines(_data, _scenarios, _title, _xlim, _ylim):
    """
    Density plot of the lines and number of congestion avoided.

    :param _data: Dictionary of the required data.

    :param _scenarios: List of scenarios to be adopted.

    :param _title: Title of the figure.

    :param _xlim: Limits of the x-axis.

    :param _ylim: Limits of the y-axis.
    """
    lines_loadPerc_pre = _data['lines_ldg_perc_pre']
    i_max_perc_pre = _data['i_max_perc_pre']

    lines_loadPerc_post = _data['lines_ldg_perc_post']
    i_max_perc_post = _data['i_max_perc_post']

    colours = ['orange', 'green', 'red', 'purple', 'brown', 'maroon']
    clr = 0
    n_lw = 1

    loadp_above_max = {}

    flat = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    for name_tab in _scenarios:

        if flat == 0:
            loadp_pre = (lines_loadPerc_pre[name_tab].T.loc[lines_loadPerc_post[name_tab].T.index]).T
            loadp_pre = loadp_pre.drop(loadp_pre[(loadp_pre == 0).any(axis=1)].index)
            loadp_pre_f = pd.DataFrame(loadp_pre.values.flatten())[0]

            Labelf = name_tab + '_pre'  # comb[1] + '_' + comb[2] + comb[3] + '_pre'
            # Histogram Plots
            counts, bins, _ = ax1.hist(loadp_pre_f, bins=30, alpha=0.0)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            ax1.plot(bin_centers, counts, color='lightblue', linewidth=3, linestyle='--', label=Labelf)

            # Bar Plots
            loadp_above_max[Labelf] = max(np.count_nonzero(loadp_pre_f[loadp_pre_f > i_max_perc_pre]), 0)

            flat = 1

        loadp_post = lines_loadPerc_post[name_tab]
        loadp_post = loadp_post.drop(loadp_post[(loadp_post == 0).any(axis=1)].index)
        loadp_post_f = pd.DataFrame(loadp_post.values.flatten())[0]

        # Histogram Plots
        Labelf2 = name_tab + '_post'
        counts, bins, _ = ax1.hist(loadp_post_f, bins=30, alpha=0.0)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax1.plot(bin_centers, counts, color=colours[clr], linewidth=n_lw, linestyle='-', label=Labelf2)
        n_lw = n_lw + 0.25
        clr = clr + 1
        if clr > (len(colours) - 1):
            clr = 0

        # Bar Plots
        loadp_above_max[Labelf2] = max(np.count_nonzero(loadp_post_f[loadp_post_f > i_max_perc_post]), 0)

    df2 = pd.DataFrame.from_dict(loadp_above_max, orient='index',
                                 columns=['Overloaded Lines'])  # @Eliana: only lines or elements?
    df2.rename(columns={'Overloaded Lines': 'Cumulative nº of overloaded lines'},
               inplace=True)  # @Eliana: only lines or elements?
    df2.plot(kind='bar', ax=ax2, color='red', alpha=0.5, legend=True, position=0.5, width=0.2)

    ax1.axvline(x=100, color='red', linestyle='--')

    ax1.set_xlabel('Loading Percentage [%]', color='mediumblue', fontsize=14)
    ax1.set_ylabel('Number of occurrences [nº]', color='mediumblue', fontsize=14)
    ax2.set_xlabel('Cases', color='mediumblue', fontsize=14)
    ax2.set_ylabel('Number of occurrences [nº]', color='mediumblue', fontsize=14)

    fig.suptitle(_title, fontsize=16, fontweight='bold')
    ax1.set_title('Loading percentage for all elements', color='darkblue', fontsize=14)
    ax2.set_title('Element operating status', color='darkblue', fontsize=14)
    ax1.legend()  # bbox_to_anchor=(1.4, 1)
    ax2.legend()  # bbox_to_anchor=(1.4, 1)

    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=10, ha='center', fontsize=10)

    if _ylim[0]:
        ax1.set_ylim(_ylim[0])

    if _ylim[1]:
        ax2.set_ylim(_ylim[1])

    if _xlim:
        ax1.set_xlim(_xlim)

    fig.set_facecolor('ghostwhite')
    ax1.set_facecolor('white')  # ghostwhite
    ax1.patch.set_alpha(0.9)
    ax1.grid(color='lightgray', alpha=0.5)  # lightgray

    ax2.set_facecolor('white')  # ghostwhite
    ax2.patch.set_alpha(0.9)
    ax2.grid(color='lightgray', alpha=0.5, axis='y')  # lightgray

    plt.tight_layout()
    plt.show()

    return fig
