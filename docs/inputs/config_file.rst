.. _config_file:

==================
Configuration File
==================

The `.yaml` configuration file controls various aspects of the simulation, including the network configuration,
market model settings, and simulation parameters.


Simulation Tag
----------------
The simulation tag is a tag added at the end of the result outputs to identify the simulation.
This helps differentiate between various simulations when running sensitivity analyses.

**Type**:  
String

**Examples**::

    simulation_tag: ''
    simulation_tag: 'my_sim_tag'


Simulation Window
-------------------
The simulation window defines the start and end time intervals to be analyzed during the simulation.
It requires two parameters; the initial interval and the last (plus one) interval.
The simulation window is agnostic of the granularity of the simulation, but it depends on the specific intervals.

**Type**:  
List of two integers

**Example**::

    simulation_window: [0, 24]


Network Parameters
--------------------

Inside the parameter `network` have been defined several sub-parameters.
In the following each sub-parameter is described.

Net Folder
------------
The net folder parameter defines the name of the folder where the network data are stored.

**Type**:  
String

**Example**::

    network:
        net_folder: 'my_folder_name'


Net Filename
----------------
The net file defines the name of the file containing network information.
The extension of the file can be '.csv' or '.xlsx'.

**Type**:  
String

**Example**::

    network:
        net_filename: 'my_network_file.xlsx'


Profiles
----------
.. _profiles:

The Enginite platform defines 4 categories of resources in the electrical network:

* Load
* Static Generator
* Generator
* Storage

Each category may be composed of different resources. For instance, the Load category can be composed by simple load,
flexible load, heatpump etc.

In the configuration file, nested in the `network` parameter is defined the `profile` parameter. In addition, nested in
the `profile` parameter have been defined the resource category parameters:

* `load` - Load
* `sgen` - Static Generator
* `gen` - Generator
* `storage` - Storage

Nested in the resource category parameters are nested the product of the market:

* `p` - Active Power
* `q` - Reactive Power

Finally, nested on the market's products are defined the profile type:

* `denorm` - Denormalized Profile
* `norm` - Normalized Profile
* `external` - External Profile


The platform will load these profiles to set up the simulation.
The extension of the file can be '.csv' or '.xlsx'.

**Type**:  
Dictionary

**Example**::

    network:
        profile:
            load:
                p:
                    denorm: 'denorm_profile_load_p.xlsx'
                    norm: 'norm_profile_load_p.xlsx'
                    external: ''
                q:
                    denorm: 'denorm_profile_load_q.xlsx'
                    norm: 'norm_profile_load_q.xlsx'
                    external: ''
            sgen:
                p:
                    denorm: 'denorm_profile_sgen_p_maxgen.xlsx'
                    norm: ''
                    external: ''
                q:
                    denorm: 'denorm_profile_sgen_p_maxgen.xlsx'
                    norm: ''
                    external: ''
            gen:
                p:
                    denorm: ''
                    norm: ''
                    external: ''
                q:
                    denorm: ''
                    norm: ''
                    external: ''
            storage:
                p:
                    denorm: ''
                    norm: ''
                    external: 'storage_initprof_param.xlsx'
                q:
                    denorm: ''
                    norm: ''
                    external: 'storage_initprof_param.xlsx'


Normalize Profile
-------------------
The normalize profile parameter specifies whether the corresponding profiles,
defined in the `profile_filename` parameter above, should be normalized.

**Type**:  
Dictionary of Booleans

**Example**::

    network:
        profile:
            norm_profile:
                load_prof_p: True
                load_prof_q: True
                sgen_prof_p: False
                sgen_prof_q: False
                storage_prof_p: False
                storage_prof_q: False


Timeseries Output File Type
-----------------------------
The timeseries output file type defines the file format for storing time-series simulation output.

**Type**:  
String

**Example**::

    network:
        ts_output_file_type: '.csv'


Flexibility Service Providers
-------------------------------
The Enginite platform allows the user to simulate different flexibility markets per market model, market product,
market voltage limitation, market storage capacity and finally per flexibility service provider (aka fsp) category.
In the configuration file, nested in the `network` parameter is defined the `fsp` parameter.
This parameter allows to configure parameter related to the flexibility service providers, including their data source,
bidding strategy, and input file.

**Sub-Parameters**:

- ``fsp_sheetname`` - (*string*): Name of the sheet where FSP data is saved.
- ``fsp_bidding_strategy`` - (*string*): Strategy used by FSPs to submit bids in the market. Options are 'initial_value' or 'mva_value'.
- ``fsp_input_filename`` - (*string*): File containing the FSP data. The extension of the file can only be '.xlsx'.

**Example**::

    network:
        fsp:
            fsp_sheetname: 'fsp_data'
            fsp_bidding_strategy: 'initial_value'
            fsp_input_filename: 'FSP_data_PL.xlsx'


Model Tag
-----------
The model tag parameter defines the market model used in the simulation.
Available options:

* *'CMVC'*: Combined Congestion Management and Voltage Control
* *'CM'*: Congestion Management only
* *'VC'*: Voltage Control only

**Type**:  
String

**Example**::

    ModelTAG: 'CMVC'


Product Tag
-------------
The product tag parameter specifies the type of market products considered in the simulation.
It can be for Active Power (P), Reactive Power (Q), or both.

**Type**:  
List of Strings

**Example**::

    ProductTAG: ['P', 'Q', 'PQ']


Voltage Limitation Tag
------------------------
The voltage limitation parameter defines the voltage limitations for the simulation.
Each tag represents a different voltage range.

**Type**:  
List of Strings

**Available Options**:

- *'VL01'*: 0.95 - 1.05 pu
- *'VL02'*: 0.93 - 1.07 pu
- *'VL03'*: 0.9 - 1.1 pu

**Example**::

    VLTAG: ['VL01', 'VL02', 'VL03']


Flexibility Service Provider Tag
----------------------------------
The flexibility service provider tag parameter defines the percentage increments for FSP capacity activation.

**Type**:  
List of Strings

**Available Options**:

- *'F01'*: 5% capacity
- *'F02'*: 10% capacity
- *'F03'*: 15% capacity
- *'F04'*: 20% capacity
- *'F05'*: 25% capacity

**Example**::

    FspTAG: ['F01', 'F02', 'F03', 'F04', 'F05']


Storage Tag
-------------
The storage tag parameter defines a multiplier for storage capacity activation.
If requires a string (i.e., 'SK01') where the number at the end defines the multiplier. For instance 'SK02' define a
storage capacity two times the initial value of the storage present in the network file.

**Type**:  
String

**Example**::

    StorageTag: 'SK01'


Scenario and Cost Parameters
------------------------------
For each resource category in the electrical network, the Enginite platform allows the user to define a multipliers
that is applied to the load, static generation, generator and storage profiles for a given scenario.
In case one category is not included in the network, the user can avoid define the parameter for that specific resource.

**Type**:  
Integer

**Example**::

    scen_factor_load: 1
    scen_factor_sgen: 2
    scen_factor_gen: 2
    scen_factor_storage: 1


BetaCost
----------
The beta cost parameter represents the Value of Lost Load (VOLL), measured in EUR/MWh, based on the country in question.

* For Spain: 5890 [EUR/MWh]
* For Germany: 12410 [EUR/MWh]
* For Poland: 6260 [EUR/MWh]
* For Portugal: 7880 [EUR/MWh]

.. note::
    Reference for VOLL values `here`_.

.. _here: https://www.acer.europa.eu/en/Electricity/Infrastructure_and_network%20development/Infrastructure/Documents/CEPA%20s

**Type**:  
Integer

**Example**::

    BetaCost: 6260


Frequency
-----------
Frequency setting for the power flow algorithm.

**Type**:  
Integer

**Example**::

    f_hz: 50


Power Flow Algorithm
----------------------
The power flow algorithm parameter specifies the power flow algorithm to be used.
Available options include:

* *'nr'*: Newton-Raphson
* *'bfsw'*: Backward-Forward Sweep

**Type**:  
String

**Example**::

    par_pf_algo: 'nr'


Sensitivity Factors Mode
--------------------------
Defines the mode for sensitivity factor calculation.

**Type**:  
String

**Example**::

    Sens_factors_mode: 'EMPIRICAL'


Sensitivity Factors Threshold
-------------------------------
If the number of hours multiplied by the number of pilot bus exceeds this threshold,
the matrix of the worst hour will be used.

**Type**:  
Integer

**Example**::

    Sens_factors_problem_size_threshold: 100000000000000000


Debug Power Flow Results to Excel
-----------------------------------
When set to `True`, power flow results are saved to Excel for debugging.

**Type**:  
Boolean

**Example**::

    debug_pf_res_to_excel: False


Debug Pilot Bus
-----------------
When set to `True`, the file `VPilotBus_FULLH_FULLBUS.csv` is saved in the Market_input folder.

**Type**:  
Boolean

**Example**::

    debug_save_VPilotBus_FULLH: True


KPI Parameters
----------------
Tolerances for voltage and current simulations. These parameters impact both the market and KPI evaluation.

**Type**:  
Float

**Example**::

    tol_v: 0.005
    tol_i: 0.005

