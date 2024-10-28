.. _network_file:

==============
Network Files
==============

The network file describes the distribution network structure.
This file must contain information about nodes, lines, and other system elements.

This file must be equal to the structure of the pandapower network.
To learn more about the pandapower network structure refer to the `Pandapower documentation`_.

.. _Pandapower documentation: https://pandapower.readthedocs.io/en/latest/elements.html#

In the EngIITe platform adopts the pandapower libraries and integrates them into the platform framework.
Even though the core of the platform is based on the pandapower library, some elements are different in order to align
them to the EnginIITe platform.

Due to the scope of the EnginIITe platform only 4 categories of resources of the electrical network are useful for the
goal of the EnginIITe platform. In particular, the 4 categories are:

* Load
* Static Generator
* Generator
* Storage

These resource categories are the only elements that are different from the pandapower network format.
In the following, the input parameters for these 4 resource categories are defined according to the pandapower
format and the EnginIITe format.


Load Element
----------------------------
.. csv-table::
   :file: load_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Static Generator Element
----------------------------
.. csv-table::
   :file: sgen_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40


Generator Element
----------------------------
.. csv-table::
   :file: gen_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40



Storage Element
----------------------------
.. csv-table::
   :file: storage_par.csv
   :delim: ;
   :widths: 10, 10, 25, 40



\* optional parameters of pandapower Load model.

\** additional parameters of the EnginIITe platform.


Saving Folder
----------------------------
All network files need to be stored in the correct folder.
Properly setting the location of network files helps ensure the platform can locate and read them during execution.
For the network files folder the correct folder is in the following path:

::

    C:\Users\...\EnginIITe\data_in\my_network_name\network

For the sake of clarity, the following root tree represents the initial root tree of the platform.

::

    EnginIITe
    ├── config
    └── data_in
        └── my_network_name
            ├── fsp_files
            ├── network
            └── profiles
    ├── data_out
    └── functions

