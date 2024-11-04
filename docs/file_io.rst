.. _file_io:

========================
Import and Export Data
========================

The EnginIITe platform provides functions for importing and exporting data from and to the platform.
These functions enable users to read and save data in two formats: `i`) `.csv` and `ii`) `.xlsx`.
These functions are meant to manage network profiles, network data, configuration files and in general terms any information
saved in the form of `.csv` and `.xlsx`.


Import Configuration File
--------------------------

.. autofunction:: functions.file_io.import_yaml


Import Pandapower Network
--------------------------

.. autofunction:: functions.file_io.import_network


Import General Data
--------------------------

.. autofunction:: functions.file_io.import_data


Import Profiles
--------------------------

The action of importing profiles can be done in different ways according to the type of profiles the user wants to read.
In the following each function that read the profiles data is explained in detail:

.. toctree::
    :maxdepth: 1

    file_profiles


Export Data
--------------------------

.. autofunction:: functions.file_io.save_excel

.. autofunction:: functions.file_io.save_cvs
