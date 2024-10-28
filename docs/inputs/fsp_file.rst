.. _fsp_file:

==========================================
Flexibility Service Provider Files
==========================================

The Flexibility Service Provider (FSP) files are essential for configuring flexibility options in the platform.
These files contain data about each provider's capabilities, locations, and costs associated with various flexibility services.

File Format and Requirements
----------------------------

FSP files **must be saved in Excel format** (`.xlsx`).
The platform is designed to read and interpret data only from Excel files, so using a different format will cause errors
during processing.

The Excel file used for defining FSPs should contain **one row for each FSP** and **one column for each attribute**.
Below is a detailed breakdown of the required attributes and their functions:

.. csv-table::
   :file: fsp_par.csv
   :delim: ;
   :widths: 10, 10, 10, 40


Saving Folder
----------------------------
All fsp files need to be stored in the correct folder.
Properly setting the location of fsp files helps ensure the platform can locate and read them during execution.
For the fsp files folder the correct folder is in the following path:

::

    C:\Users\...\EnginIITe\data_in\my_network_name\fsp_files

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

