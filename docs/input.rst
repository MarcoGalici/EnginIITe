==========================
Input Files and Parameters
==========================

This section of the documentation describes the various input files needed to configure the platform,
along with the key parameters to define in each file.

Proper use of inputs ensures that the platform can perform simulations and calculations properly.
Below is a list of the four main input files, each explained in more detail on their dedicated pages.

Overview of Inputs
------------------

To configure the platform, the following input files (or parameters) are required:

- **Configuration File**: This file, in the format of `.yaml` defines the general simulation parameters and configuration details.
- **Network File**: An Excel file containing distribution network details, including node and connection data.
- **FSP File**: An Excel file containing information on flexibility service providers, such as IDs, capacities, and prices.
- **Profile File**: An Excel file that define load, generation and storage profiles for various network resources.

To learn more about each input file and see configuration examples, refer to the following sections.

.. toctree::
   :maxdepth: 1

   inputs/config_file
   inputs/profiles_file


.. note::

   Ensure all input files are correctly configured to avoid errors during platform execution.
   The files must be consistent with each other, and parameters should adhere to the specifications described in the documentation.
