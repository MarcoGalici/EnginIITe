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

- **Configuration File**: This file, in the format of `.yaml` defines the general simulation parameters and configuration
details.
- **Network File**: An Excel file containing distribution network details, including node and connection data.
- **FSP File**: An Excel file containing information on flexibility service providers, such as IDs, capacities, and prices.
- **Profile File**: An Excel file that define load, generation and storage profiles for various network resources.

To learn more about each input file and see configuration examples, refer to the following sections.

.. toctree::
   :maxdepth: 1

   inputs/config_file
   inputs/network_file
   inputs/fsp_file
   inputs/profiles_file

.. note::

   Ensure all input files are correctly configured to avoid errors during platform execution.
   The files must be consistent with each other, and parameters should adhere to the specifications described in the documentation.

Input Details
-------------

**Configuration File**

The configuration file in `.yaml` format contains the general parameters that control the platform’s behavior
These include the simulation time window, network parameters, and market settings. For a complete description of the supported entries and configuration examples, see the documentation on :doc:`input_yaml`.

**Network File**

The network file is an Excel file describing the distribution network structure. This file must contain information about nodes, lines, and other system elements. For details on how to structure and format this file, refer to the page :doc:`input_network`.

**FSP File**

The FSP file contains data on flexibility service providers (FSPs). This file includes information such as IDs, capacities, and activation costs. See the documentation on :doc:`input_fsp` for further details.

**Profile Files**

Profile files contain load and generation data for network elements. Each resource type (loads, static generators, flexible generators, and storage) has its own profile, which can be normalized or specified in absolute terms. For specifications, see the section :doc:`input_profiles`.

Guide to Input Configuration
----------------------------

1. **Prepare the required files**: Verify that you have the necessary files (configuration `.yaml`, network, FSP, and profiles) before proceeding with platform execution.

2. **Set the `.yaml` file**: Define the main parameters in the configuration file. In this phase, it’s important to set general simulation parameters, such as the time window, market model, and product options.

3. **Define the network**: Configure the network file to accurately represent the network topology. This file must be consistent with the settings in the `.yaml` file.

4. **Configure the FSP file**: Add the details for flexibility service providers (FSP), including bidding strategies.

5. **Load profiles**: Finally, define the load and generation profiles. Ensure they are consistent with the network structure and the specifications defined in the configuration file.

6. **Verification and consistency**: Before running the simulation, check that all input files are consistent with each other to avoid errors.

For additional information, refer to the dedicated sections for each specific input file.