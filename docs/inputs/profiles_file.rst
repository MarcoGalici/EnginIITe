.. _profiles_file:

===================
Profiles Files
===================

The EnginIITe platform requires profile files to define the load, generation, and storage characteristics of various resources.
Profile files can be of three types:

1. **Denormalized Profiles (denorm_profile)**
2. **Normalized Profiles (norm_profile)**
3. **External Profiles (external_module)**

The third option does not involve an actual profile file, but rather an additional piece of code that evaluates the user-defined profiles.


File Format
-----------

Profile files must be provided in either `.csv` or `.xlsx` format, regardless of whether they are denormalized or normalized.

.. note::

    When the profiles are saved as `.xlsx` format; it is important that the sheet name is equal to the name of the profile file.

Each profile file should contain a **table structure** where:

- Rows represent **timesteps**.

- Columns represent **profile's names** associated with each resource.

Between denormalized and normalized profile tables there are some differences. In particular:

- **Denormalized Profiles**: Each column represents a specific **resource index**.
The number of columns should match the number of resources for that resource category.

- **Normalized Profiles**: Each column represents a **normalized profile name**, allowing multiple resources to share
the same profile pattern.

Profiles Definition Examples
-----------------------------

In denormalized profiles, each column corresponds to an individual resource by index.
This means that each column name should be the index of the resource it is associated with.
The platform expects the number of columns in a denormalized profile file to be at least equal to the number of
resources in that category.

- **Example of denormalized profile file structure**:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Timestep
     - 0
     - 1
   * - 0
     - 0.300601
     - 1.975223
   * - 1
     - 0.29319
     - 1.932226

In normalized profiles, each column represents a profile that can be applied to multiple resources.
Each column should be named according to the **profile name** rather than resource indices, as the profile applies
generally to all resources sharing that profile.

- **Example of normalized profile file structure**:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Timestep
     - Profile_A
     - Profile_B
   * - 0
     - 0.300601
     - 1.975223
   * - 1
     - 0.29319
     - 1.932226

For profiles defined via an external module, there is no need for a profile file unless required by the user.
This option is used when profiles require more complex calculations or are generated according to the user preferences.
In case the user need to pass input parameters to the external module that defines the profiles,
as presented as example of profiles definition in the :ref:`Config File Documentation <_profiles>`,
the user can add the name of the input parameters file for the external module. This file must be saved in the following
folder:

::

    C:\Users\...\EnginIITe\data_in\my_network_name\fsp_files

In addition, the external module must be defined within the resource module.


Saving Folder
----------------------------
All profile files need to be stored in the correct folder.
Properly setting the location of profile files helps ensure the platform can locate and read them during execution.
For the profile files folder the correct folder is in the following path:

::

    C:\Users\...\EnginIITe\data_in\my_network_name\profiles

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

