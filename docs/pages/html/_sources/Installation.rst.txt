Installation
============

Installation of ``bnlearn`` is straightforward. 

It is advisable to create a new environment which is done as following:

.. code-block:: console

   conda create -n env_bnlearn python=3.8
   conda activate env_bnlearn


.. _installation step 1:

.. figure:: ../figs/01_installation.png

  Create environment.


Notice the last line. You need to see that your environment is now set as ``bnlearn``. In my case it is as following:

.. code-block:: console

   (env_bnlearn) D:\>


Deactivate and then activate your environment if the packages are not recognized.

.. code-block:: console

   conda deactivate
   conda activate env_bnlearn


After creating the environment, install ``bnlearn`` with pip:

.. code-block:: python

   # Install bnlearn and if a version is readily present on your local machine, this version will be installed.
   pip install bnlearn

   # Make sure to have the latest version from pypi by using the -U (update) argument.
   pip install -U bnlearn


.. _installation step 3:

.. figure:: ../figs/03_installation.png

  Install bnlearn.


Validate
========

Lets checkout whether it works by a simple example. Start python in your console:

.. code-block:: console

   python

Run the following lines which should result in a figure:

.. code-block:: python

   import bnlearn as bn
   df = bn.import_example()
   model = bn.structure_learning.fit(df)
   G = bn.plot(model)


.. _installation step 4:

.. figure:: ../figs/04_installation.png



Uninstalling
============

If you want to remove your ``bnlearn`` installation with your environment, it can be as following:

.. code-block:: python

   # List all the active environments. BNLEARN should be listed.
   conda env list

   # Remove the env_bnlearn environment
   conda env remove --name env_bnlearn

   # List all the active environments. env_bnlearn should be absent.
   conda env list
