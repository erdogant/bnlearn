Installation
============

Installation of bnlearn is straightforward. 

It is advisable to create a new environment which is done as following:

.. code-block:: console

   conda create -n BNLEARN python=3.6
   conda activate BNLEARN


.. _installation step 1:

.. figure:: ../figs/01_installation.png

  Create environment.


Notice the last line. You need to see that your environment is now set as BNLEARN. In my case it is as following:

.. code-block:: console

   (BNLEARN) D:\>


bnlearn is build on top of the pgmpy library. Recommended is a conda installation for pgmpy:

.. code-block:: console
   
   conda install -c ankurankan pgmpy


.. _installation step 2:

.. figure:: ../figs/02_installation.png

  Install pgmpy.


Deactivate and then activate your environment because it can occur that the package is not directly recognized.

.. code-block:: console

   conda deactivate
   conda activate BNLEARN


After creating the environment and installing pgmpy, you can simply install bnlearn with pip:

.. code-block:: console

   # Install bnlearn and if a version is readily present on your local machine, this version will be installed.
   pip install bnlearn

   # Make sure to have the latest version from pypi by using the -U (update) argument.
   pip install -U bnlearn


.. _installation step 3:

.. figure:: ../figs/03_installation.png

  Install bnlearn.


Validate working
================

Lets checkout whether it works by a simple example. Start python in your console:

.. code-block:: console

   python

Run the following lines which should result in a figure:

.. code-block:: python

   import bnlearn
   df = bnlearn.import_example()
   model = bnlearn.structure_learning.fit(df)
   G = bnlearn.plot(model)


.. _installation step 4:

.. figure:: ../figs/04_installation.png

  import bnlearn.



Uninstalling
============

If you want to remove your bnlearn installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. BNLEARN should be listed.
   conda env list

   # Remove the BNLEARN environment
   conda env remove --name BNLEARN

   # List all the active environments. BNLEARN should be absent.
   conda env list
