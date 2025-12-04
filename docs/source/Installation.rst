Installation
============

Install from PyPI (pip)
-----------------------

The simplest way to install bnlearn is using pip:

.. code-block:: console

    pip install bnlearn

To force install the latest version, use the -U (update) argument:

.. code-block:: console

    pip install -U bnlearn

Install from GitHub
-------------------

To install the latest development version directly from GitHub:

.. code-block:: console

    pip install git+https://github.com/erdogant/bnlearn

Create Environment
------------------

For better dependency management, it's recommended to install ``bnlearn`` in an isolated Python environment using conda:

.. code-block:: console

    conda create -n env_bnlearn python=3.13
    conda activate env_bnlearn

.. _installation step 1:

.. figure:: ../figs/01_installation.png

   Create a new conda environment.

After activation, your command prompt should show the environment name. For example:

.. code-block:: console

   (env_bnlearn) D:\>

Uninstall
---------

To remove the ``bnlearn`` installation and its environment:

.. code-block:: console

   # List all active environments
   conda env list

   # Remove the bnlearn environment
   conda env remove --name env_bnlearn

   # Verify removal by listing environments again
   conda env list

Validate Installation
---------------------

To verify your installation, start Python in your console:

.. code-block:: console

   python

Then run the following code, which should generate a figure:

.. code-block:: python

   import bnlearn as bn
   df = bn.import_example()
   model = bn.structure_learning.fit(df)
   G = bn.plot(model)

.. _installation step 4:

.. figure:: ../figs/04_installation.png

Troubleshooting Import Errors
-----------------------------

If you're using Jupyter Notebook or Google Colab, you might encounter a NumPy version compatibility error:

.. code-block:: python

    import bnlearn as bn
    # Error message:
    RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf
    ImportError: numpy.core.multiarray failed to import

This error occurs because ``bnlearn`` requires NumPy version 1.24.1 or higher. To resolve this:

1. To fix this, you need an installation of *numpy version=>1.24.1* which is installed during the ``bnlearn`` installation.
   However, when you are using colab or a jupyter notebook, you need to reset your kernel first to let it work. 
2. If using Colab or Jupyter Notebook:
   - Go to the menu
   - Click **Runtime -> Restart runtime**
   - Re-import bnlearn


.. include:: add_bottom.add