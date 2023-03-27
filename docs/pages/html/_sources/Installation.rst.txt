Install from Pypi (pip)
########################

.. code-block:: console

    pip install bnlearn

   # Force install the latest version by using the -U (update) argument.
   pip install -U bnlearn


Install from github
#####################################

.. code-block:: console

    pip install git+https://github.com/erdogant/bnlearn


Create environment
#####################


If desired, install ``bnlearn`` from an isolated Python environment using conda:

.. code-block:: python

    conda create -n env_bnlearn python=3.10
    conda activate env_bnlearn


.. _installation step 1:

.. figure:: ../figs/01_installation.png

  Create environment.


Notice the last line. You need to see that your environment is now set as ``bnlearn``. In my case it is as following:

.. code-block:: console

   (env_bnlearn) D:\>


Uninstall
###############

If you want to remove your ``bnlearn`` installation with your environment, it can be as following:

.. code-block:: console

   # List all the active environments. bnlearn should be listed.
   conda env list

   # Remove the bnlearn environment
   conda env remove --name env_bnlearn

   # List all the active environments. *env_bnlearn* should be absent.
   conda env list




Validate
#####################


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



Import Error
#####################

If you are using jupyter notebook or colab, you can get a ``numpy`` error because by default, an older version of ``numpy`` is installed.


.. code-block:: python
	
	# Import
	import bnlearn as bn

	# The following error occurs
	RuntimeError : Traceback (most recent call last)
	RuntimeError: module compiled against API version 0x10 but this version of numpy is 0xf . Check the section C-API incompatibility at the Troubleshooting ImportError section at https://numpy.org/devdocs/user/troubleshooting-importerror.html#c-api-incompatibility for indications on how to solve this problem .
	ImportError: numpy.core.multiarray failed to import


To fix this, you need an installation of *numpy version=>1.24.1* which is installed during the ``bnlearn`` installation.
However, when you are using colab or a jupyter notebook, you need to reset your kernel first to let it work. 
Go to the menu and click **Runtime -> restart runtime**. Now again import bnlearn, and it should work.



.. include:: add_bottom.add