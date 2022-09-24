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




.. raw:: html

	<hr>
	<center>
		<script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CEADP27U&placement=erdogantgithubio" id="_carbonads_js"></script>
	</center>
	<hr>

