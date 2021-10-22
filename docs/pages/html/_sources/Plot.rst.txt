Interactive plot
=================

``bnlearn`` contains **interactive** and **static** plotting functionalities with :func:`bnlearn.bnlearn.plot` for which many network and figure properties can be adjusted, such as node colors and sizes. 
To make interactive plots, it simply needs to set the ``interactive=True`` parameter in :func:`bnlearn.bnlearn.plot`. 
The interactive plots are created using the ``pyvis`` library for which various input parameters can be specified. The static plots are created using matplotlib and networkx.
Lets make some interactive and static examples. 


**Interactive plot examples**

.. code-block:: bash

	# Install the pyvis library first if you want interactive plots
	pip install pyvis


.. code-block:: python
    
	# Example of interactive plotting
	import bnlearn as bn

	# Load example dataset
	df = bn.import_example(data='asia')

	# Structure learning
	model = bn.structure_learning.fit(df)

	# Add some parameters for the interactive plot
	bn.plot(model, interactive=True, params = {'height':'600px'})

	# Add more parameters for the interactive plot
	bn.plot(model, interactive=True, params_interactive = {'height':'800px', 'width':'70%', 'notebook':False, 'heading':'bnlearn causal diagram', 'layout':None, 'font_color': False, 'bgcolor':'#ffffff'})


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/pyvis/bnlearn_asia_causal_network.html" height="1300px" width="800px", frameBorder="0"></iframe>


All the parameters to specify the interactive plot can be found here:
https://pyvis.readthedocs.io/en/latest/documentation.html



**Create interactives plot with default settings.**

.. code-block:: python

   # Import dataset
   df = bn.import_example(data='asia')
   # Structure learning
   model = bn.structure_learning.fit(df)
   # Make simple interactive plot
   bn.plot(model, interactive=True)

.. _fig-plot_interactive_simple:

.. figure:: ../figs/_fig-plot_interactive_simple.png


**Create interactive plots with a specific node-color across the entire network.**

.. code-block:: python

   bn.plot(model, node_color='#8A0707', interactive=True)

.. _fig-plot_interactive_simple:

.. figure:: ../figs/_fig-plot_interactive_simple_color.png


**Create interactive plots with a specific node-color, and node-size across the entire network.**

.. code-block:: python

   bn.plot(model, node_color='#8A0707', node_size=25, interactive=True)

.. _fig-plot_interactive_simple:

.. figure:: ../figs/_fig-plot_interactive_simple_color_size.png



**Create interactive plots with user-defined node-colors and node-sizes.**

.. code-block:: python

    # First retrieve node properties
    node_properties = bn.get_node_properties(model)

    # Make some changes
    node_properties['xray']['node_color']='#8A0707'
    node_properties['xray']['node_size']=50
    node_properties['smoke']['node_color']='#000000'
    node_properties['smoke']['node_size']=35

    # Make plot with the specified node properties
    bn.plot(model, node_properties=node_properties, interactive=True)


.. _fig-plot_interactive_simple:

.. figure:: ../figs/_fig-plot_interactive_user_colors.png


**The ``params_interactive`` parameter allows you to adjust more figure properties.**

.. code-block:: python

    bn.plot(model, interactive=True, params_interactive = {'height':'800px', 'width':'70%', 'layout':None, 'bgcolor':'#0f0f0f0f'})


Static plot
=================

To create static plots simply set the ``interactive=False`` in all the above examples. The only difference is in ``params_static`` for which the dict contains more variables that adjust the figure properties.

.. code-block:: python

    # Add parameters for the static plot
    bn.plot(model, interactive=False, params_static = {'width':15, 'height':8, 'font_size':14, 'font_family':'times new roman', 'alpha':0.8, 'node_shape':'o', 'facecolor':'white', 'font_color':'#000000'})
