Plotting
=========================================

``bnlearn`` provides both **interactive** and **static** plotting capabilities through the :func:`bnlearn.bnlearn.plot` function. These visualization tools allow for extensive customization of network and figure properties, including node colors, sizes, and layout configurations. Interactive plots are created using the ``D3Blocks`` library, while static plots utilize matplotlib and networkx.

Interactive Plotting
----------------------------------------

Prerequisites
^^^^^^^^^^^^^

Before creating interactive plots, you need to install the ``d3blocks`` library:

.. code-block:: bash

   pip install d3blocks

Basic Usage
^^^^^^^^^^^

The simplest way to create an interactive plot is by setting ``interactive=True`` in the plot function:

.. code-block:: python
    
   import bnlearn as bn

   # Load example dataset
   df = bn.import_example(data='asia')

   # Learn the network structure
   model = bn.structure_learning.fit(df)

   # Create interactive plot with default settings
   bn.plot(model, interactive=True)

   # Customize the interactive plot with specific parameters
   bn.plot(model, 
          interactive=True, 
          params_interactive={
              'height': '800px',
              'width': '70%',
              'layout': None,
              'bgcolor': '#0f0f0f0f'
          })

.. raw:: html

   <iframe src="https://erdogant.github.io/docs/pyvis/bnlearn_asia_causal_network.html" height="1300px" width="800px", frameBorder="0"></iframe>

Customizing Node Properties
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize the appearance of nodes in several ways:

1. **Uniform Node Properties**:
   Apply the same color and size to all nodes in the network:

.. code-block:: python

   # Set uniform node color
   bn.plot(model, interactive=True, node_color='#8A0707')
   
   # Set uniform node color and size
   bn.plot(model, interactive=True, node_color='#8A0707', node_size=25)

.. |figIP1| image:: ../figs/_fig-plot_interactive_simple_color.png
.. |figIP2| image:: ../figs/_fig-plot_interactive_simple_color_size.png

.. table:: Examples of uniform node customization
   :align: center

   +----------+----------+
   | |figIP1| | |figIP2| |
   +----------+----------+

2. **Individual Node Properties**:
   Customize specific nodes with different colors and sizes:

.. code-block:: python

    # Retrieve current node properties
    node_properties = bn.get_node_properties(model)

    # Customize specific nodes
    node_properties['xray']['node_color'] = '#8A0707'
    node_properties['xray']['node_size'] = 50
    node_properties['smoke']['node_color'] = '#000000'
    node_properties['smoke']['node_size'] = 35

    # Create plot with customized node properties
    bn.plot(model, node_properties=node_properties, interactive=True)

.. |figIP3| image:: ../figs/_fig-plot_interactive_user_colors.png

.. table:: Example of individual node customization
   :align: center

   +----------+
   | |figIP3| |
   +----------+

Static Plotting
----------------------------------------

Networkx Static Plots
^^^^^^^^^^^^^^^^^^^^^

Networkx provides a flexible way to create static network visualizations:

.. code-block:: python

    # Create basic static plot
    bn.plot(model, interactive=False)
    
    # Customize static plot with specific parameters
    bn.plot(model, 
           interactive=False, 
           params_static={
               'width': 15,
               'height': 8,
               'font_size': 14,
               'font_family': 'times new roman',
               'alpha': 0.8,
               'node_shape': 'o',
               'facecolor': 'white',
               'font_color': '#000000'
           })

.. |figIP7| image:: ../figs/asia_networkx.png

.. table:: Example of a networkx static plot
   :align: center

   +----------+
   | |figIP7| |
   +----------+

Graphviz Static Plots
^^^^^^^^^^^^^^^^^^^^^

Graphviz provides a more structured and hierarchical visualization style:

.. code-block:: python

    # Create graphviz plot
    bn.plot_graphviz(model)

.. |figIP6| image:: ../figs/asia_graphviz.png

.. table:: Example of a graphviz static plot
   :align: center

   +----------+
   | |figIP6| |
   +----------+

Network Comparison
----------------------------------------

The library provides tools to compare different networks, which is particularly useful when comparing learned structures against ground truth or different learning methods:

.. code-block:: python

   # Load ground truth network
   model = bn.import_DAG('asia')

   # Plot ground truth
   G = bn.plot(model)
   
   # Generate synthetic data
   df = bn.sampling(model, n=10000)
   
   # Learn structure from data
   model_sl = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
   
   # Compute edge strengths
   model_sl = bn.independence_test(model_sl, df, test='chi_square', prune=True)
   
   # Plot learned structure
   bn.plot(model_sl, pos=G['pos'])
   
   # Compare networks
   bn.compare_networks(model, model_sl, pos=G['pos'])

.. |fig_cn1| image:: ../figs/fig2a_asia_groundtruth.png
.. |fig_cn2| image:: ../figs/fig2b_asia_structurelearning.png

.. table:: Comparison of ground truth and learned networks
   :align: center

   +----------+
   | |fig_cn1||
   +----------+
   | |fig_cn2||
   +----------+

.. |fig_cn3| image:: ../figs/fig2c_asia_comparion.png
.. |fig_cn4| image:: ../figs/fig2d_confmatrix.png

.. table:: Detailed comparison showing edge differences
   :align: center

   +----------+
   | |fig_cn3||
   +----------+
   | |fig_cn4||
   +----------+

Advanced Customization
----------------------------------------

Node Properties
^^^^^^^^^^^^^^^

Node properties can be customized using the :func:`bnlearn.bnlearn.get_node_properties` function:

.. code-block:: python

    import bnlearn as bn
    # Load example data
    df = bn.import_example(data='asia')
    # Learn structure
    model = bn.structure_learning.fit(df)
    # Get current node properties
    node_properties = bn.get_node_properties(model)

    # Customize specific nodes
    node_properties['xray']['node_color'] = '#8A0707'
    node_properties['xray']['node_size'] = 2000
    node_properties['smoke']['node_color'] = '#000000'
    node_properties['smoke']['node_size'] = 2000

    # Create plot with customized nodes
    bn.plot(model, node_properties=node_properties, interactive=False)

.. |figIP4| image:: ../figs/node_properties_1.png

.. table:: Example of advanced node customization
   :align: center

   +----------+
   | |figIP4| |
   +----------+

Edge Properties
^^^^^^^^^^^^^^^

Edge properties can be customized using the :func:`bnlearn.bnlearn.get_edge_properties` function. These customizations can be combined with node properties for comprehensive network visualization.

.. code-block:: python

    import bnlearn as bn
    # Load asia DAG
    df = bn.import_example(data='asia')
    # Structure learning of sampled dataset
    model = bn.structure_learning.fit(df)
    # Test for significance
    model = bn.independence_test(model, df)
    # plot static
    G = bn.plot(model)

    # Set some edge properties
    # Because the independence_test is used, the -log10(pvalues) from model['independence_test']['p_value'] are scaled between minscale=1 and maxscale=10
    edge_properties = bn.get_edge_properties(model)

    # Make some changes
    edge_properties['either', 'xray']['color']='#8A0707'
    edge_properties['either', 'xray']['weight']=4
    edge_properties['bronc', 'smoke']['weight']=15
    edge_properties['bronc', 'smoke']['color']='#8A0707'
    
    # Plot
    params_static={'edge_alpha':0.6, 'arrowstyle':'->', 'arrowsize':60}
    bn.plot(model, interactive=False, edge_properties=edge_properties, params_static=params_static)

.. |figIP5| image:: ../figs/edge_properties_1.png

.. table:: Plot with user defined edge properties
   :align: center

   +----------+
   | |figIP5| |
   +----------+

.. include:: add_bottom.add