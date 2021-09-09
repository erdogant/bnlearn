Interactive plotting
========================

Each plot can be made interactive by setting the interactive parameter. The plots are created using the ```pyvis library for which various input parameters can be specified. Let's create a few examples with interactive plots.


**Interactive plot example **

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
	bn.plot(model, interactive=True, params = {'directed':True, 'height':'800px', 'width':'70%', 'notebook':False, 'heading':'bnlearn causal diagram', 'layout':None, 'font_color': False, 'bgcolor':'#ffffff'})


.. raw:: html

   <iframe src="https://erdogant.github.io/docs/pyvis/bnlearn_asia_causal_network.html" height="1300px" width="800px", frameBorder="0"></iframe>



All the parameters to specify the interactive plot can be found here:
https://pyvis.readthedocs.io/en/latest/documentation.html
