Saving and Loading
========================

All models can be saved and loading using the :func:`bnlearn.bnlearn.save` and :func:`bnlearn.bnlearn.load` functionality.

**Example of saving and loading models**

.. code-block:: python
    
	# Load data
	df = bn.import_example(data='asia')

	# Learn structure
	model = bn.structure_learning.fit(df, methodtype='tan', class_node='lung')

	# Save model
	bn.save(model, filepath='bnlearn_model', overwrite=True)

	# Load model
	model = bn.load(filepath='bnlearn_model')



.. include:: add_bottom.add