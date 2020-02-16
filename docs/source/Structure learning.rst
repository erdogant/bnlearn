.. _code_directive:

-------------------------------------


Structure learning
''''''''''''''''''

Let's bring in our dataset.

.. code-block:: python

  import bnlearn
  df = bnlearn.import_example()
  df.head()


.. table::

  Cloudy  Sprinkler  Rain  Wet_Grass
  0         0          1     0          1
  1         1          1     1          1
  2         1          0     1          1
  3         0          0     1          1
  4         1          0     1          1
  ..      ...        ...   ...        ...
  995       0          0     0          0
  996       1          0     0          0
  997       0          0     1          0
  998       1          1     0          1
  999       1          0     1          1


From the *bnlearn* library, we'll need the
:class:`~bnlearn.fitters.structure_learning.fit` for this exercise:


.. code-block:: python

  model = bnlearn.structure_learning.fit(df)
  G = bnlearn.plot(model)


