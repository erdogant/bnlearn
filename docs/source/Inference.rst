Inference
=========

The basic concept of variable elimination is same as doing marginalization over Joint Distribution.
But variable elimination avoids computing the Joint Distribution by doing marginalization over much smaller factors.
So basically if we want to eliminate **X** from our distribution, then we compute
the product of all the factors involving **X** and marginalize over them,
thus allowing us to work on much smaller factors.


The main categories for inference algorithms:
  1. Exact Inference: These algorithms find the exact probability values for our queries.
  2. Approximate Inference: These algorithms try to find approximate values by saving on computation.

**Two common Inference algorithms with variable Elimination**
  a. Clique Tree Belief Propagation
  b. Variable Elimination


Example Inference 1
'''''''''''''''''''

Lets load the Sprinkler data set and make some inferences.


What is the probability of *wet grass* given that it *Rains*, and the *sprinkler* is off and its *cloudy*: P(wet grass | rain=1, sprinkler=0, cloudy=1)?

>>> import bnlearn
>>> model = bnlearn.import_DAG('sprinkler')
>>> q1 = bnlearn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain':1, 'Sprinkler':0, 'Cloudy':1})

The probability having wet grass is 0.9 and not-wet-gras is 0.1.

  +--------------+------------------+
  | Wet_Grass    |   phi(Wet_Grass) |
  +==============+==================+
  | Wet_Grass(0) |           0.1000 |
  +--------------+------------------+
  | Wet_Grass(1) |           0.9000 |
  +--------------+------------------+


Example Inference 2
'''''''''''''''''''

What is the probability of wet grass given and Rain given that the *Sprinkler* is on?

>>> q2 = bnlearn.inference.fit(model, variables=['Wet_Grass','Rain'], evidence={'Sprinkler':1})

The highest probability is that in these condition, there is wet grass and no rain (P=0.63)

  +--------------+---------+-----------------------+
  | Wet_Grass    | Rain    |   phi(Wet_Grass,Rain) |
  +==============+=========+=======================+
  | Wet_Grass(0) | Rain(0) |                0.0700 |
  +--------------+---------+-----------------------+
  | Wet_Grass(0) | Rain(1) |                0.0030 |
  +--------------+---------+-----------------------+
  | Wet_Grass(1) | Rain(0) |                0.6300 |
  +--------------+---------+-----------------------+
  | Wet_Grass(1) | Rain(1) |                0.2970 |
  +--------------+---------+-----------------------+


Example Inference 3
'''''''''''''''''''

Given our model, what is the probability on lung cancer given that the person is a smoker and xray is negative?
P(lung | smoker=1, xray=0)

>>> # Lets create the dataset
>>> model = bnlearn.import_DAG('asia')

Lets make the inference:

>>> q1 = bnlearn.inference.fit(model, variables=['lung'], evidence={'xray':0, 'smoke':1})

  +---------+-------------+
  | lung    |   phi(lung) |
  +=========+=============+
  | lung(0) |      0.1423 |
  +---------+-------------+
  | lung(1) |      0.8577 |
  +---------+-------------+

