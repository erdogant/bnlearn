Causal Generative Modeling with Bayesian Networks
=================================================

This example is a (twin of the R based bnlearn package)[https://rpubs.com/osazuwa/causaldag1] but now in Python!
You first need to install bnlearn. Checkout the installation pages if you did not do this yet.


Overview
^^^^^^^^

* Build DAG
* plot DAG
* Learning a (causal) DAG from data
* Specifying your own probability distributions
* Estimating parameters of CPDs
* Inference on the causal generative model


Understanding the directed acyclic graph (DAG) representation
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Lets start with a causal model for transportation data which is based on surveys.
Here, we show how we can visualize it with bnlearn package.

	* The survey data dataset
	
	* survey data is a data set that focuses on how public transport varies across social groups. It includes the following factors (discrete variables):

		* Age (A): It is recorded as young (young) for individuals below 30 years, adult (adult) for individuals between 30 and 60 years old, and old (old) for people older than 60.

		* Sex (S): The biological sex of individual, recorded as male (M) or female (F).

		* Education (E): The highest level of education or training completed by the individual, recorded either high school (high) or university degree (uni).

		* Occupation (O): It is recorded as an employee (emp) or a self employed (self) worker.

		* Residence (R): The size of the city the individual lives in, recorded as small (small) or big (big).

		* Travel (T): The means of transport favoured by the individual, recorded as car (car), train (train) or other (other)

		* Travel is the target of the survey, the quantity of interest whose behaviour is under investigation.



Building a causal DAG
'''''''''''''''''''''

If you readily know (or have expections) of the relationships between variables, we can setup the (causal) relationships between the variables with a directed graph (DAG). 
Each node corresponds to a variable and each edge represents conditional dependencies between pairs of variables.

In bnlearn, we can graphically represent the relationships between variables in survey data like this:

.. code-block:: python

   import bnlearn
   from pgmpy.models import BayesianModel


   # Define the network structure
   edges = [('A', 'E'),
            ('S', 'E'),
            ('E', 'O'),
            ('E', 'R'),
            ('O', 'T'),
            ('R', 'T'),
	    ]

   DAG = bnlearn.DAG(edges)

   DAG = BayesianModel(DAG)
