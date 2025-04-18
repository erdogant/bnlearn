
<p align="center">
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/logo.png" width="300" />
</p>


# bnlearn - Library for Causal Discovery using Bayesian Learning

[![Python](https://img.shields.io/pypi/pyversions/bnlearn)](https://img.shields.io/pypi/pyversions/bnlearn)
[![PyPI Version](https://img.shields.io/pypi/v/bnlearn)](https://pypi.org/project/bnlearn/)
![GitHub Repo stars](https://img.shields.io/github/stars/erdogant/bnlearn)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/bnlearn/blob/master/LICENSE)
[![Forks](https://img.shields.io/github/forks/erdogant/bnlearn.svg)](https://github.com/erdogant/bnlearn/network)
[![Open Issues](https://img.shields.io/github/issues/erdogant/bnlearn.svg)](https://github.com/erdogant/bnlearn/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/bnlearn/month)](https://pepy.tech/project/bnlearn/)
[![Downloads](https://pepy.tech/badge/bnlearn)](https://pepy.tech/project/bnlearn)
[![DOI](https://zenodo.org/badge/231263493.svg)](https://zenodo.org/badge/latestdoi/231263493)
[![Docs](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/bnlearn/)
[![Medium](https://img.shields.io/badge/Medium-Blog-black)](https://erdogant.github.io/bnlearn/pages/html/Documentation.html#medium-blog)
![GitHub repo size](https://img.shields.io/github/repo-size/erdogant/bnlearn)
[![Donate](https://img.shields.io/badge/Support%20this%20project-grey.svg?logo=github%20sponsors)](https://erdogant.github.io/bnlearn/pages/html/Documentation.html#)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://erdogant.github.io/bnlearn/pages/html/Documentation.html#colab-notebook)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->


### 

``bnlearn`` is Python package for causal discovery by learning the graphical structure of Bayesian networks, parameter learning, inference and sampling methods.
Because probabilistic graphical models can be difficult in usage, Bnlearn for python (this package) is build on the <a href="https://github.com/pgmpy/pgmpy">pgmpy</a> package and contains the most-wanted pipelines. Navigate to [API documentations](https://erdogant.github.io/bnlearn/) for more detailed information.

# 
**⭐️ Star this repo if you like it ⭐️**
# 

### Read the Medium blog for more details.

<p align="left">
  <a href="https://towardsdatascience.com/a-step-by-step-guide-in-detecting-causal-relationships-using-bayesian-structure-learning-in-python-c20c6b31cee5" target="_blank">
    <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/medium_blog2.png?raw=true" width="150" />
  </a>

  <a href="https://towardsdatascience.com/a-step-by-step-guide-in-designing-knowledge-driven-models-using-bayesian-theorem-7433f6fd64be" target="_blank">
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/medium_blog1.png?raw=true" width="150" />
  </a>

</p>


---

# 


### [Documentation pages](https://erdogant.github.io/bnlearn/)

On the [documentation pages](https://erdogant.github.io/bnlearn/) you can find detailed information about the working of the ``bnlearn`` with many examples. 

# 

### Installation

##### It is advisable to create a new environment (e.g. with Conda). 
```bash
conda create -n env_bnlearn python=3.10
conda activate env_bnlearn
```

##### Install bnlearn from PyPI
```bash
pip install bnlearn
```

##### Install bnlearn from github source
```bash
pip install git+https://github.com/erdogant/bnlearn
```

##### The following functions are available after installation:

```python
# Import library
import bnlearn as bn

# Structure learning
bn.structure_learning.fit()

# Compute edge strength with the test statistic
bn.independence_test(model, df, test='chi_square', prune=True)

# Parameter learning
bn.parameter_learning.fit()

# Inference
bn.inference.fit()

# Make predictions
bn.predict()

# Based on a DAG, you can sample the number of samples you want.
bn.sampling()

# Load well-known examples to play around with or load your own .bif file.
bn.import_DAG()

# Load simple data frame of sprinkler dataset.
bn.import_example()

# Compare 2 graphs
bn.compare_networks()

# Plot graph
bn.plot()
bn.plot_graphviz()

# To make the directed graph undirected
bn.to_undirected()

# Convert to one-hot datamatrix
bn.df2onehot()

# Derive the topological ordering of the (entire) graph 
bn.topological_sort()

# See below for the exact working of the functions
```

##### The following methods are also included:
* inference
* sampling
* comparing two networks
* loading bif files
* Conversion of directed to undirected graphs


# 

### Method overview
Learning a Bayesian network can be split into the underneath problems which are all implemented in this package for both discrete, continuous and mixed data sets:

* **Structure learning**: Given the data: Estimate a DAG that captures the dependencies between the variables.
   * There are multiple manners to perform structure learning.
      * Constraintsearch or PC
      * Exhaustivesearch
      * Hillclimbsearch
      * NaiveBayes
      * TreeSearch
          * Chow-liu
          * Tree-augmented Naive Bayes (TAN)
      * Direct-LiNGAM (for continuous and hybrid datasets)
      * ICA-LiNGAM (for continuous and hybrid datasets)

* **Parameter learning**: Given the data and DAG: Estimate the (conditional) probability distributions of the individual variables.
* **Inference**: Given the learned model: Determine the exact probability values for your queries.

# 


### Examples

A structured overview of all examples are now available on the [documentation pages](https://erdogant.github.io/bnlearn/).

##### Structure learning

* [Example: Learn structure on the Sprinkler dataset based on a simple dataframe](https://erdogant.github.io/bnlearn/pages/html/Examples.html#example-1)

* [Example: Comparison method and scoring types types for structure learning](https://erdogant.github.io/bnlearn/pages/html/Examples.html#example-2)

* [Example: Learn structure on  more complex dataset (Asia)](https://erdogant.github.io/bnlearn/pages/html/Examples.html#example-3)

##### Parameter learning

* [Example: Parameter learning using a DAG and dataframe](https://erdogant.github.io/bnlearn/pages/html/Examples.html#parameter-learning)


##### Inferences

* [Example: Make predictions on a dataframe using inference](https://erdogant.github.io/bnlearn/pages/html/Predict.html)


##### Sampling

* [Example: Sampling to create datasets](https://erdogant.github.io/bnlearn/pages/html/Sampling%20and%20datasets.html)


##### Complete examples

* [Example: Create a Bayesian Network, learn its parameters from data and perform the inference](https://erdogant.github.io/bnlearn/pages/html/Examples.html#create-a-bayesian-network-learn-its-parameters-from-data-and-perform-the-inference)

* [Example: Use case in the medical domain](https://erdogant.github.io/bnlearn/pages/html/UseCases.html)

* [Example: Use case Titanic](https://erdogant.github.io/bnlearn/pages/html/UseCases.html#)



##### Plotting  
* [Example: Interactive plotting](https://erdogant.github.io/bnlearn/pages/html/Plot.html#)

* [Example: Static plotting](https://erdogant.github.io/bnlearn/pages/html/Plot.html#static-plot)

* [Example: Comparison of two networks](https://erdogant.github.io/bnlearn/pages/html/Plot.html#comparison-of-two-networks)

##### Various

* [Example: Saving and loading of bnlearn models](https://erdogant.github.io/bnlearn/pages/html/saving%20and%20loading.html)

* [Example: Data conversions such as creating sparse datamatrix from source-target and weights](https://erdogant.github.io/bnlearn/pages/html/dataframe%20conversions.html?highlight=target#)

* [Example: Load DAG from BIF files](https://erdogant.github.io/bnlearn/pages/html/Examples.html?highlight=comparison#import-from-bif)


 #

### Various basic examples


```python

    import bnlearn as bn
    # Example dataframe sprinkler_data.csv can be loaded with: 
    df = bn.import_example()
    # df = pd.read_csv('sprinkler_data.csv')
```

##### df looks like this

```python

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

```

```python

    model = bn.structure_learning.fit(df)
    # Compute edge strength with the chi_square test statistic
    model = bn.independence_test(model, df)
    G = bn.plot(model)
```

<p align="center">
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/fig_sprinkler_sl.png" width="600" />
</p>

* Choosing various methodtypes and scoringtypes:

```python

    model_hc_bic  = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model_hc_k2   = bn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
    model_hc_bdeu = bn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
    model_ex_bic  = bn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
    model_ex_k2   = bn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
    model_ex_bdeu = bn.structure_learning.fit(df, methodtype='ex', scoretype='bdeu')
    model_cl      = bn.structure_learning.fit(df, methodtype='cl', root_node='Wet_Grass')
    model_tan     = bn.structure_learning.fit(df, methodtype='tan', root_node='Wet_Grass', class_node='Rain')
```

## Example: Parameter Learning
```python
    import bnlearn as bn
    # Import dataframe
    df = bn.import_example()
    # As an example we set the CPD at False which returns an "empty" DAG
    model = bn.import_DAG('sprinkler', CPD=False)
    # Now we learn the parameters of the DAG using the df
    model_update = bn.parameter_learning.fit(model, df)
    # Make plot
    G = bn.plot(model_update)
```

## Example: Inference
```python
    import bnlearn as bn
    model = bn.import_DAG('sprinkler')
    query = bn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1,'Sprinkler':0, 'Wet_Grass':1})
    print(query)
    print(query.df)
    
    # Lets try another inference
    query = bn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1})
    print(query)
    print(query.df)

```

<hr>

### References
* https://erdogant.github.io/bnlearn/
* http://www.bnlearn.com/bnrepository/
* http://pgmpy.org


### Contributors
Setting up and maintaining bnlearn has been possible thanks to users and contributors. Thanks to:

<p align="left">
  <a href="https://github.com/erdogant/bnlearn/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=erdogant/bnlearn" />
  </a>
</p>


### Citation
Please cite ``bnlearn`` in your publications if this is useful for your research. See column right for citation information.

### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)

<a href="https://www.buymeacoffee.com/erdogant"><img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=&slug=erdogant&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" /></a>
