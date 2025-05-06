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

<div>

<a href="https://erdogant.github.io/bnlearn/"><img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/logo.png" width="175" align="left" /></a>
``bnlearn`` is Python package for causal discovery by learning the graphical structure of Bayesian networks, parameter learning, inference, and sampling methods.
Because probabilistic graphical models can be difficult to use, ``Bnlearn`` contains the most-wanted pipelines. Navigate to [API documentations](https://erdogant.github.io/bnlearn/) for more detailed information. **⭐️ Star it if you like it ⭐️**
</div>

---

### Key Pipelines

| Feature | Description |
|--------|-------------|
| [**Causal Discovery / Structure Learning**](https://erdogant.github.io/bnlearn/pages/html/Structure%20learning.html) | Learn the model structure from data or with expert knowledge. |
| [**Parameter Learning**](https://erdogant.github.io/bnlearn/pages/html/Parameter%20learning.html) | Estimate model parameters (e.g., conditional probability distributions) from observed data. |
| [**Causal Inference**](https://pgmpy.org/examples/Causal%20Inference.html) | Compute interventional and counterfactual distributions using do-calculus. |
| [**Synthetic Data**](https://erdogant.github.io/bnlearn/pages/html/Sampling.html) | Generate synthetic data. |
| [**Discretize Data**](https://erdogant.github.io/bnlearn/pages/html/Discretizing.html) | Discretize continuous datasets. |

---

### Resources and Links
- **Example Notebooks:** [Examples](https://erdogant.github.io/bnlearn/pages/html/Documentation.html#google-colab-notebooks)
- **Blog Posts:** [Medium](https://erdogant.medium.com/)
- **Documentation:** [Website](https://erdogant.github.io/bnlearn)
- **Bug Reports and Feature Requests:** [GitHub Issues](https://github.com/erdogant/bnlearn/issues)

---

### The following functions are available after installation:

| Feature | Description |
|--------|-------------|
| **Key Pipelines** | |
|  [Structure learning](https://erdogant.github.io/bnlearn/pages/html/Structure%20learning.html#exhaustivesearch) | ```bn.structure_learning.fit()``` |
| [Parameter learning](https://erdogant.github.io/bnlearn/pages/html/Parameter%20learning.html#parameter-learning-examples) | ```bn.parameter_learning.fit()``` |
| [Inference](https://erdogant.github.io/bnlearn/pages/html/Inference.html#examples-inference) | ```bn.inference.fit()``` |
| [Make predictions](https://erdogant.github.io/bnlearn/pages/html/Predict.html) | ```bn.predict()``` |
| [Generate Synthetic Data](https://erdogant.github.io/bnlearn/pages/html/Sampling.html) | ```bn.sampling()``` |
| [Compute Edge Strength](https://erdogant.github.io/bnlearn/pages/html/independence_test.html) | ```bn.independence_test(model, df, test='chi_square', prune=True)``` |
| **Key Functions** | |
| [Imputations](https://erdogant.github.io/bnlearn/pages/html/impute.html#) | ```bn.knn_imputer()``` |
| [Discretizing](https://erdogant.github.io/bnlearn/pages/html/Discretizing.html#) | ```bn.discretize()``` |
| [Check Model Parameters](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.check_model) | ```bn.check_model()``` |
| [Create DAG](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.make_DAG) | ```bn.make_DAG()``` |
| [Get Node Properties](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.get_node_properties) | ```bn.get_node_properties()``` |
| [Get Edge Properties](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.get_edge_properties) | ```bn.get_edge_properties()``` |
| [Get Parents From Edges](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.get_parents) | ```bn.get_parents()``` |
| [Generate Default CPT per Node](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.generate_cpt) | ```bn.generate_cpt()``` |
| [Generate Default CPTs for All Edges](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.build_cpts_from_structure) | ```bn.build_cpts_from_structure()``` |
| **Make Plots** | |
| [Plotting](https://erdogant.github.io/bnlearn/pages/html/Plot.html) | ```bn.plot()``` |
| [Plot Graphviz](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.plot_graphviz) | ```bn.plot_graphviz()``` |
| [Compare 2 Networks](https://erdogant.github.io/bnlearn/pages/html/Plot.html#comparison-of-two-networks) | ```bn.compare_networks()``` |
| [Load DAG (bif files)](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.import_DAG) | ```bn.import_DAG()``` |
| [Load Examples](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.import_example) | ```bn.import_example()``` |
| **Transformation Functions** | |
| [Convert DAG to Undirected](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.to_undirected) | ```bn.to_undirected()``` |
| [Convert to one-hot](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.df2onehot) | ```bn.df2onehot()``` |
| [Convert Adjacency Matrix to Vector](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.adjmat2vec) | ```bn.adjmat2vec()``` |
| [Convert Adjacency Matrix to Dictionary](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.adjmat2dict) | ```bn.adjmat2dict()``` |
| [Convert Vector to Adjacency Matrix](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.vec2adjmat) | ```bn.vec2adjmat()``` |
| [Convert DAG to Adjacency Matrix](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.dag2adjmat) | ```bn.dag2adjmat()``` |
| [Convert DataFrame to Onehot](https://erdogant.github.io/bnlearn/pages/html/Example%20Datasets.html) | ```bn.df2onehot()``` |
| [Convert Query to DataFrame](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.query2df) | ```bn.query2df()``` |
| [Convert Vector to DataFrame](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.vec2df) | ```bn.vec2df()``` |
| **Metrics** | |
| [Compute Topological Ordering](https://erdogant.github.io/bnlearn/pages/html/topological_sort.html) | ```bn.topological_sort()``` |
| [Compute Structure Scores](https://erdogant.github.io/bnlearn/pages/html/Structure_scores.html) | ```bn.structure_scores()``` |
| **General** | |
| [Save Model](https://erdogant.github.io/bnlearn/pages/html/saving%20and%20loading.html) | ```bn.save()``` |
| [Load Model](https://erdogant.github.io/bnlearn/pages/html/saving%20and%20loading.html) | ```bn.load()``` |
| [Print CPTs](https://erdogant.github.io/bnlearn/pages/html/bnlearn.bnlearn.html#bnlearn.bnlearn.print_CPD) | ```bn.print_CPD()``` |

---
### Installation

##### Install bnlearn from PyPI
```bash
pip install bnlearn
```
##### Install bnlearn from github source
```bash
pip install git+https://github.com/erdogant/bnlearn
```
##### Load library
```python
# Import library
import bnlearn as bn
```
---

### Code Examples

```python

    import bnlearn as bn
    # Example dataframe sprinkler_data.csv can be loaded with: 
    df = bn.import_example()
    # df = pd.read_csv('sprinkler_data.csv')

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


    model = bn.structure_learning.fit(df)
    # Compute edge strength with the chi-square test statistic
    model = bn.independence_test(model, df)
    G = bn.plot(model)
```

<p align="center">
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/fig_sprinkler_sl.png" width="600" />
</p>

* Choosing various methodtypes and scoringtypes:

```python

# Example: Structure Learning

    model_hc_bic  = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model_hc_k2   = bn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
    model_hc_bdeu = bn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
    model_ex_bic  = bn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
    model_ex_k2   = bn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
    model_ex_bdeu = bn.structure_learning.fit(df, methodtype='ex', scoretype='bdeu')
    model_cl      = bn.structure_learning.fit(df, methodtype='cl', root_node='Wet_Grass')
    model_tan     = bn.structure_learning.fit(df, methodtype='tan', root_node='Wet_Grass', class_node='Rain')

# Example: Parameter Learning

    import bnlearn as bn
    # Import dataframe
    df = bn.import_example()
    # As an example we set the CPD at False which returns an "empty" DAG
    model = bn.import_DAG('sprinkler', CPD=False)
    # Now we learn the parameters of the DAG using the df
    model_update = bn.parameter_learning.fit(model, df)
    # Make plot
    G = bn.plot(model_update)

# Example: Inference

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
