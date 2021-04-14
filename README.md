# bnlearn - Graphical structure of Bayesian networks

[![Python](https://img.shields.io/pypi/pyversions/bnlearn)](https://img.shields.io/pypi/pyversions/bnlearn)
[![PyPI Version](https://img.shields.io/pypi/v/bnlearn)](https://pypi.org/project/bnlearn/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/bnlearn/blob/master/LICENSE)
[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)
[![Github Forks](https://img.shields.io/github/forks/erdogant/bnlearn.svg)](https://github.com/erdogant/bnlearn/network)
[![GitHub Open Issues](https://img.shields.io/github/issues/erdogant/bnlearn.svg)](https://github.com/erdogant/bnlearn/issues)
[![Project Status](http://www.repostatus.org/badges/latest/active.svg)](http://www.repostatus.org/#active)
[![Downloads](https://pepy.tech/badge/bnlearn/month)](https://pepy.tech/project/bnlearn/)
[![Downloads](https://pepy.tech/badge/bnlearn)](https://pepy.tech/project/bnlearn)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/bnlearn/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/erdogant/bnlearn/blob/master/notebooks/bnlearn.ipynb)

    Star it if you like it!

``bnlearn`` is Python package for learning the graphical structure of Bayesian networks, parameter learning, inference and sampling methods. This work is inspired by the R package (bnlearn.com) that has been very usefull to me for many years. Although there are very good Python packages for probabilistic graphical models, it still can remain difficult (and somethimes unnecessarily) to (re)build certain pipelines. Bnlearn for python (this package) is build on the <a href="https://github.com/pgmpy/pgmpy">pgmpy</a> package and contains the most-wanted pipelines. Navigate to [API documentations](https://erdogant.github.io/bnlearn/) for more detailed information.

### Method overview
Learning a Bayesian network can be split into two problems which are both implemented in this package:
* Structure learning: Given a set of data samples, estimate a DAG that captures the dependencies between the variables.
* Parameter learning: Given a set of data samples and a DAG that captures the dependencies between the variables, estimate the (conditional) probability distributions of the individual variables.

#### The following functions are available after installation:

```python
# Import library
import bnlearn as bn

# Structure learning
bn.structure_learning.fit()

# Parameter learning
bn.parameter_learning.fit()

# Inference
bn.inference.fit()

# Based on a DAG, you can sample the number of samples you want.
bn.sampling()

# Load well known examples to play arround with or load your own .bif file.
bn.import_DAG()

# Load simple dataframe of sprinkler dataset.
bn.import_example()

# Compare 2 graphs
bn.compare_networks()

# Plot graph
bn.plot()

# To make the directed grapyh undirected
bn.to_undirected()

# Convert to one-hot datamatrix
bn.df2onehot()
 
# See below for the exact working of the functions
```

#### The following methods are also included:
* inference
* sampling
* comparing two networks
* loading bif files
* conversion of directed to undirected graphs

## Conda installation
It is advisable to create a new environment. 
```bash
conda create -n env_bnlearn python=3.8
conda activate env_bnlearn
```

## Conda installation
```python
conda install -c ankurankan pgmpy
pip install -U bnlearn # -U is to force download latest version
```

## Pip installation
```python
pip install -U pgmpy>=0.1.13
pip install -U bnlearn # -U is to force to overwrite current version
```

* Alternatively, install bnlearn from the GitHub source:
```bash
git clone https://github.com/erdogant/bnlearn.git
cd bnlearn
pip install -U .
```  

## Import bnlearn package
```python
import bnlearn as bn
```

## Example: Structure Learning
```python
# Example dataframe sprinkler_data.csv can be loaded with: 
df = bn.import_example()
# df = pd.read_csv('sprinkler_data.csv')
model = bn.structure_learning.fit(df)
G = bn.plot(model)
```

#### df looks like this
```
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
```

## Example: Parameter Learning
```python
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
model = bn.import_DAG('sprinkler')
q_1 = bn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1,'Sprinkler':0, 'Wet_Grass':1})
q_2 = bn.inference.fit(model, variables=['Rain'], evidence={'Cloudy':1})
```

## Example: Sampling to create dataframe
```python
model = bn.import_DAG('sprinkler')
df = bn.sampling(model, n=1000)
```

* Output of the model:
```
[bnlearn] Model correct: True
CPD of Cloudy:
+-----------+-----+
| Cloudy(0) | 0.5 |
+-----------+-----+
| Cloudy(1) | 0.5 |
+-----------+-----+
CPD of Sprinkler:
+--------------+-----------+-----------+
| Cloudy       | Cloudy(0) | Cloudy(1) |
+--------------+-----------+-----------+
| Sprinkler(0) | 0.5       | 0.9       |
+--------------+-----------+-----------+
| Sprinkler(1) | 0.5       | 0.1       |
+--------------+-----------+-----------+
CPD of Rain:
+---------+-----------+-----------+
| Cloudy  | Cloudy(0) | Cloudy(1) |
+---------+-----------+-----------+
| Rain(0) | 0.8       | 0.2       |
+---------+-----------+-----------+
| Rain(1) | 0.2       | 0.8       |
+---------+-----------+-----------+
CPD of Wet_Grass:
+--------------+--------------+--------------+--------------+--------------+
| Sprinkler    | Sprinkler(0) | Sprinkler(0) | Sprinkler(1) | Sprinkler(1) |
+--------------+--------------+--------------+--------------+--------------+
| Rain         | Rain(0)      | Rain(1)      | Rain(0)      | Rain(1)      |
+--------------+--------------+--------------+--------------+--------------+
| Wet_Grass(0) | 1.0          | 0.1          | 0.1          | 0.01         |
+--------------+--------------+--------------+--------------+--------------+
| Wet_Grass(1) | 0.0          | 0.9          | 0.9          | 0.99         |
+--------------+--------------+--------------+--------------+--------------+
[bnlearn] Nodes: ['Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass']
[bnlearn] Edges: [('Cloudy', 'Sprinkler'), ('Cloudy', 'Rain'), ('Sprinkler', 'Wet_Grass'), ('Rain', 'Wet_Grass')]
[bnlearn] Independencies:
(Cloudy _|_ Wet_Grass | Rain, Sprinkler)
(Sprinkler _|_ Rain | Cloudy)
(Rain _|_ Sprinkler | Cloudy)
(Wet_Grass _|_ Cloudy | Rain, Sprinkler)
```

## Example: Loading DAG from bif files
```python
bif_file= 'sprinkler'
bif_file= 'alarm'
bif_file= 'andes'
bif_file= 'asia'
bif_file= 'pathfinder'
bif_file= 'sachs'
bif_file= 'miserables'
bif_file= 'filepath/to/model.bif'

# Loading example dataset
model = bn.import_DAG(bif_file)
```

## Example: Comparing networks
```python
# Load asia DAG
model = bn.import_DAG('asia')
# plot ground truth
G = bn.plot(model)
# Sampling
df = bn.sampling(model, n=10000)
# Structure learning of sampled dataset
model_sl = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
# Plot based on structure learning of sampled data
bn.plot(model_sl, pos=G['pos'])
# Compare networks and make plot
bn.compare_networks(model, model_sl, pos=G['pos'])
```

#### Graph of ground truth
<p align="center">
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/fig2a_asia_groundtruth.png" width="600" />
</p>

#### Graph based on Structure learning
<p align="center">
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/fig2b_asia_structurelearning.png" width="600" />
</p>

#### Graph comparison ground truth vs. structure learning
<p align="center">
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/fig2c_asia_comparion.png" width="600" />
  <img src="https://github.com/erdogant/bnlearn/blob/master/docs/figs/fig2d_confmatrix.png" width="400" />
</p>


## Example: Titanic example

```python

    import bnlearn as bn
    # Load example mixed dataset
    df_raw = bn.import_example(data='titanic')
    # Convert to onehot
    dfhot, dfnum = bn.df2onehot(df_raw)
    # Structure learning
    DAG = bn.structure_learning.fit(dfnum, methodtype='cl', black_list=['Embarked','Parch','Name'], root_node='Survived', bw_list_method='filter')
    # Plot
    G = bn.plot(DAG)
    # Parameter learning
    model = bn.parameter_learning.fit(DAG, dfnum)
    # Make inference
    q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':True, 'Pclass':True})
    q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':0})
    # Print model
    bn.print_CPD(model)
```

## Citation
Please cite bnlearn in your publications if this is useful for your research. Here is an example BibTeX entry:
```BibTeX
@misc{erdogant2019bnlearn,
  title={bnlearn},
  author={Erdogan Taskesen},
  year={2019},
  howpublished={\url{https://github.com/erdogant/bnlearn}},
}
```

## References
* https://erdogant.github.io/bnlearn/
* http://pgmpy.org
* https://programtalk.com/python-examples/pgmpy.factors.discrete.TabularCPD/
* http://www.bnlearn.com/
* http://www.bnlearn.com/bnrepository/


### Maintainer
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
