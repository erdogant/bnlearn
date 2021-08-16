# bnlearn - Library for Bayesian network learning and inference

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
Learning a Bayesian network can be split into the underneath problems which are all implemented in this package:
* Structure learning: Given the data: Estimate a DAG that captures the dependencies between the variables.
* Parameter learning: Given the data and DAG: Estimate the (conditional) probability distributions of the individual variables.
* Inference: Given the learned model: Determine the exact probability values for your queries.

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

# Make predictions
bn.predict()

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

# Derive the topological ordering of the (entire) graph 
bn.topological_sort()

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
import bnlearn as bn
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

## Example: Sampling to create dataframe
```python
import bnlearn as bn
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
import bnlearn as bn

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
    query = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':True, 'Pclass':True})
    print(query)
    print(query.df)

    # Another inference using only sex for evidence
    q1 = bn.inference.fit(model, variables=['Survived'], evidence={'Sex':0})
    print(query)
    print(query.df)

    # Print model
    bn.print_CPD(model)

```

## Example: Make predictions on a dataframe using inference

```python

    # Import bnlearn
    import bnlearn as bn
    
    # Load example DataFrame
    df = bn.import_example('asia')
    print(df)
    #       smoke  lung  bronc  xray
    # 0         0     1      0     1
    # 1         0     1      1     1
    # 2         1     1      1     1
    # 3         1     1      0     1
    # 4         1     1      1     1
    #     ...   ...    ...   ...
    # 9995      1     1      1     1
    # 9996      1     1      1     1
    # 9997      0     1      1     1
    # 9998      0     1      1     1
    # 9999      0     1      1     0

    # Create some edges for the DAG
    edges = [('smoke', 'lung'),
             ('smoke', 'bronc'),
             ('lung', 'xray'),
             ('bronc', 'xray')]
    
    # Construct the Bayesian DAG
    DAG = bn.make_DAG(edges, verbose=0)
    # Plot DAG
    bn.plot(DAG)

    # Learn CPDs using the DAG and dataframe
    model = bn.parameter_learning.fit(DAG, df, verbose=3)
    bn.print_CPD(model)

    # CPD of smoke:
    # +----------+----------+
    # | smoke(0) | 0.500364 |
    # +----------+----------+
    # | smoke(1) | 0.499636 |
    # +----------+----------+
    # CPD of lung:
    # +---------+---------------------+----------------------+
    # | smoke   | smoke(0)            | smoke(1)             |
    # +---------+---------------------+----------------------+
    # | lung(0) | 0.13753633720930233 | 0.055131004366812224 |
    # +---------+---------------------+----------------------+
    # | lung(1) | 0.8624636627906976  | 0.9448689956331878   |
    # +---------+---------------------+----------------------+
    # CPD of bronc:
    # +----------+--------------------+--------------------+
    # | smoke    | smoke(0)           | smoke(1)           |
    # +----------+--------------------+--------------------+
    # | bronc(0) | 0.5988372093023255 | 0.3282387190684134 |
    # +----------+--------------------+--------------------+
    # | bronc(1) | 0.4011627906976744 | 0.6717612809315866 |
    # +----------+--------------------+--------------------+
    # CPD of xray:
    # +---------+---------------------+---------------------+---------------------+---------------------+
    # | bronc   | bronc(0)            | bronc(0)            | bronc(1)            | bronc(1)            |
    # +---------+---------------------+---------------------+---------------------+---------------------+
    # | lung    | lung(0)             | lung(1)             | lung(0)             | lung(1)             |
    # +---------+---------------------+---------------------+---------------------+---------------------+
    # | xray(0) | 0.7787162162162162  | 0.09028393966282165 | 0.7264957264957265  | 0.07695139911634757 |
    # +---------+---------------------+---------------------+---------------------+---------------------+
    # | xray(1) | 0.22128378378378377 | 0.9097160603371783  | 0.27350427350427353 | 0.9230486008836525  |
    # +---------+---------------------+---------------------+---------------------+---------------------+
    # [bnlearn] >Independencies:
    # (smoke ⟂ xray | bronc, lung)
    # (lung ⟂ bronc | smoke)
    # (bronc ⟂ lung | smoke)
    # (xray ⟂ smoke | bronc, lung)
    # [bnlearn] >Nodes: ['smoke', 'lung', 'bronc', 'xray']
    # [bnlearn] >Edges: [('smoke', 'lung'), ('smoke', 'bronc'), ('lung', 'xray'), ('bronc', 'xray')]


    # Generate some example data based on DAG
    Xtest = bn.sampling(model, n=1000)
    print(Xtest)
    #      smoke  lung  bronc  xray
    # 0        1     1      1     1
    # 1        1     1      1     1
    # 2        0     1      1     1
    # 3        1     0      0     1
    # 4        1     1      1     1
    # ..     ...   ...    ...   ...
    # 995      1     1      1     1
    # 996      1     1      1     1
    # 997      0     1      0     1
    # 998      0     1      0     1
    # 999      0     1      1     1
    

    # Make predictions
    Pout = bn.predict(model, Xtest, variables=['bronc','xray'])
    print(Pout)

    #         xray  bronc         p
    # 0       1      0  0.542757
    # 1       1      1  0.624117
    # 2       1      0  0.542757
    # 3       1      1  0.624117
    # 4       1      0  0.542757
    # ..    ...    ...       ...
    # 995     1      0  0.542757
    # 996     1      0  0.542757
    # 997     1      1  0.624117
    # 998     1      1  0.624117
    # 999     1      0  0.542757

    

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
