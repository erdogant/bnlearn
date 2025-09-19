import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bnlearn as bn

def test_example_3():
    import bnlearn as bn
    # Load sprinkler dataset
    df = bn.import_example('sprinkler')

    # 'hc' or 'hillclimbsearch'
    model_hc_bic  = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model_hc_k2   = bn.structure_learning.fit(df, methodtype='hc', scoretype='k2')
    model_hc_bdeu = bn.structure_learning.fit(df, methodtype='hc', scoretype='bdeu')
    
    # 'ex' or 'exhaustivesearch'
    model_ex_bic  = bn.structure_learning.fit(df, methodtype='ex', scoretype='bic')
    model_ex_k2   = bn.structure_learning.fit(df, methodtype='ex', scoretype='k2')
    model_ex_bdeu = bn.structure_learning.fit(df, methodtype='ex', scoretype='bdeu')
    
    # 'cs' or 'constraintsearch'
    model_cs_k2   = bn.structure_learning.fit(df, methodtype='cs', scoretype='k2')
    model_cs_bdeu = bn.structure_learning.fit(df, methodtype='cs', scoretype='bdeu')
    model_cs_bic  = bn.structure_learning.fit(df, methodtype='cs', scoretype='bic')
    
    # 'cl' or 'chow-liu' (requires setting root_node parameter)
    model_cl      = bn.structure_learning.fit(df, methodtype='cl', root_node='Wet_Grass')

    # Examples to illustrate how to manually compute MLE for the node Cloudy and Rain:
    
    # Compute CPT for the Cloudy Node:
    # This node has no conditional dependencies and can easily be computed as following:
    
    # P(Cloudy=0)
    sum(df['Cloudy']==0) / df.shape[0] # 0.488
    
    # P(Cloudy=1)
    sum(df['Cloudy']==1) / df.shape[0] # 0.512
    
    # Compute CPT for the Rain Node:
    # This node has a conditional dependency from Cloudy and can be computed as following:
    
    # P(Rain=0 | Cloudy=0)
    sum( (df['Cloudy']==0) & (df['Rain']==0) ) / sum(df['Cloudy']==0) # 394/488 = 0.807377049
    
    # P(Rain=1 | Cloudy=0)
    sum( (df['Cloudy']==0) & (df['Rain']==1) ) / sum(df['Cloudy']==0) # 94/488  = 0.192622950
    
    # P(Rain=0 | Cloudy=1)
    sum( (df['Cloudy']==1) & (df['Rain']==0) ) / sum(df['Cloudy']==1) # 91/512  = 0.177734375
    
    # P(Rain=1 | Cloudy=1)
    sum( (df['Cloudy']==1) & (df['Rain']==1) ) / sum(df['Cloudy']==1) # 421/512 = 0.822265625

def test_example_2():
    import bnlearn as bn
    # Load sprinkler dataset
    df = bn.import_example('sprinkler')
    # The edges can be created using the available variables.
    print(df.columns)
    # ['Cloudy', 'Sprinkler', 'Rain', 'Wet_Grass']
    
    # Define the causal dependencies based on your expert/domain knowledge.
    # Left is the source, and right is the target node.
    edges = [('Cloudy', 'Sprinkler'),
             ('Cloudy', 'Rain'),
             ('Sprinkler', 'Wet_Grass'),
             ('Rain', 'Wet_Grass')]
    
    
    # Create the DAG
    DAG = bn.make_DAG(edges)
    
    # Plot the DAG. This is identical as shown in Figure 3
    bn.plot(DAG)
    
    # Print the Conditional probability Tables
    bn.print_CPD(DAG)
    # [bnlearn] >No CPDs to print. Tip: use bnlearn.plot(DAG) to make a plot.
    # This is correct, we did not learn any CPTs yet! We only defined the graph without defining any probabilities.
    
    # Parameter learning on the user-defined DAG and input data using maximumlikelihood
    model_mle = bn.parameter_learning.fit(DAG, df, methodtype='maximumlikelihood')
    
    # Print the learned CPDs
    bn.print_CPD(model_mle)
    
def test_example_1():
    import bnlearn as bn
    # Load sprinkler dataset
    df = bn.import_example('sprinkler')
    # Print to screen for illustration
    print(df)
    
    # Learn the DAG in data using Bayesian structure learning:
    DAG = bn.structure_learning.fit(df)
    
    # print adjacency matrix
    print(DAG['adjmat'])
    # target     Cloudy  Sprinkler   Rain  Wet_Grass
    # source                                        
    # Cloudy      False      False   True      False
    # Sprinkler    True      False  False       True
    # Rain        False      False  False       True
    # Wet_Grass   False      False  False      False
    
    # Plot
    G = bn.plot(DAG)
    bn.plot_graphviz(DAG)

    # Interactive plotting
    G = bn.plot(DAG, interactive=True)

def test_hypergeo_test():
    import bnlearn
    import pandas as pd
    from scipy.stats import hypergeom
    
    # Load titanic dataset
    df = bnlearn.import_example(data='titanic')
    
    print(df[['Survived','Sex']])
    #     Survived     Sex
    #0           0    male
    #1           1  female
    #2           1  female
    #3           1  female
    #4           0    male
    #..        ...     ...
    #886         0    male
    #887         1  female
    #888         0  female
    #889         1    male
    #890         0    male
    #[891 rows x 2 columns]
    
    # Total number of samples
    N=df.shape[0]
    
    # Number of success in the population
    K=sum(df['Survived']==1)
    
    # Sample size/number of draws
    n=sum(df['Sex']=='female')
    
    # Overlap between female and survived
    x=sum((df['Sex']=='female') & (df['Survived']==1))
    
    print(x-1, N, n, K)
    # 232 891 314 342
    
    # Compute
    P = hypergeom.sf(x, N, n, K)
    P = hypergeom.sf(232, 891, 314, 342)
    
    # 3.5925132664684234e-60
    assert P==3.5925132664694183e-60
    

def test_bnlearn_example_1():
    # Import data set and drop continous and sensitive features
    df = bn.import_example(data='census_income')
    
    # Data cleaning
    drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
    df.drop(labels=drop_cols, axis=1, inplace=True)
    
    # Print
    df.head()
    
    #          workclass  education  ... native-country salary
    #0         State-gov  Bachelors  ...  United-States  <=50K
    #1  Self-emp-not-inc  Bachelors  ...  United-States  <=50K
    #2           Private    HS-grad  ...  United-States  <=50K
    #3           Private       11th  ...  United-States  <=50K
    #4           Private  Bachelors  ...           Cuba  <=50K
    #
    #[5 rows x 7 columns]

    # Structure learning
    model = bn.structure_learning.fit(df, methodtype='hillclimbsearch', scoretype='bic')
    
    # Test edges significance and remove.
    model = bn.independence_test(model, df, test="chi_square", alpha=0.05, prune=True)
    
    # Make plot
    G = bn.plot(model, interactive=False)

    # Ceate dotgraph plot
    dotgraph = bn.plot_graphviz(model)
    # Create pdf
    dotgraph.view(filename=r'c:/temp/bnlearn_plot') 
   
    # Make plot interactive
    G = bn.plot(model, interactive=True)
    
    # Show edges
    print(model['model_edges'])
    # [('education', 'salary'),
    # ('marital-status', 'relationship'),
    # ('occupation', 'workclass'),
    # ('occupation', 'education'),
    # ('relationship', 'salary'),
    # ('relationship', 'occupation')]

    # Learn the CPD by providing the model and dataframe
    model = bn.parameter_learning.fit(model, df)
    
    # Print the CPD
    CPD = bn.print_CPD(model)
    
    query = bn.inference.fit(model, variables=['salary'], evidence={'education':'Doctorate'})
    query = bn.inference.fit(model, variables=['salary'], evidence={'education':'HS-grad'})
    query = bn.inference.fit(model, variables=['workclass'], evidence={'education':'Doctorate', 'marital-status':'Never-married'})
    

def test_pgmpy():
    # Import functions from pgmpy
    from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
    from pgmpy.models import BayesianNetwork, NaiveBayes
    from pgmpy.inference import VariableElimination
    
    # Import data set and drop continous and sensitive features
    df = bn.import_example(data='census_income')
    drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
    df.drop(labels=drop_cols, axis=1, inplace=True)
    
    # Create estimator
    est = HillClimbSearch(df)
    
    # Create scoring method
    scoring_method = BicScore(df)
    
    # Create the model and print the edges
    model = est.estimate(scoring_method=scoring_method)
    
    # Show edges
    print(model.edges())
    # [('education', 'salary'),
    # ('marital-status', 'relationship'),
    # ('occupation', 'workclass'),
    # ('occupation', 'education'),
    # ('relationship', 'salary'),
    # ('relationship', 'occupation')]

    vec = {
        'source': ['education', 'marital-status', 'occupation', 'relationship', 'relationship', 'salary'],
        'target': ['occupation', 'relationship', 'workclass', 'education', 'salary', 'education'],
        'weight': [True, True, True, True, True, True]
    }
    vec = pd.DataFrame(vec)

    # Create Bayesian model
    bayesianmodel = BayesianNetwork(vec)
    
    # Fit the model
    bayesianmodel.fit(df, estimator=BayesianEstimator, prior_type='bdeu', equivalent_sample_size=1000)
    
    # Create model for variable elimination
    model_infer = VariableElimination(bayesianmodel)
    
    # Query
    query = model_infer.query(variables=['salary'], evidence={'education':'Doctorate'})
    print(query)


def test_causalnex():
    from causalnex.structure.notears import from_pandas
    from causalnex.network import BayesianNetwork
    import networkx as nx
    import datazets as dz
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
    le = LabelEncoder()

    # Import data set and drop continous and sensitive features
    df = dz.get(data='census_income')

    # Clean
    drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
    df.drop(labels=drop_cols, axis=1, inplace=True)
    
    # Next, we want to make our data numeric, since this is what the NOTEARS expect.
    df_num = df.copy()
    for col in df_num.columns:
        df_num[col] = le.fit_transform(df_num[col])

    # Structure learning
    sm = from_pandas(df_num)
    
    # Thresholding
    sm.remove_edges_below_threshold(0.8)
    
    # Use positions from Bnlearn
    # pos=G['pos']
    
    # Make plot
    plt.figure(figsize=(15,10));
    edge_width = [ d['weight']*0.3 for (u,v,d) in sm.edges(data=True)]
    nx.draw_networkx(sm, node_size=400, arrowsize=20, alpha=0.6, edge_color='b', width=edge_width)
    
    # If required, remove spurious edges and relearn structure.
    sm = from_pandas(df_num, tabu_edges=[("relationship", "native-country")], w_threshold=0.8)

    # Step 1: Create a new instance of BayesianNetwork
    bn = BayesianNetwork(sm)
    
    # Step 2: Reduce the cardinality of categorical features
    # Use domain knowledge or other manners to remove redundant features.
    
    # Step 3: Create Labels for Numeric Features
    # Create a dictionary that describes the relation between numeric value and label.
    
    # Step 4: Specify all of the states that each node can take
    bn = bn.fit_node_states(df)
    
    # Step 5: Fit Conditional Probability Distributions
    bn = bn.fit_cpds(df, method="BayesianEstimator", bayes_prior="K2")
    
    # Return CPD for education
    result = bn.cpds["education"]
    
    # Extract any information and probabilities related to education.
    print(result)
    
    # marital-status  Divorced              ...   Widowed            
    # salary             <=50K              ...      >50K            
    # workclass              ? Federal-gov  ... State-gov Without-pay
    # education                             ...                      
    # 10th            0.077320    0.019231  ...  0.058824      0.0625
    # 11th            0.061856    0.012821  ...  0.117647      0.0625
    # 12th            0.020619    0.006410  ...  0.058824      0.0625
    # 1st-4th         0.015464    0.006410  ...  0.058824      0.0625
    # 5th-6th         0.010309    0.006410  ...  0.058824      0.0625
    # 7th-8th         0.056701    0.006410  ...  0.058824      0.0625
    # 9th             0.067010    0.006410  ...  0.058824      0.0625
    # Assoc-acdm      0.025773    0.057692  ...  0.058824      0.0625
    # Assoc-voc       0.046392    0.051282  ...  0.058824      0.0625
    # Bachelors       0.097938    0.128205  ...  0.058824      0.0625
    # Doctorate       0.005155    0.006410  ...  0.058824      0.0625
    # HS-grad         0.278351    0.333333  ...  0.058824      0.0625
    # Masters         0.015464    0.032051  ...  0.058824      0.0625
    # Preschool       0.005155    0.006410  ...  0.058824      0.0625
    # Prof-school     0.015464    0.006410  ...  0.058824      0.0625
    # Some-college    0.201031    0.314103  ...  0.058824      0.0625
    # [16 rows x 126 columns]

def test_dowhy():
    # Import libraries
    from dowhy import CausalModel
    import dowhy.datasets
    import datazets as dz
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    le = LabelEncoder()

    # Import data set and drop continous and sensitive features
    df = dz.get(data='census_income')

    drop_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'race', 'sex']
    df.drop(labels=drop_cols, axis=1, inplace=True)
    
    # Treatment variable must be binary
    df['education'] = df['education']=='Doctorate'
    
    # Next, we need to make our data numeric.
    df_num = df.copy()
    for col in df_num.columns:
        df_num[col] = le.fit_transform(df_num[col])
    
    # Specify the treatment, outcome, and potential confounding variables
    treatment = "education"
    outcome = "salary"
    
    # Step 1. Create a Causal Graph
    model= CausalModel(
            data=df_num,
            treatment=treatment,
            outcome=outcome,
            common_causes=list(df.columns[~np.isin(df.columns, [treatment, outcome])]),
            graph_builder='ges',
            alpha=0.05,
            )
    
    # Display the model
    model.view_model()

    # Step 2: Identify causal effect and return target estimands
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    # Results
    print(identified_estimand)
    # Estimand type: EstimandType.NONPARAMETRIC_ATE
    # ### Estimand : 1
    # Estimand name: backdoor
    # Estimand expression:
    #      d                                                                        
    # ────────────(E[salary|workclass,marital-status,native-country,relationship,occupation])
    # d[education]                                                                  
    #        
    # Estimand assumption 1, Unconfoundedness: If U→{education} and U→salary then P(salary|education,workclass,marital-status,native-country,relationship,occupation,U) = P(salary|education,workclass,marital-status,native-country,relationship,occupation)
    #
    # ### Estimand : 2
    # Estimand name: iv
    # No such variable(s) found!
    #
    # ### Estimand : 3
    # Estimand name: frontdoor
    # No such variable(s) found!


    # Step 3: Estimate the target estimand using a statistical method.
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_stratification")
    
    # Results
    print(estimate)
    #*** Causal Estimate ***
    #
    ## Identified estimand
    # Estimand type: EstimandType.NONPARAMETRIC_ATE
    #
    ### Estimand : 1
    # Estimand name: backdoor
    # Estimand expression:
    #      d                                                                        
    # ────────────(E[salary|workclass,marital-status,native-country,relationship,occupation])
    # d[education]                                                                 
    #       
    # Estimand assumption 1, Unconfoundedness: If U→{education} and U→salary then P(salary|education,workclass,marital-status,native-country,relationship,occupation,U) = P(salary|education,workclass,marital-status,native-country,relationship,occupation)
    #
    ## Realized estimand
    # b: salary~education+workclass+marital-status+native-country+relationship+occupation
    # Target units: ate
    #
    ## Estimate
    # Mean value: 0.4697157228651772
    # Step 4: Refute the obtained estimate using multiple robustness checks.
    refute_results = model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")

def test_causalimpact():
    # Import libraries
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.arima_process import arma_generate_sample
    import matplotlib.pyplot as plt
    from causalimpact import CausalImpact
    
    # Generate samples
    np.random.seed(42)  # for reproducibility
    x1 = arma_generate_sample(ar=[1, -0.999], ma=[1, 0.9], nsample=100) + 100
    y = 1.2 * x1 + np.random.randn(100)
    y[71:] = y[71:] + 10  # introduce intervention
    
    # Create time-indexed DataFrame
    data = pd.DataFrame({"y": y, "x1": x1}, index=pd.date_range("2020-01-01", periods=100))
    
    # Define pre- and post-period
    pre_period = [data.index[0], data.index[70]]
    post_period = [data.index[71], data.index[-1]]
    
    # Run CausalImpact
    impact = CausalImpact(data, pre_period, post_period)
    
    # Create inferences
    impact.run()
    
    # Plot
    impact.plot()
    
    # Results
    impact.summary()
    #                              Average    Cumulative
    # Actual                           130          3773
    # Predicted                        120          3501
    # 95% CI                    [118, 122]  [3447, 3555]
                                                      
    # Absolute Effect                    9           272
    # 95% CI                       [11, 7]    [326, 218]
                                                      
    # Relative Effect                 7.8%          7.8%
    # 95% CI                  [9.3%, 6.2%]  [9.3%, 6.2%]
                                                      
    # P-value                         0.0%              
    # Prob. of Causal Effect        100.0%
    
    # Summary report
    impact.summary(output="report")
