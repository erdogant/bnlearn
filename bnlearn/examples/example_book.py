# %%
# =============================================================================
# CHAPTER 2
# =============================================================================
# Load library
import bnlearn as bn
# Fetch model
model = bn.fetch('sprinkler')
print(list(model.keys()))
# ['model', 'adjmat']

# Fetch model
truth = bn.fetch('sprinkler')
df = bn.sampling(truth, n=1000)
print(df.head())

G = bn.plot(truth)

model = bn.structure_learning.fit(df)
print(model['adjmat'])

scores, adjmat_diff = bn.compare(truth, model, showfig=False)
print(scores)
# [[11  1]
#  [ 1  3]]
model = bn.independence_test(model, df, prune=False)
print(model['independence_test'])

model = bn.parameter_learning.fit(model, df, methodtype='maximumlikelihood')
CPDs = bn.print_CPD(model)

q = bn.inference.fit(model, variables=['Wet_Grass'], evidence={'Rain': 1, 'Sprinkler': 0})
print(q.df)

pred = bn.predict(model, df.head(5).drop(columns=['Wet_Grass']), variables=['Wet_Grass'])
print(pred)

df_new = bn.sampling(model, n=5, evidence={'Rain': 1})
print(df_new)

G = bn.plot(truth)
bn.plot(model, pos=G['pos'])
dot = bn.plot(model, output='graphviz')
a, b = bn.compare(truth, model, side_by_side=True, edge_label='auto', figsize=(15,8))

# %%

