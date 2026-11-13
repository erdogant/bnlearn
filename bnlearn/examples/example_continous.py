import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate 1,000 continuous observations
np.random.seed(42)
n = 1000
temperature = np.random.uniform(10, 40, n)

# Pressure depends continuously on temperature, with noise
pressure = 2 + 0.8 * temperature + np.random.normal(0, 3, n)
# Create dataframe
df = pd.DataFrame({"Temperature": temperature, "Pressure": pressure})
# Creat bins
bins = [-np.inf, 20, 30, np.inf]
# Set labels
labels = ["Low", "Medium", "High"]
# Discritize
df["Temperature_state"] = pd.cut(df["Temperature"], bins=bins, labels=labels, right=False)


# Create three panels
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
# A. Original continuous data
axes[0].scatter(df["Temperature"], df["Pressure"], alpha=0.45, s=22)
axes[0].set_title("A. Original data")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Pressure (bar)")
axes[0].set_xlim(10, 40)

# B. After discretization
x_categories = pd.Categorical(df["Temperature_state"], categories=labels, ordered=True).codes
# Add small horizontal jitter so all 1,000 observations remain visible
rng = np.random.default_rng(42)
x_jitter = x_categories + rng.uniform(-0.13, 0.13, n)
axes[1].scatter(x_jitter, df["Pressure"], alpha=0.45, s=22)
axes[1].set_title("B. After discretization")
axes[1].set_xlabel("Temperature state")
axes[1].set_ylabel("Pressure (bar)")
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(labels)
axes[1].set_xlim(-0.5, 2.5)

# C. What was lost: original values within each category
for i, label in enumerate(labels):
    values = df.loc[df["Temperature_state"] == label, "Temperature"]
    y = rng.uniform(-0.08, 0.08, len(values))
    axes[2].scatter(np.full(len(values), i) + y, values, alpha=0.35, s=20)

axes[2].set_title("C. What was lost")
axes[2].set_xlabel("Discretized temperature")
axes[2].set_ylabel("Original temperature (°C)")
axes[2].set_xticks(range(3))
axes[2].set_xticklabels(labels)
axes[2].set_xlim(-0.5, 2.5)
axes[2].axhline(20, linestyle="--", linewidth=1)
axes[2].axhline(30, linestyle="--", linewidth=1)

# Highlight representative observations
examples = [20.1, 24.7, 29.9]
for value in examples:
    axes[2].annotate(f"{value:.1f}°C", xy=(1, value), xytext=(1.22, value), arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9, va="center")

fig.suptitle("Discretization Changes the Representation of Continuous Data", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

# Show a compact check of the three example values
examples_df = pd.DataFrame({"Temperature (°C)": examples})
examples_df["Discretized state"] = pd.cut(examples_df["Temperature (°C)"], bins=bins, labels=labels, right=False)

print(examples_df.to_string(index=False))



# %%
import numpy as np
import pandas as pd
from lingam.utils import make_dot

# Number of samples
n = 10000

# Step 1: Initialize root node
x3 = np.random.uniform(size=n)

# Step 2: Create dependent variables
x0 = 3.0 * x3 + np.random.uniform(size=n)
x2 = 6.0 * x3 + np.random.uniform(size=n)

# Step 3: Create further dependencies
x5 = 4.0 * x0 + np.random.uniform(size=n)

# Step 4: Create final dependencies
x1 = 3.0 * x0 + 2.0 * x2 + np.random.uniform(size=n)
x4 = 8.0 * x0 - 1.0 * x2 + np.random.uniform(size=n)

# Create DataFrame
df = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T,
                 columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
df.head()

# Define adjacency matrix
m = np.array([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
             [3.0, 0.0, 2.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [8.0, 0.0,-1.0, 0.0, 0.0, 0.0],
             [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
dot = make_dot(m)
# dot


# %%


import bnlearn as bn

# Fit the model
# model = bn.structure_learning.fit(df, methodtype='direct-lingam')
# model = bn.structure_learning.fit(df, methodtype='ica-lingam')
model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic-g')
# model = bn.structure_learning.fit(df, methodtype='hc', scoretype='aic-g')
# model = bn.structure_learning.fit(df, methodtype='hc', scoretype='loglik-g')


# Examine the output to see how well the dependency values are recovered
print(model['adjmat'])
# target        x0        x1       x2   x3        x4       x5
# source
# x0      0.000000  2.987320  0.00000  0.0  8.057757  3.99624
# x1      0.000000  0.000000  0.00000  0.0  0.000000  0.00000
# x2      0.000000  2.010043  0.00000  0.0 -0.915306  0.00000
# x3      2.971198  0.000000  5.98564  0.0 -0.704964  0.00000
# x4      0.000000  0.000000  0.00000  0.0  0.000000  0.00000
# x5      0.000000  0.000000  0.00000  0.0  0.000000  0.00000

# Visualize the model
bn.plot(model, output='graphviz')

