import pandas as pd
from juliacall import Main as jl, convert as jlconver

jl.println("Hello from Julia!")

import julia
from julia.api import Julia
# jl = Julia(compiled_modules=False)

Main.data = data
Main.token_data = jl.eval("tokens_data = TokenDocument(data) ; return(tokens_data)")


# julia.install()               # install PyCall.jl etc.
# 
# from julia import Base        # short demo

# pip install julia
# from julia.api import Julia

# There are few methods to embed your Julia code into Python
# 1. Run whole expressions of Julia
# 2. Run Julia with “magic” command
# 3. Run Julia with a script
    
df = pd.read_csv('auto_mpg.csv', header=None)


x = jl.rand(range(10), 3, 5)

jl.eval("readcsv")

jl.eval('include("L_DVBN.jl")')
A = jl.eval('data = readcsv("auto_mpg.csv") ; return(data)')
A = jl.eval('f = readcsv("auto_mpg.csv")')

A = jl.eval(Array(Any,392,8))

# read data and set as a matrix



Main.data = data
stem_list = jl.eval("stemming_document(data)")
