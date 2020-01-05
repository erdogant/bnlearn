""" A= ismember(list1, list2)

 INPUT:
   list1:           numpy array
   list2:           numpy array

 OUTPUT
	output

 DESCRIPTION
   MATLAB equivalent ismember function
   [LIA,LOCB] = ISMEMBER(A,B) also returns an array LOCB containing the
   lowest absolute index in B for each element in A which is a member of
   B and 0 if there is no such index.
   https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function

 EXAMPLE
    %reset -f
    import sys, os, importlib
    sys.path.append('D://Dropbox/ETA/toolbox_PY/general/')
    print(os.getcwd())
    import ismember as ETA
    importlib.reload(ETA)
    import numpy as np
    import pandas as pd

    a_vec  = pd.DataFrame([1,2,3,None])
    b_vec  = pd.DataFrame([None,1,2])
    [I,idx] = ETA.ismember(a_vec,b_vec)
    a_vec.values[I]
    b_vec.values[idx]

    a_vec   = pd.DataFrame(['aap','None','mies','aap','boom','mies',None,'mies','mies','pies',None])
    b_vec   = pd.DataFrame([None,'mies','mies','pies',None])
    [I,idx] = ETA.ismember(a_vec,b_vec)
    a_vec.values[I]
    b_vec.values[idx]

    a_vec   = np.array([1,2,3,None])
    b_vec   = np.array([1,2,4])
    [I,idx] = ETA.ismember(a_vec,b_vec)
    a_vec[I]
    b_vec[idx]

    a_vec   = np.array(['boom','aap','mies','aap'])
    b_vec   = np.array(['aap','boom','aap'])
    [I,idx] = ETA.ismember(a_vec,b_vec)
    a_vec[I]
    b_vec[idx]

    [I,idx] = ETA.ismember(b_vec,a_vec)
    b_vec[I]
    a_vec[idx]

"""
#print(__doc__)

#--------------------------------------------------------------------------
# Name        : ismember.py
# Version     : 1.0
# Author      : E.Taskesen
# Contact     : erdogan@gmail.com
# Date        : Oct. 2017
#--------------------------------------------------------------------------

#%%
def ismember(a_vec, b_vec):
	#%% DECLARATIONS
    I   = []
    idx = []

    #%% Libraries
    import numpy as np
#    from numpy import nan

    #%% Check type
    if 'pandas' in str(type(a_vec)):
         a_vec.values[np.where(a_vec.values==None)]='NaN'
         a_vec = np.array(a_vec.values)
     #end
    if 'pandas' in str(type(b_vec)):
         b_vec.values[np.where(b_vec.values==None)]='NaN'
         b_vec = np.array(b_vec.values)
     #end

    #%% 
    bool_ind = np.isin(a_vec,b_vec)
    common = a_vec[bool_ind]
    [common_unique, common_inv]  = np.unique(common, return_inverse=True)     # common = common_unique[common_inv]
    [b_unique, b_ind] = np.unique(b_vec, return_index=True)  # b_unique = b_vec[b_ind]
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]

    I=bool_ind
    idx=common_ind[common_inv]

    #%%
    return(I,idx)
    
    #%% END