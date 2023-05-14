# read data and set as a matrix
f = readcsv("auto_mpg.csv")

data = Array(Any,392,8)

discrete_index = [2,7,8]
continuous_index = [1,3,4,5,6]

for i = 1 : 392
    for j = 1 : 8
        if j in discrete_index
            data[i,j] = round(Int64,f[i,j])
        else
            data[i,j] = f[i,j]
        end
    end
end

##### Case 1 ######
##### Network Structure is known in advance #####

graph = [2,(2,3),(3,5),(5,1),(1,5,3,7),(3,4),(4,6),8]
# the given network structure
# Each element in the graph set (p1,p2,...pN,x) means that p1,...pN are parent variables to x
u_cycle = 8
# Number of cycles for discretization process


e1 = Learn_Discrete_BayesNet.Discretize_All(data,graph,continuous_index,u_cycle)

@test e1 == Any[
               [9.0,15.25,17.65,20.9,25.65,28.9,46.6],
               [68.0,70.5,93.5,109.0,159.5,259.0,284.5,455.0],
               [46.0,71.5,99.0,127.0,230.0],
               [1613.0,2115.0,2480.5,2959.5,3657.5,5140.0],
               [8.0,12.35,13.75,16.05,22.85,24.8]
              ]
# e1[i] is the discretization on ith continuous variable

##### Case 2 #####
##### Network structure is not known in advance #####
##### A topological order of variables is given #####

order = [2,3,5,1,7,4,6,8]
# the given topological order
u_parent = 2
# Limit of number of parent variable
u_cycle = 5
# Number of cycles for discretization process
k2_times = 1
# Learn discrete Bayesian net only one K2 procedure

e2 =  Learn_Discrete_BayesNet.Learn_DVBN(data,continuous_index,order,u_parent,u_cycle)

@test e2[1] == [2,(2,3),(3,2,5),(5,2,1),(1,2,7),(3,2,4),(4,5,6),(3,2,8)]
#test learned_graph
@test e2[2] == Any[
                   [9.0,17.25,20.9,26.2,46.6],
                   [68.0,93.5,106.0,134.5,159.5,169.5,259.0,284.5,455.0],
                   [46.0,71.5,115.5,127.0,230.0],
                   [1613.0,2217.0,2959.5,3657.5,5140.0],
                   [8.0,13.45,16.05,22.85,24.15,24.8]
                  ]
#test learned discretization policy on each continuous variable

###### Case 3 #####
##### Network structure is not known in advance #####
times = 5
# run K2 along with the Bayesian discretization method 5 times to learn a discrete Bayesian network
# Each time with a random topological order of variables
e3 = Learn_Discrete_BayesNet.Learn_Discrete_Bayesian_Net(data,continuous_index,u_parent,u_cycle,times)

@test e3 != 0
