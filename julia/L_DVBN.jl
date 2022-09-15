function combine_spouses_data(spouse_value_matrix)
        n_ins = length(spouse_value_matrix[:,1])

        combined_data = Array(Any,n_ins)
        for i = 1 : n_ins
                combined_data[i] = tuple(spouse_value_matrix[i,:]...)
        end

        return combined_data
end

function parent_class_combine(data_matrix,parent_set)
        n_parent = length(parent_set)
        N = length(data_matrix[:,1])

        parent_data_set = Array(Int64,N,n_parent)

        for i = 1 : n_parent
                parent_data_set[:,i] = data_matrix[:,parent_set[i]]
        end

        parent_data_set = combine_spouses_data(parent_data_set)

        class_combine = length(unique(parent_data_set))
        return class_combine
end

function log_prob_single_edge_last_term(class)
        n = length(class)
        uniq_cl = unique(class)
        val_cl = length(uniq_cl)

        distr_table = Array(Array,n)

        # Survey distribution for each data point

        for index_data = 1 : n
                class_cl = findfirst(uniq_cl,class[index_data])
                z = zeros(Int64,val_cl)
                z_modify = zeros(Int64,val_cl)
                z_modify[class_cl] = 1
                distr_table[index_data] = z_modify
        end

        # Survey class distribution between point i and point j

        distr_intval_table = Array(Array,n,n)

        for ind_init = 1 : n
                for ind_end = ind_init : n
                        if ind_init == ind_end
                                current_distr = distr_table[ind_init]
                        else
                                current_distr = distr_intval_table[ind_init,ind_end-1] +
                                                distr_table[ind_end]
                        end
                        distr_intval_table[ind_init,ind_end] = current_distr
                end
        end
        # println(distr_intval_table[1,n])
        # Get 1/P(D|M) for single edge case and count only last term in objective func

        inv_p = Array(Float64,n,n)

        for ind_init = 1 : n
                for ind_end = 1 : n
                        if ind_end < ind_init
                                inv_p[ind_init,ind_end] = Inf
                        else
                                distr_in_intval = distr_intval_table[ind_init,ind_end]
                                current = lfact(ind_end-ind_init+1)
                                for ind_cl = 1 : val_cl
                                        current -= lfact(distr_in_intval[ind_cl])
                                end

                                inv_p[ind_init,ind_end] = current
                        end
                end
        end

        return inv_p
end

function log_prob_spouse_child_data(child,spouse)
        n = length(child)
        uniq_sp = unique(spouse)
        uniq_ch = unique(child)
        val_sp = length(uniq_sp)
        val_ch = length(uniq_ch)

        distr_table = Array(Array,n,val_sp)

        # Survey distribution for each data point

        for index_data = 1 : n
                class_ch = findfirst(uniq_ch,child[index_data])
                class_sp = findfirst(uniq_sp,spouse[index_data])
                z = zeros(Int64,val_ch)
                z_modify = zeros(Int64,val_ch)
                z_modify[class_ch] = 1

                for class_sp_index = 1 : val_sp
                        if class_sp_index == class_sp
                                distr_table[index_data,class_sp_index] = z_modify
                        else
                                distr_table[index_data,class_sp_index] = z
                        end
                end
        end

        # Survey class distribution in between point i and point j

        distr_intval_table = Array(Array,n,n,val_sp)

        for ind_val_sp = 1 : val_sp
                for ind_init = 1 : n
                for ind_end = ind_init : n
                        if ind_init == ind_end
                                current_distr = distr_table[ind_end,ind_val_sp]
                        else
                                current_distr = distr_intval_table[ind_init,ind_end-1,ind_val_sp] +
                                                distr_table[ind_end,ind_val_sp]
                        end
                        #println(current_distr)
                        distr_intval_table[ind_init,ind_end,ind_val_sp] = current_distr
                end
                end
        end

        # Get the 1/P(D|M) for this child-spouse set

        inv_p = Array(Float64,n,n)

        for ind_init = 1 : n
        for ind_end = 1 : n
                if ind_end < ind_init
                        inv_p[ind_init,ind_end] = Inf
                else
                        current_val = 0.0
                        for ind_val_sp = 1 : val_sp
                                distr_in_intval = distr_intval_table[ind_init,ind_end,ind_val_sp]
                                total_num = sum(distr_in_intval)

                                # The forth term in objective function
                                current = lfact(total_num)
                                for ind_val_ch = 1 : val_ch
                                        current -= lfact(distr_in_intval[ind_val_ch])
                                end

                                # The third term in objective function
                                current += lfact(total_num + val_ch -1) - lfact(val_ch-1) - lfact(total_num)

                                # The 3rd and 4th terms for this value of spouse
                                current_val += current
                        end
                        inv_p[ind_init,ind_end] = current_val
                end
        end
        end



        return inv_p
end

function prior_of_intval(continuous,lambda)
        N = length(continuous)
        d_1_N = continuous[N] - continuous[1]
        prior = Array(Float64,N)
        for i = 1 : N-1
                d_i = continuous[i+1] - continuous[i]
                prior[i] = 1 - exp(-lambda * d_i / d_1_N)
        end
        prior[N] = 1

        return prior
end

function largest_class_value(data_matrix)
        n = length(data_matrix[1,:])
        largest = 0
        for i = 1 : n
                class_value = length(unique(data_matrix[:,i]))
                if class_value > largest
                        largest = class_value
                end
        end

        return largest
end

function BN_discretizer_p_data_model(data_matrix,parent_set,child_spouse_set,approx = true)

        N = length(data_matrix[:,1])
        n = length(data_matrix[1,:])
        n_p = length(parent_set)
        n_c = length(child_spouse_set)
        n_r = n_p + n_c

        parent_class_number = 1
        if length(parent_set) > 0
                for i = 1 : n_p
                        multi = length(unique(data_matrix[:,parent_set[i]]))
                        parent_class_number = parent_class_number * multi
                end

                parent_class_number_2 = parent_class_combine(data_matrix,parent_set)
        end

        # -log(P(D|M)) part:

        log_P_data_model = zeros(Float64,N,N)
        nearest_var_set = [parent_set;child_spouse_set]
        if approx == true
        for ind_parent = 1 : n_p
                table = 0
                single_variable_data = data_matrix[:,parent_set[ind_parent]]
                table = log_prob_single_edge_last_term(single_variable_data)
                log_P_data_model += table
        end

        else
                if n_p > 0
                table = 0
                parent_data = Array(Int64,N,n_p)
                for p = 1 : n_p
                        parent_data[:,p] = data_matrix[:,parent_set[p]]
                end
                combined_parent = combine_spouses_data(parent_data)
                table = log_prob_single_edge_last_term(combined_parent)
                log_P_data_model += table
                end
        end

        for ind_child = 1 : n_c
                table = 0
                if length(child_spouse_set[ind_child]) == 1
                        child_data = data_matrix[:,child_spouse_set[ind_child]]
                        spouse_data = zeros(Int64,N)
                        table = log_prob_spouse_child_data(child_data,spouse_data)

                elseif length(child_spouse_set[ind_child]) == 2
                        (child,spouse) = child_spouse_set[ind_child]
                        child_data = data_matrix[:,child]
                        spouse_data = data_matrix[:,spouse]
                        table = log_prob_spouse_child_data(child_data,spouse_data)
                else
                        # A child has more than 2 parents
                        child = child_spouse_set[ind_child][1]
                        child_data = data_matrix[:,child]
                        spouse_set = child_spouse_set[ind_child][2:end]
                        spouse_matrix = Array(Int64,N,length(spouse_set))
                        for spouse_index = 1 : length(spouse_set)
                                spouse_matrix[:,spouse_index] = data_matrix[:,spouse_set[spouse_index]]
                        end
                        spouse_data = combine_spouses_data(spouse_matrix)
                        table = log_prob_spouse_child_data(child_data,spouse_data)
                end

                log_P_data_model += table
        end

        # Add -log(p_class_distribution_for_parent)


        for i = 1 : N
                for j = i : N
                        log_P_data_model[i,j] += lfact(j - i + parent_class_number) -
                                                 lfact(j-i+1) - lfact(parent_class_number-1)
                end
        end


        return log_P_data_model

end

function BN_discretizer_free_number_rep(continuous,data_matrix,parent_set,child_spouse_set,approx = true)
        p_data_model = BN_discretizer_p_data_model(data_matrix,parent_set,child_spouse_set,approx)
        #lambda = div(largest_class_value(data_matrix),2)
        lambda = largest_class_value(data_matrix)
        split_on_intval = prior_of_intval(continuous,lambda)
        N = length(continuous)
        not_split_on_intval = ones(Float64,N) - split_on_intval

        conti_norep = unique(continuous)
        conti_head = Array(Int64,length(conti_norep))
        conti_tail = Array(Int64,length(conti_norep))
        N_norep = length(conti_norep)


        conti_head[1] = 1
        index_head = 1
        for i = 2 : N
                if continuous[i] != continuous[i-1]
                        index_head += 1
                        conti_head[index_head] = i
                end
        end

        conti_tail[end] = N
        index_tail = length(conti_norep)
        for i = N-1 : -1 : 1
                if continuous[i] != continuous[i+1]
                        index_tail -= 1
                        conti_tail[index_tail] = i
                end
        end

        #println(conti_tail==conti_head)
        smallest_value = Array(Float64,N_norep)
        optimal_disc = Array(Array,N_norep)
        length_conti_data = continuous[N] - continuous[1]

        for a = 1 : N_norep
                if a == 1
                        smallest_value[1] = -log(split_on_intval[conti_tail[1]]) +
                                                        p_data_model[conti_head[1],conti_tail[1]]
                        optimal_disc[1] = [conti_tail[1]]
                else
                        current_value = Inf
                        current_disc = [0]
                        for b = 1 : a
                                if b == a
                                        value =
                                        ( ( (continuous[conti_tail[a]] - continuous[1])/
                                                length_conti_data)*lambda) -
                                        log(split_on_intval[conti_tail[a]]) +
                                        p_data_model[1,conti_tail[a]]
                                else

                                        value = smallest_value[b] +
                                        ( ( (continuous[conti_tail[a]] - continuous[conti_head[b+1]])/
                                        length_conti_data)*lambda) -
                                        log(split_on_intval[conti_tail[a]]) +
                                        p_data_model[conti_head[b+1],conti_tail[a]]
                                end

                                if value < current_value
                                        current_value = value
                                        if b == a
                                                current_disc = [conti_tail[a]]
                                        else
                                                current_disc = [optimal_disc[b];conti_tail[a]]
                                        end
                                end
                        end
                        smallest_value[a] = current_value
                        optimal_disc[a] = current_disc
                end
        end

        #println(optimal_disc[end])
        optimal_disc_value = Array(Float64,length(optimal_disc[end])+1)

        for i = 0 : length(optimal_disc[end])
                if i == 0
                        optimal_disc_value[i+1] = continuous[1]
                elseif i == length(optimal_disc[end])
                        optimal_disc_value[i+1] = continuous[end]
                else
                        optimal_disc_value[i+1] = 0.5 * (continuous[optimal_disc[end][i]]
                                                      +continuous[optimal_disc[end][i]+1])
                end
        end
        #println(smallest_value[end])
        return optimal_disc_value

end

# Equal width discretization
function equal_width_disc(continuous,m)

        N = length(continuous)
        min_value = minimum(continuous)
        max_value = maximum(continuous)
        span = (max_value - min_value)/m
        class_list = Array(Int64,N)

        for i = 1 : N
                class = div((continuous[i] - min_value),span)
                if class >= m
                        class -= 1
                end
                class_list[i] = class + 1
        end
        return class_list
end

function equal_width_edge(continuous,m)
        min_value = continuous[1]
        max_value = continuous[end]
        span = max_value - min_value
        edge = [continuous[1]]
        for i = 1 : m
                edge = [edge, min_value + (span/m)*i]
        end
        return edge
end

#######################################################################################
#######################################################################################

function cartesian_product(product_set)
        N = length(product_set)
        M = 1
        for i = 1 : N
                M *= length(product_set[i])
        end

        product_returned = [product_set[1]...]

        for class_index = 2 : N
                class_number = length(product_set[class_index])
                current_length = length(product_returned[:,1])

                enlarged_matrix = Array(Int64,current_length*class_number,class_index)

                if class_number == 1
                        enlarged_matrix[:,1:class_index-1] = product_returned
                        for i = 1 : current_length
                                enlarged_matrix[i,class_index] = product_set[class_index][1]
                        end
                else

                        for enlarge_times = 1 : class_number
                                enlarged_matrix[(enlarge_times-1)*current_length+1:enlarge_times*current_length,1:class_index-1] =
                                product_returned
                        end

                        for i = 1 : class_number * current_length
                                item_index = div(i-1,current_length) + 1
                                enlarged_matrix[i,class_index] = product_set[class_index][item_index]
                        end
                end
                product_returned = enlarged_matrix

        end

        return product_returned
end

function find_intval(x,disc)
    if x < disc[2]
        return 1
    end
    if x >= disc[end-1]
        return length(disc)-1
    end

    for i = 2 : length(disc)-2
           if (x >= disc[i])&(x < disc[i+1])
                   return i
           end
    end
end

function continuous_to_discrete(data,bin_edge_1)
        #bin_edge_extend
        bin_edge = copy(bin_edge_1)
        bin_edge[1] = -Inf
        bin_edge[end] = Inf
        data_discrete = Array(Int64,length(data))
        for i = 1 : length(data)
                index = 0
                for j = 2 : length(bin_edge)
                        if (data[i] > bin_edge[j-1])&(data[i] <= bin_edge[j])
                                index = j-1
                        end
                end
                data_discrete[i] = index
        end
        return data_discrete
end

function sort_disc_by_vorder(continuous_order,disc_edge)
      reorder_disc_edge = Array(Any,length(continuous_order))
      for i = 1 : length(continuous_order)
            num_less = 1
            for j = 1 : length(continuous_order)
                  if continuous_order[j] < continuous_order[i]
                        num_less += 1
                  end
            end
            reorder_disc_edge[num_less] = disc_edge[i]
    end
    return reorder_disc_edge
end

function rand_seq(N)

        seq = Array(Int64,N)

        i = 1
        seq[1] = 1 + round(Int64, div(N * rand(),1) )

        while i < N
                number = 1 + round(Int64,div( N*rand(),1))
                if ~(number in seq[1:i])
                        i += 1
                        seq[i] = number
                end
        end
        return seq
end

#######################################################################################
#######################################################################################


function disc_intval_seq(table)
        N = length(table)
        seq = Array(Int64,N)
        for i = 1 : N
                seq[i] = length(table[i])
        end
        return seq
end

function graph_to_reverse_order(graph)
        order = Array(Int64,length(graph))
        for i = 1 : length(graph)
                order[length(graph) - i + 1] = graph[i][end]
        end
        return order
end

function graph_to_reverse_conti_order(graph,continuous_index)
        order = graph_to_reverse_order(graph)
        Order = Array(Int64,length(continuous_index))
        ind = 0
        for i = 1 : length(order)
                if order[i] in continuous_index
                        ind += 1
                        Order[ind] = order[i]
                end
        end
        return Order
end

function graph_to_markov(graph,target)
        parent_set = []; child_spouse_set = [];
        for i = 1 : length(graph)

                condi_graph = graph[i]
                if length(condi_graph) == 1
                        if target == condi_graph
                                parent_set = []
                        end

                elseif target in condi_graph

                        index = findfirst(condi_graph,target)
                        if index == length(condi_graph)
                                parent_set = [condi_graph[1:end-1]...]
                        else

                                child = condi_graph[end]
                                spouse = []

                                for j = 1 : length(condi_graph)-1
                                        if j != index
                                                spouse = [spouse,condi_graph[j]]
                                        end
                                end
                                child_spouse_set = [child_spouse_set,tuple([child,spouse]...)] ###### change to ; ######

                        end
                else

                end
        end

        for i = 1 : length(child_spouse_set)
                if length(child_spouse_set[i]) == 1
                        child_spouse_set = [child_spouse_set[1:i-1],child_spouse_set[i][1],child_spouse_set[i+1:end]]
                end
        end

        return (parent_set,child_spouse_set)
end

function one_iteration(data,data_integer,graph,discrete_index,continuous_index,lcard,approx = true)


        # Save discretization edge
        disc_edge_collect = Array(Any,length(continuous_index))

        for i = 1 : length(continuous_index)
                target = continuous_index[i]

                increase_order = sortperm(data[:,target])
                conti = data[:,target][increase_order]
                data_integer_sort = Array(Int64,size(data))

                # sort data_integer properly
                for j = 1 : length(data_integer[1,:])
                        data_integer_sort[:,j] = data_integer[:,j][increase_order]
                end
                sets = graph_to_markov(graph,target)
                parent_set = sets[1]; child_spouse_set = sets[2];

                if (length(sets[1]) + length(sets[2])) == 0
                        disc_edge = equal_width_edge(conti,lcard)
                        disc_edge_collect[i] = disc_edge
                else
                        disc_edge = BN_discretizer_free_number_rep(conti,data_integer_sort,parent_set,child_spouse_set,approx)
                        disc_edge_collect[i] = disc_edge
                end

                # Update by current discretization
                data_integer[:,target] = continuous_to_discrete(data[:,target],disc_edge)
        end


        return (data_integer,disc_edge_collect)
end

function BN_discretizer_iteration_converge(data,graph,discrete_index,continuous_index,cut_time,approx = true)
        # intital the first data_integer
        l_card = 0
        for i = 1 : length(discrete_index)
                card = length(unique(data[:,discrete_index[i]]))
                if card > l_card
                        l_card = card
                end
        end

        # pre-equal-width-discretize on continuous variables
        data_integer = zeros(Int64,size(data))
        for i = 1 : length(discrete_index)
                index = discrete_index[i]
                data_integer[:,index] = data[:,index]
        end

        for i = 1 : length(continuous_index)
                index = continuous_index[i]
                data_integer[:,index] = equal_width_disc(data[:,index],l_card)
        end

        disc_edge_collect = Array(Any,length(continuous_index))

        disc_edge_previous = Array(Any,length(continuous_index))

        for i = 1 : length(continuous_index)
                disc_edge_previous[i] = []
                disc_edge_collect[i] = [0]
        end

        times = 0
        while (disc_edge_previous != disc_edge_collect)&(times<cut_time)

            times += 1
            #println(("iteration times = ",times))
            disc_edge_previous = disc_edge_collect

            X = one_iteration(data,data_integer,graph,discrete_index,continuous_index,l_card,approx)
            data_integer = X[1]
            disc_edge_collect = X[2]

        end

        return (data_integer,disc_edge_collect)
end


function K2_f(x,parent)
        N = length(x)
        n = length(parent[1,:])

        # build carti product
        carti_set = [tuple(unique(parent[:,1])...)]

        for i = 2 : n
                carti_set = [carti_set;tuple(unique(parent[:,i])...)]
        end

        uni_x = unique(x); r = length(uni_x)
        carti_set = [carti_set;tuple(uni_x...)]

        CP = cartesian_product(carti_set)

        r_p = length(CP[:,1]) / r
        r_p = round(Int64,r_p)

        # Survey distribution
        count = zeros(Int64,length(CP[:,1]))

        for i = 1 : N
                reform_data = [parent[i,:] x[i]]
                for j = 1 : length(CP[:,1])
                        if reform_data == CP[j,:]
                                count[j] += 1
                        end
                end
        end

        # Sum over same parent value
        count_p = zeros(Int64,r_p)
        for i = 1 : r_p
                count_pp = 0
                for j = 1 : r
                        index = i + r_p * (j-1)
                        count_pp += count[index]
                end
                count_p[i] = count_pp
        end

        # Evaluate
        f = 0
        for i = 1 : r_p
                f += lfact(count_p[i]+r-1) - lfact(r-1)
        end
        for i = 1 : length(CP[:,1])
                f -= lfact(count[i])
        end


        return f
end

function parent_table_to_graph(parent_table)
        m = length(parent_table)
        # Parent_table -> graph
        graph = []
        for i = 1 : m
                child = i
                if length(parent_table[i]) == 0
                        graph = [graph;child]
                else
                parent = []
                for j = 1 : length(parent_table[i])
                        parent = [parent;parent_table[i][j]]
                end

                add = [parent,child]
                graph = [graph;tuple(add...)]
                end
        end
        return graph
end

function K2_one_iteration_discretization(order,u,data_matrix,continuous_index,cut_time,approx = true)
        N = length(data_matrix[:,1])
        m = length(data_matrix[1,:]) # number of variables
        mc = length(continuous_index)

        # continuous/discrete index in the new order
        conti_index = Array(Int64,mc)
        disc_index = Array(Int64,m-mc)
        for i = 1 : mc
                conti_index[i] = findfirst(order,continuous_index[i])
        end
        conti_index = sort(conti_index, rev=true)

        index_disc = 0
        for i = 1 : m
                if ~(i in continuous_index)
                        index_disc += 1
                        disc_index[index_disc] = findfirst(order,i)
                end
        end

        # data that is sorted by order
        data = Array(Any,size(data_matrix))
        for i = 1 : m
                data[:,i] = data_matrix[:,order[i]]
                #println(i)
        end

        # Initialize graph structure
        initial_lcard= 0
        data_discretized = Array(Int64,size(data_matrix))
        if (m - length(conti_index)) == 0
            initial_lcard = round(Int64, log(m)+1)
        else
            for i = 1 : m
                    if ~(i in conti_index)
                            data_discretized[:,i] = data[:,i]
                            current_initial_lcard = length(unique(data[:,i]))

                            if current_initial_lcard > initial_lcard
                                    initial_lcard = current_initial_lcard
                            end
                    end
            end
        end
        for i = 1 : m
                if i in conti_index
                        data_discretized[:,i] = equal_width_disc(data[:,i],initial_lcard)
                end
        end

        # Initialize graph structure
        parent_table = Array(Any,m)
        for i = 1 : m
                parent_table[i] = [] # No parent in the begining
        end
        # store discretization edges
        disc_edge = 0

        # Iterate for all nodes
        for i = 1 : m
                p_old = K2_f(data_discretized[:,i],zeros(Int64,N))
                OKToProceed = true
                current_parent = []

                while OKToProceed & (length(parent_table[i]) < u)
                        # find node z for iteration
                        iteration_list = []
                        for j = 1 : i-1
                                if ~(j in current_parent)
                                        iteration_list = [iteration_list;j]
                                end
                        end

                        # iteration to find most probable
                        current_value = Inf
                        current_best_parent = []


                        for j = 1 : length(iteration_list)

                                iteration_parent = [current_parent;iteration_list[j]]
                                iteration_parent_data = data_discretized[:,iteration_parent[1]]
                                #if i ==4 ; println(iteration_parent); end;

                                for k = 2 : length(iteration_parent)
                                        iteration_parent_data = [iteration_parent_data data_discretized[:,iteration_parent[k]]]
                                end
                                this_iteration_value = K2_f(data_discretized[:,i],iteration_parent_data)


                                if this_iteration_value < current_value
                                        current_value = this_iteration_value
                                        current_best_parent = iteration_parent
                                end
                        end


                        if current_value < p_old
                                p_old = current_value
                                current_parent = current_best_parent
                                parent_table[i] = current_parent
                                ###### Rediscretize ######
                                #println((i,"th varaible"))
                                current_graph = parent_table_to_graph(parent_table)
                                disc_result = BN_discretizer_iteration_converge(data,current_graph,
                                                                               disc_index,conti_index,cut_time,approx)

                                ##### Replace data_discretized #####

                                data_discretized = disc_result[1]
                                disc_edge = disc_result[2]

                        else

                                OKToProceed = false

                        end

                end
                #println(("parent of",i,"is",parent_table[i]))
        end

        score = 0
        # Calculate score
        for i = 1 : m
                if length(parent_table[i]) == 0
                        score += K2_f(data_discretized[:,i],zeros(Int64,N))
                else
                        parent_data_final = Array(Int64,N,length(parent_table[i]))
                        for j = 1 : length(parent_table[i])
                                parent_data_final[:,j] = data_discretized[:,parent_table[i][j]]
                        end
                        score += K2_f(data_discretized[:,i],parent_data_final)
                end
        end

        # Parent_table -> graph
        graph = []
        for i = 1 : m
                child = order[i]
                if length(parent_table[i]) == 0
                        graph = [graph,child]
                else
                parent = []
                for j = 1 : length(parent_table[i])
                        parent = [parent,order[parent_table[i][j]]]
                end

                add = [parent,child]
                graph = [graph,tuple(add...)]
                end
        end

        # Disc_edge is modified to correct order
        conti_index_correspond = Array(Any,mc,3)
        for i = 1 : mc
              conti_index_correspond[i,2] = continuous_index[i]
              conti_index_correspond[i,1] = findfirst(order,continuous_index[i])
        end

        zz = sortperm(conti_index_correspond[:,1], rev = true)
        conti_index_correspond[:,1] = conti_index_correspond[:,1][zz]
        conti_index_correspond[:,2] = conti_index_correspond[:,2][zz]

        for i = 1 : mc
              conti_index_correspond[i,3] = disc_edge[i]
        end

        zzz = sortperm(conti_index_correspond[:,2])
        disc_edge_result = conti_index_correspond[:,3][zzz]

        return (score,graph,disc_edge_result)
end

function K2_w_discretization(data_matrix,u,continuous_index,times,cut_time,approx = true)

        score = Inf
        graph = 0
        disc_edge = 0

        for time = 1 : times
                # Produce random sequence of indexes
                println(("Times of K2 along with discretization method:",time))
                order = rand_seq(length(data_matrix[1,:]))
                iteration_result = K2_one_iteration_discretization(order,u,data_matrix,
                                                          continuous_index,cut_time,approx)

                if iteration_result[1] < score
                        score = iteration_result[1]
                        graph = iteration_result[2]
                        disc_edge = iteration_result[3]
                end
        end

        return (score,graph,disc_edge)
end

function Discretize_All(data_matrix,graph,continuous_index,cut_time)
    discrete_index = Array(Int64,length(data_matrix[1,:]) - length(continuous_index))
    num = 0
    for i = 1 : length(data_matrix[1,:])
        if ~(i in continuous_index)
            num += 1
            discrete_index[num] = i
        end
    end
    sort_continuous = graph_to_reverse_conti_order(graph,continuous_index)
    X = BN_discretizer_iteration_converge(data_matrix,graph,discrete_index,sort_continuous,cut_time,false)
    edge = X[2]
    reorder_edge = sort_disc_by_vorder(sort_continuous,edge)

    return reorder_edge
end



function Learn_DVBN(data_matrix,continuous_index,order,u,cut_time)
    X = K2_one_iteration_discretization(order,u,data_matrix,continuous_index,cut_time,false)
    return (X[2],X[3])
end

function Learn_Discrete_Bayesian_Net(data_matrix,continuous_index,u,cut_time,times)
    X = K2_w_discretization(data_matrix,u,continuous_index,times,cut_time,false)
    return (X[2],X[3],X[1])
end
