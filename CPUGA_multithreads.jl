using BenchmarkTools
using Plots
using Random
using Base.Threads


# Fitness Function
function fitness_function!(fitness::Vector{Float32}, sxx::Vector{Float32}, syy::Vector{Float32}, sxy::Vector{Float32}, x::Matrix{Bool})
    
    for i in eachindex(fitness)
        rxx = 0.0f0
        ryy = 0.0f0
        rxy = 0.0f0
        for g in eachindex(view(x, :, i)) # Each gene
            rxx += sxx[g] * x[g, i]
            ryy += syy[g] * x[g, i]
            rxy += sxy[g] * x[g, i]
        end

        #fitness[i] = rxx + ryy + rxy
        fitness[i] = rxx^2 + ryy^2 - rxx * ryy + 3 * rxy^2
    end
    return nothing
end

# Find Best Individuals Function
function find_best_function!(fitness::Vector{Float32}, best_fitness::Vector{Float32}, best_index::Vector{Int64}, best_individual::Matrix{Bool}, population::Matrix{Bool}, p::Int64)
    # Select best individuals
    
    for i in eachindex(fitness)
        if best_fitness[p] < fitness[i]
            best_fitness[p] = fitness[i]
            best_index[p] = i

            best_individual[p,:] .= population[:,i]
        end
    end
    return nothing

end

# Tournament Selection Function
function selection_function!(fitness::Vector{Float32}, idx1::Vector{Int64}, idx2::Vector{Int64}, selected_index::Vector{Int64})

    for i in eachindex(fitness)
        # select two random teams of individuals        
                
        # Extract the best teammates of the two teams
        if fitness[idx1[i]] > fitness[idx2[i]] # member of the first team performed better
            selected_index[i] = idx1[i]           
        else  # member of the second team performed better
            selected_index[i] = idx2[i]
        end
    end
    return nothing
end

# Crossover Function
function crossover_function!(parent1::Matrix{Bool},parent2::Matrix{Bool},child1::Matrix{Bool},child2::Matrix{Bool},crossover_probability::Float32)
    # Index for the problem
    
    for i in eachindex(view(parent1,1,:))
        if rand() < crossover_probability
            crossover_point = rand(1:size(parent1,1)-1)
            #println(crossover_point)
            child1[:,i] .= vcat(parent1[1:crossover_point,i],parent2[crossover_point+1:end,i])
            child2[:,i] .= vcat(parent2[1:crossover_point,i],parent1[crossover_point+1:end,i])
        else
            child1[:,i] .= parent1[:,i]
            child2[:,i] .= parent2[:,i]
        end
    end
    return nothing

end

# Selection function
function select_parent_function!(population::Matrix{Bool},selected_index::Vector{Int64},parent1::Matrix{Bool},parent2::Matrix{Bool})

    # Consider only odd indices (it's like doing 1:2:num_individuals)
    for i in 1:2:size(population,2)
        parent1[:, i÷2+1] .= population[:, selected_index[i]]
        parent2[:, i÷2+1] .= population[:, selected_index[i+1]]
    end
    return nothing
end


# Mutation Function 
function mutate_function!(population::Matrix{Bool},mutation_probability::Float32)
    # Index for the problem
    for i in eachindex(view(population,1,:))
        if rand() <= mutation_probability
            #population[i,mutation_point,p]   =   !population[i,mutation_point,p]
            mutation_point = rand(1:size(population,1))
            population[mutation_point,i] ⊻= true
        end
    end
    return nothing

end

# Update population function
function update_population_function!(population::Matrix{Bool},child1::Matrix{Bool},child2::Matrix{Bool})

    for i in 1:2:size(population,2)
    # Consider only even indices 
        population[:, i]   .=   child1[:, i÷2+1]
        population[:, i+1] .=   child2[:, i÷2+1]
    end
    return nothing
end

function Evolve!(num_individuals::Int64, num_genes::Int64, num_problems::Int64, crossover_probability::Float32, mutation_probability::Float32, num_generations::Int64)

    # Fitness function preallocation 
    sxx = ones(Float32, num_genes)
    syy = ones(Float32, num_genes)
    sxy = ones(Float32, num_genes)

    # Best individuals preallocation
    best_fitness = zeros(Float32, num_problems)
    best_index = zeros(Int64, num_problems)
    best_individual = zeros(Bool, num_problems, num_genes)

    # Counter for threads iteration
    thread_usage = zeros(Int64, nthreads())

    Threads.@threads for p in 1:num_problems  # Each thread handles a separate problem

        tid = threadid()  # Get thread ID
        thread_usage[tid] += 1  # Count how many iteration is the current thread doing 

        # Create a local copy of the variables in order to avoid conflicts between threads
        local_population = rand(Bool, num_genes, num_individuals)  # Each thread has a separate population 
        local_fitness = zeros(Float32, num_individuals)
        local_parent1 = zeros(Bool, num_genes, num_individuals ÷ 2)
        local_parent2 = similar(local_parent1)
        local_child1 = similar(local_parent1)
        local_child2 = similar(local_parent1)
        local_selected_index = zeros(Int64, num_individuals)
        local_idx1 = zeros(Int64, num_individuals)
        local_idx2 = zeros(Int64, num_individuals)

        # Evolution for current problem
        for generation in 1:num_generations
            fitness_function!(local_fitness, sxx, syy, sxy, local_population)
            find_best_function!(local_fitness, best_fitness, best_index, best_individual, local_population, p)
            rand!(local_idx1, 1:num_individuals)
            rand!(local_idx2, 1:num_individuals)
            selection_function!(local_fitness, local_idx1, local_idx2, local_selected_index)
            select_parent_function!(local_population, local_selected_index, local_parent1, local_parent2)
            crossover_function!(local_parent1, local_parent2, local_child1, local_child2, crossover_probability)
            update_population_function!(local_population, local_child1, local_child2)
            mutate_function!(local_population, mutation_probability)
        end

        if p % 100 == 0
            println("Problem $p solved by thread $tid.")
        end
    end

    # Print load distribution between threads
    println("\nLoad distribution between threads:")
    for i in 1:nthreads()
        println("Thread $i executed $(thread_usage[i]) iterations")
    end

    return best_fitness, best_individual
end


# Genetic algorithm parameters
num_genes = 200
num_individuals = 100
num_problems = 1000
num_generations = 250
crossover_probability = 0.8f0;
mutation_probability = 0.5f0;


@time best_fitness, best_individual = Evolve!(num_individuals,num_genes,num_problems,crossover_probability,mutation_probability,num_generations)
# @show CUDA.synchronize()
# Move CUDA arrays back to CPU
# best_fitnessCPU = Array(best_fitness)
#plot(best_fitness)