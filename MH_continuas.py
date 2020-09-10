import random
import math
import numpy as np

################################## Leaders and Followers ##############################################
#  Metaheurística "Leaders and Followers" (LaF)
def leaders_and_follower(popsize, dim, lbound, ubound, maxFEs):

    leaders = np.random.uniform(low=lbound, high=ubound, size=(popsize, dim))   #Inicializa la población de Leaders
    followers = np.random.uniform(low=lbound, high=ubound, size=(popsize, dim)) #Inicializa la población de Followers

    leaders_cost = []
    followers_cost = []
    for i in range(0, popsize):
        leaders_cost.append(evaluate(leaders[i]))
        followers_cost.append(evaluate(followers[i]))

    median_leaders=np.median(leaders_cost)

    FEs = 2*popsize
    while FEs<maxFEs:  # Mientras no se alcance el número máximo de evaluaciones (condición de parada)

        for i in range(0, len(followers)):
            trial = create_trial(leaders[random.randint(0, len(leaders)-1)], followers[i], lbound, ubound)
            trial_cost = evaluate(trial)
            if trial_cost < followers_cost[i]:
                followers[i] = trial
                followers_cost[i] = trial_cost
            FEs += 1

        median_followers = np.median(followers_cost)

        if median_followers < median_leaders:
            leaders, leaders_cost = merge_population(leaders, leaders_cost, followers, followers_cost, popsize)
            median_leaders = np.median(leaders_cost)
            followers = np.random.uniform(low=lbound, high=ubound, size=(popsize, dim))
            for i in range(0, len(followers)):
                followers_cost[i] = evaluate(followers[i])
            FEs += 30

    best_i = np.argmin(leaders_cost)
    return leaders[best_i], leaders_cost[best_i]

# Método para crear una nueva solución en LaF
def create_trial(l, f, lbound, ubound):

    trial = f + (l-f)*np.random.uniform(low=0, high=2, size=len(l))

    for i in range(0, len(trial)):
        if trial[i] > ubound:
            trial[i] = ubound
        if trial[i] < lbound:
            trial[i] = lbound

    return trial

# Métdodo para combinar poblaciones en LaF
def merge_population(leaders, leaders_costs, followers, followers_costs, popsize):

    populations = []
    populations.append(leaders)
    populations.append(followers)

    costs = []
    costs.append(leaders_costs)
    costs.append(followers_costs)
    full_population = np.vstack(populations)
    full_costs = np.hstack(costs)

    indices = np.argsort(full_costs)
    return full_population[indices][:popsize], full_costs[indices][:popsize]
############################## Leaders and Followers ###################################


############################## Búsqueda Aleatoria ###################################
def random_search(dim, lbound, ubound, maxFEs):
    best_solution = np.random.uniform(low=lbound, high=ubound, size=(1, dim))[0]
    best_cost = evaluate(best_solution)

    for i in range(maxFEs-1):
        solution = np.random.uniform(low=lbound, high=ubound, size=(1, dim))[0]
        cost = evaluate(solution)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution

    return  best_solution, best_cost
############################## Búsqueda Aleatoria ###################################


############################## Evolución Diferencial ###################################
def differential_evolution(popsize, dim, lbound, ubound, maxFEs, f, cr):
	# Inicializar la población de manera aleaoria y uniforme
	population = np.random.uniform(low=lbound, high=ubound, size=(popsize, dim))

	# Almacenar la evaluaciones de todos los individuos de la población
	population_cost = np.empty(popsize)
	for i in range(0, popsize):
		population_cost[i] = evaluate(population[i])

	FEs = popsize  # Ya se evaluó la función una vez por cada individuo de la población

	# Mientras no se alcance el número máximo de evaluaciones (condición de
	# parada)
	while FEs < maxFEs:
		new_population = np.empty((popsize, dim))
		new_cost = np.empty(popsize)
		
		for i in range(0, popsize): #Generar un descendiente por cada individuo de la población

			#Se eligen aleatorios 3 individuos diferentes de la población
			index1, index2, index3 = np.random.choice(np.delete(np.arange(popsize), i), replace = False, size = 3)
			
			#Se mutan para formar un nuevo individuo
			trial = population[index1] + f * (population[index2] - population[index3])	

			offspring = np.empty(dim)

			#Para forzar siempre al menos un elemento del trial para que no sea igual al padre
			fix_index = np.random.randint(dim)
			#Se aplica el operador de recombinación entre el nuevo individuo y el padre
			for j in range(0,dim):		
			 	if j == fix_index:
			 		offspring[j] = trial[j]
			 	else:
			 		#Si es menor que el índice de recombinación se va a poner el elemento de la nueva
			 		#solución en el descendiente
			 		if np.random.uniform() < cr:
			 			offspring[j] = trial[j]
			 		#Sino se pone el elemento del individuo original
			 		else:
			 			offspring[j] = population[i][j]

			#Se evalúa el descendiente obtenido
			cost = evaluate(offspring)

			FEs += 1

			#Se añade a la próxima generación el que tenga mejor evaluación entre el padre
			#y el descendiente
			if cost < population_cost[i]:
				new_population[i] = offspring
				new_cost[i] = cost

			else:
				new_population[i] = population[i]
				new_cost[i] = population_cost[i]

		#Actualizamos la población y los costos
		population = new_population
		population_cost = new_cost

		#Se modifican los parámetros f y cr, con probabilidad 1/10 haçiéndolos variar entre 
		#[0.1,1] y [0,1] respectivamente de manera uniforme
		if np.random.binomial(1, 0.1):
			f = 0.1 + np.random.uniform() * 0.9

		if np.random.binomial(1, 0.1):
			cr = np.random.uniform()

	best_solution =  np.argmin(population_cost)
	return  population[best_solution], population_cost[best_solution]

############################## Evolución Diferencial ###################################


######################### Función Objetivo ####################
def evaluate(s):
    # Función Rastrigin
    f=0
    for d in range(0, len(s)):
        f = f + 10 + math.pow(s[d], 2) - 10*math.cos(2*math.pi*s[d])
    return f


######################### Optimización ####################
dim = 10            # Número de dimensiones
popsize = dim       # Tamaño de la población (solo para LaF)
lbound = -100       # Cota inferior de la función objetivo (la misma para cada variable)
ubound = 100        # Cota superior de la función objetivo (la misma para cada variable)
maxFEs = 10000*dim  # Número máximo de evaluaciones (Condición de parada)
popsize_DE = 2*dim  # Tamaño de la población (solo para DE)
f = 0.8				# Factor de amplificación del vector diferencia (solo para DE)
cr = 0.5			# Controlador de la recombinación (solo para DE)

best_solution, best_cost = differential_evolution(popsize_DE, dim, lbound, ubound, maxFEs, f, cr)
print("Mejor solución de Evolución Diferencial: "+ str(best_cost))

best_solution, best_cost = random_search(dim, lbound, ubound, maxFEs)
print("Mejor solución de Búsqueda Aleatoria: "+ str(best_cost))

best_solution, best_cost = leaders_and_follower(popsize, dim, lbound, ubound, maxFEs)
print("Mejor solución de Leaders and Followers: "+ str(best_cost))
