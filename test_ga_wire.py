from __future__ import division
from deap import algorithms, base, creator, tools
import random
import networkx as nx

per_dir_cell_count = 10 # number of cell being copied in each direction

max_size = 256 # maximum size to put numbers

switch_count =0
for i in range(max_size):
    if i < per_dir_cell_count:
        switch_count += 1+ 2 * i
    else:
        switch_count += 1 + 2 * per_dir_cell_count



flow_capacity = 6


#nodes = list(self.get_nodes())
#nodes.remove(self._fake_root)
#node_dic = self._generate_standard_index_dictionary()
#switch_map = self.get_connectivity_matrix(node_dic)

toolbox = base.Toolbox()
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox.register("indices", random.sample, range(max_size * max_size), switch_count)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.indices)
toolbox.register("population", tools.initRepeat, list,
                 toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)

###
network_graph = nx.DiGraph()
network_graph.add_nodes_from([(i,j) for i in range(max_size) for j in range(max_size)])
edges_list = []
for i in range(max_size):
    for j in range(max_size):
        curr_node=(i,j)
        left_neighb = (i,j-1)
        right_neighb = (i, j + 1)
        up_neighb = (i + 1, j)
        down_neighb = (i -1, j - 1)

        if left_neighb[1] >= 0:
            edges_list.append((curr_node, left_neighb,{'capacity':flow_capacity}))
        if right_neighb[1] < max_size:
            edges_list.append((curr_node, right_neighb, {'capacity': flow_capacity}))
        if up_neighb[0] < max_size:
            edges_list.append((curr_node, up_neighb, {'capacity': flow_capacity}))
        if down_neighb[0] >= 0:
            edges_list.append((curr_node, down_neighb, {'capacity': flow_capacity}))

network_graph.add_edges_from(edges_list)

network_graph.add_node('S')
network_graph.add_node('T')

def evaluation(individual):
    src_edges = []
    dst_edges = []




toolbox.register("evaluate", evaluation)
toolbox.register("select", tools.selTournament, tournsize = 3)

fit_stats = tools.Statistics(key=operator.attrgetter("fitness.values"))
fit_stats.register('mean', np.mean)
fit_stats.register('min', np.min)

pop = toolbox.population(n=5000)

bfs_set = set(
    self.get_BFS_label_dictionary().values())
bfs_set.update(range(num_nodes))
pop.insert(0, creator.Individual(list(bfs_set))) # adding bfs solution as an initial guess
pop.insert(1, creator.Individual(list(bfs_set)))  # adding bfs solution as an initial guess
pop.insert(2, creator.Individual(list(bfs_set)))  # adding bfs solution as an initial guess

result, log = algorithms.eaSimple(pop, toolbox,
                                  cxpb=0.5, mutpb=0.2,
                                  ngen=2000, verbose=False,
                                  stats=fit_stats)

best_individual = tools.selBest(result, k=1)[0]
print('Fitness of the best individual: ', evaluation(best_individual)[0])

plt.figure(figsize=(11, 4))
plots = plt.plot(log.select('min'), 'c-', log.select('mean'), 'b-')
plt.legend(plots, ('Minimum fitness', 'Mean fitness'), frameon=True)
plt.ylabel('Fitness')
plt.xlabel('Iterations')
plt.show()

return dict(zip(nodes, best_individual))

