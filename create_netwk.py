import igraph, random
# For best results, max_synapses should be ~10% num_neurons
def create_network(num_neurons, max_synapses, print_stats=False):
    graph = igraph.Graph.Watts_Strogatz(1, int(num_neurons), int(max_synapses/2), 0.05)
    edge_list = graph.get_adjlist()
    # Prune excess edges
    for i in xrange(len(edge_list)):
        while len(edge_list[i]) > max_synapses:
            edge_list[i].pop(random.randint(0,max_synapses-1))
            """
            for j in xrange(max_synapses-len(edge_list[i])):
                edge_list[i].pop(random.randint(0,max_synapses-1))
            """

    # Transform into a list of 2-tuple edges
    # edges = [item for sublist in edge_list for item in sublist]
    edges = [ (i,j) for i in xrange(len(edge_list)) for j in edge_list[i] ]

    graph = igraph.Graph(edges=edges, directed=True)

    if print_stats:
        print "After pruning"
        print graph.transitivity_undirected()
        print graph.average_path_length()
        adjl = graph.get_adjlist()
        q = [len(i) for i in adjl]
        print float(sum(q))/len(q)
        print max(q)

    return graph.get_adjacency()

create_network(1000,100)
