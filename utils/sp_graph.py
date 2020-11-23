'''
This module create a square grid with dimension (dim,dim) using package networkx and output the incidence matrix for
the grid graph.

This is how the incidence matrix is built
for i  in edges:
  A_mat[sources[i], i] = -1
  A_mat[destinations[i], i] = 1

'''
import networkx as nx


class SpGraph:
    def __init__(self, dim):
        self.dim = dim
        self.A = self.get_incidence_matrix()  # we store the incidence matrix of the grid in A
        self.n_nodes, self.n_edges = self.A.shape

    def get_incidence_matrix(self):
        '''

        :return: the incidence matrix A which is a dense numpy array
        '''
        G = nx.grid_graph(dim=(self.dim, self.dim))
        sparse_A = nx.incidence_matrix(G,
                                       oriented=True)  # if oriented = False the matrix get populated with only ones. But we need specify the source of each edge with -1
        A = sparse_A.A
        return A
