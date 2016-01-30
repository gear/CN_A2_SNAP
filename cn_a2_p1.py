"""
Utility functions and implementation for assignment 2 (Part 1) of
the Tokyo Insitute of Technology - Fall 2015 Complex Network
- Instructor: Assoc. Prof. Tsuyoshi Murata
- Date: Jan 26, 2016
- Deadline: Feb 01, 2015
- Student: NGUYEN T. Hoang - M1
- StudentID: 15M54097
- Python version: 2.7.10
- SNAP version: 2.4 - SNAP: snap.stanford.edu
"""

import snap as sn
import numpy as np

SOURCE_URL = 'https://github.com/gear/CN_A2_SNAP/blob/master/edgelist.txt'

def maybe_download(filename, workdirectory):
    """ Download the data, unless it's already here """
    if not os.path.exists(workdirectory):
        os.mkdir(workdirectory)
    filepath = os.path.join(workdirectory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(SOURCE_URL, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, ' bytes.')
    return filepath

def edge_list_to_np(edge_list):
    # For now
    return np.array([0,1,1,1,1,0,0,0,1,0,1,1,1,0,0,0,1,1,0,1,1,0,0,0,1,1,1,0,1,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0]).reshape([8,8])

class UnweightedUndirectedGraph(object):
    # Initialize
    def __init__ (self, edge_list, graph_name):
        if not os.path.exists(edge_list_file):
            print('File not found: %s' % edge_list)
        else:
            self._filepaht = edge_list
            self._graph = sn.LoadEdgeList(sn.PUNGraph, edge_list, 0, 1, ' ')
        self._file = edge_list
        self._num_nodes = self._graph.GetNodes()
        self._num_edges = self._graph.GetEdges()
        self._graph_name = str(graph_name)
        self._adj_matrix = edge_list_to_np(edge_list)

    def EigenvectorCentrality(self):
        # Create a hashmap to store result
        # Mapping node ID to Eigenvector Centrality score.
        NIdEigenH = sn.TIntFltH()
        sn.GetEigenVector(self._graph, NIdEigenH)
        return NIdEigenH

    def BetweennessCentrality(self, isNode = true):
        # Create hashmaps to store Node betweenness and
        # Vertex betweenness. Return Node score by default.
        NodeScore = sn.TIntFltH()
        EdgeScore = sn.TIntFltH()
        sn.GetBetweennessCentr(self._graph, NodeScore, EdgeScore, 1.0)
        if (isNode):
            return NodeScore
        else
            return EdgeScore

    def LaplacianMatrix(self):
        # Get diag of adj matrix A and then turn it into diagonal matrix D
        D = np.diag(np.sum(self._adj_matrix, 0))
        return D - self._adj_matrix

    def SpectralBisection(self):
        L = self.LaplacianMatrix();
        # Get eigenvector of L
        u, v = np.linalg.eig(L)
        # Get the second smallest eigenvalue index
        # and its corresponding eigenvector
        i = np.argsort(u)[1]
        eigv = v[:,i]
        partition = np.ones(eigv.shape[0])
        index_sorted_eigv = np.argsort(eigv)
        for i in range(partition.shape[0] / 2, partition.shape[0]):
            partition[index_sorted_eigv[i]] = -1
        return partition

    def ModularityMatrix(self):
        B = np.zeros(self._adj_matrix.shape)
        d = np.sum(self._adj_matrix, 0)
        m = sum(d)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i,j] = A[i,j] - (d[i] * d[j]) / float(m)
        return B

    def SpectralModularityMaximization(self):
        B = self.ModularityMatrix()
        # Get eigenvector of B
        u, v = np.linalg.eig(B)
        # Get the largest eigvenvalue and its vector
        i = np.argsort(u)[u.shape[0]-1]
        eigv = v[:,i]
        partition = np.ones(eigv.shape[0])
        for i in range(eigv.shape[0]):
            if (eigv[i] < 0):
                partition[i] = -1
        return partition
