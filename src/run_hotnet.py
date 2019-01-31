# Imports

import shutil
import os
import sys, argparse, time
from tqdm import tqdm
import subprocess
import uuid
import math, numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
import networkx as nx
from itertools import combinations
import h5py
#PATH WHERE THE HIERARCHICAL HOTNET SCRIPTS ARE
sys.path.insert(0, '/aloy/home/afernandez/ntwrk_project/hotnet/hierarchical-hotnet/src/')
#For running into the cluster
os.environ['MKL_NUM_THREADS'] = '1'

# Parser

def get_parser():
    description = 'Diffuse heat on a network, given scores and networks.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n', '--network_folders', type=str, required=True, nargs='*', help='Networks folders. They should contain edgelist.tsv, idx2node.tsv and similarity_matrix.h5 files.')
    parser.add_argument('-s', '--scores_files', type=str, required=True, nargs='*', help='Scores files. These scores refer to the *same* study (i.e. different types of score).')
    parser.add_argument('-bs', '--baseline_score', type=float, required=False, default=0.0, help="Baseline heat. If 0, non-scored nodes will not appear in clusters.")
    parser.add_argument('-p', '--perm', type=int, required=False, default=100, help="Permutations to perform")
    parser.add_argument('-t', '--consensus_t', type=int, required=False, default=2, help="Consensus cutoff")
    parser.add_argument('-e', '--erase', type=bool, required=False, default=False, help="Erase results and redo")
    parser.add_argument('-o', '--output_folder', type=str, required=False, default='results/', help='Output folder')
    return parser

#HHIO & COMMON

def load_edge_list(filename, dictionary=None):
    '''
    Load edge list.
    '''
    edge_list = list()
    with open(filename, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                arrs = l.strip().split()
                i = int(arrs[0])
                j = int(arrs[1])
                edge_list.append((i, j))

    if dictionary:
        edge_list = convert_edge_list(edge_list, dictionary)

    return edge_list

def load_gene_score(filename, score_threshold=0.0):
    '''
    Load gene scores.
    '''
    gene_to_score = dict()
    with open(filename, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                arrs = l.strip().split()
                gene = arrs[0]
                score = float(arrs[1])
                if score>=score_threshold:
                    gene_to_score[gene] = score
    return gene_to_score

def load_index_gene(filename):
    '''
    Load index-gene associations.
    '''
    index_to_gene = dict()
    gene_to_index = dict()
    with open(filename, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                arrs = l.strip().split()
                index = int(arrs[0])
                gene = arrs[1]
                index_to_gene[index] = gene
                gene_to_index[gene] = index
    return index_to_gene, gene_to_index

def save_gene_score(filename, gene_to_score):
    '''
    Save gene scores.
    '''
    gene_score_list = sorted(gene_to_score.items())
    with open(filename, 'w') as f:
        f.write('\n'.join('{}\t{}'.format(gene, score) for gene, score in gene_score_list))

def load_matrix(filename, matrix_name='PPR'):
    '''
    Load matrix.
    '''
    f = h5py.File(filename, 'r')
    if matrix_name in f:
        A = f[matrix_name].value
    else:
        raise KeyError('Matrix {} is not in {}.'.format(matrix_name, filename))
    f.close()
    return A

def save_weighted_edge_list(filename, edge_list, dictionary=None):
    '''
    Save weighted edge list as a TSV file in (source, target, weight) format.
    '''
    if dictionary:
        edge_list = convert_weighted_edge_list(edge_list, dictionary)

    with open(filename, 'w') as f:
        f.write('\n'.join('\t'.join(map(str, edge)) for edge in edge_list))

def save_index_gene(filename, index_to_gene):
    '''
    Save index-gene associations.
    '''
    index_gene_list = sorted(index_to_gene.items())
    with open(filename, 'w') as f:
        f.write('\n'.join('{}\t{}'.format(index, gene) for index, gene in index_gene_list))

def load_weighted_edge_list(filename, dictionary=None):
    '''
    Load weighted edge list from a TSV file in (source, target, weight) format.
    '''
    edge_list = list()
    with open(filename, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                arrs = l.rstrip('\n').split('\t')
                source = int(arrs[0])
                target = int(arrs[1])
                weight = float(arrs[2])
                edge_list.append((source, target, weight))

    if dictionary:
        edge_list = convert_weighted_edge_list(edge_list, dictionary)

    return edge_list

def combined_similarity_matrix(P, gene_to_index, gene_to_score):
    '''
    Find the combined similarity submatrix from the topological similarity matrix and the gene
    scores.
    '''
    m, n = np.shape(P)
    min_index = min(gene_to_index.values())
    max_index = max(gene_to_index.values())

    # We do not currently support nodes without edges, which seems to be the most likely way for
    # someone to encounter this error.
    if m!=n:
        raise IndexError('The similarity matrix is not square.')
    if max_index-min_index+1!=n:
        raise IndexError('The similarity matrix and the index-gene associations have different dimensions.')

    topology_genes = set(gene_to_index)
    score_genes = set(gene_to_score)
    common_genes = sorted(topology_genes & score_genes)
    common_indices = [gene_to_index[gene]-min_index for gene in common_genes]

    f = np.array([gene_to_score[gene] for gene in common_genes])
    S = P[np.ix_(common_indices, common_indices)]*f
    common_index_to_gene = dict((i+1, gene) for i, gene in enumerate(common_genes))
    common_gene_to_index = dict((gene, i) for i, gene in common_index_to_gene.items())

    return S, common_index_to_gene, common_gene_to_index

def convert_edge_list_to_adjacency_matrix(edge_list):
    '''
    Convert an edge list to an adjacency matrix for an undirected graph.
    '''
    k = min(min(edge) for edge in edge_list)
    l = max(max(edge) for edge in edge_list)

    A = np.zeros((l-k+1, l-k+1), dtype=np.int)
    for i, j in edge_list:
        A[i-k, j-k] = A[j-k, i-k] = 1

    return A

def convert_edge_list(edge_list, dictionary):
    '''
    Convert an edge list with indices/names to an edge list with names/indices.
    '''
    return [(dictionary[i], dictionary[j]) for i, j in edge_list]

def convert_weighted_edge_list(edge_list, dictionary):
    '''
    Convert an edge list with indices/names to an edge list with names/indices.
    '''
    return [(dictionary[i], dictionary[j], w) for i, j, w in edge_list]

def combined_similarity_matrix(P, gene_to_index, gene_to_score):
    '''
    Find the combined similarity submatrix from the topological similarity matrix and the gene
    scores.
    '''
    m, n = np.shape(P)
    min_index = min(gene_to_index.values())
    max_index = max(gene_to_index.values())

    # We do not currently support nodes without edges, which seems to be the most likely way for
    # someone to encounter this error.
    if m!=n:
        raise IndexError('The similarity matrix is not square.')
    if max_index-min_index+1!=n:
        raise IndexError('The similarity matrix and the index-gene associations have different dimensions.')

    topology_genes = set(gene_to_index)
    score_genes = set(gene_to_score)
    common_genes = sorted(topology_genes & score_genes)
    common_indices = [gene_to_index[gene]-min_index for gene in common_genes]

    f = np.array([gene_to_score[gene] for gene in common_genes])
    S = P[np.ix_(common_indices, common_indices)]*f
    common_index_to_gene = dict((i+1, gene) for i, gene in enumerate(common_genes))
    common_gene_to_index = dict((gene, i) for i, gene in common_index_to_gene.items())

    return S, common_index_to_gene, common_gene_to_index

def progress(message=''):
    '''
    Write status message to screen; overwrites previous message and does not advance line.
    '''
    try:
        rewind = progress.rewind
    except AttributeError:
        rewind = 0

    sys.stdout.write('\r'+' '*rewind)
    sys.stdout.flush()
    sys.stdout.write('\r'+str(message))
    sys.stdout.flush()
    progress.rewind = max(len(str(message).expandtabs()), rewind)

###################################################################################################################################
#
# Hierarchical clustering generation functions
#
####################################################################################################################################

def tarjan_HD(A, reverse=True, verbose=False):
    '''
    Compute the hierarchical decomposition of a graph into strongly connected components using Tarjan's algorithm (1983).
    This function is the driver function for the implementation of this algorithm.
    '''
    # Check for Fortran module:
    if verbose:
        if imported_fortran_module:
            print('- Fortran module imported successfully...')
        else:
            print('- Fortran module not imported successfully; falling back to Python-only functions...')

    # Initialize variables.
    n = np.shape(A)[0]
    V = range(n)
    T = list()
    i = 0
    root = n

    # Compute hierarchical decomposition.
    # We reduce memory consumption by temporarily reducing the precision of the weights in the adjacency matrix, which is usually
    # fine because we do not perform arithmetic on the weights; another strategy would replace the weights by their ranks.
    if not reverse:
        weights = np.unique(A)
        m = len(weights)
        B = np.asarray(A, dtype=np.float32)

        T, root = tarjan_HD_recursive(V, B, T, i, root)

        S = list()
        for u, v, a in T:
            j = min(max(np.searchsorted(weights, a), 1), m-1)
            x = weights[j-1]
            y = weights[j]
            if abs(x-a)<abs(y-a):
                w = x
            else:
                w = y
            S.append((u, v, w))
        T = S

    else:
        # When adding edges in reverse, we "reverse" the weights in the adjacency matrix and then "reverse" the corresponding heights.
        weights = np.unique(A)
        m = len(weights)
        max_weight = weights[m-1]
        B = np.asarray(2*max_weight-A, dtype=np.float32)
        np.fill_diagonal(B, 0)

        T, root = tarjan_HD_recursive(V, B, T, i, root)

        S = list()
        for u, v, a in T:
            b = 2*max_weight-a
            j = min(max(np.searchsorted(weights, b), 1), m-1)
            x = weights[j-1]
            y = weights[j]
            if abs(x-b)<abs(y-b):
                w = x
            else:
                w = y
            S.append((u, v, w))
        T = S

    T = [(u+1, v+1, w) for u, v, w in T]

    return T

def find_height_to_clusters(preordered_T, index_to_gene, reverse=True):
    '''
    Find clusters for every distinct height of the dendrogram.
    '''
    # Associate nodes of dendrogram with indices.
    index_to_node = defaultdict(set)
    for index, gene in index_to_gene.items():
        index_to_node[index] = frozenset([gene])

    # Initialize clusters with leaf nodes.
    height_to_clusters = dict()
    height = float('inf')
    clusters = set(index_to_node.values())
    height_to_clusters[height] = clusters.copy()

    # Update clusters while ascending dendrogram.
    T = sorted(preordered_T, key=lambda x: x[2], reverse=reverse)
    m = len(T)

    for i, edge in enumerate(T):
        source, target, height = edge

        a = index_to_node[source]
        b = index_to_node[target]
        c = frozenset(set(a) | set(b))

        clusters.discard(a)
        clusters.discard(b)
        clusters.add(c)

        del index_to_node[source]
        index_to_node[target] = c

        if i==m-1 or height!=T[i+1][2]:
            height_to_clusters[height] = clusters.copy()

    # Add cluster for root node.
    height = 0.0
    clusters = set(frozenset(index_to_node.values()))
    height_to_clusters[height] = clusters.copy()

    return height_to_clusters

def tarjan_HD_recursive(V, A, T, i, root):
    '''
    Compute the hierarchical decomposition of a graph into strongly connected components using Tarjan's algorithm (1983).
    This function is not the driver function for the implementation of this algorithm; call the tarjan_HD function directly instead
    of this function.
    '''
    weights = find_distinct_weights(A)
    m = len(weights)-1
    r = m-i

    if r==1:
        # Case 1
        weight_m = weights[m]
        del weights

        for v in V:
            T.append((v, root, weight_m))
        return T, root

    else:
        # Case 2
        j = int(math.ceil(0.5*float(i+m)))
        weight_i = weights[i]
        weight_j = weights[j]
        del weights

        A_j = threshold_edges(A, weight_j)
        components = strongly_connected_components(A_j)

        if len(components)==1:
            # Case 2a
            return tarjan_HD_recursive(V, A_j, T, i, root)
        else:
            # Case 2b
            Y = list()
            for component in components:
                if len(component)>1:
                    X = index_vertices(V, component)
                    B = slice_array(A_j, component, component)
                    k = subproblem_index(B, weight_i)
                    subtree, root = tarjan_HD_recursive(X, B, list(), k, root)
                    T.extend(subtree)
                    Y.append(root)
                    root += 1
                else:
                    Y.extend(index_vertices(V, component))

            B = condense_graph(A, components)
            k = subproblem_index(B, weight_j)
            return tarjan_HD_recursive(Y, B, T, k, root)

def condense_graph(A, components):
    if imported_fortran_module:
        n = len(components)
        nodes = np.array([i for component in components for i in component], dtype=np.int64)
        sizes = np.array([len(component) for component in components], dtype=np.int64)
        indices = np.array([np.sum(sizes[:i]) for i in range(n+1)], dtype=np.int64)
        return fortran_module.condense_adjacency_matrix(A, nodes+1, indices+1)
    else:
        n = len(components)
        B = np.zeros((n, n), dtype=A.dtype)
        for i in range(n):
             for j in range(n):
                if i!=j:
                    C = slice_array(A, components[j], components[i])
                    nonzero_indices = np.nonzero(C)
                    if np.size(nonzero_indices)>0:
                        B[i, j] = np.min(C[nonzero_indices])
        return B

def find_distinct_weights(A):
    if imported_fortran_module:
        B, l = fortran_module.unique_entries(A)
        return B[:l]
    else:
        return np.unique(A)

def index_vertices(vertices, indices):
    return [vertices[index] for index in indices]

def slice_array(A, rows, columns):
    if imported_fortran_module:
        return fortran_module.slice_array(A, np.array(columns, dtype=np.int)+1, np.array(rows, dtype=np.int)+1)
    else:
        return A[np.ix_(rows,columns)]

def strongly_connected_components(A):
    if imported_fortran_module:
        indices = fortran_module.strongly_connected_components(A)
    else:
        indices = strongly_connected_components_from_adjacency_matrix(A)

    index_to_component = defaultdict(list)
    for i, j in enumerate(indices):
        index_to_component[j].append(i)
    return index_to_component.values()

def strongly_connected_components_from_adjacency_matrix(A):
    m, n = np.shape(A)
    nodes = range(n)

    index = -np.ones(n, dtype=np.int64)
    lowlink = -np.ones(n, dtype=np.int64)
    found = np.zeros(n, dtype=np.bool)
    queue = np.zeros(n, dtype=np.int64)
    subqueue = np.zeros(n, dtype=np.int64)
    component = np.zeros(n, dtype=np.int64)

    neighbors = np.zeros((n, n), dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    for v in nodes:
        neighbors_v = np.where(A[v]>0)[0]
        degree_v = np.size(neighbors_v)
        neighbors[v, 0:degree_v] = neighbors_v
        degree[v] = degree_v

    i = 0
    j = 0
    k = 0
    l = 0

    for u in nodes:
        if not found[u]:
            queue[k] = u
            k += 1

            while k>=1:
                v = queue[k-1]
                if index[v]==-1:
                    i += 1
                    index[v] = i

                updated_queue = False
                for w in neighbors[v, 0:degree[v]]:
                    if index[w]==-1:
                        queue[k] = w
                        k += 1
                        updated_queue = True
                        break

                if not updated_queue:
                    lowlink[v] = index[v]
                    for w in neighbors[v, 0:degree[v]]:
                        if not found[w]:
                            if index[w]>index[v]:
                                lowlink[v] = min(lowlink[v], lowlink[w])
                            else:
                                lowlink[v] = min(lowlink[v], index[w])
                    k -= 1

                    if lowlink[v]==index[v]:
                        found[v] = True
                        j += 1
                        component[v] = j
                        while l>=1 and index[subqueue[l-1]]>index[v]:
                            w = subqueue[l-1]
                            l -= 1
                            found[w] = True
                            component[w] = j
                    else:
                        subqueue[l] = v
                        l += 1

    return component

def subproblem_index(A, weight):
    B = find_distinct_weights(A)[1:]
    i = np.searchsorted(B, weight, side='right')
    return i

def threshold_edges(A, weight):
    if imported_fortran_module:
        return fortran_module.threshold_matrix(A, weight)
    else:
        B = A.copy()
        B[B>weight] = 0
        return B

#FUNCTIONS FOR PLOOTING

# Define functions.
def statistic(sizes):
    return max(sizes)

def compute_height_to_statistic(height_to_sizes):
    return dict((height, statistic(sizes)) for height, sizes in height_to_sizes.items())

def load_height_to_statistic(edge_list_file, index_gene_file):
    T = load_weighted_edge_list(edge_list_file)
    index_to_gene, gene_to_index = load_index_gene(index_gene_file)
    height_to_sizes = find_height_to_sizes(T, index_to_gene)
    height_to_statistic = compute_height_to_statistic(height_to_sizes)
    return height_to_statistic

def load_height_to_statistic_wrapper(arrs):
    return load_height_to_statistic(*arrs)

def combine_sizes(height_to_size_collection):
    set_heights_collection = [set(x.keys()) for x in height_to_size_collection]
    set_heights = set.union(*set_heights_collection)
    heights_collection = [sorted(x, reverse=True) for x in set_heights_collection]
    heights_union = sorted(set_heights, reverse=True)

    indices = [0 for x in heights_collection]
    lengths = [len(x) for x in heights_collection]

    height_to_sizes = dict()
    for height in heights_union:
        sizes = []
        for i, heights in enumerate(heights_collection):
            while indices[i]+1<lengths[i] and heights[indices[i]+1]>=height:
                indices[i] += 1
            sizes.append(height_to_size_collection[i][heights[indices[i]]])
        height_to_sizes[height] = sizes

    return height_to_sizes

#FUNCTIONS FOR CUTTING HIERARCHY
# Define functions.
def statistic(sizes):
    return max(sizes)

def compute_height_to_statistic(height_to_sizes):
    return dict((height, statistic(sizes)) for height, sizes in height_to_sizes.items())

def find_height_to_sizes(T, index_to_gene, reverse=True):
    '''
    Find composition of clusters for every distinct height of the dendrogram.
    '''
    index_to_index = dict((index, index) for index, gene in index_to_gene.items())
    height_to_clusters = find_height_to_clusters(T, index_to_index, reverse)
    height_to_sizes = dict((height, map(len, clusters)) for height, clusters in height_to_clusters.items())
    return height_to_sizes

def load_height_to_statistic(edge_list_file, index_gene_file):
    T = load_weighted_edge_list(edge_list_file)
    index_to_gene, gene_to_index = load_index_gene(index_gene_file)
    height_to_sizes = find_height_to_sizes(T, index_to_gene)
    height_to_statistic = compute_height_to_statistic(height_to_sizes)
    return height_to_statistic

def load_height_to_statistic_wrapper(arrs):
    return load_height_to_statistic(*arrs)

def combine_sizes(height_to_size_collection):
    set_heights_collection = [set(x.keys()) for x in height_to_size_collection]
    set_heights = set.union(*set_heights_collection)
    heights_collection = [sorted(x, reverse=True) for x in set_heights_collection]
    heights_union = sorted(set_heights, reverse=True)

    indices = [0 for x in heights_collection]
    lengths = [len(x) for x in heights_collection]

    height_to_sizes = dict()
    for height in heights_union:
        sizes = []
        for i, heights in enumerate(heights_collection):
            while indices[i]+1<lengths[i] and heights[indices[i]+1]>=height:
                indices[i] += 1
            sizes.append(height_to_size_collection[i][heights[indices[i]]])
        height_to_sizes[height] = sizes

    return height_to_sizes

def find_max_size_difference(observed_height_to_size, permuted_height_to_size):
    height_to_size_collection = [observed_height_to_size, permuted_height_to_size]

    set_heights_collection = [set(x.keys()) for x in height_to_size_collection]
    set_heights = set.union(*set_heights_collection)
    heights_collection = [sorted(x, reverse=True) for x in set_heights_collection]
    heights_union = sorted(set_heights, reverse=True)

    indices = [0 for x in heights_collection]
    lengths = [len(x) for x in heights_collection]

    height_to_difference = dict()
    for height in heights_union:
        sizes = []
        for i, heights in enumerate(heights_collection):
            while indices[i]+1<lengths[i] and heights[indices[i]+1]>=height:
                indices[i] += 1
            sizes.append(height_to_size_collection[i][heights[indices[i]]])
        difference = float(sizes[0])/float(sizes[1])
        height_to_difference[height] = difference

    max_difference = max(height_to_difference.values())
    height = max(height for height, difference in height_to_difference.items() if difference==max_difference)

    #Modified by Adria Fernandez
    return height, max_difference
    #end

def find_max_size_difference_plot(observed_height_to_size, permuted_height_to_size):
    height_to_size_collection = [observed_height_to_size, permuted_height_to_size]
    set_heights_collection = [set(x.keys()) for x in height_to_size_collection]
    set_heights = set.union(*set_heights_collection)
    heights_collection = [sorted(x, reverse=True) for x in set_heights_collection]
    heights_union = sorted(set_heights, reverse=True)

    indices = [0 for x in heights_collection]
    lengths = [len(x) for x in heights_collection]

    height_to_difference = dict()
    for height in heights_union:
        sizes = []
        for i, heights in enumerate(heights_collection):
            while indices[i]+1<lengths[i] and heights[indices[i]+1]>=height:
                indices[i] += 1
            sizes.append(height_to_size_collection[i][heights[indices[i]]])
        difference = float(sizes[0])/float(sizes[1])
        height_to_difference[height] = difference

    max_difference = max(height_to_difference.values())
    height = max(height for height, difference in height_to_difference.items() if difference==max_difference)

    return height

def find_cut(preordered_T, index_to_gene, threshold, reverse=True):
    '''
    Find clusters for specific height in dendrogram.
    '''
    # Associate nodes of dendrogram with indices.
    index_to_node = defaultdict(set)
    for index, gene in index_to_gene.items():
        index_to_node[index] = frozenset([gene])

    # Initialize clusters with leaf nodes.
    clusters = set(index_to_node.values())

    # Update clusters while ascending dendrogram.
    T = sorted(preordered_T, key=lambda x: x[2], reverse=reverse)
    m = len(T)

    for k, edge in enumerate(T):
        source, target, height = edge
        if (not reverse and height>threshold) or (reverse and height<threshold):
            break

        a = index_to_node[source]
        b = index_to_node[target]
        c = frozenset(set(a) | set(b))

        clusters.discard(a)
        clusters.discard(b)
        clusters.add(c)

        del index_to_node[source]
        index_to_node[target] = c

    return clusters

# Define functions.
def load_components(component_file):
    components = []
    with open(component_file, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                arrs = l.rstrip('\n').split('\t')
                component = sorted(arrs)
                components.append(component)
                break
    return components

#SINGLE HOTNET  & PERMUTED SCORES & CONSTRUCT HIERARCHY & CUT HIERARCHY & PERFORM CONSENSUS
def single_hotnet(score_file,index_file,similarity_matrix,perm,output_folder):

    # Output folder structure

    outfolder = "%s/" % str(uuid.uuid4())
    os.mkdir(outfolder)
    permuted_folder = outfolder + "/permuted/"
    os.mkdir(permuted_folder)
    hierarchies_folder = outfolder + "/hierarchies/"
    os.mkdir(hierarchies_folder)
    hierarchies_permuted_folder = hierarchies_folder + "/permuted/"
    os.mkdir(hierarchies_permuted_folder)

    # Permute scores.

    print "1 - Permuting scores"

    for i in tqdm(xrange(perm)):

        #PERMUTE SCORES
        permute_scores(score_file,index_file,i,permuted_folder + "/perm_scores_%d" % i)

    # Construct hierarchies.
    print "2 - Constructing hierarchies"

    ## ADDED BY ADRIA FERNANDEZ
    score_threshold = 0.1
    ##
    # CONSTRUCT HIREARCHIES
    construct_hierarchy(similarity_matrix, index_file, score_file, score_threshold, hierarchies_folder + "/edge_list.tsv", hierarchies_folder + "/index_gene.tsv")

    for i in tqdm(xrange(perm)):
        construct_hierarchy(similarity_matrix, index_file, permuted_folder + "/perm_scores_%d" % i, score_threshold, hierarchies_permuted_folder + "/edge_list_%d.tsv" % i, hierarchies_permuted_folder + "/index_gene_%d.tsv" % i )

    # Assemble results

    print "3 - Assembling results"

    oelf = hierarchies_folder + "/edge_list.tsv"
    oigf = hierarchies_folder + "/index_gene.tsv"
    #pelf = " ".join(hierarchies_permuted_folder + "/edge_list_%d.tsv" % i for i in xrange(args.perm))
    #pigf = " ".join(hierarchies_permuted_folder + "/index_gene_%d.tsv" % i for i in xrange(args.perm))
    pelf = [hierarchies_permuted_folder + "/edge_list_%d.tsv" % i for i in xrange(perm)]
    pigf = [hierarchies_permuted_folder + "/index_gene_%d.tsv" % i for i in xrange(perm)]
    # Print statistics plot

    print "    Statistics plot"
    cores = 1
    #---Modified by Adria Fernandez
    # I keep the returned diff area
    obs_auc, prm_auc = plot_hierarchy_statistic(oelf, oigf, pelf, pigf, cores, output_folder+'/clusters.pdf')
    hotnet_areas = (obs_auc, prm_auc)
    #---end----

    # Cut hierarchy

    print "    Cutting hierarchy"
    #---Modified by Adria Fernandez
    # I introduce a new variable "diff area" to be returned in the output
    cut_hierarchy(oelf,oigf,pelf,pigf,hotnet_areas,output_folder+'/clusters.tsv',num_cores=1,verbose=False)
    #---end----
    # Remove folder

    print "    Cleaning"

    cmd = "rm -r %s" % outfolder
    subprocess.Popen(cmd, shell = True).wait()

def cut_hierarchy(observed_edge_list_file,observed_index_gene_file,permuted_edge_list_files,permuted_index_gene_files,hotnet_areas,output_file,num_cores=1,verbose=False):

    # Load data.
    if verbose:
        progress('Loading data...')

    assert len(permuted_edge_list_files)==len(permuted_index_gene_files)

#    if args.num_cores!=1:
#        pool = mp.Pool(None if args.num_cores==-1 else args.num_cores)
#        map_fn = pool.map
#    else:
    map_fn = map

    map_input = [(observed_edge_list_file, observed_index_gene_file)] + list(zip(permuted_edge_list_files, permuted_index_gene_files))
    map_output = list(map_fn(load_height_to_statistic_wrapper, map_input))

    observed_height_to_statistic = map_output[0]
    permuted_height_to_statistic_collection = map_output[1:]

#    if args.num_cores!=1:
#        pool.close()
#        pool.join()

    # Process data.
    if verbose:
        progress('Processing data...')

    permuted_height_to_sizes = combine_sizes(permuted_height_to_statistic_collection)
    permuted_height_to_average_size = dict((height, np.mean(sizes)) for height, sizes in permuted_height_to_sizes.items())

    #---Modified by Adria Fernandez
    height,diff_delta = find_max_size_difference(observed_height_to_statistic, permuted_height_to_average_size)
    obs_auc, prm_auc = hotnet_areas
    #---end----

    statistic = observed_height_to_statistic[height]

    T = load_weighted_edge_list(observed_edge_list_file)
    index_to_gene, gene_to_index = load_index_gene(observed_index_gene_file)
    clusters = find_cut(T, index_to_gene, height)

    # Save data.
    if verbose:
        progress('Saving data...')

    permuted_height = min(x for x in permuted_height_to_sizes.keys() if x>=height)
    permuted_sizes = permuted_height_to_sizes[permuted_height]
    p_value = float(sum(1 for x in permuted_sizes if x>=statistic))/float(len(permuted_sizes))

    #---Modified by Adria Fernandez
    header_string = '#delta: {}\n#Statistic: {}\n#p-value: {}\n#Diff: {}\n#Observed_auc: {}\n#Permuted_auc: {}'.format(height, statistic, p_value,diff_delta,obs_auc,prm_auc)
    #---end----

    sorted_clusters = sorted(sorted(map(sorted, clusters)), key=len, reverse=True)
    output_string = '\n'.join('\t'.join(cluster) for cluster in sorted_clusters)

    with open(output_file, 'w') as f:
        f.write(header_string + '\n' + output_string)

    if verbose:
        progress()

def plot_hierarchy_statistic(observed_edge_list_file, observed_index_gene_file, permuted_edge_list_files, permuted_index_gene_files,num_cores ,output_file, label='',verbose=False):
    #---added by Adria Fernandez -----
    import numpy as np
    from sklearn.metrics import auc
    #-----

    if verbose:
        progress('Loading data...')

    assert len(permuted_edge_list_files)==len(permuted_index_gene_files)

#    if args.num_cores!=1:
#        pool = mp.Pool(None if args.num_cores==-1 else args.num_cores)
#        map_fn = pool.map
#    else:
    map_fn = map

    map_input = [(observed_edge_list_file, observed_index_gene_file)] + list(zip(permuted_edge_list_files, permuted_index_gene_files))
    map_output = list(map_fn(load_height_to_statistic_wrapper, map_input))

    observed_height_to_statistic = map_output[0]
    permuted_height_to_statistic_collection = map_output[1:]

#   if args.num_cores!=1:
#       pool.close()
#       pool.join()

    # Process data.
    if verbose:
        progress('Processing data...')

    permuted_height_to_sizes = combine_sizes(permuted_height_to_statistic_collection)
    permuted_height_to_min_size = dict((height, min(sizes)) for height, sizes in permuted_height_to_sizes.items())
    permuted_height_to_average_size = dict((height, np.mean(sizes)) for height, sizes in permuted_height_to_sizes.items())
    permuted_height_to_max_size = dict((height, max(sizes)) for height, sizes in permuted_height_to_sizes.items())

    observed_heights = np.sort(list(observed_height_to_statistic.keys()))
    permuted_average_heights = np.sort(list(permuted_height_to_average_size.keys()))
    selected_height = find_max_size_difference_plot(observed_height_to_statistic, permuted_height_to_average_size)


    # Plot results.
    if verbose:
        progress('Plotting data...')

    ### Plot sizes.
    for height_to_statistic in permuted_height_to_statistic_collection:
        heights, statistics = zip(*sorted(height_to_statistic.items()))
        heights = [1.0/height if height>0 else float('inf') for height in heights]
        heights = [height if height<float('inf') else 1e10 for height in heights]
        plt.step(heights, statistics, c='b', linewidth=0.5, alpha=0.1)

    heights, statistics = zip(*sorted(permuted_height_to_min_size.items()))
    heights = [1.0/height if height>0 else float('inf') for height in heights]
    heights = [height if height<float('inf') else 1e10 for height in heights]
    plt.step(heights, statistics, c='b', lw=1, alpha=0.3, linestyle=':')

    heights, statistics = zip(*sorted(permuted_height_to_max_size.items()))
    heights = [1.0/height if height>0 else float('inf') for height in heights]
    heights = [height if height<float('inf') else 1e10 for height in heights]
    plt.step(heights, statistics, c='b', lw=1, alpha=0.3, linestyle=':')

    heights, statistics = zip(*sorted(observed_height_to_statistic.items()))
    heights = [1.0/height if height>0 else float('inf') for height in heights]
    heights = [height if height<float('inf') else 1e10 for height in heights]
    plt.step(heights, statistics, c='r', lw=1, label='Observed', zorder=5)
    #---added by Adria Fernandez -----
    #Calculate AUC
    obs_auc  = auc([np.log(x) if x>1 else 0 for x in heights],[np.log(x) if x>1 else 0 for x in statistics])
    #---

    heights, statistics = zip(*sorted(permuted_height_to_average_size.items()))
    heights = [1.0/height if height>0 else float('inf') for height in heights]
    heights = [height if height<float('inf') else 1e10 for height in heights]
    plt.step(heights, statistics, c='b', lw=1, label='Permuted', zorder=4)
    #---added by Adria Fernandez -----
    #Calculate AUC
    prm_auc  = auc([np.log(x) if x>1 else 0 for x in heights],[np.log(x) if x>1 else 0 for x in statistics])
    #---

    ### Plot difference between observed and average permuted sizes at selected height.
    observed_larger_height = min(observed_heights[observed_heights>=selected_height])
    permuted_larger_height = min(permuted_average_heights[permuted_average_heights>=selected_height])

    plt.plot((1.0/selected_height, 1.0/selected_height),
             (permuted_height_to_average_size[permuted_larger_height], observed_height_to_statistic[observed_larger_height]),
             c='k', ls='--', alpha=0.75, zorder=6)

    ### Set plot properties, save plot, etc.
    plt.xlim(0.8*min(1/x for x in observed_height_to_statistic if 0.0<x<float('inf')), 1.2*max(1/x for x in observed_height_to_statistic if 0.0<x<float('inf')))
    plt.ylim(0.8, 1.2*max(observed_height_to_statistic.values()))
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel(r'$1/\delta$')
    plt.ylabel(r'Statistic')
    if label:
        plt.title(r'Test statistic across $\delta$ thresholds for' + '\n' + r'{}'.format(' '.join(label)))

    legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.125), ncol=2)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.tight_layout()

    if verbose:
        progress('Saving data...')

    if output_file.endswith('.png'):
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(output_file, bbox_inches='tight')
    plt.close()

    if verbose:
        progress()

    #---added by Adria Fernandez -----
    #Calculate area difference
    return obs_auc,prm_auc
    #----

def construct_hierarchy(similarity_matrix_file,index_gene_file, gene_score_file, score_threshold,hierarchy_edge_list_file, hierarchy_index_gene_file,similarity_matrix_name='PPR', verbose=False):

    # Load data.
    if verbose:
        print('Loading data...')

    P = load_matrix(similarity_matrix_file, similarity_matrix_name)
    index_to_gene, gene_to_index = load_index_gene(index_gene_file)
    gene_to_score = load_gene_score(gene_score_file, score_threshold)

    # Process data.
    if verbose:
        print('Processing data...')

    S, common_index_to_gene, common_gene_to_index = combined_similarity_matrix(P, gene_to_index, gene_to_score)

    # If the digraph associated with S is not strongly connected, then restrict to a largest strongly connected component.
    components = strongly_connected_components(S)
    if len(components)>1:
        component = sorted(max(components, key=len))
        S = S[np.ix_(component, component)]
        common_index_to_gene = dict((i+1, common_index_to_gene[j+1]) for i, j in enumerate(component))
        common_gene_to_index = dict((gene, i) for i, gene in common_index_to_gene.items())

    # Construct hierarchical decomposition.
    if verbose:
        print('Constructing hierarchical decomposition...')

    T = tarjan_HD(S, reverse=True, verbose=verbose)

    # Save results.
    if verbose:
        print('Saving results...')

    save_weighted_edge_list(hierarchy_edge_list_file, T)
    save_index_gene(hierarchy_index_gene_file, common_index_to_gene)


def permute_scores(gene_score_file,index_gene_file,seed,output_file):

    # Load edges.
    gene_to_score = load_gene_score(gene_score_file)
    score_genes = set(gene_to_score)
    if index_gene_file:
        index_to_gene, gene_to_index = load_index_gene(index_gene_file)
        network_genes = set(gene_to_index)
    else:
        network_genes = score_genes

    # Permute scores.
    genes = sorted(score_genes)
    scores = np.array([gene_to_score[gene] for gene in genes])

    np.random.seed(seed)
    permute_indices = [i for i, gene in enumerate(genes) if gene in network_genes]   # Only permute scores for genes in network if given.
    scores[permute_indices] = np.random.permutation(scores[permute_indices])

    gene_to_score = dict((gene, score) for gene, score in zip(genes, scores))

    # Save permuted_scores.
    save_gene_score(output_file, gene_to_score)

def perform_consensus(component_files,index_gene_files,edge_list_files,networks,scores,threshold,output_file,verbose=False):
    # Load data.
    if verbose:
        progress('Loading data...')

    assert len(component_files)==len(index_gene_files)==len(edge_list_files)==len(networks)==len(scores)

    index_to_gene_collection = dict()
    gene_to_index_collection = dict()
    edge_list_collection = dict()
    components_collection = dict()

    for network_label, score_label, index_gene_file, edge_list_file, component_file in zip(networks, scores, index_gene_files, edge_list_files, component_files):
        index_to_gene, gene_to_index = load_index_gene(index_gene_file)
        edge_list = set(frozenset((index_to_gene[i], index_to_gene[j])) for i, j in load_edge_list(edge_list_file))
        components = load_components(component_file)

        index_to_gene_collection[(network_label, score_label)] = index_to_gene
        gene_to_index_collection[(network_label, score_label)] = gene_to_index
        edge_list_collection[(network_label, score_label)] = edge_list
        components_collection[(network_label, score_label)] = components

    # Process data.
    if verbose:
        progress('Processing data...')

    edge_to_networks = defaultdict(set)
    edge_to_scores = defaultdict(set)
    edge_to_pairs = defaultdict(set)
    edge_to_tally = defaultdict(int)

    for network_label, score_label in zip(networks, scores):
        edge_list = edge_list_collection[(network_label, score_label)]
        components = components_collection[(network_label, score_label)]
        for component in components:
            for u, v in combinations(component, 2):
                edge = frozenset((u, v))
                if edge in edge_list:
                    edge_to_tally[edge] += 1

    thresholded_edges = set(edge for edge, tally in edge_to_tally.items() if tally>=threshold)

    G = nx.Graph()
    G.add_edges_from(thresholded_edges)
    consensus_results = sorted(sorted([sorted(x) for x in nx.connected_components(G)]), key=len, reverse=True)

    # Save data.
    if verbose:
        progress('Saving data...')

    output_string = '\n'.join('\t'.join(x) for x in consensus_results)
    with open(output_file, 'w') as f:
        f.write(output_string)

    if verbose:
        progress()

#********************************************************************#
if __name__=='__main__':

    # Variables

    python_path = sys.executable
    hotnet_path =  "../src/"

    try:
        import fortran_module
    except:
        cwd = os.getcwd()
        os.chdir(hotnet_path)
        subprocess.Popen("f2py -c fortran_module.f95 -m fortran_module > /dev/null", shell=True).wait()
        print "fortran compiled"
        os.chdir(cwd)

    if 'fortran_module' in sys.modules:
        imported_fortran_module = True
    else:
        imported_fortran_module = False

    args = get_parser().parse_args(sys.argv[1:])
    # Generate folders and a scores file where everything else is 0, and scores not present in the interactome are removed.

    if not os.path.exists(args.output_folder): os.mkdir(args.output_folder)

    def get_dirname(netfold, scorefile):
        net = netfold.rstrip("/").split("/")[-1]
        score = os.path.splitext(os.path.basename(scorefile))[0]
        dirname = args.output_folder + "/" + net + "_" + score + "/"
        return dirname

    for netfold in args.network_folders:
        nodes = set()
        with open(netfold + "/idx2node.tsv", "r") as f:
            for l in f: nodes.update([l.rstrip("\n").split("\t")[1]])

        for scorefile in args.scores_files:
            dirname = get_dirname(netfold, scorefile)
            if os.path.exists(dirname):
                if args.erase:
                    shutil.rmtree(dirname)
                else:
                    continue
            os.mkdir(dirname)
            scores = {}
            with open(scorefile, "r") as f:
                for l in f:
                    l = l.rstrip("\n").split("\t")
                    scores[l[0]] = float(l[1])
            f = open(dirname + "/scores.tsv", "w")
            for k in sorted(scores, key=scores.get, reverse=True):
                if k not in nodes: continue
                f.write("%s\t%.3f\n" % (k, scores[k]))
            for n in sorted(nodes.difference(scores.keys())):
                f.write("%s\t%.3f\n" % (n, args.baseline_score))
            f.close()

    # Run single hotnets

    for netfold in args.network_folders:
        for scorefile in args.scores_files:
            dirname = get_dirname(netfold, scorefile)
            scorefile = dirname + "/scores.tsv"
            if args.erase or not os.path.exists(dirname+"/clusters.tsv"):
                #SINGLE HOTNET
                single_hotnet(scorefile, netfold+"/idx2node.tsv", netfold+"/similarity_matrix.h5", args.perm, dirname)
                print "SINGLE HOTNET RUN: %s" % os.path.basename(dirname.rstrip("/"))
    # Run consensus

    print "Consensus results"

    cf, igf, elf, n, s = [], [], [], [], []

    for netfold in args.network_folders:
        for scorefile in args.scores_files:
            dirname = get_dirname(netfold, scorefile)
            cf += [dirname+"/clusters.tsv"]
            igf += [netfold+"/idx2node.tsv"]
            elf += [netfold+"/edgelist.tsv"]
            n += [dirname.split("_")[0]]
            s += [dirname.split("_")[1]]

    perform_consensus(cf,igf,elf,n,s,args.consensus_t,args.output_folder+"/consensus.tsv")
