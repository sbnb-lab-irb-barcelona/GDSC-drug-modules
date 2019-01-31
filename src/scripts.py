import sys
import os
import copy as cp
import random
import numpy as np
import pandas as pd
import subprocess
import codecs
from scipy import stats
from collections import Counter
from tqdm import tqdm
import networkx as nx
from diamond import DIAMOnD

def make_output_folders():
    folders = ['../results/','../results/diamond/','../results/diamond/PCM','../results/diamond/NCM','../results/hotnet/','../results/hotnet/PCM','../results/hotnet/','../results/reactome_gsea','../results/final_modules']
    for folder in folders:
        try:
            os.mkdir(folder)
        except:
            continue
def get_correlations(expr_df,drug_df,write=None):

    stack = []
    drugs = sorted(drug_df.columns)
    genes = sorted(expr_df.columns)
    #Iterating across drugs
    for drug in tqdm(drugs):
        drug_vector = drug_df[drug]

        #removing nan cls
        nan_cl = pd.isnull(drug_vector)
        drug_vector = list(drug_vector[~nan_cl])
        X_expr = expr_df.loc[~nan_cl]

        #Iterating across genes
        for gene in genes:
            gene_vector = X_expr[gene]

            #Pearson correlation
            r,pval = stats.pearsonr(gene_vector,drug_vector)
            N = len(gene_vector)
            SE = 1/(np.sqrt(N-3))

            stack.append([drug,gene,r,SE])
    stack = np.asarray(stack,dtype=object)

    #Normalizing the correlactions
    stack = np.c_[stack,np.arctanh(list(stack[:,2]))]

    #Getting Z-corr and building the matrix
    n_genes = len(genes)
    M = []
    for i in np.arange(0,len(stack),n_genes):
        v = stack[i:i+n_genes]
        M.append(v[:,4]/v[:,3])

    df = pd.DataFrame(M,columns=genes,index=drugs)
    if write is not None:
        df.to_csv(write)
    return df

def find_zscore_cutoff(corr_df,prop=0.025,cutoff_window = (2,5),step=0.1):

    cutoff_range = np.asarray([round(x,2) for x in np.arange(cutoff_window[0],cutoff_window[1]+step,step)])
    corr_vc = corr_df.values.ravel()

    prop_gn_pairs_up = []
    prop_gn_pairs_dw = []
    for cutoff in cutoff_range:
        up = len(np.where(corr_vc > cutoff)[0])/len(corr_vc)
        dw = len(np.where(corr_vc < -cutoff)[0])/len(corr_vc)
        prop_gn_pairs_up.append(up)
        prop_gn_pairs_dw.append(dw)

    prop_gn_pairs_up = np.asarray(prop_gn_pairs_up)
    prop_gn_pairs_dw = np.asarray(prop_gn_pairs_dw)

    ix_0025_r = min(np.where(prop_gn_pairs_up <=prop)[0])
    ix_0025_l = min(np.where(prop_gn_pairs_dw <=prop)[0])
    zscore_r = cutoff_range[ix_0025_r]
    zscore_l = cutoff_range[ix_0025_l]
    return zscore_l,zscore_l

def get_drug2gene_correlations(corr_df,cutoff):
    #Getting up/down genes
    genes = np.asarray(corr_df.columns)
    up_genes = {}
    dw_genes = {}
    for drug in corr_df.index.values:
        up_ix = np.where(corr_df.loc[drug]>=cutoff)[0]
        dw_ix = np.where(corr_df.loc[drug]<=-cutoff)[0]
        up_genes[drug] = set(genes[up_ix])
        dw_genes[drug] = set(genes[dw_ix])
    return up_genes, dw_genes

def get_drug_count_matrix(corr_df,drug2genes_corr):
    a = []
    for drug,genes in drug2genes_corr.items():
        a+= list(genes)

    gene_drug_count = Counter(a).most_common()
    my_set = set(np.array(gene_drug_count)[:,0])
    [gene_drug_count.append((x,0)) for x in tqdm(corr_df.columns) if x not in my_set]
    return np.array(gene_drug_count,dtype=object)

def get_fcg_set(gene_drug_count,prop):
    cs = np.cumsum(np.bincount(sorted(gene_drug_count[:,1], reverse =False)))
    cutoff = np.searchsorted(cs,cs[-1]*prop)
    return set([x[0] for x in gene_drug_count if x[1]>=cutoff])

def get_fcg(corr_df,drug2genes_corr,prop=0.95,write=None):
    gene_drug_count = get_drug_count_matrix(corr_df,drug2genes_corr)
    fcg = get_fcg_set(gene_drug_count,prop)
    if write is not None:
        with open(write,'w') as o:
            for g in fcg:
                o.write('%s\n'%g)
    return fcg

#GSEA Functions
def get_ensbl2uniAC(path='../data/ensbl2AC.tsv'):
    ensbl2uniAC = {}
    with open(path,'r') as f:

        for line in f:
            h = line.rstrip('\n').split('\t')
            if h[0] not in ensbl2uniAC:
                ensbl2uniAC[h[0]] = set([])
            ensbl2uniAC[h[0]].add(h[1])
    return ensbl2uniAC

def map_and_return_universe(v,mapping_dict):
    u = set([])
    for x in v:
        try:
            u.update(mapping_dict[x])
        except KeyError:
            continue
    return u

def get_reactome_genesets():
    d = {}
    with codecs.open('../data/reactome.tsv','r',encoding='utf8') as f:
        f.readline()
        for l in f:
            r,n,gs = tuple(l.rstrip('\n').split('\t'))
            d[r] = set(gs.split(','))

    return d

def map_corr_df_to_uniprot(corr_df,the_universe):
    ensbl2uniAC = get_ensbl2uniAC()
    m = []
    genes = []
    for gn in corr_df:
        v = list(corr_df[gn])

        try:
            ups = ensbl2uniAC[gn] & the_universe
        except KeyError:
            continue

        for u in ups:
            m.append(v)
            genes.append(u)
    return pd.DataFrame(m,index=genes,columns=corr_df.index.values).T

def es_score(idxs, ref_expr, p = 1):

    if len(idxs) < 10: return 0.
    N  = len(ref_expr)
    Nh = len(idxs)
    norm = 1. / (N - Nh)
    miss = np.empty(N)
    miss[:] = norm
    miss[idxs] = 0.
    hit_nums = np.zeros(N)
    for i in idxs:
        hit_nums[i] = np.abs(ref_expr[i])**p
    hit = hit_nums / np.sum(hit_nums)
    P_hit = hit
    P_miss = miss
    P_hit  = np.cumsum(hit)
    P_miss = np.cumsum(miss)
    ES = P_hit - P_miss

    return ES[np.argmax(np.abs(ES))]

def gsea(geneset, gene_expr_matrix, p = 1000):

    #Keeping the idxs where the geneset is and the sorted expr
    idxs = [idx for idx, x in enumerate(gene_expr_matrix) if x[0] in geneset]
    ref_expr = np.array([x[1] for x in gene_expr_matrix])

    # ES
    es = es_score(idxs, ref_expr)

    # Pvalue
    c = 0
    for _ in range(p):
        idx_rand = [random.randint(0, len(ref_expr) - 1) for _ in range(len(idxs))]
        es_rand = es_score(idx_rand,ref_expr)
        if es > 0:
            if es_rand >= es:
                c += 1
        else:
            if es_rand <= es:
                c += 1
    pval = c / p

    return es, pval

def run_gsea(sorted_matrix,genesets,output_path):

    # Interating across pathways
    with open(output_path,'w') as out:
        out.write('pathID\tES\tpvalue\n')
        for pathway,geneset in genesets.items():

            #Running GSEA
            es,pval = gsea(geneset,sorted_matrix)

            #Writing output
            out.write('%s\t%.4f\t%.4f\n'%(pathway,es,pval))

def get_drug2pwy(path = '../results/reactome_gsea/',pvalue_cutoff = 0.01):
    d2pwy = {}
    for file in [path+x for x in os.listdir(path)]:
        drug = file.split('/')[-1][:-4]
        d2pwy[drug] = set([])
        with open(file,'r') as f:
            f.readline()
            for l in f:
                h = l.rstrip().split('\t')
                if float(h[2]) < pvalue_cutoff:
                    d2pwy[drug].add(h[0])
    return d2pwy

#HotNet Functions
def write_hotnet_inputs(drug2genes,corr_df,drug2pwy,reactome_genesets,network_set,output_path,abs_res=True):

    ensbl2uniAC = get_ensbl2uniAC()
    drug_list = sorted(drug2genes)

    for drug in drug_list:
        genes = sorted(drug2genes[drug])

        # 1) Getting drug matrix (gene, score)
        my_matrix = list(zip(genes,list(corr_df.loc[drug,genes])))

        # 2) Getting Gene universe
        # -- Genes in Enriched pathways
        if drug2pwy is not None and reactome_genesets is not None:
            gsea_genes = set([])
            for pwy in drug2pwy[drug]:
                gsea_genes.update(reactome_genesets[pwy])
            #-- Getting intersection gsea_genes & network_set
            gene_universe = gsea_genes & network_set
        else:
            gene_universe = network_set

        # 3) Mapping from ensemble to unip and applying Gene Universe
        out_matrix = []
        for gene,score in my_matrix:
            if abs_res is True:
                score = abs(float(score))
            if gene not in ensbl2uniAC: continue

            unips = ensbl2uniAC[gene] & gene_universe

            for unip in unips:
                out_matrix.append([unip,score])
        out_matrix = sorted(out_matrix, key=lambda x: x[1],reverse=True)
        # 4) Writing
        with open(output_path+'/%s.tsv'%drug,'w') as out:
            for row in out_matrix:
                gene = row[0]

                score = row[1]
                out.write('%s\t%s\n'%(gene,score))


def run_iteratively_hotnet(input_file_path,output_path,p=100,min_size=5,n_iter=10):

    drug = input_file_path.split('/')[-1][:-4]


    #Check if there are score for the given file
    my_input = []
    with open(input_file_path,'r') as f:
        for line in f:
            hit = line.rstrip().split('\t')
            my_input.append(hit)

    my_genes = set([x[0] for x in my_input])

    if len(my_input) < min_size:
        #sys.exit('There are less than 10 genes for this file (%i). Process interrumped.'%len(my_input))
        return None
    network = '../data/string/'

    #Starting the iterations
    for it in range(n_iter):
        it +=1

        #First itertation
        if it == 1:

            #Making the output folders
            if not os.path.isdir(output_path):
                os.mkdir(output_path)
            if not os.path.isdir(output_path+'/1/'):
                 os.mkdir(output_path+'/1/')

            outfolder = output_path+"/1/"

        else:
            if it == 2:
                #reading previous results
                with open(outfolder+'/string_%s/clusters.tsv'%(drug),'r') as f:
                    #stats
                    f.readline()
                    f.readline()
                    f.readline()
                    f.readline()
                    f.readline()
                    f.readline()

                    #First module
                    module = f.readline().rstrip().split('\t')
                    if len(module) < min_size:
                        #sys.exit('The best module of the last iteration have less than %i genes. Finishing recursive_hotnet'%min_size)
                        return None
                input_file_path = input_file_path[:-4] + '+%i.tsv'%it

            elif it > 2:
                #reading previous results
                with open(outfolder+'/string_%s+%i/clusters.tsv'%(drug,it-1),'r') as f:

                    #stats
                    f.readline()
                    f.readline()
                    f.readline()
                    f.readline()
                    f.readline()
                    f.readline()

                    #First module
                    module = f.readline().rstrip().split('\t')
                    if len(module) < min_size:
                        #sys.exit('The best module of the last iteration have less than %i genes. Finishing recursive_hotnet'%min_size)
                        return None

                input_file_path = input_file_path[:-5] + '%i.tsv'%it

            #Removing genes from input_file_path
            my_genes = my_genes.difference(set(module))

            if len(my_genes) < min_size:
                #sys.exit('There are less than %i genes for this file (%i). Process interrumped.'%(min_size,len(my_input)))
                return None
            #Writing new input file
            with open(input_file_path,'w') as o:
                for row in my_input:
                    gene = row[0]
                    score = row[1]
                    if gene in my_genes:
                         o.write('%s\t%s\n'%(gene,score))

            #Making new folder
            if not os.path.isdir(output_path+'/%i/'%it):
                os.mkdir(output_path+'/%i/'%it)
            outfolder = output_path+"/%i/"%it

        #Executing HotNet
        cmd = "python ../src/run_hotnet.py -p %s -n %s -s %s -o %s" % (p,network, input_file_path, outfolder)

        subprocess.Popen(cmd, shell = True).wait()
        #print('Iteration: %i\n'%it)

#DIAMOnD
def diamond_list(G, seeds, diamond_max_added_nodes = 200, diamond_alpha = 10):
    """Given a network and a set of nodes(seeds) it returns a list of nodes (until max_nodes) sorted based on
    the DIAMOnD algorithm

    ArgKeys:
    G -- network in networkx.Graph() format
    seeds -- list of interesting nodes that are forming your initial subset
    diamond_max_added_nodes -- (default=200) max number of nodes included in the output list (see paper)
    diamond_alpha -- (default=1) DIAMOnD parameter that gives weights to the former seeds (see paper)
    """
    nodes = []
    if not seeds: return []
    added_nodes = DIAMOnD(G, seeds, diamond_max_added_nodes, diamond_alpha)
    for DIAMOnD_node_info in added_nodes:
        DIAMOnD_node = DIAMOnD_node_info[0]
        p = float(DIAMOnD_node_info[3])
        nodes += [DIAMOnD_node]
    return nodes

def diamond_distance(node, mod, diamond_list):
    """Given a specific node,module and diamond_list returns the distance of the node from the module

    ArgKeys:
    node -- specific node in which we are interesting in. It could be a protein, a gene, etc., depending on the network
    mod -- list of nodes that are part of the module
    diamond_list -- list of nodes sorted using the DIAMOnD algorithm
    """
    if not mod: return np.nan
    if node in mod: return 0
    if node not in diamond_list: return len(diamond_list)
    return diamond_list.index(node) + 1

def get_diamond_dist(module_dict,G,out_path,diamond_max_nodes,diamond_alpha):

    for drug, mds in module_dict.items():
        with open(out_path+'/%s.txt'%drug,'w') as o:
            r = []
            for md in mds:
                d_list = diamond_list(G, md, diamond_max_added_nodes = diamond_max_nodes, diamond_alpha = diamond_alpha)
                r.append(','.join(d_list))
            o.write('\n'.join(r))

def read_network(network_file):
    G = nx.Graph()
    if type(network_file) == str:
        rows = []
        with open(network_file,'r') as f:
            for line in f:
                row = line.rstrip().split('\t')
                node1 = row[0]
                node2 = row[1]
                G.add_edge(node1,node2)
    elif type(network_file) == list or type(network_file) == np.ndarray:
        for line in network_file:
            node1 = line[0]
            node2 = line[1]
            G.add_edge(node1,node2)
    else:
        for line in network_file:
            line_data   = line.strip().split('\t')
            node1 = line_data[0]
            node2 = line_data[1]
            G.add_edge(node1,node2)

    return G
def retrieve_iter_hotnet(path,sample=None,min_size = 10,max_modules=2,min_pvalue=0.05,detailed=False):

    network = 'string'
    my_dict = {}
    if sample is None:
        samples = os.listdir(path)
    else:
        samples = [sample]

    stack = []

    for sample in samples:
        iterations = sorted([int(x) for x in os.listdir(path+'/'+sample)])
        my_dict[sample] = []

        for it in iterations:
            if it > max_modules: break #stop retriving modules
            if it == 1:
                file = '%s/%s/%i/%s_%s/clusters.tsv'%(path,sample,it,network,sample)
            else:
                file = '%s/%s/%i/%s_%s+%i/clusters.tsv'%(path,sample,it,network,sample,it)

            try:
                with open(file,'r') as f:
                    f.readline()
                    f.readline()
                    pval = float(f.readline().rstrip().split(':')[-1].strip())
                    f.readline()
                    f.readline()
                    f.readline()

                    #module
                    if min_pvalue:
                       if pval <= min_pvalue:
                           module = f.readline().rstrip().split('\t')
                       else:
                           break
                    else:
                        module = f.readline().rstrip().split('\t')

                    if len(module) < min_size:
                        break
                    else:
                        my_dict[sample].append(module)

            except FileNotFoundError:
                print("No such file or directory: '%s'"%file)
                continue

        if len(my_dict[sample]) == 0:
            del my_dict[sample]

    return my_dict


def run_diamond(module_path,out_path,sample=None,alpha=10,max_nodes=200):
    module = retrieve_iter_hotnet(module_path,sample=sample)
    if len(module) == 0: return None
    string = read_network('../data/string/interactions.tsv')
    get_diamond_dist(module,string,out_path=out_path,diamond_max_nodes=max_nodes,diamond_alpha=alpha)

def read_diamond_res(path,md=None):

    res = {}
    files = [path + x for x in os.listdir(path) if x.endswith('.txt')]
    for file in files:
        drug = file.split('/')[-1][:-4]
        r = []
        with open(file,'r') as f:
            for l in f:
                r.append(l.rstrip('\n').split(','))
        res[drug] = r

    return res


def add_diamond_genes(module_dict,dlist,corr_dict,dlist_cutoff=200,map_corr=True):

    ensbl2uniAC = get_ensbl2uniAC()

    new_modules = {}

    for d in module_dict:
        new_modules[d] = []
        alreadies = set([])
        if map_corr is True:
            candidates = set([y for x in corr_dict[d] if x in ensbl2uniAC for y in ensbl2uniAC[x]])
        else:
            candidates = set(corr_dict[d])

        for ix,md in enumerate(module_dict[d]):
            gns = set(md) | (set(dlist[d][ix][:dlist_cutoff]) & candidates)
            gns = gns.difference(alreadies)

            new_modules[d].append(sorted(gns))
            alreadies.update(gns)
    return new_modules
