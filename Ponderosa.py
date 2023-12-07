import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import argparse
import os
from collections import Counter
from math import floor, ceil
import itertools as it
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

from pedigree_tools import ProcessSegments, PedigreeHierarchy, SiblingClassifier, TrainPonderosa, introduce_phase_error

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ibd", help = "IBD segment file. If multiple files for each chromosome, this is the path for chromosome 1.", required=True)
    parser.add_argument("-fam", help="PLINK-formated .fam file")
    parser.add_argument("-ages", help="Age file. First column is the IID, second column is the age", default="")
    parser.add_argument("-map", help = "PLINK-formatted .map file.", default="")
    parser.add_argument("-training", help = "Path and name of the 'degree' pkl file for training.", default="")
    parser.add_argument("-populations", help="Path and file name of .txt file where col1 is the IID and col2 is their population.", default="")
    parser.add_argument("-population", help="If -populations is used, this will specify which population to run Ponderosa on.", default="pop1")
    parser.add_argument("-out", help = "Output prefix.", default="Ponderosa")
    parser.add_argument("-min_p", help="Minimum posterior probability to output the relationship.", default=0.8, type=float)
    parser.add_argument("-n_pairs", help="Max number of pairs to look at. Default: all", default=np.inf, type=float)
    parser.add_argument("-debug", action="store_true")
    args = parser.parse_args()
    return args

''' Ponderosa can take as input a file (e.g., containing IBD segments) that has all chromosomes
represented, in which case it does not expect chr1 to be in the file name. Otherwise, if the files
are split by chromosome, the user should supply the path and file name for chromosome 1, in which
case the chromosome number should be denoted as chr1'''
def get_file_names(file_name):

    if "chr1" in file_name:
        return [file_name.replace("chr1", f"chr{chrom}") for chrom in range(1, 23)]
    else:
        return [file_name]
    


''' SampleData is a class that holds a networkx graph where nodes are individuals
and edges describe the relationship between those individuals. It can be queried easily
using the get_edges and get_nodes functions, which allow the user to supply functions
that act as conditionals for returning node pairs and nodes, resp.'''
class SampleData:

    def __init__(self, fam_file, **kwargs):

        ### load the fam file
        fam = pd.read_csv(fam_file, delim_whitespace=True, dtype=str, header=None)

        ### the default information for adding a person
        self.default = {"sex": 0, "mother": -1, "father": -1, "children": [], "age": np.nan, "popl": "pop1"}

        ### init the graph and set defaults
        g = nx.Graph()
        g.add_nodes_from(fam.apply(lambda x: (x[1], {"sex": x[4],
                                                    "mother": x[3] if x[3] != "0" else -1,
                                                    "father": x[2] if x[2] != "0" else -1,
                                                    "children": [],
                                                    "age": np.nan,
                                                    "popl": "pop1"}), axis=1).values)

        ### get the population
        popln = kwargs.get("population", "pop1")

        # add the children
        for sex_col in [2, 3]:
            # assume that missing parent is coded as 0
            for parent, children_df in fam[fam[sex_col] != "0"].groupby(sex_col):
                # get a list of the children
                children = children_df[1].values.tolist()
                # add to attribute
                nx.set_node_attributes(g, {parent: children}, "children")

        # add the population data
        pop_file = kwargs.get("pop_file", "")
        if os.path.exists(pop_file):
            pops = pd.read_csv(pop_file, delim_whitespace=True, dtype=str, header=None)
            pop_attrs = {iid: pop for iid, pop in pops[[0,1]].values}
            nx.set_node_attributes(g, pop_attrs, "popl")
        else:
            print("No population file has been provided.")

        # add age data
        age_file = kwargs.get("age_file", "")
        if os.path.exists(age_file):
            ages = pd.read_csv(age_file, delim_whitespace=True, dtype={0: str, 1: float}, header=None)
            age_attrs = {iid: age for iid, age in ages[[0,1]].values}
            nx.set_node_attributes(g, age_attrs, "age")
        else:
            print("No age data has been provided.")

        ### get the genome len
        map_file = kwargs.get("map_file", "")
        if os.path.exists(map_file):
            map_files = get_file_names(map_file)

            map_df = pd.concat([pd.read_csv(filen, delim_whitespace=True, header=None) for filen in map_files])

            genome_len = 0
            for chrom, chrom_df in map_df.groupby([0]):
                genome_len += (chrom_df.iloc[-1][2] - chrom_df.iloc[0][2])
        # no map files provided, use default genome length
        else:
            print("No map files have been provided. Assuming genome length of 3545 cM.")
            genome_len = 3545
                # store the genome_len

        self.genome_len = genome_len

        ### load the king file
        king_file = kwargs.get("king_file", "")
        if os.path.exists(king_file):
            king_df = pd.read_csv(king_file, delim_whitespace=True, dtype={"ID1": str, "ID2": str})

            # get set of MZ twins; remove the 2nd twin in the pair
            mz_twins = set(king_df[king_df.InfType=="Dup/MZ"]["ID2"].values.tolist())

            # create an IBD graph and add var attrs to it
            g.add_edges_from(king_df.apply(lambda x: [x.ID1, x.ID2, 
                                                {"k_prop": x.PropIBD,
                                                "k_ibd1": x.IBD1Seg,
                                                "k_ibd2": x.IBD2Seg,
                                                "k_degree": x.InfType}],
                                                axis=1).values)

        ### subset to only samples from the population and who are not part of the twin set
        popl_samples = {nodes for nodes, attrs in g.nodes(data=True) if attrs.get("popl", "")==popln}
        self.g = g.subgraph(popl_samples - mz_twins)


        ### load ibd, process ibd segs
        ibd_file = kwargs.get("ibd_file", "")
        code_yaml = kwargs.get("code_yaml", "relationship_codes.yaml")
        if os.path.exists(ibd_file):
            print("Processing IBD segments...")
            ibd_files = get_file_names(ibd_file)
            # load ibd files
            ibd_df = pd.concat([pd.read_csv(filen, delim_whitespace=True, dtype={"id1": str, "id2": str})
                                for filen in ibd_files])

            ibd_df["l"] = ibd_df["end_cm"] - ibd_df["start_cm"]

            # iterate through the pairs of individuals, compute ibd stats
            for (id1, id2), pair_df in ibd_df.groupby(["id1", "id2"]):

                # only compute hsr, other stats for 3rd+ degree relatives
                k = self.g.get_edge_data(id1, id2, {}).get("k_prop", 0)

                # 2^-3.5 is the lower limit for 3rd degree relatives
                if k < 2**-3.5:
                    continue

                # compute the IBD data
                pair_ibd = ProcessSegments(pair_df)
                ibd_data = pair_ibd.ponderosa_data(genome_len, inter_phase=False)

                # add phase errors
                pair_ibd_pe = ProcessSegments(introduce_phase_error(pair_df, 50))
                ibd_data_pe = pair_ibd_pe.ponderosa_data(genome_len, inter_phase=False)

                # set up the pedigree hierarchy, which will store the probs and info on how the probs were computed
                hier = PedigreeHierarchy(code_yaml)
                hier.update_attr_from([[node, "ordering", sorted([id1, id2])] for node in hier.init_nodes])
                hier.update_attr_from([[node, "ordering_method", "sorted"] for node in hier.init_nodes])

                cur_edge_data = self.g.get_edge_data(id1, id2)


                cur_edge_data["ibd1"] = ibd_data.ibd1
                cur_edge_data["ibd2"] = ibd_data.ibd2
                cur_edge_data["h"] = {id1: ibd_data.h1, id2: ibd_data.h2}
                cur_edge_data["h_error"] = {id1: ibd_data_pe.h1, id2: ibd_data_pe.h2}
                cur_edge_data["n"] = ibd_data.n
                cur_edge_data["probs"] = hier


                # # add ibd1 data and initialze probs
                # self.g.add_edge(id1, id2, ibd1=ibd_data.ibd1,
                #                      ibd2=ibd_data.ibd2,
                #                      h={id1: ibd_data.h1, id2: ibd_data.h2},
                #                      h_pe={id1: ibd_data_pe.h1, id2: ibd_data_pe.h2},
                #                      n=ibd_data.n,
                #                      k=(ibd_data.ibd1/2 + ibd_data.ibd2),
                #                      probs=hier,
                #                      segments=pair_df)

        else:
            print("No IBD files have been provided.")


    # returns a list of node pair edges
    # optional func arg is a function that takes as input the data dict and returns true if wanted
    def get_edges(self, f=lambda x: True):
        return [(id1, id2) for id1, id2, data in self.g.edges(data=True) if f(data)]

    # returns a nested np array of [id1, id2, attr1, attr2, ...] for all edges that pass the function
    def get_edge_data(self, attr_list, f=lambda x: True):
        edges, out_attrs = [], []
        for id1, id2, data in self.g.edges(data=True):
            if f(data):
                edges.append([id1, id2])
                out_attrs.append([data[attr] for attr in attr_list])
        return np.array(edges, dtype=str), np.array(out_attrs)

    # updates a bunch of edges at once
    # edge_list looks like [(A, B), (B, C), (D, E)]
    # attr_list looks like [1, 2, 3]
    def update_edges(self, edge_list, attr_list, attr_name):

        attr_dict = {(id1, id2): attr for (id1, id2), attr in attr_list}
        nx.set_edge_attributes(g, attr_dict, attr_name)

    # same as above but returns a set of nodes
    # e.g., get_nodes(lambda x: x["age"] > 90) returns all nodes that are >90 yo
    def get_nodes(self, f=lambda x: True):
        return [id1 for id1, data in self.g.nodes(data=True) if f(data)]
    
    def get_subset(self, f):
        # get the nodes based on the conditional f
        node_subset = self.get_nodes(f)
        # return the subset for which the conditional is true
        return self.g.subgraph(set(node_subset))
    
    def to_dataframe(self, edges, include_edges):
        if len(edges) == 0:
            return nx.to_pandas_edgelist(self.g, source="id1", target="id2")
        # make copy of the graph
        tmp = self.g.copy()
        # subgraph of the edges
        if include_edges:
            tmp = tmp.edge_subgraph(edges)
        # subgraph of the complement of the edges
        else:
            tmp.remove_edges_from(edges)
        return nx.to_pandas_edgelist(tmp, source="id1", target="id2")

    # update the hier probabilities
    def update_probs(self, probs_list, prob_labs, method, prob_objs):
        # iterate through the probabilities
        for probs, prob_obj in zip(probs_list, prob_objs):
            prob_obj.set_attrs({i: p for i,p in zip(prob_labs, probs)}, "p_con")
            prob_obj.set_attrs({i: method for i in prob_labs}, "method")

    def node_attr(self, node, attr, nan=np.nan):
        try:
            return self.g.nodes[node][attr]
        except:
            return nan

        
def infer_second(id1, id2, pair_data, samples,
                mhs_gap, gp_gap):

    ### Re-compute probabilities based on phenotypic data (parents, age, etc)
    hier_obj = pair_data["probs"]
    hier = hier_obj.hier

    ### bool is True if we want to use the haplotype classifier
    use_hap = hier.nodes["HS"]["p"] + hier.nodes["GP/AV"]["p"] >= 0.80

    ### Use phenotype data to set certain relationships to 0

    # get the age gap
    age_gap = abs(samples.nodes[id1]["age"] - samples.nodes[id2]["age"])

    # too far in age to be MHS
    if age_gap > mhs_gap:
        hier.nodes["MHS"]["p"] = 0
        hier.nodes["MHS"]["method"] = "pheno"
        hier.nodes["PHS"]["method"] = "pheno"

    # too close in age to be GP
    if age_gap < gp_gap:
        hier_obj.set_attrs({i: 0 for i in ["PGP", "MGP", "GP"]}, "p")
        hier_obj.set_attrs({i: "pheno" for i in ["PGP", "MGP", "GP"]}, "method")

    # not a paternal relationship
    if samples.nodes[id1]["father"] != None or samples.nodes[id2]["father"] != None:
        # not PHS
        hier.nodes["PHS"]["p"] = 0
        hier.nodes["PHS"]["method"] = "pheno"
        hier.nodes["MHS"]["method"] = "pheno"

    # not MHS bc mother exists
    if samples.nodes[id1]["mother"] != None or samples.nodes[id2]["mother"] != None:
        hier.nodes["MHS"]["p"] = 0
        hier.nodes["MHS"]["method"] = "pheno"
        hier.nodes["PHS"]["method"] = "pheno"

    # re-compute probabilities
    second_nodes = ["AV", "MGP", "PGP", "PHS", "MHS"]
    prob_sum = sum([hier.nodes[i]["p"] for i in second_nodes])
    hier_obj.set_attrs({i: prob_sum and hier.nodes[i]["p"] / prob_sum or 0 for i in second_nodes}, "p")

    ### Done with pheno data

    ### Tier 2 probabilities

    # they are NOT hs; have to be GP/AV
    if hier.nodes["PHS"]["p"] + hier.nodes["MHS"]["p"] == 0:
        hier_obj.set_attrs({"HS": 0, "GP/AV": 1}, "p")
        hier_obj.set_attrs({"HS": "pheno"}, "method")

    # use haplotype data
    if use_hap:
        prob_sum = hier.nodes["HS"]["p"] + hier.nodes["GP/AV"]["p"]
        hier_obj.set_attrs({"HS": hier.nodes["HS"]["p"] / prob_sum if prob_sum > 0 else 0.50,
                            "GP/AV": hier.nodes["GP/AV"]["p"] / prob_sum if prob_sum > 0 else 0.50}, "p")

    # don't use the haplotype data
    else:
        hier_obj.set_attrs({"HS": hier.nodes["PHS"]["p"] + hier.nodes["MHS"]["p"],
                        "GP/AV": hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"] + hier.nodes["AV"]["p"]}, "p")
        hier_obj.set_attrs({"HS": "segs", "GP/AV": "segs"}, "method")

    ### Tier 3 probabilities

    # for GP vs. AV
    prob_sum = hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"] + hier.nodes["AV"]["p"]
    hier_obj.set_attrs({"GP": (hier.nodes["PGP"]["p"] + hier.nodes["MGP"]["p"])/prob_sum if prob_sum > 0 else 0.50,
                        "AV": hier.nodes["AV"]["p"]/prob_sum if prob_sum > 0 else 0.50}, "p")

    # for MHS versus PHS
    prob_sum = hier.nodes["MHS"]["p"] + hier.nodes["PHS"]["p"]
    hier_obj.set_attrs({"PHS": prob_sum and hier.nodes["PHS"]["p"]/prob_sum or 0.50,
                        "MHS": prob_sum and hier.nodes["MHS"]["p"]/prob_sum or 0.50}, "p")

    ### Tier 4 probabilities

    # for MGP vs PGP
    prob_sum = hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"]
    hier_obj.set_attrs({"PGP": hier.nodes["PGP"]["p"]/prob_sum if prob_sum > 0 else 0.50,
                        "MGP": hier.nodes["MGP"]["p"]/prob_sum if prob_sum > 0 else  0.50}, "p")

    hier_obj.set_attrs({"relatives": 1, "2nd": hier.nodes["2nd"]["p"]}, "post")
    # iterate down the tree and compute the posterior prob
    for parent, child in nx.bfs_edges(hier, "2nd"):

        hier_obj.set_attrs({child: hier.nodes[child]["p"] * hier.nodes[parent]["post"]}, "post")

        # need to order according to genetically youngest --> oldest
        if child in ["GP", "MGP", "PGP", "AV", "GP/AV"]:

            # use age
            if samples.nodes[id1]["age"] > samples.nodes[id2]["age"]:
                hier.nodes[child]["ordering"] = (id2, id1)
                hier.nodes[child]["ordering_method"] = "age"
            elif samples.nodes[id2]["age"] > samples.nodes[id1]["age"]:
                    hier.nodes[child]["ordering"] = (id1, id2)
                    hier.nodes[child]["ordering_method"] = "age"  

            # use the haplotype score      
            elif use_hap:
                if samples.edges[id1, id2]["h"][id1] > samples.edges[id1, id2]["h"][id2]:
                    hier.nodes[child]["ordering"] = (id1, id2)
                    hier.nodes[child]["ordering_method"] = "hap"
                else:
                    hier.nodes[child]["ordering"] = (id2, id1)
                    hier.nodes[child]["ordering_method"] = "hap"

def readable_results(results, min_p):

    out_results = []
    for (id1, id2), hier in results.items():

        # compute lengths of all paths from relatives
        paths = nx.single_source_shortest_path_length(hier, "relatives")

        # highly probable rels
        probable_rels = sorted([[l and 1/l or np.inf, rel] for rel, l in paths.items() if hier.nodes[rel]["post"] > min_p])

        # get the deepest rel
        deepest_rel = probable_rels[0][1]
        
        # get the degree of relative
        degree = nx.shortest_path(hier, "relatives", deepest_rel)[1] if deepest_rel != "relatives" else "relatives"
        
        rel_data = hier.nodes[deepest_rel]

        row = list(rel_data["ordering"]) + [rel_data[col] for col in ["ordering_method", "post", "p", "method"]] + [deepest_rel, degree]
        row += [hier.nodes["pair_data"]["ibd1"], hier.nodes["pair_data"]["ibd2"]]
        out_results.append(row)

    results_df = pd.DataFrame(out_results, columns=["id1", "id2", "order_method", "posterior", "conditional", "method", "relationship", "degree", "ibd1", "ibd2"])
    
    return results_df.round(4)


if __name__ == "__main__":

    args = parse_args()

    ### Samples is a SampleData obj that has all node and pairwise IBD information
    if args.debug:
        print("In debug mode.")
        i = open("Samples_debug.pkl", "rb")
        Samples = pkl.load(i)
        i.close()

    else:
        Samples = SampleData(fam_file=args.fam,
                            ibd_file=args.ibd,
                            age_file=args.ages,
                            map_file=args.map,
                            pop_file=args.populations,
                            population=args.population,
                            n_pairs=args.n_pairs)

    ### Training step
    
    # True if there is already trained data; load the pickled training objs
    if os.path.exists(args.training):
        print("Trained classifiers provided.")

        # load the degree classifier
        # proba goes: [2nd, 3rd, 4th, FS]
        infile = open(args.training, "rb")
        degree_classfr = pkl.load(infile)

        # load the n segs classifier
        # probs go ['AV', 'MGP', 'MHS', 'PGP', 'PHS']
        infile = open(args.training.replace("degree", "nsegs"), "rb")
        nsegs_classfr = pkl.load(infile)

        # load the hap classifier
        # probs go ['GP/AV', 'HS', 'PhaseError']
        infile = open(args.training.replace("degree", "hap"), "rb")
        hap_classfr = pkl.load(infile)

    # pedigree object to find existing relationships
    print("Finding pedigree relationships")
    if args.debug:
        i = open("Pedigree_debug.pkl", "rb")
        pedigree = pkl.load(i)
        i.close()
    else:
        try:
            pedigree = Pedigree(Samples.g, degree_classfr)
        except:
            pedigree = Pedigree(Samples.g, None)

        # find relationships
        pedigree.find_relationships()

    # no training provided; generate by self
    if not os.path.exists(args.training):
        print("Trained classifiers not provided. Training classifiers now.")

        # get the training data
        train = TrainPonderosa(real_data=True, samples=Samples, pedigree=pedigree)

        ### haplotype classifier
        hap_classfr = train.hap_classifier()

        ### n segs classifier
        nsegs_classfr = train.nsegs_classifier()

        ### degree classifier
        degree_classfr = train.degree_classifier()

    ### Classification step 1: degree of relatedness
    print("Beginning classification.")

    # get the input data; X has the following cols: id1, id2, ibd1, ibd2
    edges, X_degree = Samples.get_edge_data(["ibd1", "ibd2"], lambda x: (0 < x["k"] < 0.42) or (0.1 < x["ibd2"] < 0.5))
    degree_probs = degree_classfr.predict_proba(X_degree)

    # add the classification probs
    Samples.update_probs(edges, degree_probs, degree_classfr.classes_, "ibd")

    ### Classification step 2: n of segs
    edges, X_n = Samples.get_edge_data(["n", "k"], lambda x: x["probs"].hier.nodes["2nd"]["p"] > 0.80)
    n_probs = nsegs_classfr.predict_proba(X_n)

    # add the n probs
    Samples.update_probs(edges, n_probs, nsegs_classfr.classes_, "segs")

    ### Classification step 3: haps
    edges, X_hap = Samples.get_edge_data(["h"], lambda x: x["probs"].hier.nodes["2nd"]["p"] > 0.80)

    # retrieve h1 and h2 values and predict the probs
    haps = [[h[0][id1], h[0][id2]] for (id1, id2), h in zip(edges, X_hap)]
    h_probs = hap_classfr.predict_proba(haps)

    # update the hap probs
    # get the index of the GPAV and HS classes
    class_index = np.where(hap_classfr.classes_ != "PhaseError")[0]
    Samples.update_probs(edges, h_probs[:,class_index], hap_classfr.classes_[class_index] , "hap")

    ### Inference steps

    # hold the hier objects
    out_hiers = {}

    # iterate through the pairs
    for id1, id2, data in Samples.g.edges(data=True):

        # get the pair data
        pair_data = data.copy()

        # infer the rel if second
        if pair_data["probs"].hier.nodes["2nd"]["p"] > 0.80:
            infer_second(id1, id2, pair_data, Samples.g, 30, 30)

        else:
            pair_data["probs"].set_attrs({i: j.get("p" if i != "relatives" else "post", 1) for i,j in pair_data["probs"].hier.nodes(data=True)}, "post")

        hier = pair_data["probs"].hier

        # add the pair data as a node with attributes
        hier.add_node("pair_data",
                      h=pair_data["h"],
                      n=pair_data["n"],
                      ibd1=pair_data["ibd1"],
                      ibd2=pair_data["ibd2"])
        
        # add the ids as nodes so you can get the pair info
        hier.add_nodes_from(Samples.g.subgraph({id1, id2}).nodes(data=True))

        out_hiers[(id1, id2)] = hier

    # write out a pickle object of the results
    outf = open(f"{args.out}_results.pkl", "wb")
    pkl.dump(out_hiers, outf)
    outf.close()

    # get the readable results df
    results_df = readable_results(out_hiers, args.min_p)
    results_df.to_csv(f"{args.out}_results.txt", sep="\t", index=False)
