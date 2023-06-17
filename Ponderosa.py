import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import argparse
import os
from pedigree_tools import ProcessSegments, PedigreeHierarchy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ibd", help = "IBD segment file. If multiple files for each chromosome, this is the path for chromosome 1.", required=True)
    parser.add_argument("-fam", help="PLINK-formated .fam file")
    parser.add_argument("-ages", help="Age file. First column is the IID, second column is the age", default="")
    parser.add_argument("-map", help = "PLINK-formatted .map file.", default="")
    parser.add_argument("-training", help = "Path and name of the 'degree' pkl file for training.", default="")
    parser.add_argument("-out", help = "Output prefix.", default="Ponderosa")
    args = parser.parse_args()
    return args

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

        ### init the graph and set defaults
        g = nx.Graph()
        g.add_nodes_from(fam.apply(lambda x: (x[1], {"sex": x[4],
                                                    "mother": x[3] if x[3] != "0" else None,
                                                    "father": x[2] if x[2] != "0" else None,
                                                    "children": [],
                                                    "age": np.nan}), axis=1).values)

        # add the children
        for sex_col in [2, 3]:
            # assume that missing parent is coded as 0
            for parent, children_df in fam[fam[sex_col] != "0"].groupby(sex_col):
                # get a list of the children
                children = children_df[1].values.tolist()
                # add to attribute
                nx.set_node_attributes(g, {parent: children}, "children")

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
            print("No map files have been provided.")
            genome_len = 3500

        ### load ibd, process ibd segs
        ibd_file = kwargs.get("ibd_file", "")
        if os.path.exists(ibd_file):
            ibd_files = get_file_names(ibd_file)
            # load ibd files
            ibd_df = pd.concat([pd.read_csv(filen, delim_whitespace=True, dtype={"id1": str, "id2": str})
                                for filen in ibd_files])

            ibd_df["l"] = ibd_df["end_cm"] - ibd_df["start_cm"]

            # iterate through the pairs of individuals, compute ibd stats
            n_pairs = 0
            for (id1, id2), pair_df in ibd_df.groupby(["id1", "id2"]):

                # if distant relative or nodes not in the fam file, ignore
                if ibd_df["l"].sum() < 10\
                    or id1 not in g.nodes\
                    or id2 not in g.nodes\
                    or id1 == id2:
                    continue

                pair_ibd = ProcessSegments(pair_df)
                ibd_data = pair_ibd.ponderosa_data(genome_len)

                # set up the pedigree hierarchy, which will store the probs and info on how the probs were computed
                hier = PedigreeHierarchy()
                hier.set_attrs({i: np.nan for i in hier.init_nodes}, "p")
                hier.set_attrs({i: np.nan for i in hier.init_nodes}, "post")
                hier.set_attrs({i: None for i in hier.init_nodes}, "method")
                hier.set_attrs({i: tuple(sorted([id1, id2])) for i in hier.init_nodes}, "ordering")
                hier.set_attrs({i: "sorted" for i in hier.init_nodes}, "ordering_method")

                # add ibd1 data and initialze probs
                g.add_edge(id1, id2, ibd1=ibd_data.ibd1,
                                     ibd2=ibd_data.ibd2,
                                     h={id1: ibd_data.h1, id2: ibd_data.h2},
                                     n=ibd_data.n,
                                     k=(ibd_data.ibd1/2 + ibd_data.ibd2),
                                     probs=hier)

                n_pairs += 1
                # if n_pairs > 10:
                #     break

        else:
            print("No IBD files have been provided.")

        self.g = g

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

    # update the hier probabilities
    def update_probs(self, pair_list, probs_list, prob_labs, method):
        # iterate through the probabilities
        for (id1, id2), probs in zip(pair_list, probs_list):
            self.g.edges[id1, id2]["probs"].set_attrs({i: p for i,p in zip(prob_labs, probs)}, "p")
            self.g.edges[id1, id2]["probs"].set_attrs({i: method for i in prob_labs}, "method")


def infer_second(id1, id2, pair_data, samples,
                mhs_gap, gp_gap):

    ### Re-compute probabilities based on phenotypic data (parents, age, etc)
    hier_obj = pair_data["probs"]
    hier = hier_obj.hier

    # not PHS
    if samples.nodes[id1]["father"] != None or samples.nodes[id2]["father"] != None:
        hier.nodes["PHS"]["p"] = 0
        hier.nodes["PHS"]["method"] = "pheno"
    
    # not MHS bc mother exists
    if samples.nodes[id1]["mother"] != None or samples.nodes[id2]["mother"] != None:
        hier.nodes["MHS"]["p"] = 0
        hier.nodes["MHS"]["method"] = "pheno"

    # get the age gap
    age_gap = abs(samples.nodes[id1]["age"] - samples.nodes[id2]["age"])

    # too far in age to be MHS
    if age_gap > mhs_gap:
        hier.nodes["MHS"]["p"] = 0
        hier.nodes["MHS"]["method"] = "pheno"
    
    # too close in age to be GP
    if age_gap < gp_gap:
        hier.obj.set_attrs({"GP": 0, "MGP": 0, "PGP": 0}, "p")
        hier_obj.set_attrs({i: "pheno" for i in ["GP", "MGP", "PGP"]}, "method")

    # compute posterior probs for the n_segs
    second_nodes = ["AV", "MGP", "PGP", "PHS", "MHS"]
    prob_sum = sum([hier.nodes[i]["p"] for i in second_nodes])
    hier_obj.set_attrs({i: prob_sum and hier.nodes[i]["p"] / prob_sum or 0 for i in second_nodes}, "p")

    # need to reset the probabilities

    # if hap score is not good, use HSR for GP/AV vs. HS
    use_hap = hier.nodes["HS"]["p"] + hier.nodes["GP/AV"]["p"] >= 0.80
    if not use_hap:
        hier_obj.set_attrs({"HS": hier.nodes["PHS"]["p"] + hier.nodes["MHS"]["p"],
                        "GP/AV": hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"] + hier.nodes["AV"]["p"]}, "p")
        hier_obj.set_attrs({"HS": "segs", "GP/AV": "segs"}, "method")

    # update the PHS and MHS probs
    p_hs = hier.nodes["PHS"]["p"] + hier.nodes["MHS"]["p"]
    # prob of hs is zero, either bc of segments of pheno data (likely pheno data)
    if p_hs == 0:
        # there are parents present; the p(HS) is 0 and p(GPAV) is 1
        if hier.nodes["PHS"]["method"] == "pheno" and hier.nodes["MHS"]["method"] == "pheno":
            hier_obj.set_attrs({"HS": 0, "GP/AV": 1}, "p")
            hier_obj.set_attrs({"HS": "pheno", "GP/AV": "pheno"}, "method")
    else:
        hier_obj.set_attrs({"PHS": hier.nodes["PHS"]["p"] / p_hs,
                    "MHS": hier.nodes["MHS"]["p"] / p_hs}, "p")


    # update the GP and AV probs
    p_gpav = hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"] + hier.nodes["AV"]["p"]
    hier_obj.set_attrs({"GP": p_gpav and (hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"]) / p_gpav or 0,
                    "AV": p_gpav and hier.nodes["AV"]["p"] / p_gpav or 0}, "p")

    # update the sex-specific GP probs
    p_gp = hier.nodes["MGP"]["p"] + hier.nodes["PGP"]["p"]
    hier_obj.set_attrs({"MGP": p_gp and hier.nodes["MGP"]["p"] / p_gp or 0,
                    "PGP": p_gp and hier.nodes["PGP"]["p"] / p_gp or 0}, "p")

    # iterate down the tree and compute the posterior prob
    for parent, child in nx.bfs_edges(hier, "2nd"):
        # posterior prob is the post prob of the parent times the conditional of the child
        post = hier.nodes[parent]["p" if parent == "2nd" else "post"] *  hier.nodes[child]["p"]
        # update the attribute
        hier.nodes[child]["post"] = post

        # need to order according to genetically youngest --> oldest
        if child in ["GP", "MGP", "PGP", "AV", "GP/AV"]:
            if samples.nodes[id1]["age"] > samples.nodes[id2]["age"]:
                hier.nodes[child]["ordering"] = (id2, id1)
                hier.nodes[child]["ordering_method"] = "age"
            elif samples.nodes[id2]["age"] > samples.nodes[id1]["age"]:
                    hier.nodes[child]["ordering"] = (id1, id2)
                    hier.nodes[child]["ordering_method"] = "age"        
            elif use_hap:
                if samples.edges[id1, id2]["h"][id1] > samples.edges[id1, id2]["h"][id2]:
                    hier.nodes[child]["ordering"] = (id1, id2)
                    hier.nodes[child]["ordering_method"] = "hap"
                else:
                    hier.nodes[child]["ordering"] = (id2, id1)
                    hier.nodes[child]["ordering_method"] = "hap"




if __name__ == "__main__":

    args = parse_args()

    ### Samples is a SampleData obj that has all node and pairwise IBD information
    Samples = SampleData(fam_file = args.fam,
                         ibd_file = args.ibd,
                         age_file = args.ages,
                         map_file = args.map)

    ### Training step
    
    # True if there is already trained data; load the pickled training objs
    if os.path.exists(args.training):

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


    ### Classification steps

    ### Classification step 1: degree of relatedness

    # get the input data; X has the following cols: id1, id2, ibd1, ibd2
    edges, X_degree = Samples.get_edge_data(["ibd1", "ibd2"], lambda x: 0.05 < x["k"] < 0.45)
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

    second_pairs = edges

    # hold the hier objects
    out_hiers = {}

    # iterate through the pairs
    for id1, id2 in second_pairs:

        # get the pair data
        pair_data = Samples.g[id1][id2].copy()

        # infer the relationship
        rel = infer_second(id1, id2, pair_data, Samples.g, 30, 30)

        out_hiers[(id1, id2)] = pair_data["probs"].hier

    # write out a pickle object of the results
    outf = open(f"{args.out}_results.pkl", "wb")
    pkl.dump(out_hiers, outf)
    outf.close()