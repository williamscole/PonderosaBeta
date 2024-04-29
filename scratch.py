import numpy as np
import networkx as nx
import yaml
import itertools as it
from collections import Counter
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import logging
from copy import deepcopy
from sklearn.model_selection import LeaveOneOut

from Ponderosa import *
from pedigree_tools import PedigreeHierarchy, Pedigree


# class Relationship:
#     def __init__(self, data):
#         p1 = sorted([[1] + path for path in data.get(1, [])], key=lambda x: x[1:])
#         p2 = sorted([[2] + path for path in data.get(2, [])], key=lambda x: x[1:])

#         # support for inter-generational relationships
#         longest_path = max([len(path) for path in p1 + p2])
#         matrix = np.array([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2])

#         # set attrs
#         self.sex_specific = data["sex"]
#         self.mat = matrix if self.sex_specific else matrix[:,1:]

#     # returns boolean if it is the given relationship
#     def is_relationship(self, mat):
#         # convert the path dict to a matrix
#         mat = mat if self.sex_specific else mat[:,1:]

#         # not the same shape == can't be the relationship
#         if self.mat.shape != mat.shape:
#             return False

#         # returns True if the relationship matches
#         return (self.mat - mat).sum() == 0
        

# class RelationshipCodes:
#     def __init__(self, yaml_file):
#         # open the code yaml
#         i = open(yaml_file, "r")
#         code_yaml = yaml.safe_load(i)

#         # load each relationship, create the matrix
#         self.codes = []
#         for name, data in code_yaml.items():
#             r = Relationship(data)
#             self.codes.append([name, r])

#         # sort such that the sex-specific relationships come first
#         self.codes.sort(key=lambda x: int(x[1].sex_specific), reverse=True)

#     # converts the path dictionary to a matrix
#     def convert_to_matrix(self, path_dict):
#         # get the paths
#         p1 = sorted([[1] + [direction for _, direction in path] for path in path_dict[1]], key=lambda x: x[1:])
#         p2 = sorted([[2] + [direction for _, direction in path] for path in path_dict[2]], key=lambda x: x[1:])

#         # get the kinships
#         k1 = sum([0.5**(len(path)-2) for path in p1])
#         k2 = sum([0.5**(len(path)-2) for path in p2])

#         # support for inter-generational relationships
#         longest_path = max([len(path) for path in p1 + p2])
#         mat = np.array([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2])

#         # returns the matrix and expected ibd1, ibd2 values
#         return mat, k1*(1-k2) + k2*(1-k1), k1*k2

#     # returns the relationship
#     def determine_relationship(self, path_dict):
#         # get the matrix and expected ibd values
#         mat, ibd1, ibd2 = self.convert_to_matrix(path_dict)

#         # the pair are the same generation
#         same_gen = mat[:,1:].sum() == 0

#         # iterate through the relationships
#         for name, robj in self.codes:
#             # boolean if the relationship is true
#             if robj.is_relationship(mat):
#                 return name, ibd1, ibd2, mat, same_gen

#         ### haven't found a relationship, need to make sure that it's not a reversed code

#         # get the first column of the matrix
#         pcol = mat[:,:1]
#         # reverse the direction of the rest of the matrix and flip along the horizontal
#         tmp = np.flip(mat[:,1:]*-1)
#         # add the parent column to the flipped matrix
#         rv_mat = np.append(pcol, tmp, axis=1)
#         # bound at least one possible relationship that it could be
#         found = sum([robj.is_relationship(rv_mat) for _, robj in self.codes]) > 0

#         # relationship is not found
#         return "nan" if found else "unknown", ibd1, ibd2, mat, same_gen

# class Pedigree:
#     def __init__(self, **kwargs):

#         po_list = kwargs.get("po_list", [])
#         samples_g = kwargs.get("samples", None)

#         # a list of po-pairs has been supplied
#         if len(po_list) > 0:
#             po = nx.DiGraph()
#             po.add_edges_from(po_list)

#         # samples graph has been supplied
#         elif len(samples_g.g.nodes) > 0:
#             tmp = nx.DiGraph()
#             tmp.add_edges_from(it.chain(*[[[data["mother"], node], [data["father"], node]] for node, data in samples.g.nodes(data=True)]))
#             po = tmp.subgraph(set(tmp.nodes) - {np.nan})

#         self.po = po

#         # must supply the yaml file
#         self.R = RelationshipCodes(kwargs["yaml_file"])

#         # create the pedigree hierarchy
#         self.hier = PedigreeHierarchy(kwargs["yaml_file"])

#     # for a given focal individual, finds all relationships
#     def focal_relationships(self, focal):
#         # recursive function that, given a focal individual returns all paths to relatives
#         def get_paths(cur_relative, path_ids, path_dirs, paths):
#             # init the next set of relatives
#             next_set = []

#             # past the first iteration, so we can get down nodes, but only down nodes that are not in the path
#             if len(path_dirs) > 1:
#                 next_set += [(nxt_relative,-1) for nxt_relative in self.po.successors(cur_relative) if nxt_relative not in path_ids]

#             # we're still moving up, so we can get the up nodes
#             if path_dirs[-1] > 0:
#                 next_set += [(nxt_relative, 1) for nxt_relative in self.po.predecessors(cur_relative)]

#             # we can't keep going; base case
#             if len(next_set) == 0:
#                 paths.append((path_ids, path_dirs))
#                 return paths

#             # iterate through the new set of relatives
#             for nxt_relative, direction in next_set:
#                 paths = get_paths(nxt_relative, path_ids + [nxt_relative], path_dirs + [direction], paths)
#             return paths

#         # given the output of get_paths, creates all sub-paths
#         def merge_paths(focal, paths):
#             # init the dict to store each relative pair and the paths along each parental lineage
#             rel_pairs = {id2: {1: set(), 2: set()} for id2 in it.chain(*[path_ids for path_ids,_ in paths])}

#             # iterate through the paths
#             for path_ids, path_dirs in paths:
#                 # zip the path ids and the path directions
#                 path = [(id2, pdir) for id2, pdir in zip(path_ids, path_dirs)]

#                 # iterate through each person in the path
#                 for index in range(1, len(path)):
#                     # get the id of the relative
#                     id2 = path[index][0]
#                     # get the subpath from the focal to the current id2
#                     subpath = path[1:index+1]
#                     # determine which parent they are related through
#                     parent_sex = samples.g.nodes[subpath[0][0]]["sex"]
#                     # add to the rel pairs dictionary
#                     rel_pairs[id2][int(parent_sex)] |= {tuple(path[1:index+1])}
            
#             return rel_pairs

#         # get all paths
#         path_list = get_paths(focal, [focal], [1], [])

#         # keep track of unknown relationships
#         unknown_rels = []

#         # iterate through each relative of the focal individual
#         for id2, path_dict in merge_paths(focal, path_list).items():
#             if focal == id2:
#                 continue

#             # get the relationship 
#             rname, e_ibd1, e_ibd2, mat, same_gen = self.R.determine_relationship(path_dict)

#             # don't want to add if rname is nan or they are same generation and id2 > id1
#             if rname == "nan" or (same_gen and focal > id2):
#                 continue

#             if rname == "unknown":
#                 unknown_rels.append(np.array2string(mat[:,1:]))

#             self.hier.add_pair((focal, id2), rname, {"ibd1": e_ibd1, "ibd2": e_ibd2})
        
#         return unknown_rels

#     # finds all relationships for nodes in the graph
#     def find_all_relationships(self):

#         unknown_rels = it.chain(*[self.focal_relationships(focal) for focal in self.po.nodes])

#         print("The following unknown relationships were found:")

#         for unkr, n in Counter(unknown_rels).items():
#             print(f"{n} of the following were found:")
#             print(unkr + "\n")
        
# samples = SampleData(fam_file="for_dev/Himba_allPO.fam",
#                      king_file="for_dev/King_Relatedness_no2278.seg",
#                      ibd_file="for_dev/Himba_shapeit.chr1_segments.txt",
#                      map_file="for_dev/newHimba_shapeit.chr1.map",
#                      code_yaml="tree_codes.yaml")

# i = open("for_dev/sample.pkl", "wb")
# samples = pkl.dump(samples, i)
# i.close()

class Classifiers:
    def __init__(self, pairs):

        # store the training data for each classifier; purpose is to allow leave one out training
        self.training = {}

        ### Train the degree classifier
        degree_train = pairs.get_pair_df("relatives").dropna(subset=["k_ibd1"])

        self.training["degree"] = [np.array(degree_train[["k_ibd1", "k_ibd2"]].values.tolist()),
                                   np.array(degree_train["degree"].values.tolist())]
        
        ### Train the hap classifier
        hap_train = pairs.get_pair_df_from(["HS", "GPAV"]).dropna(subset=["h"])

        # get the phase error classifier
        X_train = hap_train.apply(lambda x: [x.h_error[x.pair[0]], x.h_error[x.pair[1]]], axis=1).values.tolist()
        y_train = ["Phase error" for _ in X_train]

        # now add the actual haplotype scores
        X_train += hap_train.apply(lambda x: [x.h[x.pair[0]], x.h[x.pair[1]]], axis=1).values.tolist()
        y_train += hap_train["requested"].values.tolist()

        self.training["hap"] = [np.array(X_train), np.array(y_train)]

        ### Train the degree classifier
        n_train = pairs.get_pair_df_from(["MGP", "MHS", "PHS", "PGP", "AV"]).dropna(subset=["n"])

        # also train on the kinship coefficient
        n_train["ibd_cov"] = n_train.apply(lambda x: x.k_ibd2 + x.k_ibd1, axis=1)

        self.training["n"] = [np.array(n_train[["ibd_cov", "n"]].values.tolist()), np.array(n_train["requested"].values.tolist())]

        self.loo = LeaveOneOut()

    # if n samples in X_train, will train the classifier n times, leaving a different sample out each time
    def leaveOneOut(self, X_train, y_train, func, labels):
            return_probs = []; return_labels = []
            lda = LinearDiscriminantAnalysis()

            for train, test in self.loo.split(X_train):

                lda.fit(X_train[train], y_train[train])

                # predict
                return_probs.append(func(lda, X_train[test])[0])
                return_labels.append(lda.classes_)

            # just predicting the label; don't need the probabilities
            if labels:
                return return_probs, iter(return_labels)

            return return_probs

    def return_classifier(self, classif):

        X_train, y_train = self.training[classif]

        lda = LinearDiscriminantAnalysis()

        lda.fit(X_train, y_train)

        return lda
    
    # returns an array of the probabilities and an iterator of the classes that each probability describes
    def predict_proba(self, classif, X=[]):

        # get the training data for the classifier
        X_train, y_train = self.training[classif]

        # no training data supplied --> perform leave one out
        if len(X)==0:
            return self.leaveOneOut(X_train, y_train, lambda lda, X: lda.predict_proba(X), labels=True)

        lda = self.return_classifier(classif)
 
        return lda.predict_proba(X), it.cycle([lda.classes_])

    # predicts just the label
    def predict(self, classif, X=[]):
        lda = LinearDiscriminantAnalysis()
        X_train, y_train = self.training[classif]

        # perform leave-one-out
        if len(X)==0:
            return self.leaveOneOut(X_train, y_train, lda, lambda lda, X: lda.predict(X), labels=False)

        lda = self.return_classifier(classif)

        return lda.predict(X)
    
    def write_pkl(self, classif, output):

        lda = self.return_classifier(classif)

        i = open(output, "wb")
        pkl.dump(lda, i)
        i.close()


    
class ResultsData:
    def __init__(self, samples, pairs, df=pd.DataFrame()):

        self.samples = samples
        self.pairs = pairs
        self.df = self.samples.to_dataframe([], include_edges=False) if df.shape[0]==0 else df

    # writes out a human readable output; can specificy the columns
    def write_readable(self, output, **kwargs):

        cols = kwargs.get("columns",
                          ["id1", "id2", "k_ibd1", "k_ibd2", "most_probable", "probability"])
        
        if "h1" in cols:
            self.df["h1"], self.df["h2"] = zip(*self.df.apply(lambda x: [x.h[x.id1], x.h[x.id2]], axis=1))

        self.df[cols].to_csv(output, index=False, sep="\t")

    # creates the dataframe
    def to_dataframe(self, edges, include_edges, inplace=False):

        tmp = self.samples.to_dataframe(edges, include_edges)

        if inplace:
            self.df = tmp
        else:
            return tmp

    # takes as input a function that works on the dataframe and subsets it
    def subset_dataframe(self, func, inplace=False):
        tmp = self.df[self.df.apply(lambda x: func(x), axis=1)]

        if inplace:
            self.df = tmp
        else:
            return tmp
        
    def subset_samples(self, func):

        self.subset_dataframe(func, inplace=True)

        self.samples.edge_subset(self.df[["id1", "id2"]].apply(tuple, axis=1).values, inplace=True)

    # sets the min prob for readable format; can be rerun with new probs
    def set_min_probability(self, min_p, update_attrs=False):
        self.df["most_probable"], self.df["probability"] = zip(*self.df["probs"].apply(lambda x: x.most_probable(min_p)))

        if update_attrs:
            self.samples.update_edges(self.df[["id1","id2"]].values, self.samples["most_probable"].values, "most_probable")
            self.samples.update_edges(self.df[["id1","id2"]].values, self.samples["most_probable"].values, "probability")

    # recomputes probs across all rows of df
    def compute_probs(self):
        self.df["probs"].apply(lambda x: x.compute_probs())

    # pickles the object
    def pickle_it(self, output):
        self.df = None
        # self.samples = None
        i = open(output, "wb")
        pkl.dump(self, i)
        i.close()

    def most_likely_among(self, nodes, update_attrs=False):
        likely_among = self.df["probs"].apply(lambda x: nodes[np.argmax([x.hier.nodes[node]["p"] for node in nodes])]).values

        if update_attrs:
            self.samples.update_edges(zip(self.df[["id1","id2"]].values, likely_among), "likely_among")

        else:
            return likely_among






    



def PONDEROSA(**kwargs):
    # samples = SampleData(fam_file="for_dev/Himba_allPO.fam",
    #                 king_file="for_dev/King_Relatedness_no2278.seg",
    #                 ibd_file="for_dev/Himba_shapeit.chr1_segments.txt",
    #                 map_file="for_dev/newHimba_shapeit.chr1.map",
    #                 code_yaml="tree_codes.yaml")
    
    # i = open("for_dev/sample.pkl", "wb")
    # pkl.dump(samples, i)
    # i.close()

    i = open("for_dev/sample.pkl", "rb")
    samples = pkl.load(i)

    pedigree = Pedigree(samples=samples, pedigree_file="pedigree_codes.yaml", tree_file="tree_codes.yaml")
    pedigree.find_all_relationships()

    pairs = pedigree.hier

    training = Classifiers(pairs)



    if kwargs.get("assess", True):

        unknown_df = samples.to_dataframe(pairs.get_pairs("relatives"), include_edges=True).dropna(subset=["k_prop"])

    else:

        # get all close relatives to exclude
        found_close = list(pairs.get_nodes_from(["2nd", "PO", "FS"]))

        # make a dataframe of all pairs that are not close relatives
        unknown_df = samples.to_dataframe(found_close, include_edges=False)

    unknown_df = unknown_df.reset_index()


    unknown_df["ibd_cov"] = unknown_df.apply(lambda x: x.ibd1 + x.ibd2, axis=1)

    # predict the degree of relatedness
    # unknown_df["predicted_degree"] = training.predict("degree", unknown_df[["k_ibd1", "k_ibd2"]].values)


    # get the degree probabilities
    probs, labels = training.predict_proba("degree", unknown_df[["k_ibd1", "k_ibd2"]].values)

    # add the probabilities to the tree
    for (_, row), prob in zip(unknown_df.iterrows(), probs):
        row["probs"].add_probs(list(zip(next(labels), prob)), "ibd")

    if kwargs.get("assess", True):

        second_pairs = samples.to_dataframe(pairs.get_pairs("2nd"), include_edges=True)[["id1", "id2"]]
        second = unknown_df.merge(second_pairs, on=["id1", "id2"], how="inner")

    else:
        second = unknown_df[unknown_df["probs"].apply(lambda x: x.hier.nodes["2nd"]["p_con"] > 0.2)]


    # get the n_ibd segs classifier probabilities
    probs, labels = training.predict_proba("n", second[["ibd_cov", "n"]].values)
    # add the probabilities to the tree
    for (_, row), prob in zip(second.iterrows(), probs):
        row["probs"].add_probs(list(zip(next(labels), prob)), "nsegs")

        # for each of these nodes, take the sum of the two children
        for node in ["GP", "GPAV", "HS"]:
            # array of the children probabilities
            child_probs = [row["probs"].hier.nodes[i]["p_con"] for i in row["probs"].hier.successors(node)]
            # add the probability
            row["probs"].add_probs(node, p_con=sum(child_probs), method="nsegs")


    # get the probabilities from the hap score classifier
    probs, labels = training.predict_proba("hap", second.apply(lambda x: sorted([h for _,h in x.h.items()], reverse=True), axis=1).values.tolist())
    
    for (_, row), prob in zip(second.iterrows(), probs):
        # get the index of the Phase error class
        classes = list(next(labels))
        pe_index = classes.index("Phase error"); del classes[pe_index]
        # the chance of high Phase error is high; do not update the HS or GPAV probabilities
        if prob[pe_index] < 0.2:
            row["probs"].add_probs(list(zip(classes, np.delete(prob, pe_index))), "hap")

    samples.edge_subset(unknown_df[["id1", "id2"]].apply(lambda x: tuple(x), axis=1).values, inplace=True)

    relatives_obj = ResultsDataFrame(samples=samples, pairs=pairs, df=unknown_df)
    relatives_obj.compute_probs()
    relatives_obj.set_min_probability(kwargs.get("min_p", 0.5))
    relatives_obj.write_readable(kwargs.get("output", "test.txt"))
    relatives_obj.subset_samples(lambda x: x.probs.hier.nodes["2nd"]["p"] > 0.5)
    relatives_obj.pickle_it("test.pkl")


    # relatives_obj.subset_dataframe(lambda x: x.probs.hier.nodes["2nd"]["p"]>0.9, inplace=True)

    # relatives_obj.most_likely_among(["HS", "AV", "GP"], True)

    # import pdb; pdb.set_trace()
    # relatives_obj = ResultsDataFrame(pd.concat([second, unknown_df[~unknown_df["index"].isin(second_index)]]).reset_index(drop=True))

    # relatives_obj.compute_probs()

    # relatives_obj.set_min_probability(kwargs.get("min_p", 0.5))

    # relatives_obj.pickle_it("test.pkl")

    


    # all_relatives["probs"].apply(lambda x: x.compute_probs())

    # all_relatives["most_probable"], all_relatives["probability"] = zip(*all_relatives["probs"].apply(lambda x: x.most_probable(0.8)))

    # import pdb; pdb.set_trace()


    # all_relatives[["id1", "id2", "k_ibd1", "k_ibd2", "n", "predicted_degree", "most_probable", "probability"]].to_csv("delete.txt", index=False, sep=" ")

    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    # PONDEROSA(assess=False)
    # i = open("for_dev/Ponderosa_results.pkl", "rb")
    # obj = pkl.load(i)
    # import pdb; pdb.set_trace()

    df = pd.read_csv("for_dev/Ponderosa.txt", delim_whitespace=True)
    sns.scatterplot(data=df, x="k_ibd1", y="k_ibd2", hue="degree")
    plt.savefig("for_dev/Ponderosa_Himba.png")


# print(Ped.hier.get_pair_df_from(["GPAV", "HS"]).dropna(subset="h"))

# print(Ped.hier.get_pair_df("2nd").dropna(subset="n"))


# import pdb; pdb.set_trace()


# yaml_file = "pedigree_codes.yaml"
# Ped = Pedigree(samples=samples, yaml_file=yaml_file)
# Ped.find_all_relationships()

# # merge the samples ibd data in
# Ped.hier.update_attr_from([[(id1, id2), "ibd_data", samples.g.get_edge_data(id1, id2)] for id1,id2 in Ped.hier.get_pairs("relatives")])

# import pdb; pdb.set_trace()


