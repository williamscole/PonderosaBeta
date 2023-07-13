import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import argparse
import os
from math import floor, ceil
import itertools as it
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from pedigree_tools import ProcessSegments, PedigreeHierarchy, SiblingClassifier, TrainPonderosa

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

        ### init the graph and set defaults
        g = nx.Graph()
        g.add_nodes_from(fam.apply(lambda x: (x[1], {"sex": x[4],
                                                    "mother": x[3] if x[3] != "0" else None,
                                                    "father": x[2] if x[2] != "0" else None,
                                                    "children": [],
                                                    "age": np.nan,
                                                    "pop": "pop1"}), axis=1).values)

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
            nx.set_node_attributes(g, pop_attrs, "pop")

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

        # store the genome_len
        self.genome_len = genome_len

        ### load ibd, process ibd segs
        ibd_file = kwargs.get("ibd_file", "")
        if os.path.exists(ibd_file):
            print("Processing IBD segments...")
            ibd_files = get_file_names(ibd_file)
            # load ibd files
            ibd_df = pd.concat([pd.read_csv(filen, delim_whitespace=True, dtype={"id1": str, "id2": str})
                                for filen in ibd_files])

            ibd_df["l"] = ibd_df["end_cm"] - ibd_df["start_cm"]

            # iterate through the pairs of individuals, compute ibd stats
            n_pairs = 0
            tot_pairs = len([1 for _ in ibd_df.groupby(["id1", "id2"])])
            for (id1, id2), pair_df in ibd_df.groupby(["id1", "id2"]):

                # if distant relative or nodes not in the fam file, ignore
                if ibd_df["l"].sum() < 400\
                    or id1 not in g.nodes\
                    or id2 not in g.nodes\
                    or id1 == id2\
                    or g.nodes[id1]["pop"] != popln\
                    or g.nodes[id2]["pop"] != popln:
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
                # set the root post prob to 1
                hier.hier.nodes["relatives"]["post"] = 1
                hier.hier.nodes["relatives"]["p"] = 1

                # add ibd1 data and initialze probs
                g.add_edge(id1, id2, ibd1=ibd_data.ibd1,
                                     ibd2=ibd_data.ibd2,
                                     h={id1: ibd_data.h1, id2: ibd_data.h2},
                                     n=ibd_data.n,
                                     k=(ibd_data.ibd1/2 + ibd_data.ibd2),
                                     probs=hier,
                                     segments=pair_df)

                # check to see if parents
                if ibd_data.ibd1 > 0.75:
                    hier.set_attrs({"1st": 1, "PO": 1, "FS": 0}, "p")

                n_pairs += 1
                if n_pairs % 100 == 0:
                    print(f"{n_pairs} of {tot_pairs} pairs processed.")

                if n_pairs > kwargs.get("n_pairs", np.inf):
                    break

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
    
    def get_subset(self, f):
        # get the nodes based on the conditional f
        node_subset = self.get_nodes(f)
        # return the subset for which the conditional is true
        return self.g.subgraph(set(node_subset))

    # update the hier probabilities
    def update_probs(self, pair_list, probs_list, prob_labs, method):
        # iterate through the probabilities
        for (id1, id2), probs in zip(pair_list, probs_list):
            self.g.edges[id1, id2]["probs"].set_attrs({i: p for i,p in zip(prob_labs, probs)}, "p")
            self.g.edges[id1, id2]["probs"].set_attrs({i: method for i in prob_labs}, "method")


class Pedigree:
    '''Takes as input a networkx graph where nodes have attributes:
        children: list
        father: str (None if missing)
        mother: str (None if missing)
        age: float (np.nan if missing)
        sex: str ("1" if male "2" if female)'''
    def __init__(self, g, degree_classifier=None):

        # specify the relationships
        # rels = {"relatives": ["1st", "2nd", "3rd", "4th"],
        #         "1st": ["PO", "FS"], "2nd": ["HS", "GPAV", "DCO"], "3rd": ["CO", "GGP", "HAV", "DHCO"], "4th": ["HCO", "GGGP"], "GPAV": ["GP", "AV"],
        #         "GP": ["MGP", "PGP"], "HS": ["MHS", "PHS"]}

        # init pedigree obj to store the relationships
        # self.rels = PedigreeHierarchy(hier=list(it.chain(*[[[parent, child] for child in child_l] for parent, child_l in rels.items()])))
        self.pairs = PedigreeHierarchy()
        # set attributes 
        # self.rels.set_attrs({i: set() for i in self.rels.get_nodes("relatives")}, "pairs")

        # pedigree object is a directed graph which directs parent --> child
        ped = nx.DiGraph()
        ped.add_nodes_from(g.nodes)

        # iterate through the nodes in the input graph and add their parental edges
        for node, attrs in g.nodes(data=True):
            # info about their relatives
            mother = attrs["mother"]; father = attrs["father"]; children = attrs["children"]

            # age and sex info
            age = attrs["age"]; sex = attrs["sex"]

            # set attrs
            ped.nodes[node]["age"] = age
            ped.nodes[node]["sex"] = sex

            # list of all the PO relationships to add
            po_list = [[father, node], [mother, node]] + [[node, child] for child in children]

            # iterate through the list
            for parent, child in po_list:
                # add the nodes; parent node may be None, in which case it is renamed to "remove"
                ped.add_edge({parent: parent, None: "remove"}[parent], child)

        # remove the remove dummy node and assign to the class
        self.dummy, self.ped = SiblingClassifier(ped.subgraph(set(ped.nodes) - {"remove"}), g, 0, degree_classifier)


    # adds a pair to a relationship category
    def add_pair(self, id1, id2, rel):
        # add the pair
        self.pairs.add_relative(rel, (id1, id2))
        # self.rels.hier.nodes[rel]["pairs"] |= {(id1, id2)}

    def find_relationship(self, paths):
        # keeps all of the legal paths
        out_paths = []

        # iterate through the paths
        for path in paths:

            # self looping path
            if len(path) == 1:
                continue

            # path_dir describes the path between the two nodes. 1 indicates moving up the ped, -1 indicates moving down the ped
            path_dir = [1 if self.ped.get_edge_data(path[i], path[i+1]) == None else -1 for i in np.arange(len(path)-1)]

            # is a lineal relationship
            if path_dir.count(-1) == 0 or path_dir.count(1) == 0:

                # id1 is the older individual if the path starts with -1; in this case, flip everything so id2 is listed first
                if path_dir[0] == -1:
                    path_dir = [1 for _ in path_dir]
                    path = path[::-1]

                # get the sex of the younger individual's parent
                sex = self.ped.nodes[path[1]]["sex"]

                out_paths.append((path, tuple(path_dir), sex))
                continue

            # find index of the first descent in the pedigree and then check that there are no more ascents after
            # once we go down in the pedigree, we can't go back up; if this bool is true, continue
            if path_dir[path_dir.index(-1):].count(1) > 0:
                continue

            # we want the first node to be the genetically younger individual; we we ensure that this is true
            # Rule: if id1 is genetically older they are closer to the tmrca
            if len(path_dir[:path_dir.index(-1)]) < len(path_dir[path_dir.index(-1):]):
                path_dir = list(np.array(path_dir)*-1)[::-1]
                path = path[::-1]

            # get the sex of the first individual's parent
            sex = self.ped.nodes[path[1]]["sex"]

            out_paths.append((path, tuple(path_dir), sex))

        if len(out_paths) == 0:
            return np.nan, np.nan

        # get a df of all the paths
        paths_df = pd.DataFrame(out_paths, columns=["path_ids", "path", "sex"])

        # the ibd1 from the given relationship
        paths_df["ibd1"] = paths_df["path"].apply(lambda x: 0.5**(len(x)-1))

        # id1 is the genetically younger ind, id2 is the older
        paths_df["id1"] = paths_df["path_ids"].apply(lambda x: x[0])
        paths_df["id2"] = paths_df["path_ids"].apply(lambda x: x[-1])

        # iterate through each path type
        for path, path_df in paths_df.groupby("path"):

            # lineal relationship
            if np.abs(path).sum() == sum(path):

                # default rel; change for PO andGP
                rel = (len(path)-1)*"G" + "P"
                prefix = ""

                # they are GP
                if path == (1,):
                    rel = "PO"

                # they are GP/GC add the sex
                elif path == (1, 1):
                    sex = path_df.iloc[0]["sex"]
                    prefix = "P" if sex == "1" else ("M" if sex == "2" else "")

                self.add_pair(*path_df.iloc[0][["id1", "id2"]], prefix + rel)                
        
            # they are siblings (either FS or HS)
            elif path == (1, -1):
                
                # half siblings
                if path_df.shape[0] == 1:
                    sex = path_df.iloc[0]["sex"]
                    prefix = "P" if sex == "1" else ("M" if sex == "2" else "")
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], prefix + "HS")

                # full siblings
                if path_df.shape[0] == 2:
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], "FS")

            # they are avuncular
            elif path == (1, 1, -1):

                # half-avuncular
                if path_df.shape[0] == 1:
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], "HAV")

                # full avuncular
                else:
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], "AV")

            # they are cousins
            elif path == (1, 1, -1, -1):

                # half-cousins
                if path_df.shape[0] == 1:
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], "HCO")

                # double cousins
                elif path_df.shape[0] == 4:
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], "DCO")

                # full cousins
                elif path_df.shape[0] == 2 and len(set(path_df["sex"])) == 1:
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], "CO")

                # double half-cousins
                elif path_df.shape[0] == 2 and len(set(path_df["sex"])) == 2:
                    self.add_pair(*path_df.iloc[0][["id1", "id2"]], "DHCO")

        # get the kinship through each parent
        k1, k2 = [paths_df[paths_df.sex==sex]["ibd1"].sum() for sex in ["1", "2"]]

        # return the expected IBD1, IBD2
        return (k1*(1-k2)) + (k2*(1-k1)), k1*k2

    def find_relationships(self):

        # temp graph that is undirected
        tmp = self.ped.to_undirected()

        # keep track of expected IBD sharing
        eIBD = nx.Graph()

        ### iterate through all pairs
        for id1, id2 in it.combinations(self.ped.nodes, r=2):

            # find and add the relationshipsl; return the expected IBD sharing
            eIBD1, eIBD2 = self.find_relationship(nx.all_simple_paths(tmp, min(id1, id2), max(id1, id2), cutoff=6))

            # add to the expected IBD graph
            if eIBD1 > 0:
                eIBD.add_edge(id1, id2, ibd1=eIBD1, ibd2=eIBD2)

        self.eIBD = eIBD

    # returns all pairs that are under the umbrella (inclusive) of a relationship (root)
    def get_relationships(self, root):
        # keep track of pairs
        out_pairs = self.rels.hier.nodes[root]["pairs"]
        
        # iterate through the descendants
        for child in nx.descendants(self.rels.hier, root):
            out_pairs |= self.rels.hier.nodes[child]["pairs"]

        return out_pairs



        
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
