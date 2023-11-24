import numpy as np
import networkx as nx
import yaml
import itertools as it
from collections import Counter

from Ponderosa import SampleData
from pedigree_tools import PedigreeHierarchy


class Relationship:
    def __init__(self, data):
        p1 = sorted([[1] + path for path in data.get(1, [])], key=lambda x: x[1:])
        p2 = sorted([[2] + path for path in data.get(2, [])], key=lambda x: x[1:])

        # support for inter-generational relationships
        longest_path = max([len(path) for path in p1 + p2])
        matrix = np.array([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2])

        # set attrs
        self.sex_specific = data["sex"]
        self.mat = matrix if self.sex_specific else matrix[:,1:]

    # returns boolean if it is the given relationship
    def is_relationship(self, mat):
        # convert the path dict to a matrix
        mat = mat if self.sex_specific else mat[:,1:]

        # not the same shape == can't be the relationship
        if self.mat.shape != mat.shape:
            return False

        # returns True if the relationship matches
        return (self.mat - mat).sum() == 0
        

class RelationshipCodes:
    def __init__(self, yaml_file):
        # open the code yaml
        i = open(yaml_file, "r")
        code_yaml = yaml.safe_load(i)

        # load each relationship, create the matrix
        self.codes = []
        for name, data in code_yaml.items():
            r = Relationship(data)
            self.codes.append([name, r])

        # sort such that the sex-specific relationships come first
        self.codes.sort(key=lambda x: int(x[1].sex_specific), reverse=True)

    # converts the path dictionary to a matrix
    def convert_to_matrix(self, path_dict):
        # get the paths
        p1 = sorted([[1] + [direction for _, direction in path] for path in path_dict[1]], key=lambda x: x[1:])
        p2 = sorted([[2] + [direction for _, direction in path] for path in path_dict[2]], key=lambda x: x[1:])

        # get the kinships
        k1 = sum([0.5**(len(path)-2) for path in p1])
        k2 = sum([0.5**(len(path)-2) for path in p2])

        # support for inter-generational relationships
        longest_path = max([len(path) for path in p1 + p2])
        mat = np.array([path + [0 for _ in range(longest_path - len(path))] for path in p1 + p2])

        # returns the matrix and expected ibd1, ibd2 values
        return mat, k1*(1-k2) + k2*(1-k1), k1*k2

    # returns the relationship
    def determine_relationship(self, path_dict):
        # get the matrix and expected ibd values
        mat, ibd1, ibd2 = self.convert_to_matrix(path_dict)

        # the pair are the same generation
        same_gen = mat[:,1:].sum() == 0

        # iterate through the relationships
        for name, robj in self.codes:
            # boolean if the relationship is true
            if robj.is_relationship(mat):
                return name, ibd1, ibd2, mat, same_gen

        ### haven't found a relationship, need to make sure that it's not a reversed code

        # get the first column of the matrix
        pcol = mat[:,:1]
        # reverse the direction of the rest of the matrix and flip along the horizontal
        tmp = np.flip(mat[:,1:]*-1)
        # add the parent column to the flipped matrix
        rv_mat = np.append(pcol, tmp, axis=1)
        # bound at least one possible relationship that it could be
        found = sum([robj.is_relationship(rv_mat) for _, robj in self.codes]) > 0

        # relationship is not found
        return "nan" if found else "unknown", ibd1, ibd2, mat, same_gen

class Pedigree:
    def __init__(self, **kwargs):

        po_list = kwargs.get("po_list", [])
        samples_g = kwargs.get("samples", None)

        # a list of po-pairs has been supplied
        if len(po_list) > 0:
            po = nx.DiGraph()
            po.add_edges_from(po_list)

        # samples graph has been supplied
        elif len(samples_g.g.nodes) > 0:
            tmp = nx.DiGraph()
            tmp.add_edges_from(it.chain(*[[[data["mother"], node], [data["father"], node]] for node, data in samples.g.nodes(data=True)]))
            po = tmp.subgraph(set(tmp.nodes) - {np.nan})

        self.po = po

        # must supply the yaml file
        self.R = RelationshipCodes(kwargs["yaml_file"])

        # create the pedigree hierarchy
        self.hier = PedigreeHierarchy(kwargs["yaml_file"])

    # for a given focal individual, finds all relationships
    def focal_relationships(self, focal):
        # recursive function that, given a focal individual returns all paths to relatives
        def get_paths(cur_relative, path_ids, path_dirs, paths):
            # init the next set of relatives
            next_set = []

            # past the first iteration, so we can get down nodes, but only down nodes that are not in the path
            if len(path_dirs) > 1:
                next_set += [(nxt_relative,-1) for nxt_relative in self.po.successors(cur_relative) if nxt_relative not in path_ids]

            # we're still moving up, so we can get the up nodes
            if path_dirs[-1] > 0:
                next_set += [(nxt_relative, 1) for nxt_relative in self.po.predecessors(cur_relative)]

            # we can't keep going; base case
            if len(next_set) == 0:
                paths.append((path_ids, path_dirs))
                return paths

            # iterate through the new set of relatives
            for nxt_relative, direction in next_set:
                paths = get_paths(nxt_relative, path_ids + [nxt_relative], path_dirs + [direction], paths)
            return paths

        # given the output of get_paths, creates all sub-paths
        def merge_paths(focal, paths):
            # init the dict to store each relative pair and the paths along each parental lineage
            rel_pairs = {id2: {1: set(), 2: set()} for id2 in it.chain(*[path_ids for path_ids,_ in paths])}

            # iterate through the paths
            for path_ids, path_dirs in paths:
                # zip the path ids and the path directions
                path = [(id2, pdir) for id2, pdir in zip(path_ids, path_dirs)]

                # iterate through each person in the path
                for index in range(1, len(path)):
                    # get the id of the relative
                    id2 = path[index][0]
                    # get the subpath from the focal to the current id2
                    subpath = path[1:index+1]
                    # determine which parent they are related through
                    parent_sex = samples.g.nodes[subpath[0][0]]["sex"]
                    # add to the rel pairs dictionary
                    rel_pairs[id2][int(parent_sex)] |= {tuple(path[1:index+1])}
            
            return rel_pairs

        # get all paths
        path_list = get_paths(focal, [focal], [1], [])

        # keep track of unknown relationships
        unknown_rels = []

        # iterate through each relative of the focal individual
        for id2, path_dict in merge_paths(focal, path_list).items():
            if focal == id2:
                continue

            # get the relationship 
            rname, e_ibd1, e_ibd2, mat, same_gen = self.R.determine_relationship(path_dict)

            # don't want to add if rname is nan or they are same generation and id2 > id1
            if rname == "nan" or (same_gen and focal > id2):
                continue

            if rname == "unknown":
                unknown_rels.append(np.array2string(mat[:,1:]))

            self.hier.add_pair((focal, id2), rname, {"ibd1": e_ibd1, "ibd2": e_ibd2})
        
        return unknown_rels

    # finds all relationships for nodes in the graph
    def find_all_relationships(self):

        unknown_rels = it.chain(*[self.focal_relationships(focal) for focal in self.po.nodes])

        print("The following unknown relationships were found:")

        for unkr, n in Counter(unknown_rels).items():
            print(f"{n} of the following were found:")
            print(unkr + "\n")
        
samples = SampleData(fam_file="for_dev/Himba_allPO.fam",
                     king_file="for_dev/King_Relatedness_no2278.seg")

yaml_file = "relationship_codes.yaml"
Ped = Pedigree(samples=samples, yaml_file=yaml_file)
Ped.find_all_relationships()

