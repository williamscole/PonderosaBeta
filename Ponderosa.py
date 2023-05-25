import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
import argparse
from pedigree_tools import ProcessSegments

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibd", help = "IBD segment file. If multiple files for each chromosome, this is the path for chromosome 1.", required=True)
    # parser.add_argument("--fam", "PLINK-formated .fam file")
    # parser.add_argument("--ages", "Age file. First column is the IID, second column is the age", default=None)
    parser.add_argument("--map", help = "PLINK-formatted .map file.")
    parser.add_argument("--directory", help = "directory name of the simulations. Must include the /")
    args = parser.parse_args()
    return args

class Ponderosa:

    def __init__(self, ibd, map, ages):

        ### init ibd file
        if "chr1" in ibd:

            self.ibd = pd.concat([pd.read_csv(ibd.replace("chr1", f"chr{chrom}"), delim_whitespace=True) for chrom in range(1, 23)])

        else:

            self.ibd = pd.read_csv(ibd, delim_whitespace=True)

        ### init ages
        if ages != None:

            self.ages = {iid: int(age) for iid, age in np.loadtxt(ages, dtype=str)}

        else:

            self.ages = {}

        ### get the genome len
        if "chr1" in map:

            self.l = 0
            for chrom in range(1, 23):
                tmp = pd.read_csv(map.replace("chr1", f"chr{chrom}"), delim_whitespace=True, header=None)[2].values
                self.l += (tmp[-1] - tmp[0])

        else:

            tmp = pd.read_csv(map, delim_whitespace=True, header=None)
            self.l = sum([chrom_df[2].values[-1] - chrom_df[2].values[0] for chrom, chrom_df in tmp.groupby(0)])

        ### process the ibd
        self.ibd["l"] = self.ibd["end_cm"] - self.ibd["start_cm"]

        self.ibdnet = nx.Graph()
        for pair, pair_df in self.ibd.groupby(["id1", "id2"]):

            if pair_df["l"].sum() < 1000:
                continue

            p = ProcessSegments(pair_df)
            data = p.ponderosa_data(self.l)

            id1, id2 = pair

            self.ibdnet.add_edge(id1, id2, ibd1 = data.ibd1, ibd2 = data.ibd2, h = {id1: data.h1, id2: data.h2}, n = data.n)

        print("Done loading IBD")

    def degree_classif(self, pkl_classifier):

        f = open(pkl_classifier, "rb")
        classif = pkl.load(f)

        for id1, id2, data in self.ibdnet.edges(data=True):

            # order is [FS, 2nd, 3rd, 4th]
            p = classif.predict_proba([[data["ibd1"], data["ibd2"]]])

            self.ibdnet.edges[id1, id2]["degree_p"] = p[0]

    def n_classif(self, pkl_classifier):

        f = open(pkl_classifier, "rb")
        classif = pkl.load(f)

        for id1, id2, data in self.ibdnet.edges(data=True):

            if data["degree_p"][1] < 0.80:

                self.ibdnet.edges[id1, id2]["n_p"] = [np.nan for _ in range(4)]

                continue

            p = classif.predict_proba([[data["n"], 0.5*data["ibd1"] + data["ibd2"]]])

            self.ibdnet.edges[id1, id2]["n_p"] = p[0]

        return list(classif.classes_)

    def hsr_classif(self, pkl_classifier):

        f = open(pkl_classifier, "rb")
        classif = pkl.load(f)

        for id1, id2, data in self.ibdnet.edges(data=True):

            if data["degree_p"][1] < 0.80:

                self.ibdnet.edges[id1, id2]["hsr_p"] = [np.nan for _ in range(4)]

                continue

            p = classif.predict_proba([[data["h"][id1], data["h"][id2]]])

            self.ibdnet.edges[id1, id2]["hsr_p"] = p[0]

        return list(classif.classes_)

    def classify(self, directory):

        self.degree_classif(f"{directory}degree_classifier.pkl")
        hap_class = self.hsr_classif(f"{directory}hap_classifier.pkl")
        nsegs_class = self.n_classif(f"{directory}nsegs_classifier.pkl")

        output = []
        for id1, id2, data in self.ibdnet.edges(data=True):

            r = [id1, id2, data["ibd1"], data["ibd2"], data["h"][id1], data["h"][id2], data["n"]]
            r += list(data["degree_p"])
            r += list(data["hsr_p"])
            r += list(data["n_p"])

            output.append(r)

        columns = ["id1", "id2", "ibd1", "ibd2", "h1", "h2", "n_segs"]
        columns += ["FS", "2nd", "3rd", "4th"]
        columns += hap_class
        columns += nsegs_class


        out = pd.DataFrame(output, columns=columns)
        out.to_csv("Ponderosa_results.txt", index=False, sep="\t")


if __name__ == "__main__":
    args = parse_args()

    p = Ponderosa(args.ibd, args.map, None)
    p.classify(args.directory)
    
    
# p = Ponderosa("Himba_phasedibd_segments.txt", "../ponderosa/plink_data/newHimba_shapeit.chr1.map", None)
# p.classify("ponderosa_pops/")


# # inputs
# # h1, h2, tot_ibd, hscore_bool, age1, age2, N, dad1, mom1, dad2, dad2
# # h_gmm, N_lda

# with open('/Users/cole/Desktop/brown/ponderosa/test_lda.pkl', 'rb') as f:
#     N_lda = pkl.load(f)

# with open('/Users/cole/Desktop/brown/ponderosa/test_gmm.pkl', 'rb') as f:
#     h_gmm = pkl.load(f)


# def get_probs(h1, h2, tot_cov, N):
#     return N_lda.predict_proba([[tot_cov, N]])[0], h_gmm.predict_proba([[h1, h2]])[0]


# ### age functions

# # returns the Pr of being GP versus AV using age data
# def GPAV_ages(age1, age2):
#     if abs(age1 - age2) < 30:
#         return 0
#     return 0.5

# # returns the Pr of being PHS versus MHS using age data
# def HS_ages(age1, age2):
#     if abs(age1 - age2) > 35:
#         return 1
#     return 0.5
    



# ### calculates Pr(HS); 1 - Pr(HS) = Pr(GP or AV)
# def pr_HS(hscore_bool, h_probs, N_probs, dad1, mom1, dad2, mom2, age1, age2):
#     # hscore_bool True tells us to use the h classifier
#     if hscore_bool:
#         # recalculate the probabilities
#         h_probs = h_probs[:3] / sum(h_probs[:3])
#         pr_hs = h_probs[2]
#     # not using h1, h2 here so we use the N classifier
#     else:
#         # if both of the following if statements are true, they can't be HS; this variable keeps track of that
#         n = 0
#         # if they can't be MHS, then the Pr of HS from the classifier just uses the PHS component
#         if mom1 or mom2:
#             pr_hs = N_probs[4] / (1 - N_probs[2])
#             n += 1
#         # same logic as above, but can't be PHS
#         if dad1 or dad2:
#             pr_hs = N_probs[2] / (1 - N_probs[4])
#             n += 1
#         # both statements are true; they cannot be HS
#         if n > 1:
#             pr_hs = 0
#         # if none of the statements are true, weight the probabilities by age probabilities
#         if n == 0:
#             pr_PHS_age = HS_ages(age1, age2) # prob of being PHS vs. MHS using age data only
#             pr_PHS_N = N_probs[4] # prob of being PHS sharing N segs
#             pr_MHS_N = N_probs[2]
#             # created a step function that peaks at Pr(MHS | N) + Pr(PHS | N) when pr_PHS_age = pr_MHS_age = 0.5
#             # otherwise weights it such taht Pr(HS) = Pr(PHS | N) when Pr(MHS | age) --> 0
#             if pr_PHS_age <= 0.5:
#                 pr_hs = (2 * pr_PHS_N * pr_PHS_age) + pr_MHS_N
#             else:
#                 pr_hs = (-2 * pr_MHS_N * pr_PHS_age) + (2 * pr_MHS_N) + pr_PHS_N
#     # return HS prob
#     return pr_hs

# # calculates the prob that they are PHS given they are HS; this equals 1 - Pr(MHS)
# # use existing parent data, ages, and lastly the number of segments to determine this
# def pr_PHS(N_probs, age1, age2, dad1, mom1, dad2, mom2):
#     # if one of them has a father, then Pr(PHS) = 0
#     if dad1 or dad2:
#         return 0
#     # if one of them has a mother, then Pr(MHS) = 0
#     if mom1 or mom2:
#         return 1
#     # from N, the prob of being PHS
#     pr_PHS_N = N_probs[4] / (N_probs[2] + N_probs[4])
#     # rescale the pr_PHS_N with the age prob
#     pr_PHS_age = HS_ages(age1, age2)
#     # get the prob of being MHS
#     pr_MHS = (1 - pr_PHS_N) * (1 - pr_PHS_age)
#     # return the rescaled prob of being PHS
#     return (pr_PHS_N * pr_PHS_age) / ((pr_PHS_N * pr_PHS_age) + pr_MHS)

# # calculates the prob of the pair being GP given they are GP or AV; this is 1 - Pr(AV)
# def pr_GP(N_probs, age1, age2):
#     # first use the function GPAV_ages to calculate the prob of being GP (versus AV) based on the ages
#     pr_GP_age = GPAV_ages(age1, age2)
#     # next take the prob of being GP using N, which is [Pr(MGP) + Pr(PGP)] / Pr(GP or AV)
#     pr_GP_N = (N_probs[1] + N_probs[3]) / (N_probs[0] + N_probs[1] + N_probs[3])
#     # multiply the complement of the probabilities from above to get the Pr of being avuncular
#     pr_AV = (1 - pr_GP_age) * (1 - pr_GP_N)
#     # return the prob of being GP
#     return (pr_GP_age * pr_GP_N) / ((pr_GP_age * pr_GP_N) + pr_AV)

# ### calculates the prob that ID1 is the grandparent of ID2 given they are GP
# def pr_GP1(hscore_bool, h_gmm, mom1, dad1, dad2, mom2, age1, age2):
#     # if using the haplotype score
#     if hscore_bool:
#         # Pr(GP1) / [Pr(GP1) + Pr(GP2)]
#         return h_gmm[0] / sum(h_gmm[:2])
#     # ID2 has both parents in dataset, so ID1 can't be their grandparent
#     if dad2 and mom2:
#         return 0
#     # ID1 has both parents in the dataset, so ID1 must be the grandparent
#     if dad1 and mom1:
#         return 1
#     # ID1 is older than ID2 --> ID1 must be grandparent
#     if age1 > age2 or age2 < 30:
#         return 1
#     # ID2 is older than ID1 --> ID1 cant be grandparent
#     if age1 < age2 or age1 < 30:
#         return 0
#     # if no other information, set the prob to 1/2
#     return 0.5

# # returns prob of ID1 being the uncle/aunt of ID2 given they are AV
# def pr_AV1(hscore_bool, h_probs, age1, age2, mom1, dad1, mom2, dad2, use_age):
#     # use the haplotype scores if possible
#     if hscore_bool:
#         return h_probs[0] / sum(h_probs[:2])
#     # ID1 has both parents in the dataset so it must be the aunt/uncle
#     if dad1 and mom1:
#         return 1
#     # ID2 has both parents in the dataset, so ID1 can't be the aunt/uncle
#     if dad2 and mom2:
#         return 0
#     if age1 > age2 and use_age:
#         return 1
#     if age1 < age2 and use_age:
#         return 0
#     # there's no other information
#     return 0.5

# # returns Pr of the given GP being a paternal grandparent
# def pr_PGP(N_probs, gc_dad, gc_mom):
#     # grandchild has a dad in the dataset, so can't be a paternal grandparent
#     if gc_dad:
#         return 0
#     # grandchild has a mom in the dataset, so must be a maternal grandparent
#     if gc_mom:
#         return 1
#     # otherwise use the IBD segments
#     return N_probs[3] / (N_probs[1] + N_probs[3])




    
# h1, h2 = 0.65, 0.65
# N = 80
# dad1, mom1 = False, False
# dad2, mom2 = False, False
# age1 = 39
# age2 = 50
# tot_cov = 1800
# id1, id2 = "A", "B"

# N_probs, h_probs = get_probs(h1, h2, tot_cov, N)

# def probs(id1, id2, N_probs, h_probs, h1, h2, N, dad1, mom1, dad2, mom2, age1, age2):
    
#     # boolean decides if the h classifier should be used
#     hscore_bool = h_probs[3] < 0.5

#     # calculates each of the conditional probabilities
#     HS = pr_HS(hscore_bool, h_probs, N_probs, dad1, mom1, dad2, mom2, age1, age2)
#     PHS_HS = pr_PHS(N_probs, age1, age2, dad1, mom1, dad2, mom2)
#     GP_GPAV = pr_GP(N_probs, age1, age2)
#     AV1_AV = pr_AV1(hscore_bool, h_probs, age1, age2, dad1, mom1, dad2, mom2, True)
#     GP1_GP = pr_GP1(hscore_bool, h_probs, dad1, mom1, dad2, mom2, age1, age2)
#     PGP1_GP1 = pr_PGP(N_probs, dad2, mom2)
#     PGP2_GP2 = pr_PGP(N_probs, dad1, mom1)

#     # creates the graph network to store the probabilities
#     tree = nx.DiGraph()
#     tree.add_node("2nd", P = 1)

#     # the edges in the tree and the associated probabilities
#     edges = [["2nd", "HS", HS], ["2nd", "GP/AV", 1 - HS],
#             ["HS", "PHS", PHS_HS], ["HS", "MHS", 1 - PHS_HS], ["GP/AV", "GP", GP_GPAV], ["GP/AV", "AV", 1 - GP_GPAV],
#             ["GP", "GP1", GP1_GP], ["GP", "GP2", 1 - GP1_GP], ["AV", "AV1", AV1_AV], ["AV", "AV2", 1 - AV1_AV],
#             ["GP1", "PGP1", PGP1_GP1], ["GP1", "MGP1", 1 - PGP1_GP1], ["GP2", "PGP2", PGP2_GP2], ["GP2", "MGP2", 1 - PGP2_GP2]]

#     # for each edge, add the child node and its prob by taking the probability assoc. with the edge and multiplying by the Pr of the parent node
#     for node1, node2, p in edges:
#         tree.add_edge(node1, node2)
#         tree.nodes[node2]["P"] = tree.nodes[node1]["P"] * p

#     # output the data
#     row = [id1, id2, h1, h2, N, age1, age2]
#     col_names = ["ID1", "ID2", "H1", "H2", "N", "AGE1", "AGE2", "90%", "90%_p", "75%", "75%_p", "50%", "50%_p"]

#     # give the most specific relationship whose likelihood is above b
#     for p in [0.9, 0.75, 0.5]:
#         rel_prob, rel = sorted(([[1, "2nd"]]+[[tree.nodes[node]["P"], node] for _, node in nx.bfs_edges(tree, "2nd") if tree.nodes[node]["P"] >= p])[::-1], key = lambda x: x[0])[0]
#         row += [rel, round(rel_prob, 3)]
#     row += [round(tree.nodes[node]["P"], 3) for _, node in nx.bfs_edges(tree, "2nd")]
#     col_names += [node for _, node in nx.bfs_edges(tree, "2nd")]
#     return pd.DataFrame([row], columns = col_names)

# print(probs(id1, id2, N_probs, h_probs, h1, h2, N, dad1, mom1, dad2, mom2, age1, age2))