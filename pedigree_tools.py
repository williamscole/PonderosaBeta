from cmath import phase
import networkx as nx
import pandas as pd
import itertools as it
import numpy as np
from datetime import datetime
import phasedibd as ibd
import os
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import concurrent.futures


def split_regions(region_dict, new_region):
    # returns the overlap of 2 regions (<= 0 if no overlap)
    def overlap(region1, region2):
        start1, end1 = region1
        start2, end2 = region2
        return min(end1,end2) - max(start1,start2)
    # out region will be returned; is a dict of regions mapping to members of region    
    out_region = dict()
    # overlapped keeps track of all the regions that overlap with the new region
    overlapped = {tuple(new_region[:2]):[new_region[2]]}
    # iterate through the existing regions
    for region in sorted(region_dict):
        # if overlap
        if overlap(region, new_region[:2]) > 0:
            # the regions completely overlap, just add the member and return region dict
            if tuple(region) == tuple(new_region[:2]):
                region_dict[region] += [new_region[2]]
                return region_dict
            # bc the region overlaps, add it to overlapped
            overlapped[region] = region_dict[region]
        # no overlap, but add the region to the out_region dict
        else:
            out_region[region] = region_dict[region]
    # all the segments in overlapped overlap, so each consecutive pairs of coordinates in sites should/could have different members
    sites = sorted(set(it.chain(*overlapped)))
    # iterate thru consecutive sites
    for start, stop in zip(sites, sites[1:]):
        # get the members of the regions that overlap the consecutive sites
        info = [j for i, j in overlapped.items() if overlap((start, stop), i) > 0]
        # unpack the membership
        out_region[(start,stop)] = sorted(it.chain(*info))
    return out_region

# perform various computations on ibd segments
class ProcessSegments:
    def __init__(self, pair_df):
        self.segs = pair_df

    def split_regions(self, region_dict, new_region):
        # returns the overlap of 2 regions (<= 0 if no overlap)
        def overlap(region1, region2):
            start1, end1 = region1
            start2, end2 = region2
            return min(end1,end2) - max(start1,start2)
        # out region will be returned; is a dict of regions mapping to members of region    
        out_region = dict()
        # overlapped keeps track of all the regions that overlap with the new region
        overlapped = {tuple(new_region[:2]):[new_region[2]]}
        # iterate through the existing regions
        for region in sorted(region_dict):
            # if overlap
            if overlap(region, new_region[:2]) > 0:
                # the regions completely overlap, just add the member and return region dict
                if tuple(region) == tuple(new_region[:2]):
                    region_dict[region] += [new_region[2]]
                    return region_dict
                # bc the region overlaps, add it to overlapped
                overlapped[region] = region_dict[region]
            # no overlap, but add the region to the out_region dict
            else:
                out_region[region] = region_dict[region]
        # all the segments in overlapped overlap, so each consecutive pairs of coordinates in sites should/could have different members
        sites = sorted(set(it.chain(*overlapped)))
        # iterate thru consecutive sites
        for start, stop in zip(sites, sites[1:]):
            # get the members of the regions that overlap the consecutive sites
            info = [j for i, j in overlapped.items() if overlap((start, stop), i) > 0]
            # unpack the membership
            out_region[(start,stop)] = sorted(it.chain(*info))
        return out_region

    # stitches together segments that are at most max_gap apart
    def segment_stitcher(self, segment_list, max_gap = 1):
        regions = {}
        for start, stop in segment_list:
            overlapped = {start, stop}
            updated_regions = set()
            for r1, r2 in regions:
                if min(stop, r2) - max(start, r1) > -max_gap:
                    overlapped |= {r1, r2}
                else:
                    updated_regions |= {(r1, r2)}
            updated_regions |= {(min(overlapped), max(overlapped))}
            regions = updated_regions
        return regions

    # returns ibd1, ibd2 values for the pair
    def get_ibd1_ibd2(self):
        ibd1, ibd2 = 0, 0
        for chrom, chrom_df in self.segs.groupby("chromosome"):
            r = {}
            for _, row in chrom_df.iterrows():
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id1_haplotype"]])
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id2_haplotype"]+2])
            for (start, end), hap in r.items():
                l = end - start
                if 0 in hap and 1 in hap and 2 in hap and 3 in hap:
                    ibd2 += l
                else:
                    ibd1 += l
        return ibd1, ibd2

    # returns the number of IBD segments
    def get_n_segments(self):
        n = 0
        for _, chrom_df in self.segs.groupby("chromosome"):
            n += len(self.segment_stitcher(chrom_df[["start_cm", "end_cm"]].values))
        return n

    # returns the haplotype score of the pair
    def get_h_score(self):
        hap, tot = {0:0, 1:0}, 0
        for _, chrom_df in self.segs.groupby("chromosome"):
            r= {}
            for _, row in chrom_df.iterrows():
                r = self.split_regions(r, [row["start_cm"], row["end_cm"], row["id1_haplotype"]])
                r = self.split_regions(r, [row["start_cm"], row["end_cm"], row["id2_haplotype"]+2])
            temp = {0:0, 1:0, 2:0, 3:0}
            for (start, end), hapl in r.items():
                if len(hapl) > 2:
                    continue
                l = end - start
                tot += l
                for h in hapl:
                    temp[h] += l
            hap[0] += max(temp[0], temp[1])
            hap[1] += max(temp[2], temp[3])
        return hap[0]/tot, hap[1]/tot

    def run_ponderosa(self):
        class ponderosa: pass
        ponderosa = ponderosa()
        ibd1, ibd2 = self.get_ibd1_ibd2()
        ponderosa.ibd1 = ibd1
        ponderosa.ibd2 = ibd2
        ponderosa.n = self.get_n_segments()
        h1, h2 = self.get_h_score()
        ponderosa.h1 = h1
        ponderosa.h2 = h2
        return ponderosa



# takes as input a map_file (either all chrom together or separate) and a vcf file
def interpolate_map(map_file, vcf_file):
    # take all the sites from the vcf
    vcf_pos = []
    with open(vcf_file) as vcf:
        for line in vcf:
            if "#" in line:
                continue
            chrom, pos, rsid = line.split()[:3]
            vcf_pos.append([int(chrom), int(pos), rsid])

    # create dataframe from the vcf sites
    out_map = pd.DataFrame(vcf_pos, columns=["CHROM", "MB", "rsID"])

    # we have multiple map files
    if "chr1" in map_file:
        # create dict where chrom maps to its map_df
        chrom_map = {chrom: pd.read_csv(map_file.replace("chr1", f"chr{chrom}"), delim_whitespace=True, names=["CHROM", "rsID", "cM", "MB"]) for chrom in range(1,23)}
    # only one map file with all chrom
    else:
        # load the map file
        map_df = pd.read_csv(map_file, delim_whitespace=True, names=["CHROM", "rsID", "cM", "MB"])
        # subset into dict mapping to its df
        chrom_map = {chrom: chrom_df for chrom, chrom_df in map_df.groupby("chrom")}
    
    # iterate through the chromosomes of the vcf file
    for chrom, chrom_df in out_map.groupby("CHROM"):
        # load the map reference to interpolate chrom
        map_df = chrom_map[chrom]
        # linear interpolation of sites
        chrom_df["cM"] = np.interp(chrom_df["MB"], map_df["MB"], map_df["cM"])
        # convert df to str and reorder
        chrom_df = chrom_df[["CHROM", "rsID", "cM", "MB"]].astype(str)
        # write out the map files individually
        out = open(f"sim_chr{chrom}.map", "w")
        _ = out.write("\n".join(chrom_df.apply(lambda x: "\t".join(x), axis=1).values.tolist()) + "\n")
        
class RemoveRelateds:

    def __init__(self):
        self.seed = np.random.choice(np.arange(20000))
        np.random.seed = self.seed

    # takes as input a king file
    # threshold_func is a func that takes as input PropIBD, IBD1Seg, IBD2Seg, InfType and returns True if the input is considered to be related
    def king_graph(self, king_file: str, threshold_func):

        # read in king file
        king = pd.read_csv(king_file, delim_whitespace = True, dtype = {"ID1": str, "ID2": str})

        # create graph structure with the kinship coeff
        self.kinG = nx.Graph()
        self.kinG.add_weighted_edges_from(king[["ID1", "ID2", "PropIBD"]].values)

        # build the kinship graph
        G = nx.Graph()
        king_related = king[king[["PropIBD", "IBD1Seg", "IBD2Seg", "InfType"]].apply(lambda x: threshold_func(*x), axis = 1)]
        G.add_edges_from(king_related[["ID1", "ID2"]].values)

        return G

    def unrelated_family(self, g):

        # keep track of the unrelated nodes
        unrelated_nodes = list()

        # dict maps an ID to the number of close relatives
        degree_d = dict(g.degree())

        # each iteration removes a node from degree_d so will be executed len(degree_d) times
        while len(degree_d) > 0:

            # create function that returns the num of close relatives +- random noise for tie-breakers
            randmin = lambda x: degree_d[x] + np.random.normal(0, 0.00001)

            # picks the node with the fewest close relatives
            node1 = min(degree_d, key = randmin)

            # add the node to unrelated_nodes but only if not already related
            add = True
            for node2 in unrelated_nodes:
                if g.has_edge(node1, node2):
                    add = False
            if add:
                unrelated_nodes.append(node1)

            # delete the node from degree_d regardless of if it's added or not
            del degree_d[node1]

        return unrelated_nodes

    def get_unrelateds(self, G):
        # object to store various components of the run
        # n_comp is number of distinct families, unrelateds holds the unrelateds, max k holds the highest kinship value of the set
        run = type('run', (object,), {"n_comp": 0, "unrelateds": [], "max_k": 0})

        # iterate through each "family" (clusters of relatives entirely unrelated)
        for i in nx.connected_components(G):
            g = G.subgraph(i)
            run.unrelateds += self.unrelated_family(g)
            run.n_comp += 1

        # for a sanity check, keeps track of closest relative pair in the set
        for id1, id2 in it.combinations(run.unrelateds, r = 2):

            # return the edge
            edge = self.kinG.get_edge_data(id1, id2)

            # get the edge weight (kinship) if the edge exists
            k = 0 if edge == None else edge["weight"]

            # only update max_k if k > max_k
            run.max_k = run.max_k if run.max_k > k else k

        # return the run object
        return run

    # run it over multiple iterations using different seeds to get more individuals included
    # target is the min len of unrelateds for the loop to stop
    # max_iter is the max num of iterations before it stops
    def multiple_runs(self, G, target, max_iter = 10):

        # keep track of the runs
        run_list = []

        # only run it max_iter times
        for i in range(max_iter):

            # choose and set new seed
            seed = np.random.choice(np.arange(20000))
            np.random.seed = seed

            # run the unrelateds algorithm
            run = self.get_unrelateds(G)
            run.seed = seed

            # add the new run to run_list
            run_list.append(run)

            # if the most recent run exceeds our target, stio
            if len(run.unrelateds) >= target:
                print(f"Target of {target} relatives found")
                break

            print(f"Running iteration {i+1}...found a set of {len(run.unrelateds)}")

        # sort by length and get run with longest unrelateds list
        run_list.sort(key = lambda x: len(x.unrelateds), reverse = True)
        run = run_list[0]

        # set the class seed
        self.seed = run.seed

        return run


    def write_out(self, run, prefix = "unrelateds"):

        # write log file
        log = open(f"{prefix}.log", "w")
        log.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n")
        log.write(f"Random seed: {self.seed}\n\n")
        log.write(f"Found {run.n_comp} distinct families/family at the given kinship threshold\n")
        log.write(f"Wrote a total of {len(run.unrelateds)} relatives to {prefix}.txt\n")
        log.write(f"Of these, the max proportion IBD is {run.max_k}\n")
        log.close()

        # write out unrelateds
        out = open(f"{prefix}.txt", "w")
        out.write("\n".join(run.unrelateds) + "\n")
        out.close()

    def plink_unrelateds(self, king_file: str, max_k: float):

        # create the relatedness network from the king file
        G = self.king_graph(king_file, lambda propIBD, a, IBD2Seg, c: propIBD > 0.1 or IBD2Seg > 0.03)

        # run 10 iterations to find the largest set
        run = self.multiple_runs(G, target = np.inf, max_iter = 10)

        # write out the file
        outfile = open("sim_keep.txt", "w")
        _ = outfile.write("\n".join(run.unrelateds))
        
        print(f"Wrote out {len(run.unrelateds)} IDs to keep to 'sim_keep.txt'")

class PedSims:

    # takes as input the path where the all sim data is
    def __init__(self, path):
        self.path = path
        self.ibd = pd.DataFrame()
        self.log = "simulated_pedigrees.log"
        out = open(self.log, "w")
        _ = out.write(f"Pedigree simulations with ped-sim\n{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n\n")
        out.close()

    def write_log(self, msg):
        out = open(self.log, "a")
        _ = out.write(msg)
        out.close()          

    def run_ibd_iteration(self, map_file, rel, iter, chrom):
        # converts indices to the actual iid
        def return_ids(vcff):
            with open(vcff) as vcf:
                    for lines in vcf:
                        if "#CHROM" in lines:
                            return lines.split()[9:]

        # dict that stores the IDs for each relationship type
        relative_ids = {"av": {"AV": ("g2-b2-i1", "g3-b1-i1"),
                                "FS": ("g2-b1-i1", "g2-b2-i1"),
                                "CO": ("g3-b1-i1", "g3-b2-i1"),
                                "CORM": ("g3-b2-i1", "g4-b1-i1")},
                        "mhs": {"MHS": ("g2-b1-i1", "g2-b2-i1"),
                                "MHCO": ("g3-b1-i1" , "g3-b2-i1"),
                                "MHAV1": ("g2-b1-i1", "g3-b2-i1"),
                                "MHAV2": ("g2-b2-i1", "g3-b1-i1")},
                        "phs": {"PHS": ("g2-b1-i1", "g2-b2-i1"),
                                "PHCO": ("g3-b1-i1" , "g3-b2-i1"),
                                "PHAV1": ("g2-b1-i1", "g3-b2-i1"),
                                "PHAV2": ("g2-b2-i1", "g3-b1-i1")},
                        "mgp": {"MGP": ("g1-b1-i1", "g3-b1-i1"),
                                "MGGP": ("g1-b1-s1", "g4-b1-i1")},
                        "pgp": {"PGP": ("g1-b1-i1", "g3-b1-i1"),
                                "PGGP": ("g1-b1-s1", "g4-b1-i1")}}

        def rename_id(cur_id, rel_name, index):
            cur_id = cur_id.split("_")[0]
            sim_iter = int("".join([i for i in cur_id if i.isnumeric()]))
            return f"i{iter}{rel_name}{sim_iter}_{index}"

        # hap = ibd.VcfHaplotypeAlignment(f"sim_chr{chrom}.vcf", map_file.replace("chr1", f"chr{chrom}"))
        hap = ibd.VcfHaplotypeAlignment(f"sim_chr{chrom}.vcf", map_file.replace("chr1", f"chr{chrom}"))
        tpbwt = ibd.TPBWTAnalysis()
        ibd_results = tpbwt.compute_ibd(hap, use_phase_correction=False)

        # convert the index IDs to their actual IDs
        convert = {index:i.split()[0] for index,i in enumerate(return_ids(f"sim_chr{chrom}.vcf"))}
        ibd_results["id1"] = ibd_results.apply(lambda x: convert[x.id1],axis=1)
        ibd_results["id2"] = ibd_results.apply(lambda x: convert[x.id2],axis=1)

        # extract the relative segments
        relative_segs = pd.DataFrame()

        for relative, (id1, id2) in relative_ids[rel].items():
            df = ibd_results[ibd_results.apply(lambda x: id1 in x.id1 and id2 in x.id2 and x.id1.split("_")[0] == x.id2.split("_")[0], axis = 1)].copy()
            df["id1"] = df["id1"].apply(lambda x: rename_id(x, relative, 1))
            df["id2"] = df["id2"].apply(lambda x: rename_id(x, relative, 2))
            df["relative"] = relative
            relative_segs = pd.concat([relative_segs, df.drop(["start", "end"], axis=1)])

        return relative_segs

    # takes as input a pair's segments and returns IBD1 and IBD2 cM
    def ibd1_ibd2(self, seg_df):
        ibd1, ibd2 = 0, 0
        for chrom, chrom_df in seg_df.groupby("chromosome"):
            r = {}
            for _, row in chrom_df.iterrows():
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id1_haplotype"]])
                r = split_regions(r, [row["start_cm"], row["end_cm"], row["id2_haplotype"]+2])
            for (start, end), hap in r.items():
                l = end - start
                if 0 in hap and 1 in hap and 2 in hap and 3 in hap:
                    ibd2 += l
                else:
                    ibd1 += l
        return ibd1, ibd2
    
    def load_ibd(self):
        # load ibd segments
        ibd = pd.read_feather("simulated_segments.f")

        # initialize ibd graph
        ibdG = nx.Graph()

        # add eches to ibd graph
        for (id1, id2), pair_df in ibd.groupby(["id1", "id2"]):
            ibd1, ibd2 = self.ibd1_ibd2(pair_df)
            rel = pair_df["relative"].values[0]
            ibdG.add_edge(id1, id2, ibd = pair_df, ibd1 = ibd1, ibd2 = ibd2, rel = rel)

        # store the graph
        self.ibd = ibdG

        return ibdG

    def segment_stitcher(self, segment_list, max_gap = 1):
        regions = {}
        for start, stop in segment_list:
            overlapped = {start, stop}
            updated_regions = set()
            for r1, r2 in regions:
                if min(stop, r2) - max(start, r1) > -max_gap:
                    overlapped |= {r1, r2}
                else:
                    updated_regions |= {(r1, r2)}
            updated_regions |= {(min(overlapped), max(overlapped))}
            regions = updated_regions
        return regions

    def pair_segments(self, id1, id2, pair_df, max_gap = 1):
        pair_data = type('pair_data', (object,), {"n_segs": 0, "tot_cov": 0, "tot_all": 0, "hap": {id1: 0, id2: 0}})
        for chrom, chrom_df in pair_df.groupby("chromosome"):
            regions = self.segment_stitcher(chrom_df[["start_cm", "end_cm"]].values, max_gap)
            pair_data.n_segs += len(regions)
            pair_data.tot_cov += sum([stop - start for start, stop in regions])
            pair_data.tot_all += chrom_df["l"].sum()
            temp = {id1:{0:0, 1:0}, id2:{0:0, 1:0}}
            for _, row in chrom_df.iterrows():
                temp[row["id1"]][row["id1_haplotype"]] += row["l"]
                temp[row["id2"]][row["id2_haplotype"]] += row["l"]
            pair_data.hap[id1] += max([temp[id1][0], temp[id1][1]])
            pair_data.hap[id2] += max([temp[id2][0], temp[id2][1]])
        return pair_data

    # pair_df is a dataframe of a pair of relatives
    # mean_d is the mean distance between switch errors
    def introduce_phase_error(self, pair_df, mean_d):
        
        # given a mean distance between switch error returns a list of randomly drawn sites
        def generate_switches(mean_d, index):
            #start the switch at 0
            switches, last_switch = [], 0
            
            # longest chrom is 287 cM 
            while last_switch < 300:
                
                # add the new site to the previous site
                switches.append(np.random.exponential(mean_d) + last_switch)
                
                # previous site is now the new site
                last_switch = switches[-1]
                
            # return
            return [(i, index) for i in switches]
        
        # store the newly create segments
        new_segments = []
        
        for chrom, chrom_df in pair_df.groupby("chromosome"):
        
            # generate the switch locations
            s1 = generate_switches(mean_d, 0)
            s2 = generate_switches(mean_d, 1)
            switches = np.array(sorted(s1 + s2))
            switch_index, switches = switches[:,1], switches[:,0]

            # old segments
            segments = chrom_df[["start_cm", "end_cm", "id1_haplotype", "id2_haplotype", "id1", "id2", "chromosome"]].values

            # iterate through the segments
            for start, stop, hap1, hap2, id1, id2, chrom in segments:

                # get number of switches before the segment
                n_dict = {0: len(np.where(np.logical_and(switches<start, switch_index==0))[0]),
                        1: len(np.where(np.logical_and(switches<start, switch_index==1))[0])}


                # get the index of switches within the segment
                b = np.where(np.logical_and(switches>=start, switches<=stop))[0]

                # iterate through the switches and the switch index in the segment
                for s, index in zip(switches[b], switch_index[b]):
                    # add the broken segment as the current start --> s and add n, which is the number of preceding switches
                    new_segments.append([chrom, id1, id2, hap1, hap2, start, s, n_dict[0], n_dict[1]])

                    # new start
                    start = s

                    # increase the number of switches by 1 but only on the relevant switch
                    n_dict[index] += 1

                # add the final segment
                new_segments.append([chrom, id1, id2, hap1, hap2, start, stop, n_dict[0], n_dict[1]])

        pair_df = pd.DataFrame(new_segments, columns = ["chromosome", "id1", "id2", "id1_haplotype", "id2_haplotype", "start_cm", "end_cm", "n1", "n2"])
        pair_df["l"] = pair_df["end_cm"] - pair_df["start_cm"]
        pair_df = pair_df[pair_df.l >= 2.5]
        pair_df["id1_haplotype"] = pair_df[["id1_haplotype", "n1"]].apply(lambda x: x[0] if x[1]%2 == 0 else (x[0]+1)%2, axis = 1)
        pair_df["id2_haplotype"] = pair_df[["id2_haplotype", "n2"]].apply(lambda x: x[0] if x[1]%2 == 0 else (x[0]+1)%2, axis = 1)

        return pair_df


    def train_H_classifier(self, training_data, error_data, title, covariance_type = "tied"):
        error_data["drel"] = "No inference"

        pair_df = pd.concat([error_data, training_data])

        # get haplotype scores
        pair_df["h1"] = (pair_df["id1_hap"] / pair_df["tot_cov"]).apply(lambda x: min(x, 1))
        pair_df["h2"] = (pair_df["id2_hap"] / pair_df["tot_cov"]).apply(lambda x: min(x, 1))

        # get the haplotype scores to train
        X = pair_df[["h1", "h2"]].values.tolist() + pair_df[["h2", "h1"]].values.tolist()

        # get the means to initialize the gmm
        hs_mean = pair_df[pair_df.drel == "HS"]["h1"].mean()
        gpav1_mean, gpav2_mean = pair_df[pair_df.drel != "HS"][["h1", "h2"]].mean()
        init_means = [[gpav1_mean, gpav2_mean], [gpav2_mean, gpav1_mean], [hs_mean, hs_mean], [0.7, 0.7]]

        # init and fit the gmm
        gmm = GaussianMixture(n_components = 4, means_init = init_means, covariance_type = "tied")
        gmm.fit(X)
        gmm.weights_ = [0.25, 0.25, 0.25, 0.25]

        # plot the classifier
        x = np.linspace(0.65, 1, 50)
        y = np.linspace(0.65, 1, 50)
        X, Y = np.meshgrid(x, y)
        for cat in range(4):
            Z = []
            for row in range(50):
                Z.append([p[cat] for p in gmm.predict_proba([[x, y] for x, y in zip(X[row], Y[row])])])
            plt.contourf(X, Y, np.array(Z), cmap = ["Greens", "Greens", "Blues", "Reds"][cat], levels=np.linspace(0.5, 1, 10))

        plt.colorbar().set_label(label = "Probability", size=12)

        pair_df["shuffle"] = np.random.binomial(1, 0.5, pair_df.shape[0])
        pair_df["temp_h1"] = pair_df.apply(lambda x: x.h1 if x.shuffle else x.h2, axis=1)
        pair_df["temp_h2"] = pair_df.apply(lambda x: x.h2 if x.shuffle else x.h1, axis=1)
        pair_df["Relationship"] = pair_df["drel"]

        ax = sns.scatterplot(data = pair_df, x = "temp_h1", y = "temp_h2", hue = "Relationship", alpha = 0.5, palette = ["palegreen", "skyblue", "lightsalmon"], hue_order = ["GP/AV", "HS", "No inference"])

        sns.move_legend(ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False, fontsize = 10)

        plt.xlabel("Haplotype score 1", size = 12)
        plt.ylabel("Haplotype score 2", size = 12)
        
        plt.savefig(f"{title}_gmm.png", dpi = 500)
        plt.close()

        with open(f"{title}_gmm.pkl", "wb") as f:
            pickle.dump(gmm, f)

    def train_N_classifier(self, training_data, title):
    
        # initialize and train the classifier
        lda = LinearDiscriminantAnalysis()
        lda.fit(training_data[["tot_all", "n_segs"]].values, training_data["rel"].values)
        
        # for plotting the probabilities, plot at fixed k, which is set to the mean k value here
        mean_k = training_data["tot_all"].mean()
        
        # over a range of num of segments from 22 to 81
        X = [[mean_k, n] for n in range(22, 81)]
        
        # dataframe of the different probabilities
        pred = pd.DataFrame(lda.predict_proba(X), columns = lda.classes_)
        pred["n_segs"] = np.arange(22, 81)
        
        # the prob of being MGP or PGP is the prob of being GP, eg
        pred["GP"] = pred["MGP"] + pred["PGP"]
        pred["HS"] = pred["MHS"] + pred["PHS"]
        
        # change shape of df
        pred = pred.melt("n_segs", var_name = "Relationship", value_name = "Probability")
        
        # plot each relationship
        sns.lineplot(data = pred[~pred.Relationship.isin(["GP", "HS"])], x = "n_segs", y = "Probability", hue = "Relationship",
                    hue_order = ["PGP", "MGP", "PHS", "MHS", "AV"], palette = ["firebrick", "lightcoral", "mediumblue", "cornflowerblue", "mediumseagreen"]) 
        
        # plot only the higher order relationships (GP, HS, AV)
        ax = sns.lineplot(data = pred[pred.Relationship.isin(["GP", "HS", "AV"])], x = "n_segs", y = "Probability", hue = "Relationship",
                    hue_order = ["GP", "HS", "AV"], palette = ["tomato", "dodgerblue", "mediumseagreen"], legend = False, linewidth = 4, style = "Relationship", dashes = {"GP": (3, 3), "AV": (3, 3), "HS": (3, 3)})
        
        # plot formatting code
        sns.move_legend(ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=5, title=None, frameon=False, fontsize = 10)
        plt.xlabel("Number of IBD segments shared", fontsize = 12)
        plt.ylabel("Probability", fontsize = 12)
        
        plt.savefig(f"{title}_lda.png", dpi = 500)
        plt.close()

        with open(f"{title}_lda.pkl", "wb") as f:
            pickle.dump(lda, f)

    def generate_training_data(self, training_nodes, mean_d):

        if mean_d == None:
            data = [[node, self.famG.nodes[node]["data"]] for node in training_nodes]

        else:
            data = []
            for id1, id2 in training_nodes:
                ibd = self.ibd.get_edge_data(id1, id2)["ibd"]
                ibd_phase_error = self.introduce_phase_error(ibd, mean_d)
                data.append([(id1, id2), self.pair_segments(id1, id2, ibd_phase_error)])
        df_data = []
        for node, info in data:
            rel = self.famG.nodes[node]["rel"]
            id1, id2 = node
            df_data.append([info.hap[id1], info.hap[id2], info.tot_cov, info.tot_all, info.n_segs, rel, "HS" if "HS" in rel else "GP/AV"])

        pair_df = pd.DataFrame(df_data, columns=["id1_hap", "id2_hap", "tot_cov", "tot_all", "n_segs", "rel", "drel"])

        return pair_df

    def analyze_ibd(self, max_couple_k, max_inlaw_k):
        # add the pairs
        self.add_fam()

        # iterate through each pair and store data
        for id1, id2 in self.famG.nodes():
            ibd = self.ibd.get_edge_data(id1, id2)
            pair_data = self.pair_segments(id1, id2, pd.DataFrame(columns=["chromosome"]) if ibd == None else ibd["ibd"])
            self.famG.nodes[(id1, id2)]["k"] = pair_data.tot_all
            self.famG.nodes[(id1, id2)]["data"] = pair_data

        # only keep training pairs whose founders are unrelated
        keep_pairs = []
        for node1 in self.root_nodes:
            add = True
            for node2 in self.famG.successors(node1):
                k = self.famG.nodes[node2]["k"]
                rel = self.famG.nodes[node2]["rel"]
                if (rel == "couple" and k > max_couple_k) or (rel == "inlaws" and k > max_inlaw_k):
                    add = False
                    break
            if add:
                keep_pairs.append(node1)

        training_data = self.generate_training_data(keep_pairs, None)

        # introduce phase error
        error_data = self.generate_training_data(keep_pairs, 75)

        # look across different phase errors
        phase_errors = {"Perfect phase": training_data, 75: error_data}
        for mean_d in [5, 25, 50, 100, 125]:
            phase_errors[mean_d] = self.generate_training_data(keep_pairs, mean_d)
        
        self.train_H_classifier(training_data, error_data, "test")
        self.train_N_classifier(training_data, "test")

        return phase_errors

    def get_items(self):
        return self.famG, self.ibd, self.root_nodes

class PedigreeNetwork:
    
    def __init__(self, fam):

        # can pass the fam pandas df or the file path to open the pandas df
        if type(fam) == str:
            fam = pd.read_csv(fam, delim_whitespace=True, header=None, dtype = str)

        # name the columns
        fam.columns = ["FID", "IID", "Father", "Mother", "Sex", "Pheno"]

        # convert sex to int
        fam["Sex"] = fam["Sex"].apply(int)

        # pedigree structure is a directed graph
        self.pedigree = nx.DiGraph()
        # add all IID as nodes, with the sex as an attribute of the node
        self.pedigree.add_nodes_from([[row["IID"], dict(sex=row["Sex"])] for _, row in fam.iterrows()])
        # write edges from father --> child and mother -->; edge attribute is down
        self.pedigree.add_edges_from([list(i) + [dict(dir="down")] for i in fam[fam.Father != "0"][["Father", "IID"]].values])
        self.pedigree.add_edges_from([list(i) + [dict(dir="down")] for i in fam[fam.Mother != "0"][["Mother", "IID"]].values])
        # write edges from child --> parent; edge attribute is up
        self.pedigree.add_edges_from([list(i) + [dict(dir="up")] for i in fam[fam.Father != "0"][["IID", "Father"]].values])
        self.pedigree.add_edges_from([list(i) + [dict(dir="up")] for i in fam[fam.Mother != "0"][["IID", "Mother"]].values])

        # these codes tell us how to get between different relationship types
        # e.g., grandchild --> grandparent has you traverse up twice
        self.rel_code = {('up', 'up'): 'GP',
                        ('up', 'up', 'up'): 'GGP',
                        ('up', 'up', 'up', 'up'): 'GGGP',
                        ('up', 'up', 'up', 'up', 'up'): 'GGGGP',
                        ('up', 'up', 'down'): 'AV',
                        ('up', 'up', 'down', 'down'): 'CO',
                        ('up', 'down'): 'sib',
                        ('up',): 'PO'}

        # relative df
        self.relatives = pd.DataFrame(columns = ["id1", "id2", "E_ibd1", "E_ibd2", "maternal", "paternal"])

    def get_pedigree(self):
        return self.pedigree

    def find_relationship(self, id1, id2):

        # a path is not legit if it goes up after it goes down; prevents relationship with a spouse/inlaw
        def legit_path(dir_path):
            down = False
            for d in dir_path:
                down = down or d == "down"
                if down and d == "up":
                    return False
            return True
        
        # we want to order id1, id2 as younger generation, older generation
        def reverse_path(id1, id2, path, path_dir):
            # True if id1 is in an older generation
            reversed = path_dir.count("down") > path_dir.count("up")
            # if id1 is in an older generation
            if reversed:
                # reverse the direction
                path_dir = [{"down": "up", "up": "down"}[i] for i in path_dir[::-1]]
                # path currently goes id1 --> id2, so reverse the list
                path = path[::-1]
                # switch the ids
                id1, id2 = id2, id1
            return id1, id2, path, path_dir, reversed

        # get all paths between the two nodes, cutoff of 5
        paths = list(nx.all_simple_paths(self.pedigree, source = id1, target = id2, cutoff = 5))

        # iterate through the paths
        for index, path in enumerate(paths):
            # gets the directions of the edges in the path
            path_dir = [self.pedigree.get_edge_data(path[index], path[index+1])["dir"] for index in range(len(path)-1)]
            # reverse the path if id1 is in an older generation; if so, switches id1 and id2
            id1_temp, id2_temp, path, path_dir, reversed = reverse_path(id1, id2, path, path_dir)
            # update the path and get the sex of id1_temp's parent
            paths[index] = [id1_temp, id2_temp, self.pedigree.nodes[path[1]]["sex"], path_dir]

        # keep track of how much IBD shared with each parent
        k = {1: 0, 2: 0}
        # list to store paths to be returned
        out_paths = []
        # iterate through each path
        for id1, id2, sex, path_dir in paths:
            # check to see if the path is legal through the pedigree
            if legit_path(path_dir):
                # get the prop IBD1 for the number of meioses between the two
                ibd1 = 0.5**(len(path_dir)-1)
                # add the amount of ibd to the parent
                k[sex] += ibd1
                # add the path and get the relationship
                out_paths.append([id1, id2, sex, self.rel_code.get(tuple(path_dir), "Other"), ibd1])
        # expected proportion of genome IBD1
        ibd1 = k[1]*(1-k[2]) + (1-k[1])*k[2]
        # expected proportion of the genome IBD2
        ibd2 = k[1] * k[2]
        # expected prop IBD
        propIBD = 0.5*ibd1 + ibd2

        return [ibd1, ibd2, propIBD, out_paths]

    # finds all the relationships
    def get_paths(self):
        # first get all the paths in the graph that are, at most, 5 edges long
        paths = dict(nx.all_pairs_shortest_path(self.pedigree, 5))
        rels = []
        for id1 in paths:
            for id2 in paths[id1]:
                if id1 >= id2:
                    continue
                ibd1, ibd2, propIBD, out_paths = self.find_relationship(id1, id2)
                for id1_temp, id2_temp, sex, rel, _ in out_paths:
                    rels.append([(id1, id2), id1_temp, id2_temp,])
                
                rels.append(relationships)
        

class Karyogram:
    def __init__(self, map_file, cm = True):
        if type(map_file) != list:
            map_file = [map_file]

        df = pd.DataFrame()
        for mapf in map_file:
            temp = pd.read_csv(mapf, delim_whitespace=True, header = None)
            df = pd.concat([df, temp])

        self.chrom_ends = {}
        self.max_x = 0
        for chrom, chrom_df in df.groupby(0):
            self.chrom_ends[chrom] = (min(chrom_df[2 if cm else 3]), max(chrom_df[2 if cm else 3])-min(chrom_df[2]))
            self.max_x = self.max_x if sum(self.chrom_ends[chrom]) < self.max_x else sum(self.chrom_ends[chrom])

        self.chrom_y = {(chrom, hap): (chrom - 1)*9 + 4*hap for chrom, hap in it.product(np.arange(1, 23), [0, 1])}

    def plot_segments(self, segments, **kwargs):

        # init the figure
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 20)

        # add the chromosome templates
        for chrom, hap in it.product(np.arange(1, 23), [0, 1]):
            rect = patches.Rectangle((self.chrom_ends[chrom][0], self.chrom_y[(chrom, hap)]),
                                    self.chrom_ends[chrom][1], 3, edgecolor = "black",
                                    facecolor = "darkgrey" if hap == 0 else "grey")
            ax.add_patch(rect)

        # add the segments
        for chrom, start, stop, hap in segments:
            facecolor = kwargs.get("hap0_color", "cornflowerblue") if hap == 0 else kwargs.get("hap1_color", "tomato")
            rect = patches.Rectangle((start, self.chrom_y[(chrom, hap)]), stop - start, 3,
                                    edgecolor = "black", facecolor = facecolor, alpha = 0.8)
            ax.add_patch(rect)

        # re-label the y ticks
        ax.set_yticks([self.chrom_y[(chrom, 0)] + 3.5 for chrom in range(1, 23)])
        ax.set_yticklabels([str(chrom) for chrom in range(1, 23)])

        # set axes limits, remove spines, modify ticks
        plt.xlim(0, self.max_x)
        plt.ylim(-2, self.chrom_y[(22, 1)] + 10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=16)
        plt.tick_params(left = False)

        plt.savefig(f"{kwargs.get('file_name', 'karyogram')}.png", dpi = kwargs.get('dpi', 500))

# p = PedigreeNetwork("Himba_allPO.fam")
# p.get_paths()

# p = PedSims("")
# p.subset_vcf("plink_keep.txt", "../ponderosa/plink_data/Himba_shapeit.chr1.vcf")
if __name__ == "__main__":
    if sys.argv[-1] == "plink_unrelateds":
        r = RemoveRelateds()
        r.plink_unrelateds(sys.argv[-3], float(sys.argv[-2]))

    if sys.argv[-1] == "concat_rel":
        p = PedSims("")
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(p.run_ibd_iteration, sys.argv[-4], sys.argv[-3], sys.argv[-2], chrom) for chrom in range(1, 23)]
            ibd_results = pd.concat([f.result() for f in concurrent.futures.as_completed(results)]).reset_index(drop=True)
        try:
            df = pd.read_feather("simulated_segments.f")
            df = pd.concat([df, ibd_results]).reset_index(drop=True)
        except:
            df = ibd_results

        df.to_feather("simulated_segments.f")

    if sys.argv[-1] == "interpolate":
        interpolate_map(sys.argv[-3], sys.argv[-2])
    if sys.argv[-1] == "subset_vcf":
        p = PedSims("")
        p.subset_vcf(sys.argv[-3], sys.argv[-2])
