import pandas as pd
import numpy as np
import pickle as pkl

def load_segments(rel, map_df, glen):

    segs = pd.read_csv(f"{rel}.seg", delim_whitespace=True, header=None, names=["id1", "id2", "chromosome", "mb1", "mb2", "ibdtype", "cm1", "cm2", "l"])

    tmp = []
    for chrom, chrom_df in segs.groupby("chromosome"):
        chrom_map = map_df[map_df.chromosome==chrom]

        chrom_df["cm1"] = np.interp(chrom_df["mb1"].values, chrom_map["mb"].values, chrom_map["cm"].values)
        chrom_df["cm2"] = np.interp(chrom_df["mb2"].values, chrom_map["mb"].values, chrom_map["cm"].values)
        chrom_df["l"] = chrom_df.cm2 - chrom_df.cm1

        tmp.append(chrom_df)

    segs = pd.concat(tmp)

    segs["pair"] = segs.apply(lambda x: (x.id1, x.id2), axis=1)

    tmp = []
    for pair, pair_df in segs.groupby("pair"):
        tmp.append([pair, pair_df["l"].sum()])

    pairs = pd.DataFrame(tmp, columns=["pair", "l"])
    pairs["k"] = pairs["l"] / glen

    mean = pairs["k"].mean()
    std = pairs["k"].std()

    keep_pairs = pairs[pairs.k.apply(lambda x : x > mean - std*0.1 and x < mean + std*0.1)]["pair"].values

    segs = segs[segs.pair.isin(keep_pairs)]

    return [pair_df[["chromosome", "cm1", "cm2"]].values.tolist() for _, pair_df in segs.groupby("pair")]

map_df = pd.concat([pd.read_csv(f"/gpfs/data/sramacha/ukbiobank_jun17/cwilli50/cluster_phasing/oscar/ukb_rephase/newbeagle_chr{chrom}.map", header=None, delim_whitespace=True,
                    names=["chromosome", "rsid", "cm", "mb"]) for chrom in range(1, 23)])

glen = sum([chrom_df.iloc[-1]["cm"]-chrom_df.iloc[0]["cm"] for _, chrom_df in map_df.groupby("chromosome")])

regions = {r: load_segments(r, map_df, glen) for r in ["hs", "co", "hco"]}

i = open("regions.pkl", "wb")
pkl.dump(regions, i)
i.close()