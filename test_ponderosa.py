from Ponderosa import SampleData, Pedigree
from pedigree_tools import TrainPonderosa
import pickle as pkl
import networkx as nx

fam = "Himba_allPO.fam"
ibd = "Himba_segments.tsv"
ages = "Himba_Age.txt"
mapf = "../ponderosa/plink_data/newHimba_shapeit.chr1.map"
n_pairs = 200

Samples = SampleData(fam_file=fam,
                         ibd_file=ibd,
                         age_file=ages,
                         map_file=mapf)#, n_pairs=n_pairs)


o = open("Samples_debug.pkl", "wb")
pkl.dump(Samples, o)
o.close()

pedigree = Pedigree(Samples.g)
pedigree.find_relationships()

# i = open("Samples_debug.pkl", "rb")
# Samples = pkl.load(i)
# Samples.genome_len = 3500
# i.close()

# i = open("Pedigree_debug.pkl", "rb")
# pedigree = pkl.load(i)
# i.close()



o = open("Pedigree_debug.pkl", "wb")
pkl.dump(pedigree, o)
o.close()

train = TrainPonderosa(real_data=True, samples=Samples, pedigree=pedigree)


import pdb; pdb.set_trace()
# ped.lineal_relationships()

# import pdb; pdb.set_trace()
# g = nx.Graph()
# g.add_edges_from(ped.edges)

# import pdb; pdb.set_trace()