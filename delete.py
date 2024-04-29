from Ponderosa import *
import pickle as pkl

i = open("for_dev/sample.pkl", "rb")
samples = pkl.load(i)

PONDEROSA(samples)
