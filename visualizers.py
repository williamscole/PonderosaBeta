#!/usr/bin/env python3

import networkx as nx
import pandas as pd
import itertools as it
import numpy as np
from datetime import datetime
import time
from math import floor, ceil
# import phasedibd as ibd
import os
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import sys
import concurrent.futures
import subprocess
from matplotlib import colors


'''
Given either a hap, degree, or nsegs classifier, will plot contours of the classifier.
'''
def visualize_classifiers(classif_name, classif, ax):

    def plot_degree_classifier(classif, ax):

        labs = list(classif.classes_)

        XX, YY = np.meshgrid(np.linspace(0, 1, 500), np.linspace(0, 0.5, 250))

        Z = classif.predict(np.c_[XX.ravel(), YY.ravel()])

        # plot colors
        Z = np.array([labs.index(i) for i in Z])
        Z = Z.reshape(XX.shape)
        ax.pcolormesh(XX, YY, Z, cmap="rainbow")

        # plot countour lines
        Z = classif.predict_proba(np.c_[XX.ravel(), YY.ravel()])
        for i in range(len(labs)):
            Zi = Z[:,i].reshape(XX.shape)

            ax.contour(XX, YY, Zi, [0.5], linewidths=4, colors="white")

        ax.set_xlabel("IBD1")
        ax.set_ylabel("IBD2")
        ax.set_title("degree classifier")

    def plot_hap_classifier(classif, ax):

        labs = list(classif.classes_)

        XX, YY = np.meshgrid(np.linspace(0.5, 1, 500), np.linspace(0.5, 1, 500))

        Z = classif.predict(np.c_[XX.ravel(), YY.ravel()])

        colorsList = [(198, 30, 49),(150, 130, 88),(38, 1, 90)]
        cmap = colors.ListedColormap(colorsList)

        Z = np.array([labs.index(i) for i in Z])
        Z = Z.reshape(XX.shape)
        ax.pcolormesh(XX, YY, Z, cmap=cmap)

        Z = classif.predict_proba(np.c_[XX.ravel(), YY.ravel()])
        for i in range(len(labs)):
            Zi = Z[:,i].reshape(XX.shape)
            ax.contour(XX, YY, Zi, [0.5], linewidths=4, colors="white")
            
        ax.set_xlabel("h1")
        ax.set_ylabel("h2")
        ax.set_title("hap classifier")

    def plot_nsegs_classifier(classif, ax):

        X = np.arange(10, 120)

        labs = classif.classes_

        for index, lab in enumerate(labs):

            Y = classif.predict_proba([[0.5, i] for i in X])[:, index]

            ax.plot(X, Y, label=lab)

        ax.legend()
        ax.set_xlabel("Number of segments")
        ax.set_ylabel("Probability")
        ax.set_title("nsegs classifier")

    if classif_name == "degree":
        plot_degree_classifier(classif, ax)

    elif classif_name == "hap":
        plot_hap_classifier(classif, ax)

    elif classif_name == "nsegs":
        plot_nsegs_classifier(classif, ax)


'''
This takes as input a pandas dataframe with two columns, x and y and a classifier. It will predict the
class of each line of the dataframe and plot the (x,y) coordinate, colored by the predicted class. Will
also plot the max probability of each point.
'''
def plot_prediction(df, classif, x, y):
    df["predicted"] = classif.predict(df[[x, y]].values)
    df["probability"] = [max(i) for i in classif.predict_proba(df[[x, y]].values)]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))

    sns.scatterplot(data=df, x=x, y=y, hue="probability", ax=axs[1])
    sns.scatterplot(data=df, x=x, y=y, hue="predicted", legend=False, alpha=0.4, ax=axs[0])

    for lab, tmp in df.groupby("predicted"):
        axs[0].text(x=tmp["ibd1"].mean(), y=tmp["ibd2"].mean(), s=lab, fontsize="medium")

    return fig, axs


'''
For plotting IBD. Can plot IBD segments on their haplotypes and has flexibility for coloring the segments.
'''
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
