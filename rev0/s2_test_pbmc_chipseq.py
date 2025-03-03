# %%
# get ChIP-seq data from relevant TFs
# test iDDN on PBMC two cell types TF-gene network
# measure accuracy, and determine hyper-parameters

import pbmc_util as pu
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# from scipy.sparse import coo_array

reload(pu)

rna_obj = "../../iddn_data/muon/data/pbmc10k.h5mu/rna"
cis_folder = "../../iddn_data/cistrome/"
tf_human_file = "../../iddn_data/tf_list/TF_names_v_1.01.txt"

# %%
# PBMC scRNA-seq data, processed using the pipeline in muon tutorial
# this uses the gene name, instead of ensembl id

dat1, dat2, gene_names, gene_dict = pu.load_rna_data(file=rna_obj)

# %%
# ground truth

# TF-target pair dict
# choose top 1000 targets for each TF
tgt1_dict = pu.read_cis_genes(cis_folder, pu.tf_list1, pu.tf_cis1, gene_names, thr=1.5)
tgt2_dict = pu.read_cis_genes(cis_folder, pu.tf_list2, pu.tf_cis2, gene_names, thr=1.5)

# Adjacency matrix of these TFs for each cell type
tf_idx1 = np.where(np.isin(gene_names, pu.tf_list1) > 0)[0]
tf_idx2 = np.where(np.isin(gene_names, pu.tf_list2) > 0)[0]
# net1_gt = pu.tf_gene_pair_to_adj_mat(gene_dict, tgt1_dict)[tf_idx1]
# net2_gt = pu.tf_gene_pair_to_adj_mat(gene_dict, tgt2_dict)[tf_idx2]
# print(np.sum(net1_gt))
# print(np.sum(net2_gt))

# %%
# iDDN, use different lambda1

tf_human_all = pd.read_csv(tf_human_file).to_numpy().flatten()
tf_human_idx = np.where(np.isin(gene_names, tf_human_all) > 0)[0]
net_out = pu.iddn_work_basic(
    dat1, dat2, tf_human_idx, l1_mat=0.01, l2_mat=0.001, n_cores=12
)

# n_gene = dat1.shape[1]
# n_edge = np.sum(net_out[:,tf_human_idx])/2
# n_full_edge = n_gene*len(tf_human_idx)
# print(n_edge/n_full_edge)

# net1_est = net_out[0][tf_idx1]
# net2_est = net_out[1][tf_idx2]
# print(np.sum(net1_est))
# print(np.sum(net2_est))

# %%
# evaluation based on each TF, focus on ranking
# we use IoU for a given number of targets
# not better than random guess

for grp in range(2):
    print(grp)
    if grp == 0:
        tf_list_now = pu.tf_list1
        tgt_dict_now = tgt1_dict
    else:
        tf_list_now = pu.tf_list2
        tgt_dict_now = tgt2_dict

    for i in range(len(tf_list_now)):
        gene = tf_list_now[i]
        idx = gene_dict[gene]

        tgt_est = gene_names[np.where(net_out[grp][idx] > 0)[0]]
        tgt_gt = tgt_dict_now[gene]
        # if len(tgt_gt) > len(tgt):
        #     tgt_gt = tgt_gt[:len(tgt)]
        # else:
        #     tgt = tgt[:len(tgt_gt)]

        tgt_union = np.union1d(tgt_est, tgt_gt)
        overlaps = np.intersect1d(tgt_gt, tgt_est)
        prec = len(overlaps) / len(tgt_est)
        recall = len(overlaps) / len(tgt_gt)
        if prec > 0 or recall > 0:
            f1 = 2 * prec * recall / (prec + recall)
        else:
            f1 = 0.0
        print(gene, len(tgt_est), prec, recall, f1)
