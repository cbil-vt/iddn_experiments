# %%
# test iDDN on PBMC two cell types TF-ATAC
# TF can go to any ATAC, and ATAC go to gene on the same chromosome
# In real application, we should limit each ATAC site to 1M neighbors

import pickle
import time
from importlib import reload
import numpy as np
import pbmc_util as pu

import pandas as pd
import muon as mu
import matplotlib.pyplot as plt
from scipy.sparse import coo_array
from iddn import iddn

reload(pu)

# %%
# dataset and information of features

l1_mat = 0.025
l2_mat = 0.005

file="../../iddn_data/muon/data/pbmc10k.h5mu/"

mdata = mu.read(file)
mu.pp.intersect_obs(mdata)

gene_anno = pu.split_interval(mdata['rna'])
atac_anno = pu.split_interval(mdata['atac'])
gene_chr_idx = pu.get_chr_index(gene_anno)
atac_chr_idx = pu.get_chr_index(atac_anno)

# %%
# choose cell types of interest

ct1="CD4+ naïve T"  # 1513
ct2="CD14 mono"  # 1241

m1 = mdata[mdata.obs['rna:celltype']==ct1]
m2 = mdata[mdata.obs['rna:celltype']==ct2]

rna1 = m1['rna'].X
atac1 = m1['atac'].X
rna2 = m2['rna'].X
atac2 = m2['atac'].X

# %%
# filter features

idx_rna = ((np.std(rna1, axis=0)>0.1) * (np.std(rna2, axis=0)>0.1)) * (gene_chr_idx>0)
rna1f = rna1[:,idx_rna]
rna2f = rna2[:,idx_rna]
gene_anno_f = gene_anno[idx_rna]
gene_chr_idx_f = gene_chr_idx[idx_rna]

idx_atac = ((np.std(atac1, axis=0)>0.1) * (np.std(atac2, axis=0)>0.1)) * (atac_chr_idx>0)
atac1f = atac1[:,idx_atac]
atac2f = atac2[:,idx_atac]
atac_anno_f = atac_anno[idx_atac]
atac_chr_idx_f = atac_chr_idx[idx_atac]

dat1 = np.hstack((rna1f, atac1f))
dat2 = np.hstack((rna2f, atac2f))
dat1 = dat1.astype(np.float64)
dat2 = dat2.astype(np.float64)

# %%
# iDDN dependency matrix

n_rna = len(gene_chr_idx_f)
n_atac = len(atac_chr_idx_f)

gene_ens_names = gene_anno_f.gene_id.to_numpy()
human_tf_idx = pu.get_human_tf_idx(gene_ens_names=gene_ens_names)

n_node = n_rna + n_atac
dep_mat = np.zeros((n_node, n_node), dtype=np.int8)
dep_mat[human_tf_idx, n_rna:] = 1

for i in range(n_rna):
    chr_idx0 = gene_chr_idx_f[i]
    atac_idx0 = np.where(atac_chr_idx_f==chr_idx0)[0] + n_rna
    dep_mat[atac_idx0, i] = 1

np.fill_diagonal(dep_mat, 0)

# plt.imshow(dep_mat[::100,::100])

print(
    f"n1={len(dat1)}, n2={len(dat2)}, n_tf={len(human_tf_idx)}\n", 
    f"n_node={n_node}, n_rna={n_rna}, n_atac={n_atac}", 
    f"l1={l1_mat}, l2={l2_mat}",
)

# %%

# dat1 = dat1[:,::50]
# dat2 = dat2[:,::50]
# dep_mat = dep_mat[::50,::50]

# %%

t0 = time.time()
out_iddn = iddn.iddn_parallel(
    dat1,
    dat2,
    dep_mat=dep_mat,
    lambda1=l1_mat,
    lambda2=l2_mat,
    n_process=12,
    output_sparse=True,
)
t1 = time.time()
tdif = t1 - t0
print(f"Running time: {tdif}")

# %%

# out = dict(out_iddn=out_iddn)
out = dict(out_iddn=out_iddn, gene_anno=gene_anno_f, atac_anno=atac_anno_f, tf_idx=human_tf_idx, n_dep=np.sum(dep_mat))
# out = dict(out_iddn=out_iddn, gene_anno=gene_anno_f, atac_anno=atac_anno_f, tf_idx=human_tf_idx, dat1=dat1, dat2=dat2)
with open(f"iddn_tf_atac_{l1_mat}.pickle", 'wb') as file:
    pickle.dump(out, file)

# %%


