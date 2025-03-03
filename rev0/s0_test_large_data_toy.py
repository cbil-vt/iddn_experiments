# %%
# test iDDN on a large data

from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
import muon as mu
from iddn import iddn, solver
from iddn_data import load_data

reload(iddn)
reload(solver)


# %%

rna = mu.read("../../iddn_data/muon/data/pbmc10k.h5mu/rna")
# atac = mu.read("../../iddn_data/muon/data/pbmc10k.h5mu/atac")

# %%

# rna = mdata.mod['rna']
data:np.ndarray = rna.X

n_per_grp = 5000

dat1 = data[:n_per_grp].astype(np.float64)
dat2 = data[n_per_grp:2*n_per_grp].astype(np.float64)

s1 = np.std(dat1,axis=0)
s2 = np.std(dat2,axis=0)

idx_good = (s1>0.1) * (s2>0.1)
dat1 = dat1[:,idx_good]
dat2 = dat2[:,idx_good]

# %%

n_gene = dat1.shape[1]
print(n_gene)
dep_mat = np.zeros((n_gene, n_gene), dtype=np.int8)
dep_mat[:2000,:] = 1
l1_mat = 0.15
l2_mat = 0.02
# l1_mat = np.zeros((n_gene, n_gene))+0.15
# l2_mat = np.zeros((n_gene, n_gene))+0.02

# %%

out_iddn = iddn.iddn_parallel(
    dat1,
    dat2,
    dep_mat=dep_mat,
    lambda1=l1_mat,
    lambda2=l2_mat,
    n_process=6,
    output_sparse=True,
)


# %%

