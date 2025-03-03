# %%
# test iDDN on PBMC two cell types TF-gene network

import numpy as np
import pbmc_util as pu

# import pandas as pd
# import muon as mu
# from iddn import iddn
# from importlib import reload
# import matplotlib.pyplot as plt
# from iddn import solver
# from scipy.sparse import coo_array
# reload(iddn)
# reload(solver)

l1_mat = 0.05
l2_mat = 0.005

use_constraints = True
cell_ratio = 1.0
gene_ratio = 1.0

dat1, dat2, gene_ens_names = pu.load_rna_data_for_benchmark(
    cell_ratio=cell_ratio,
    gene_ratio=gene_ratio,
)

# # for JGL comparison
# dat1 = dat1[:,:1000]
# dat2 = dat2[:,:1000]
# gene_ens_names = gene_ens_names[:1000]

# import pandas as pd
# df1 = pd.DataFrame(dat1)
# df2 = pd.DataFrame(dat2)
# df1.to_csv("pbmc_rna_ct1_0p25_0p25.csv")
# df2.to_csv("pbmc_rna_ct2_0p25_0p25.csv")

human_tf_idx = pu.get_human_tf_idx(gene_ens_names=gene_ens_names)

# %%

out_iddn = pu.iddn_work_basic(
    dat1,
    dat2,
    tf_idx=human_tf_idx,
    l1_mat=l1_mat,
    l2_mat=l2_mat,
    n_cores=12,
    use_constraints=use_constraints
)

n_gene = dat1.shape[1]
n_edge = np.sum(out_iddn[:,human_tf_idx])/2
n_full_edge = n_gene*len(human_tf_idx)
print(
    f"n1={len(dat1)}, n2={len(dat2)}, n_tf={len(human_tf_idx)}\n", 
    f"n_node={n_gene}, l1={l1_mat}, l2={l2_mat}, constraints={use_constraints}\n",
    f"n_edge={n_edge}, x_sparsity={n_edge/n_full_edge}",
)

# %%
