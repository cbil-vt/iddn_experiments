# %%
import pickle
import numpy as np
from scipy.sparse import coo_array
import matplotlib.pyplot as plt

# %%

with open(f"iddn_tf_atac_0.025.pickle", 'rb') as file:
    out = pickle.load(file)

out_iddn = out['out_iddn']
gene_anno = out['gene_anno']
atac_anno = out['atac_anno']
n_gene = len(gene_anno)
n_atac = len(atac_anno)
tf_idx = out['tf_idx']
n_node = n_gene + n_atac

# %%

net_out = np.zeros((2,n_node,n_node), dtype=np.int8)
n_edge = np.zeros(2)
for n in range(n_node):
    for k in range(2):
        x: coo_array = out_iddn[n][k]
        idx = x.coords[0]
        dat = x.data
        idx = idx[np.abs(dat) > 1e-4]
        net_out[k, idx, n] = 1
        # net_out[k, n, idx] = 1
        n_edge[k] += len(idx)

# plt.imshow(net_out[0,::100,::100])

# %%

# n_full0 = len(gene_anno) * len(atac_anno) + len(tf_idx)*len(atac_anno)
n_full = 194520320
print(
    f"n_edge={n_edge}, density={np.mean(n_edge)/n_full}",
)

# %%
# TFs with more differential edges

net_tf1 = net_out[0][tf_idx,n_gene:]
net_tf2 = net_out[1][tf_idx,n_gene:]
tf_sum1 = np.sum(net_tf1, axis=1)
tf_sum2 = np.sum(net_tf2, axis=1)

# %%

net_dif = np.abs(net_tf1 - net_tf2)
tf_dif_sum = np.sum(net_dif, axis=1)
s_idx = np.argsort(-tf_dif_sum)
top_idx = tf_idx[s_idx][:20]
print(top_idx)

# %%

gene_anno.iloc[top_idx]

# %%

plt.plot(np.sort(tf_dif_sum))
