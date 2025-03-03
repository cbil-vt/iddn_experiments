import numpy as np
import pandas as pd
from iddn import iddn
import muon as mu
from scipy.sparse import coo_array
import time

# Cistrome TF names and file index for PBMC
tf_list1 = ["STAT1", "IRF1", "CTCF", "RUNX1", "SPI1"]  # classical monocytes
tf_cis1 = [41287, 41301, 45444, 81223, 85986]
tf_list2 = ["ETS1", "RUNX1", "FOXP3", "REST"]  # naive CD4 T cells
tf_cis2 = [44092, 44097, 44098, 47435]


def load_rna_data(
    file="../../iddn_data/muon/data/pbmc10k.h5mu/rna",
    ct1="CD14 mono",  # 1241
    ct2="CD4+ naïve T",  # 1513
    thr=0.1,
):
    rna = mu.read(file)
    cell_types = rna.obs["celltype"].to_numpy()

    data: np.ndarray = rna.X
    dat1 = data[cell_types == ct1].astype(np.float64)
    dat2 = data[cell_types == ct2].astype(np.float64)

    s1 = np.std(dat1, axis=0)
    s2 = np.std(dat2, axis=0)
    idx_good = (s1 > thr) * (s2 > thr)  # or *
    dat1 = dat1[:, idx_good]
    dat2 = dat2[:, idx_good]
    gene_names = rna.var_names.to_numpy()[idx_good]

    gene_dict = dict()
    for i in range(len(gene_names)):
        gene_dict[gene_names[i]] = i

    return dat1, dat2, gene_names, gene_dict


def get_human_tf_idx(
    gene_ens_names,
    tf_list_file="../../iddn_data/tf_list/TFs_Ensembl_v_1.01.txt",
):
    tf_list = pd.read_csv(tf_list_file).to_numpy().flatten()
    tf_msk = np.isin(gene_ens_names, tf_list)
    tf_idx = np.where(tf_msk > 0)[0]
    return tf_idx


def load_rna_data_for_benchmark(
    file="../../iddn_data/muon/data/pbmc10k.h5mu/rna",
    cell_ratio=0.25,
    gene_ratio=0.5,
    ct1="CD4+ naïve T",  # 1513
    ct2="CD14 mono",  # 1241
    thr=0.1,
):
    rna = mu.read(file)
    cell_types = rna.obs["celltype"].to_numpy()
    # ct3 = "naïve B"  # 331

    data: np.ndarray = rna.X
    if cell_ratio <= 1:
        dat1 = data[cell_types == ct1].astype(np.float64)
        dat2 = data[cell_types == ct2].astype(np.float64)
        n1 = int(len(dat1) * cell_ratio)
        n2 = int(len(dat2) * cell_ratio)
        idx1 = np.random.choice(len(dat1), n1, replace=False)
        idx2 = np.random.choice(len(dat2), n2, replace=False)
        dat1 = dat1[idx1]
        dat2 = dat2[idx2]
    else:
        n1 = int((len(data) - 1) / 2)
        dat1 = data[:n1].astype(np.float64)
        dat2 = data[n1:].astype(np.float64)

    s1 = np.std(dat1, axis=0)
    s2 = np.std(dat2, axis=0)
    idx_good = (s1 > thr) * (s2 > thr)
    dat1 = dat1[:, idx_good]
    dat2 = dat2[:, idx_good]
    gene_ens_names = rna.var["gene_ids"].to_numpy()[idx_good]

    if gene_ratio <= 1:
        m1 = int(dat1.shape[1] * gene_ratio)
        idx1 = np.random.choice(dat1.shape[1], m1, replace=False)
        dat1 = dat1[:, idx1]
        dat2 = dat2[:, idx1]
        gene_ens_names = gene_ens_names[idx1]

    return dat1, dat2, gene_ens_names


def read_cis_genes(cis_folder, tf_list, tf_cis, gene_names, thr=3.0, nn=1000000):
    tgt_dict = dict()
    for i in range(len(tf_list)):
        gene = tf_list[i]
        if not gene in gene_names:
            continue
        cis_idx = tf_cis[i]
        tgt = _read_cis(cis_folder, cis_idx, thr, nn=nn)
        tgt = tgt[np.isin(tgt, gene_names)]
        if len(tgt) > nn:
            tgt = tgt[:nn]
        tgt_dict[gene] = tgt
        print(gene, len(tgt))
    return tgt_dict


def _read_cis(cis_folder, cis_idx, thr, nn):
    file = f"{cis_folder}/{cis_idx}_gene_score_5fold.txt"
    df_chip = pd.read_csv(file, sep="\t", skiprows=5)
    score = df_chip["score"].to_numpy()
    symbol = df_chip["symbol"].to_numpy()
    symbol_filter = symbol[score > thr]
    _, idx = np.unique(symbol_filter, return_index=True)
    tgt_gene = symbol_filter[np.sort(idx)]
    print(len(symbol_filter), len(tgt_gene))
    if len(tgt_gene) > 3 * nn:
        tgt_gene = tgt_gene[: 3 * nn]
    return tgt_gene


def tf_gene_pair_to_adj_mat(gene_dict, tgt_dict):
    n_gene = len(gene_dict)
    net_gt = np.zeros((n_gene, n_gene), dtype=np.int8)
    for key, values in tgt_dict.items():
        i = gene_dict[key]
        for v in values:
            j = gene_dict[v]
            net_gt[i, j] = 1
            net_gt[j, i] = 1
    return net_gt


def iddn_work_basic(
    dat1,
    dat2,
    tf_idx=None,
    l1_mat=0.15,
    l2_mat=0.02,
    n_cores=12,
    use_constraints=True,
):

    t0 = time.time()

    n_gene = dat1.shape[1]
    if use_constraints:
        dep_mat = np.zeros((n_gene, n_gene), dtype=np.int8)
        dep_mat[tf_idx, :] = 1
    else:
        dep_mat = np.ones((n_gene, n_gene), dtype=np.int8)
    np.fill_diagonal(dep_mat, 0)

    out_iddn = iddn.iddn_parallel(
        dat1,
        dat2,
        dep_mat=dep_mat,
        lambda1=l1_mat,
        lambda2=l2_mat,
        n_process=n_cores,
        output_sparse=True,
    )

    net_out = np.zeros((2, n_gene, n_gene), dtype=np.int8)
    for n in range(n_gene):
        for k in range(2):
            x: coo_array = out_iddn[n][k]
            idx = x.coords[0]
            dat = x.data
            idx = idx[np.abs(dat) > 1e-4]
            net_out[k, idx, n] = 1
            net_out[k, n, idx] = 1

    t1 = time.time()
    tdif = t1 - t0
    print(f"Running time: {tdif}")

    return net_out


def split_interval(adata):
    features = pd.DataFrame([s.replace(":", "-", 1).split("-") for s in adata.var.interval])
    features.columns = ["Chromosome", "Start", "End"]
    features["gene_id"] = adata.var.gene_ids.values
    features["gene_name"] = adata.var.index.values
    features.index = adata.var.index
    return features


def get_chr_index(gene_anno):
    gene_chr = gene_anno.Chromosome.to_list()
    gene_chr_idx = np.zeros(len(gene_chr), dtype=int)-1
    for i in range(len(gene_chr_idx)):
        if gene_chr[i][:3] == 'chr':
            chr_idx = gene_chr[i][3:]
            if chr_idx=="X":
                gene_chr_idx[i] = 24
            elif chr_idx=="Y":
                gene_chr_idx[i] = 25
            else:
                gene_chr_idx[i] = int(chr_idx)
    return gene_chr_idx
