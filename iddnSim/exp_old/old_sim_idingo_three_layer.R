# run iDINGO ----
library(iDINGO)
library(rhdf5)
library(parallel)

# Settings ----

sim_data_folder <- "../../sim_iddn_data/sim_input/"
exp_name = "sim3_ggm_three_layer_batch_849165"

n_rep = 5
n_sample_work = 200  # 100
sigma_add = 0.0  # 0

# R and Python HDF5 packages use opposite dimension orders
sim_filename = paste(sim_data_folder, exp_name, ".hdf5", sep = "")
print(h5ls(sim_filename))
dat1_lst = h5read(sim_filename, "dat1")
dat2_lst = h5read(sim_filename, "dat2")
dat1_lst <- aperm(dat1_lst, c(3,2,1))
dat2_lst <- aperm(dat2_lst, c(3,2,1))

layer_count = h5read(sim_filename, "layer_count")
n1 = layer_count[1]
n2 = layer_count[2]
n3 = layer_count[3]

n_sample = dim(dat1)[1]
n_feature = dim(dat1)[2]
plats = c("mRNA", "TF", "miRNA")

diffscore_mat_all = array(0.0, c(n_rep, n_feature, n_feature))

for (n in 1:n_rep) {
  print(n)

  dat1 = dat1_lst[n,,]
  dat2 = dat2_lst[n,,]
  dat1_sel = dat1[1:n_sample_work, 1:n_feature]
  dat2_sel = dat2[1:n_sample_work, 1:n_feature]

  dat_class = c(rep(0,n_sample_work), rep(1,n_sample_work))
  dat12_sel = rbind(dat1_sel, dat2_sel)

  # The dataset has order mRNA, protein, and miRNA
  dat_mRNA = dat12_sel[,1:n1]
  dat_TF = dat12_sel[,(n1+1):(n1+n2)]
  dat_miRNA = dat12_sel[,(n1+n2+1):(n1+n2+n3)]

  colnames(dat_mRNA) = paste0("mol_", 1:n1)
  colnames(dat_TF) = paste0("mol_", (n1+1):(n1+n2))
  colnames(dat_miRNA) = paste0("mol_", (n1+n2+1):(n1+n2+n3))

  fit <- idingo(dat_miRNA, dat2 = dat_TF, dat3=dat_mRNA,
                x = dat_class, plats = plats,
                diff.score = TRUE, B = 20, cores = 10)

  genepair = fit$genepair
  diffscore = fit$diff.score

  molname_mRNA = paste0(plats[[1]], "||", colnames(dat_mRNA))
  molname_TF = paste0(plats[[2]], "||", colnames(dat_TF))
  molname_miRNA = paste0(plats[[3]], "||", colnames(dat_miRNA))

  # The dataset has order mRNA, protein, and miRNA
  molname = c(molname_mRNA, molname_TF, molname_miRNA)

  n_mol = length(molname)
  molidx = 1:length(molname)
  names(molidx) = molname  # name to idx
  names(molname) = molidx  # idx to name

  diffscore_mat = array(0.0, c(n_mol, n_mol))

  for (i in 1:dim(genepair)[[1]]) {
    mol1 = genepair[i, 1]
    mol2 = genepair[i, 2]
    idx1 = molidx[[mol1]]
    idx2 = molidx[[mol2]]
    s = diffscore[i]
    diffscore_mat[idx1, idx2] = s
    diffscore_mat[idx2, idx1] = s
  }
  diffscore_mat_all[n,,] = diffscore_mat
}

fname = paste("../../sim_iddn_data/sim_output/", exp_name, "_idingo_sample_", n_sample_work,
              "_sigma_", sprintf("%.1f",sigma_add), "_3_layer.hdf5", sep = "")
h5createFile(fname)
h5write(diffscore_mat_all, fname, "diffscore")



