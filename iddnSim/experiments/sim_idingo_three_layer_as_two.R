# run iDINGO ----
# Treat TFs and miRNas as one layer

library(iDINGO)
library(rhdf5)
library(parallel)

# Settings ----

sim_data_folder <- "../../sim_iddn_data/sim_input/"
sim_out_folder <- "../../sim_iddn_data/sim_output/"
exp_name = "sim3_ggm_three_layer_v2_batch_2024_08_07_22_31_38"

n_rep = 5
n_sample_work = 200  # 100
sigma_add = 0.0  # 0
rho_lst = c(0.15, 0.175, 0.2, 0.225, 0.25)  # Based on JGL best lambda1

# R and Python HDF5 packages use opposite dimension orders
sim_filename = paste(sim_data_folder, exp_name, ".hdf5", sep = "")
print(h5ls(sim_filename))
dat1_lst = h5read(sim_filename, "dat1")
dat2_lst = h5read(sim_filename, "dat2")
dat1_lst <- aperm(dat1_lst, c(3,2,1))
dat2_lst <- aperm(dat2_lst, c(3,2,1))

layer_count = h5read(sim_filename, "layer_count")
n1 = layer_count[1]
n23 = layer_count[2] + layer_count[3]

dat1 = dat1_lst[1,,]
n_sample = dim(dat1)[1]
n_feature = dim(dat1)[2]
plats = c("mRNA", "regu")
plats_dingo = c("regu", "mRNA")

# ----

diffscore_mat_all = array(0.0, c(n_rep, n_feature, n_feature))
R1_mat_all = array(0.0, c(n_rep, n_feature, n_feature))
R2_mat_all = array(0.0, c(n_rep, n_feature, n_feature))

for (n in 1:n_rep) {
  print(n)

  dat1 = dat1_lst[n,,]
  dat2 = dat2_lst[n,,]
  dat1_sel = dat1[1:n_sample_work, 1:n_feature]
  dat2_sel = dat2[1:n_sample_work, 1:n_feature]

  dat_class = c(rep(0,n_sample_work), rep(1,n_sample_work))
  dat12_sel = rbind(dat1_sel, dat2_sel)

  # The dataset has order mRNA, TF, and miRNA
  dat_mRNA = dat12_sel[,1:n1]
  dat_regu = dat12_sel[,(n1+1):(n1+n23)]

  colnames(dat_mRNA) = paste0("mol_", 1:n1)
  colnames(dat_regu) = paste0("mol_", (n1+1):(n1+n23))

  fit <- idingo(dat_regu, dat2 = dat_mRNA,
                x = dat_class, plats = plats_dingo, rhoarray = rho_lst,
                diff.score = TRUE, B = 10, cores = 10)

  genepair = fit$genepair
  diffscore = fit$diff.score
  R1 = fit$R1
  R2 = fit$R2

  molname_mRNA = paste0(plats[[1]], "||", colnames(dat_mRNA))
  molname_regu = paste0(plats[[2]], "||", colnames(dat_regu))

  # The dataset has order mRNA, protein, and miRNA
  molname = c(molname_mRNA, molname_regu)

  n_mol = length(molname)
  molidx = 1:length(molname)
  names(molidx) = molname  # name to idx
  names(molname) = molidx  # idx to name

  diffscore_mat = array(0.0, c(n_mol, n_mol))
  R1_mat = array(0.0, c(n_mol, n_mol))
  R2_mat = array(0.0, c(n_mol, n_mol))

  for (i in 1:dim(genepair)[[1]]) {
    mol1 = genepair[i, 1]
    mol2 = genepair[i, 2]
    idx1 = molidx[[mol1]]
    idx2 = molidx[[mol2]]
    # Differential edges
    s = diffscore[i]
    diffscore_mat[idx1, idx2] = s
    diffscore_mat[idx2, idx1] = s
    # Condition 1 partial correlation
    s = R1[i]
    R1_mat[idx1, idx2] = s
    R1_mat[idx2, idx1] = s
    # Condition 2 partial correlation
    s = R2[i]
    R2_mat[idx1, idx2] = s
    R2_mat[idx2, idx1] = s
  }
  diffscore_mat_all[n,,] = diffscore_mat
  R1_mat_all[n,,] = R1_mat
  R2_mat_all[n,,] = R2_mat
}

# ----

fname = paste(sim_out_folder, exp_name, "_idingo_sample_", n_sample_work,
              "_sigma_", sprintf("%.1f",sigma_add), "_3as2_layer.hdf5", sep = "")
h5createFile(fname)
h5write(diffscore_mat_all, fname, "diffscore")
h5write(R1_mat_all, fname, "R1")
h5write(R2_mat_all, fname, "R2")

